# Готовим обучающие .npz из STHELAR_20x (открытый датасет, CC-BY-4.0).
# Берём готовый patches_overview (состав типов клеток на патч) -> метка tumor/stroma/undefined,
# стримим H&E-патчи нужных срезов, кодируем через UNI.
#
# по умолчанию - ovarian срезы (быстрая проверка). для полной базы: --tissue all
#
# pip install -U datasets pyarrow pandas
# python prep_sthelar.py                 # ovary_s0, ovary_s1
# python prep_sthelar.py --tissue all    # все раковые срезы (тяжело, ~17GB)
# потом: python -m src.train

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_config, get_device, hf_login, ensure_dir, CLASS_TO_IDX, CLASS_NAMES
from src.models.uni_encoder import build_encoder

REPO = "FelicieGS/STHELAR_20x"
OVERVIEW = "patches_overview_sthelar20x.parquet"

# колонки с числом клеток каждого типа в патче
CT_COLS = ["T_NK", "B_Plasma", "Myeloid", "Blood_vessel", "Fibroblast_Myofibroblast",
           "Epithelial", "Specialized", "Melanocyte", "Other"]

# STHELAR cells_final_label_group -> наши 3 класса (для раковых срезов Epithelial = опухоль)
CT_TO_CLASS = {
    "Epithelial": "tumor",
    "Blood_vessel": "stroma",
    "Fibroblast_Myofibroblast": "stroma",
    "Myeloid": "stroma",
    "B_Plasma": "stroma",
    "T_NK": "stroma",
    "Melanocyte": "undefined",
    "Specialized": "undefined",
    "Other": "undefined",
}

# срезы с раком (из карточки датасета)
CANCER_SLIDES = ['breast_s0', 'breast_s1', 'breast_s3', 'breast_s6', 'cervix_s0',
                 'colon_s1', 'colon_s2', 'kidney_s1', 'liver_s1', 'lung_s1', 'lung_s3',
                 'ovary_s0', 'ovary_s1', 'pancreatic_s0', 'pancreatic_s1', 'pancreatic_s2',
                 'prostate_s0', 'skin_s2', 'skin_s3', 'skin_s4']


def patch_label(row, threshold, min_cells):
    # суммируем клетки по нашим классам, берём мажоритарный
    agg = {"tumor": 0, "stroma": 0, "undefined": 0}
    for col in CT_COLS:
        agg[CT_TO_CLASS[col]] += int(row[col])
    n = sum(agg.values())
    if n < min_cells:
        return None, n, 0.0
    top = max(agg, key=agg.get)
    purity = agg[top] / n
    return (top if purity >= threshold else "undefined"), n, purity


def build_label_table(overview_path, slides, threshold, min_cells, jaccard_min):
    df = pd.read_parquet(overview_path)
    df = df[df["slide_id"].isin(slides)].copy()
    if jaccard_min > 0 and "Jaccard" in df.columns:
        df = df[df["Jaccard"].astype(float) >= jaccard_min]
    rows = {}
    for r in df.itertuples(index=False):
        rd = r._asdict()
        lab, n, pur = patch_label(rd, threshold, min_cells)
        if lab is None:
            continue
        cx = (rd["xmin"] + rd["xmax"]) / 2.0
        cy = (rd["ymin"] + rd["ymax"]) / 2.0
        rows[rd["file_name"]] = (rd["slide_id"], lab, cx, cy)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--tissue", default="ovarian",
                    help="ovarian | all (все раковые) | имя ткани")
    ap.add_argument("--slides", nargs="*", help="явный список slide_id (перебивает --tissue)")
    ap.add_argument("--max-patches", type=int, default=4000, help="лимит патчей на срез (0 = без лимита)")
    ap.add_argument("--min-cells", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--jaccard-min", type=float, default=0.0, help="фильтр качества масок")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.slides:
        slides = args.slides
    elif args.tissue.lower() == "all":
        slides = CANCER_SLIDES
    elif args.tissue.lower() == "ovarian":
        slides = ["ovary_s0", "ovary_s1"]
    else:
        slides = [s for s in CANCER_SLIDES if s.startswith(args.tissue.lower())]
    if not slides:
        raise SystemExit(f"не выбрано ни одного среза для tissue={args.tissue}")
    print("срезы:", slides)

    from huggingface_hub import hf_hub_download
    print("качаю overview-таблицу ...")
    ov = hf_hub_download(REPO, OVERVIEW, repo_type="dataset")
    labels = build_label_table(ov, slides, args.threshold, args.min_cells, args.jaccard_min)
    print(f"патчей с метками в overview: {len(labels)}")
    dist = pd.Series([v[1] for v in labels.values()]).value_counts().to_dict()
    print("распределение по меткам:", dist)

    # энкодер
    hf_login(cfg["encoder"]["hf_token_env"])
    device = get_device()
    print("device:", device)
    enc = build_encoder(cfg, device)

    # стримим патчи, собираем по срезам
    from datasets import load_dataset, Image
    ds = load_dataset(REPO, data_files={"train": "data/train-*.parquet"},
                      split="train", streaming=True).cast_column("image", Image(decode=True))

    want = {s: (args.max_patches if args.max_patches > 0 else 10**9) for s in slides}
    buf = {s: {"img": [], "y": [], "coords": [], "pid": []} for s in slides}

    seen = 0
    for ex in ds:
        seen += 1
        fn = ex["file_name"]
        meta = labels.get(fn)
        if meta is None:
            continue
        sid, lab, cx, cy = meta
        if want.get(sid, 0) <= 0:
            continue
        buf[sid]["img"].append(np.asarray(ex["image"], dtype=np.uint8))
        buf[sid]["y"].append(CLASS_TO_IDX[lab])
        buf[sid]["coords"].append([cx, cy])
        buf[sid]["pid"].append(fn)
        want[sid] -= 1
        if seen % 20000 == 0:
            left = {s: want[s] for s in slides if want[s] > 0}
            print(f"  просмотрено {seen}, осталось набрать: {left}")
        if all(v <= 0 for v in want.values()):
            break

    out_dir = ensure_dir(Path(cfg["paths"]["processed_dir"]) / "embeddings")
    for sid in slides:
        b = buf[sid]
        if not b["img"]:
            print(f"  {sid}: 0 патчей, пропускаю")
            continue
        X = enc.encode_patches(b["img"], cfg["encoder"]["batch_size"])
        out = out_dir / f"sthelar_{sid}.npz"
        np.savez_compressed(out, X=X.astype(np.float32),
                            y=np.array(b["y"], np.int64),
                            coords=np.array(b["coords"], float).astype(int),
                            patch_id=np.array(b["pid"]))
        u, c = np.unique(b["y"], return_counts=True)
        print(f"  {sid}: {len(b['y'])} патчей ->",
              {CLASS_NAMES[i]: int(n) for i, n in zip(u, c)}, "->", out.name)

    print("\nготово. теперь: python -m src.train")


if __name__ == "__main__":
    main()
