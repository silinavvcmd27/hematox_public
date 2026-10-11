# STHELAR (срезы Xenium от 10x, одиннадцать тканей) в наш формат признаков.
#
# В наборе две дорожки разметки, и они расходятся. Клеточная (cells_*) считает опухолью
# весь эпителий, включая нормальный, и в чистом виде не годится: именно эта ошибка у
# двухклассовой модели отправляла 73 % нормальной слизистой кишки в опухоль. Ядерная
# (nuclei_*) различает раковые и нормальные клетки: в лёгком это Cancerous_lung_epithelial_cell
# против Lung_epithelial_cell, в простате два раковых подтипа выделены отдельно.
#
# Метка ставится только там, где обе дорожки согласны, остальное идёт в игнор:
#
#   обе говорят «эпителий», ядерная «раковая»      опухоль
#   обе говорят «эпителий», ядерная «нормальная»   прочее (паренхима органа)
#   обе относят клетку к стромальным группам       строма
#   расхождение, less10, Other                     игнор
#
# Пиксель без метки в функцию потерь не входит и ничего не стоит, а неверная метка стоит
# дорого, поэтому требование согласия строже, чем приоритет одной дорожки над другой.
#
# Патч набора при 40x покрывает 64 мкм против наших 70. Изображение не растягивается:
# поле просто немного меньше, и токен покрывает 4,6 мкм вместо 5,0. Это оговаривается в
# тексте; интерполяция была бы хуже.
#
# Все срезы складываются в один источник, иначе при уравнивании по 4000 патчей за эпоху
# двадцать срезов задавили бы остальные наборы.
#
#   python -u pan_prep_sthelar.py --slides lung_s1 colon_s1 --per-slide 150 --name sthelar_proba
#   python -u pan_prep_sthelar.py --per-slide 800

import argparse
import io
import os

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from PIL import Image
from huggingface_hub import hf_hub_download

from src.utils import get_device, ensure_dir, TUMOR, STROMA
from src.stain import Normalizer, StainStats, fit_stats, sample_pixels
from pan_train import OTHER
from seg_extract import build_uni, encode, OUT

REPO = "FelicieGS/STHELAR_40x"
STROMA_GROUPS = ("Fibroblast_Myofibroblast", "Blood_vessel", "T_NK", "B_Plasma", "Myeloid")
CANCER = ["breast_s0", "breast_s1", "breast_s3", "breast_s6", "cervix_s0",
          "colon_s1", "colon_s2", "kidney_s1", "liver_s1", "lung_s1", "lung_s3",
          "ovary_s0", "ovary_s1", "pancreatic_s0", "pancreatic_s1",
          "pancreatic_s2", "prostate_s0", "skin_s2", "skin_s3", "skin_s4"]
NAMES = {TUMOR: "опухоль", STROMA: "строма", OTHER: "прочее", 0: "игнор"}


def build_lut(slide, report):
    """cell_id_int -> наш класс.

    Строма и нормальная паренхима берутся по согласию двух дорожек разметки. Опухоль
    берётся по ядерной дорожке независимо от клеточной, иначе теряются неэпителиальные
    опухоли: у меланомы клеточная дорожка относит меланоциты к специализированным
    клеткам, и требование согласия выбрасывало всю опухоль кожи. Защита от главной
    ошибки при этом сохраняется, потому что опухоль определяется разделением раковых и
    нормальных клеток, а не принадлежностью к эпителию.
    """
    path = hf_hub_download(REPO, f"cell_metadata/{slide}_cell_metadata.parquet",
                           repo_type="dataset")
    df = pd.read_parquet(path, columns=["cell_id_int", "nuclei_label3", "nuclei_label2",
                                        "nuclei_final_label_group", "cells_final_label_group"])
    epithelial = ((df.cells_final_label_group == "Epithelial")
                  & (df.nuclei_final_label_group == "Epithelial")).to_numpy()
    stromal = (df.cells_final_label_group.isin(STROMA_GROUPS)
               & df.nuclei_final_label_group.isin(STROMA_GROUPS)).to_numpy()
    cancer = (df.nuclei_label3 == "cancer_cell").to_numpy()
    normal = (df.nuclei_label3 == "normal_cell").to_numpy()

    cls = np.zeros(len(df), np.uint8)
    cls[stromal] = STROMA
    cls[epithelial & normal] = OTHER
    cls[cancer & ~stromal] = TUMOR

    share = pd.Series(cls).map(NAMES).value_counts(normalize=True)
    print(f"  {slide}: клеток {len(df)}, " +
          ", ".join(f"{k} {v:.0%}" for k, v in share.items()))
    if report:
        named = df.assign(класс=pd.Series(cls).map(NAMES).to_numpy())
        print(named.groupby(["класс", "nuclei_label2"]).size()
                   .sort_values(ascending=False).head(14).to_string())

    lut = np.zeros(int(df.cell_id_int.max()) + 2, np.uint8)
    lut[df.cell_id_int.to_numpy()] = cls
    return lut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", nargs="+", default=None, help="по умолчанию все раковые")
    ap.add_argument("--exclude", nargs="+", default=["ovary_s1"],
                    help="ovary_s1 это наш собственный ovary_prime")
    ap.add_argument("--name", default="sthelar_pan")
    ap.add_argument("--per-slide", type=int, default=800)
    ap.add_argument("--min-bpq", type=float, default=0.4, help="качество сегментации ядер")
    ap.add_argument("--min-label-frac", type=float, default=0.10)
    ap.add_argument("--stain-ref", default="data/processed/stain_ref_prime.npz")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out-dir", default="data/processed/seg_pan")
    args = ap.parse_args()

    slides = [s for s in (args.slides or CANCER) if s not in args.exclude]
    print(f"{args.name}: срезов {len(slides)}, до {args.per_slide} патчей с каждого\n")
    lut = {s: build_lut(s, i == 0) for i, s in enumerate(slides)}

    from datasets import load_dataset
    ds = load_dataset(REPO, split="train", streaming=True)
    buf_img = {s: [] for s in slides}
    buf_msk = {s: [] for s in slides}
    tissue, full = {}, set()

    for ex in ds:
        s = ex["slide_id"]
        if s not in lut or s in full:
            continue
        if len(buf_img[s]) >= args.per_slide:
            full.add(s)
            print(f"  набран {s} ({len(full)} из {len(slides)})")
            if len(full) == len(slides):
                break
            continue
        if ex["bPQ"] < args.min_bpq:
            continue
        ids = sp.load_npz(io.BytesIO(ex["cell_id_map"])).toarray()
        mask = lut[s][np.clip(ids, 0, len(lut[s]) - 1)]
        if (mask > 0).mean() < args.min_label_frac:
            continue
        tissue[s] = ex.get("tissue", s.split("_")[0])
        buf_img[s].append(np.asarray(ex["image"].convert("RGB")))
        buf_msk[s].append(np.asarray(Image.fromarray(mask).resize((OUT, OUT), Image.NEAREST)))

    ref = StainStats.load(args.stain_ref)
    tiles, masks, rows = [], [], []
    for s in slides:
        if not buf_img[s]:
            print(f"  {s}: ни одного годного патча, пропускаю")
            continue
        try:
            stats = fit_stats(sample_pixels(np.concatenate([t.reshape(-1, 3) for t in buf_img[s]])))
            norm = Normalizer(stats, ref)
        except ValueError as err:
            print(f"  {s}: окраска не оценилась ({err}), оставляю как есть")
            norm = None
        for img, m in zip(buf_img[s], buf_msk[s]):
            tiles.append(norm(img) if norm else img)
            masks.append(m)
            rows.append({"roi": s, "slide": s, "patient": s, "source": "sthelar",
                         "scanner": "xenium", "organ": tissue.get(s, s.split("_")[0]),
                         "tumor_frac": round(float((m == TUMOR).mean()), 3),
                         "stroma_frac": round(float((m == STROMA).mean()), 3),
                         "other_frac": round(float((m == OTHER).mean()), 3)})
        buf_img[s].clear()
        buf_msk[s].clear()

    y = np.stack(masks)
    labeled = y > 0
    print(f"\nпатчей {len(tiles)}, размечено {labeled.mean():.1%} пикселей; "
          f"среди размеченного опухоль {(y[labeled] == TUMOR).mean():.1%}, "
          f"строма {(y[labeled] == STROMA).mean():.1%}, прочее {(y[labeled] == OTHER).mean():.1%}")
    print(pd.DataFrame(rows).organ.value_counts().to_string())

    device = get_device()
    print("device:", device, "| кодируем через UNI...")
    X = encode(build_uni(device), tiles, device, bs=args.bs)

    out = ensure_dir(args.out_dir)
    np.savez_compressed(out / f"{args.name}_feat.npz", X=X, y=y, patch_px=256,
                        patch_um=64.0, stain_ref=str(args.stain_ref))
    pd.DataFrame(rows).to_csv(out / f"{args.name}_patches.csv", index=False)
    print(f"\nсохранено: {out / f'{args.name}_feat.npz'}  X={X.shape} y={y.shape}")
    os._exit(0)          # datasets и torch ссорятся при завершении процесса


if __name__ == "__main__":
    main()
