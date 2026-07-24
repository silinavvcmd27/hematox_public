# Применяем патчевую модель tumor/stroma к TCGA-OV WSI (.svs) и считаем TSP на срез.
# Быстро: на каждом срезе классифицируем выборку тканевых патчей (не весь слайд).
#
# нужен openslide: conda install -c conda-forge openslide-python
#
# python tcga_tsp.py --slides-dir data/tcga_ov --model outputs/models/ft_val_ov3.pth \
#   --out outputs/results/tcga_ov_tsp.csv

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.utils import load_config, get_device, hf_login, ensure_dir
from src.models.uni_encoder import build_encoder
from src.models.graph_head import build_head
from src.build_graph import build_edge_index

NAMES = ["tumor", "stroma"]   # как в train_binary: 0=tumor, 1=stroma


def sample_patches(path, target_mpp=0.5, ps=256, max_patches=1500, white=220):
    import openslide
    sl = openslide.OpenSlide(path)
    mpp = float(sl.properties.get("openslide.mpp-x", 0.5) or 0.5)
    ds = max(1.0, target_mpp / mpp)
    level = sl.get_best_level_for_downsample(ds)
    ldown = sl.level_downsamples[level]
    Wl, Hl = sl.level_dimensions[level]
    pos = [(x, y) for y in range(0, Hl - ps, ps) for x in range(0, Wl - ps, ps)]
    import random
    random.seed(0); random.shuffle(pos)
    patches, coords = [], []
    for x, y in pos:
        reg = np.asarray(sl.read_region((int(x * ldown), int(y * ldown)), level,
                                        (ps, ps)).convert("RGB"))
        if (reg.mean(-1) > white).mean() > 0.85:
            continue
        patches.append(reg); coords.append((x, y))
        if len(patches) >= max_patches:
            break
    sl.close()
    return patches, np.array(coords, float)


def patient_id(fname):
    m = re.search(r"TCGA-\w\w-\w{4}", fname)
    return m.group(0) if m else Path(fname).stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--max-patches", type=int, default=1500)
    ap.add_argument("--out", default="outputs/results/tcga_ov_tsp.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    hf_login(cfg["encoder"]["hf_token_env"])
    device = get_device()
    print("device:", device)
    enc = build_encoder(cfg, device)
    model = build_head(dict(cfg["train"]), cfg["encoder"]["embed_dim"], 2).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device)["state_dict"])
    model.eval()

    files = sorted(glob.glob(os.path.join(args.slides_dir, "*.svs")))
    print("срезов:", len(files))
    rows = []
    for i, f in enumerate(files, 1):
        try:
            patches, coords = sample_patches(f, max_patches=args.max_patches)
        except Exception as e:
            print(f"  [{i}] {os.path.basename(f)}: ошибка {e}"); continue
        if len(patches) < 50:
            print(f"  [{i}] {os.path.basename(f)}: мало ткани, пропуск"); continue
        X = enc.encode_patches(patches, cfg["encoder"]["batch_size"])
        edge = build_edge_index(coords, cfg["graph"]["mode"], cfg["graph"]["k"])
        with torch.no_grad():
            pred = model(torch.tensor(X, dtype=torch.float32, device=device),
                         edge.to(device)).argmax(1).cpu().numpy()
        nt = int((pred == 0).sum()); ns = int((pred == 1).sum())
        tsp = ns / (nt + ns) if (nt + ns) else float("nan")
        pid = patient_id(os.path.basename(f))
        rows.append({"slide": os.path.basename(f), "patient": pid,
                     "n_patch": len(patches), "tumor": nt, "stroma": ns,
                     "TSP_stroma": round(tsp, 4)})
        print(f"  [{i}/{len(files)}] {pid}: TSP стромы {tsp:.3f} ({len(patches)} патчей)")

    df = pd.DataFrame(rows)
    ensure_dir(Path(args.out).parent)
    df.to_csv(args.out, index=False)
    print("\nсохранено:", args.out, " срезов:", len(df))
    if len(df):
        print("медиана TSP:", round(df["TSP_stroma"].median(), 3),
              "| доля высокострома (>0.5):", round((df["TSP_stroma"] > 0.5).mean(), 3))


if __name__ == "__main__":
    main()
