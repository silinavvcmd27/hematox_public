# STHELAR 40x -> feat.npz для слоя 1 (только раковые слайды).
# Стримит шарды, фильтрует раковые + качество, id клеток -> tumor/stroma,
# кодирует UNI, кладёт по-слайдовые feat.npz в data/processed/sthelar/.
#   python -u sthelar_extract.py --per-slide 1500
import argparse
import io
import os

import numpy as np
import pandas as pd
import cv2
import scipy.sparse as sp
from huggingface_hub import hf_hub_download

from src.utils import TUMOR, STROMA, get_device, ensure_dir
from seg_extract import build_uni, encode

REPO = "FelicieGS/STHELAR_40x"
OUT_SIZE = 112

CANCER = ["breast_s0", "breast_s1", "breast_s3", "breast_s6", "cervix_s0",
          "colon_s1", "colon_s2", "kidney_s1", "liver_s1", "lung_s1", "lung_s3",
          "ovary_s0", "ovary_s1", "pancreatic_s0", "pancreatic_s1",
          "pancreatic_s2", "prostate_s0", "skin_s2", "skin_s3", "skin_s4"]

# cells_final_label_group -> наш класс; Other/Specialized/None -> 0 (фон/ignore)
LABEL2CLS = {
    "Epithelial": TUMOR,
    "Fibroblast_Myofibroblast": STROMA,
    "Blood_vessel": STROMA,
    "T_NK": STROMA,
    "B_Plasma": STROMA,
    "Myeloid": STROMA,
}


def build_lut(slide):
    p = hf_hub_download(REPO, f"cell_metadata/{slide}_cell_metadata.parquet",
                        repo_type="dataset")
    df = pd.read_parquet(p, columns=["cell_id_int", "cells_final_label_group"])
    cls = df["cells_final_label_group"].map(LABEL2CLS).fillna(0).astype(np.uint8)
    lut = np.zeros(int(df["cell_id_int"].max()) + 2, np.uint8)   # 0 -> фон
    lut[df["cell_id_int"].to_numpy()] = cls.to_numpy()
    return lut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/processed/sthelar")
    ap.add_argument("--per-slide", type=int, default=1500)
    ap.add_argument("--min-bpq", type=float, default=0.4)
    ap.add_argument("--min-tissue", type=float, default=0.1)
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()

    out = ensure_dir(args.out_dir)
    print("строю LUT меток по слайдам...")
    lut = {s: build_lut(s) for s in CANCER}

    device = get_device()
    print("device:", device, "| грузим UNI...")
    uni = build_uni(device)

    from datasets import load_dataset
    ds = load_dataset(REPO, split="train", streaming=True)

    buf_img = {s: [] for s in CANCER}
    buf_msk = {s: [] for s in CANCER}
    done = set()

    def save_slide(s):
        if not buf_img[s]:
            done.add(s)
            return
        X = encode(uni, buf_img[s], device, bs=args.bs)
        y = np.stack(buf_msk[s]).astype(np.uint8)
        np.savez_compressed(out / f"{s}_feat.npz", X=X, y=y)
        print(f"  сохранён {s}: X={X.shape} y={y.shape} "
              f"tum={int((y==TUMOR).sum())} str={int((y==STROMA).sum())}")
        buf_img[s].clear()
        buf_msk[s].clear()
        done.add(s)

    seen = 0
    for ex in ds:
        s = ex["slide_id"]
        if s not in lut or s in done:
            continue
        if len(buf_img[s]) >= args.per_slide:
            save_slide(s)
            if len(done) == len(CANCER):
                break
            continue
        if ex["bPQ"] < args.min_bpq:
            continue
        ids = sp.load_npz(io.BytesIO(ex["cell_id_map"])).toarray()
        ids = np.clip(ids, 0, len(lut[s]) - 1)
        dense = lut[s][ids]
        if (dense > 0).mean() < args.min_tissue:
            continue
        buf_img[s].append(np.asarray(ex["image"]))
        buf_msk[s].append(cv2.resize(dense, (OUT_SIZE, OUT_SIZE),
                                     interpolation=cv2.INTER_NEAREST))
        seen += 1
        if seen % 1000 == 0:
            print("  обработано %d | %s" %
                  (seen, {k: len(v) for k, v in buf_img.items() if v}))

    for s in CANCER:          # хвосты, не добравшие до cap
        if s not in done:
            save_slide(s)

    print("готово. feat.npz в", out)
    os._exit(0)               # обойти сегфолт datasets+torch при финализации


if __name__ == "__main__":
    main()