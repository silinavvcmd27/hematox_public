# Сравнение патчевой и графовой моделей в одних единицах — по клеткам.
#
# Патчевая модель мерилась по пикселям, графовая по клеткам, поэтому напрямую
# их числа не сопоставимы. Здесь предсказание патчевой модели берётся ровно в
# точках тех же клеток, что и узлы графа.
#
#   python compare_cell_level.py --slide ovary2_he --model runs/layer1/vor_noovary3.pth

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.utils import get_device, TUMOR, CLASS_NAMES
from seg_decoder import SegDecoder

GRID, OUT = 14, 112
STROMA_LIKE = (2, 3, 4, 5)
CLASSES = ("tumor", "stroma")


def to_binary(label):
    y = np.full(len(label), -100, np.int64)
    y[label == TUMOR] = 0
    y[np.isin(label, STROMA_LIKE)] = 1
    return y


def iou_dice(pred, truth):
    out = {}
    ok = truth != -100
    for ch, nm in enumerate(CLASSES):
        p = (pred == ch) & ok
        t = (truth == ch) & ok
        inter = np.logical_and(p, t).sum()
        union = np.logical_or(p, t).sum()
        den = p.sum() + t.sum()
        out[nm] = (inter / union if union else float("nan"),
                   2 * inter / den if den else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seg-dir", default="data/processed/seg")
    ap.add_argument("--graph-dir", default="data/processed/graph")
    ap.add_argument("--max-patches", type=int, default=12000)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()

    g = np.load(Path(args.graph_dir) / f"{args.slide}_graph.npz")
    pos_cells, label = g["pos"], g["label"]
    truth = to_binary(label)
    print(f"{args.slide}: клеток {len(label)}")

    man = pd.read_csv(Path(args.seg_dir) / f"{args.slide}_patches.csv")
    if args.max_patches and len(man) > args.max_patches:
        man = man.sample(args.max_patches, random_state=42).reset_index(drop=True)
    pos = man[["x0", "y0"]].to_numpy(np.int64)
    row_of = {(int(a), int(b)): i for i, (a, b) in enumerate(pos)}

    device = get_device()
    ck = torch.load(args.model, map_location=device)
    dec = SegDecoder(in_dim=1024, n_classes=int(ck["n_classes"])).to(device)
    dec.load_state_dict(ck["state_dict"])
    dec.eval()
    ch2cls = {int(k): int(v) for k, v in ck["channel_to_class"].items()}
    print("модель:", args.model, "| каналы:",
          {c: CLASS_NAMES[v] for c, v in ch2cls.items()})

    X = np.load(Path(args.seg_dir) / f"{args.slide}_feat.npz")["X"]
    maps = np.empty((len(X), OUT, OUT), np.uint8)
    with torch.no_grad():
        for i in range(0, len(X), args.bs):
            xb = torch.tensor(X[i:i + args.bs].astype(np.float32)).to(device)
            maps[i:i + args.bs] = dec(xb).argmax(1).cpu().numpy().astype(np.uint8)
    del X
    print("предсказания по патчам готовы")

    ps = args.patch_size
    cx, cy = pos_cells[:, 0].astype(float), pos_cells[:, 1].astype(float)
    px = (cx // ps).astype(np.int64) * ps
    py = (cy // ps).astype(np.int64) * ps
    rows = np.array([row_of.get((int(a), int(b)), -1)
                     for a, b in zip(px.tolist(), py.tolist())], np.int64)
    ix = np.clip(((cx - px) / ps * OUT).astype(int), 0, OUT - 1)
    iy = np.clip(((cy - py) / ps * OUT).astype(int), 0, OUT - 1)

    ok = rows >= 0
    print(f"клеток внутри патчей: {ok.sum()} ({100*ok.mean():.1f}%)")
    chan = maps[rows[ok], iy[ok], ix[ok]]
    cls = np.array([ch2cls[int(c)] for c in range(len(ch2cls))])[chan]
    pred = np.where(cls == TUMOR, 0, 1)

    m = iou_dice(pred, truth[ok])
    miou = float(np.nanmean([m[c][0] for c in CLASSES]))
    print(f"\n=== патчевая модель, метрика по клеткам ({args.slide}) ===")
    for c in CLASSES:
        print("  %-8s IoU %.3f  Dice %.3f" % (c, m[c][0], m[c][1]))
    print("  mIoU %.3f" % miou)


if __name__ == "__main__":
    main()
