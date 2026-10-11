# Кривая «число графовых слоёв против качества» из уже обученных чекпоинтов.
#
# Каждый чекпоинт помнит, какой срез был отложен, поэтому метрику можно пересчитать
# без повторного обучения: в файле лежит лучшее по валидации состояние.
# Рядом с числом прыжков печатается фактический охват в микрометрах, чтобы у оси
# был физический смысл.
#
#   python -u eval_hops.py --glob 'runs/graph/l1_h*.pth'

import argparse
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from src.utils import get_device, ensure_dir
from cell_graph_train import CellGNN, adjacency, to_binary, metrics, CLASSES


def hop_reach(xy, edges, mpp, max_hops, n_sample=500, seed=0):
    """Медианный радиус окрестности за 1..max_hops прыжков, в микрометрах."""
    n = len(xy)
    a = sp.csr_matrix((np.ones(edges.shape[1], np.int8), (edges[0], edges[1])), shape=(n, n))
    rng = np.random.default_rng(seed)
    out = {}
    for h in range(1, max_hops + 1):
        far = []
        for i in rng.choice(n, size=min(n_sample, n), replace=False):
            front = np.array([i])
            seen = front
            for _ in range(h):
                front = np.unique(np.concatenate(
                    [a.indices[a.indptr[j]:a.indptr[j + 1]] for j in front] or [np.array([], int)]))
                if not len(front):
                    break
                seen = np.union1d(seen, front)
            far.append(np.linalg.norm(xy[seen] - xy[i], axis=1).max())
        out[h] = float(np.median(far) * mpp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/graph/l1_h*.pth")
    ap.add_argument("--graph-dir", default="data/processed/graph")
    ap.add_argument("--out", default="outputs/results/hops_curve.csv")
    args = ap.parse_args()

    paths = sorted(glob(args.glob))
    if not paths:
        raise SystemExit(f"ничего не нашлось по {args.glob}")
    device = get_device()

    graphs, rows = {}, []
    for p in paths:
        ck = torch.load(p, map_location=device)
        slides = list(ck.get("val") or [])
        if len(slides) != 1:
            print(f"пропускаю {p}: в чекпоинте val={slides}")
            continue
        slide = slides[0]
        if slide not in graphs:
            g = np.load(Path(args.graph_dir) / f"{slide}_graph.npz")
            graphs[slide] = (torch.from_numpy(g["feat"].astype(np.float32)).to(device),
                             to_binary(g["label"]), g["edges"], g["pos"].astype(float),
                             float(g["mpp"]))
        feat, truth, edges, xy, mpp = graphs[slide]

        model = CellGNN(int(ck["in_dim"]), int(ck["hid"]), len(ck["classes"]), int(ck["layers"])).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        with torch.no_grad():
            pred = model(feat, adjacency(edges.astype(np.int64), len(truth), device)).argmax(1).cpu().numpy()
        m = metrics(pred, truth)
        rows.append(dict(ckpt=Path(p).name, val=slide, hops=int(ck["layers"]),
                         miou=float(np.nanmean([m[c][0] for c in CLASSES])),
                         **{f"{c}_{k}": m[c][i] for c in CLASSES for i, k in enumerate(("iou", "dice"))}))
        print("%-26s val %-16s прыжков %d  mIoU %.3f  %s"
              % (Path(p).name, slide, ck["layers"], rows[-1]["miou"],
                 "  ".join("%s Dice %.3f" % (c, m[c][1]) for c in CLASSES)))

    df = pd.DataFrame(rows).sort_values(["hops", "val"])
    ensure_dir(Path(args.out).parent)
    df.to_csv(args.out, index=False)

    max_hops = int(df["hops"].max())
    reach = {s: hop_reach(v[3], v[2], v[4], max(max_hops, 1)) for s, v in graphs.items()}
    print("\n=== кривая по числу прыжков (среднее по фолдам) ===")
    avg = df.groupby("hops")[["miou", "tumor_dice", "stroma_dice"]].mean().round(3)
    avg["охват, мкм"] = [0.0 if h == 0 else round(np.mean([reach[s][h] for s in reach]), 1)
                         for h in avg.index]
    print(avg.to_string())
    print("\nтаблица:", args.out)


if __name__ == "__main__":
    main()
