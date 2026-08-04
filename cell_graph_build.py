# Граф клеток: узел = клетка, признак = токен UNI в её позиции, рёбра = соседство.
#
# Прогонять UNI отдельно для каждой клетки нельзя (сотни тысяч клеток на слайд),
# но сетка токенов и так мельче клетки: патч 256 px делится на 14x14, то есть
# ~5 мкм на токен. Поэтому признак клетки берётся из уже посчитанного токена,
# в который она попадает.
#
#   python cell_graph_build.py --slide ovary_prime_he \
#     --cells data/seurat_csv/ovary_prime_he_cells_aligned.csv

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.utils import ensure_dir, TRAIN_CLASSES, CLASS_NAMES
from src.data.seurat_labels import CellTypeMapper

GRID = 14


def replay_positions(seg_dir, slide, max_patches, seed=42):
    man = pd.read_csv(Path(seg_dir) / f"{slide}_patches.csv")
    if max_patches and len(man) > max_patches:
        man = man.sample(max_patches, random_state=seed).reset_index(drop=True)
    return man[["x0", "y0"]].to_numpy(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--seg-dir", default="data/processed/seg")
    ap.add_argument("--map", default="config/cell_type_map.yaml")
    ap.add_argument("--max-patches", type=int, default=12000)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--max-edge-um", type=float, default=50.0)
    ap.add_argument("--mpp", type=float, default=None,
                    help="мкм/px; если не задан, берётся из slide_mpp")
    ap.add_argument("--he", default=None, help="нужен только для авто-mpp")
    ap.add_argument("--out-dir", default="data/processed/graph")
    args = ap.parse_args()

    ps = args.patch_size
    tok = ps / GRID

    df = pd.read_csv(args.cells)
    if "cell_type" not in df.columns:
        raise SystemExit(f"нет колонки cell_type; есть: {list(df.columns)}")
    mapper = CellTypeMapper(args.map)
    df["zone"] = mapper.map_series(df["cell_type"])
    name2cls = {CLASS_NAMES[c]: c for c in TRAIN_CLASSES}
    df["cls"] = df["zone"].map(name2cls)
    df = df.dropna(subset=["cls", "x", "y"]).copy()
    df["cls"] = df["cls"].astype(int)
    print(f"{args.slide}: клеток с классом {len(df)} "
          f"{df['zone'].value_counts().to_dict()}")

    pos = replay_positions(args.seg_dir, args.slide, args.max_patches)
    row_of = {(int(a), int(b)): i for i, (a, b) in enumerate(pos)}
    print(f"патчей в выборке: {len(pos)}")

    cx = df["x"].to_numpy(float)
    cy = df["y"].to_numpy(float)
    px = (cx // ps).astype(np.int64) * ps
    py = (cy // ps).astype(np.int64) * ps
    rows = np.full(len(df), -1, np.int64)
    for i, key in enumerate(zip(px.tolist(), py.tolist())):
        r = row_of.get(key)
        if r is not None:
            rows[i] = r
    keep = rows >= 0
    print(f"клеток внутри выбранных патчей: {keep.sum()} "
          f"({100*keep.mean():.1f}%)")
    if keep.sum() == 0:
        raise SystemExit("ни одна клетка не попала в патчи — проверьте выравнивание")

    df = df[keep].reset_index(drop=True)
    rows, cx, cy = rows[keep], cx[keep], cy[keep]
    jx = np.clip(((cx - (cx // ps).astype(np.int64) * ps) / tok).astype(int), 0, GRID - 1)
    iy = np.clip(((cy - (cy // ps).astype(np.int64) * ps) / tok).astype(int), 0, GRID - 1)
    tflat = iy * GRID + jx

    print("читаю признаки...")
    X = np.load(Path(args.seg_dir) / f"{args.slide}_feat.npz")["X"]
    if X.shape[1] != GRID * GRID:
        raise SystemExit(f"ожидалось {GRID*GRID} токенов, в файле {X.shape[1]}")
    feat = np.empty((len(rows), X.shape[-1]), np.float16)
    step = 200000
    for a in range(0, len(rows), step):
        b = min(a + step, len(rows))
        feat[a:b] = X[rows[a:b], tflat[a:b]]
    del X

    if args.mpp:
        mpp = args.mpp
    elif args.he:
        from slide_mpp import slide_mpp
        mpp = slide_mpp(args.he)
    else:
        mpp = 0.2738
        print("mpp не задан, беру 0.2738 (типичный для Xenium H&E)")
    r_px = args.max_edge_um / mpp

    print(f"строю граф: k={args.knn}, радиус {args.max_edge_um:.0f} мкм = {r_px:.0f} px")
    xy = np.c_[cx, cy]
    tree = cKDTree(xy)
    d, nb = tree.query(xy, k=args.knn + 1)
    d, nb = d[:, 1:], nb[:, 1:]                     # без самой себя
    ok = d <= r_px
    src = np.repeat(np.arange(len(xy)), args.knn)[ok.ravel()]
    dst = nb.ravel()[ok.ravel()]
    edges = np.vstack([np.r_[src, dst], np.r_[dst, src]]).astype(np.int32)  # неориентированный
    deg = np.bincount(edges[0], minlength=len(xy))
    print(f"рёбер {edges.shape[1]}, степень: медиана {np.median(deg):.0f}, "
          f"изолированных {int((deg == 0).sum())}")

    lab = df["cls"].to_numpy(np.uint8)
    out = ensure_dir(args.out_dir) / f"{args.slide}_graph.npz"
    np.savez_compressed(out, feat=feat, label=lab, pos=xy.astype(np.float32),
                        edges=edges, mpp=mpp)
    print(f"сохранено: {out} | узлов {len(lab)}, признак {feat.shape[1]}")
    for c in TRAIN_CLASSES:
        n = int((lab == c).sum())
        if n:
            print(f"  {CLASS_NAMES[c]:18s} {n:8d} ({100*n/len(lab):.1f}%)")


if __name__ == "__main__":
    main()
