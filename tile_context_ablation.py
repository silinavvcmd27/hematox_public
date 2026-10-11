# Сколько слой 1 теряет, когда у клетки отбирают соседей.
#
# Признаки UNI и метки берутся из готового графа (data/processed/graph), меняется
# только связность: рёбра либо строятся заново внутри изолированных тайлов и не
# пересекают границу, либо выключаются совсем. Всё остальное совпадает с обычным
# инференсом, поэтому падение Dice объясняется потерей контекста, а не доменом,
# окраской или разрешением.
#
# Строка side_um=0 (рёбра выключены) — главная: она отделяет вклад соседей от
# вклада самого клеточного токена.
#
# Для каждой стороны тайла считается и полный граф на той же подвыборке клеток
# (колонки *_full), иначе фильтр --min-cells подмешивается в разницу.
#
# Чекпоинт брать межпациентный (обучен без проверяемого среза): на in-sample
# срезе модель узнаёт клетку по её собственному токену и соседи ей не нужны.
#
#   python -u tile_context_ablation.py --slides ovary_prime_he \
#     --layer1 runs/graph/l1_ovary2.pth --out outputs/results/tile_ablation.csv

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from scipy.spatial import cKDTree

from src.utils import get_device, ensure_dir
from cell_graph_train import CellGNN, adjacency, to_binary, metrics, CLASSES

BORDER_BINS = [0, 8, 16, 24, 32, 48, 1e9]


def load_layer1(path, device):
    ck = torch.load(path, map_location=device)
    m = CellGNN(int(ck["in_dim"]), int(ck["hid"]), len(ck["classes"]), int(ck["layers"])).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    print(f"слой 1: {path} | слоёв {ck['layers']}, обучен без {ck.get('val')}")
    return m


def knn_edges(xy, knn, r_px):
    """Те же рёбра, что строит cell_graph_build: k ближайших не дальше радиуса."""
    if len(xy) < 2:
        return np.zeros((2, 0), np.int64)
    k = min(knn + 1, len(xy))
    d, nb = cKDTree(xy).query(xy, k=k)
    d, nb = d[:, 1:], nb[:, 1:]
    ok = d <= r_px
    src = np.repeat(np.arange(len(xy)), k - 1)[ok.ravel()]
    dst = nb.ravel()[ok.ravel()]
    return np.vstack([np.r_[src, dst], np.r_[dst, src]]).astype(np.int64)


def tiled_edges(xy, tile_px, knn, r_px, min_cells):
    """Рёбра внутри тайлов сетки tile_px. Возвращает ещё и маску учтённых клеток."""
    cell = (xy // tile_px).astype(np.int64)
    cell -= cell.min(0)
    flat = cell[:, 0] * (cell[:, 1].max() + 1) + cell[:, 1]
    _, tile_of, counts = np.unique(flat, return_inverse=True, return_counts=True)
    size = counts[tile_of]

    order = np.argsort(tile_of, kind="stable")
    parts = []
    for idx in np.split(order, np.cumsum(counts)[:-1]):
        if len(idx) < min_cells:
            continue
        e = knn_edges(xy[idx], knn, r_px)
        if e.shape[1]:
            parts.append(idx[e])
    edges = np.hstack(parts) if parts else np.zeros((2, 0), np.int64)
    return edges, size >= min_cells, size


def edge_stats(xy, edges, mpp, knn, r_px, n_sample=2000, seed=0):
    """Фактический охват графа в микрометрах, а не по параметру max-edge-um."""
    d, _ = cKDTree(xy).query(xy, k=knn + 1)
    nn1 = np.median(d[:, 1]) * mpp
    kth = np.median(d[:, knn]) * mpp
    clipped = float((d[:, 1:] > r_px).mean())
    length = np.median(np.linalg.norm(xy[edges[0]] - xy[edges[1]], axis=1)) * mpp

    n = len(xy)
    a = sp.csr_matrix((np.ones(edges.shape[1], np.int8), (edges[0], edges[1])), shape=(n, n))
    rng = np.random.default_rng(seed)
    reach = []
    for i in rng.choice(n, size=min(n_sample, n), replace=False):
        hop1 = a.indices[a.indptr[i]:a.indptr[i + 1]]
        if not len(hop1):
            continue
        hop2 = np.unique(np.concatenate([a.indices[a.indptr[j]:a.indptr[j + 1]] for j in hop1]))
        reach.append(np.linalg.norm(xy[hop2] - xy[i], axis=1).max())
    print(f"  соседи: ближайший {nn1:.1f} мкм, {knn}-й {kth:.1f} мкм, "
          f"медиана ребра {length:.1f} мкм, обрезано порогом {100*clipped:.1f}%")
    if reach:
        print(f"  охват за два прыжка: медиана {np.median(reach)*mpp:.1f} мкм, "
              f"90-й процентиль {np.percentile(reach, 90)*mpp:.1f} мкм")


def dice_row(pred_tiled, pred_full, truth, keep):
    t = np.where(keep, truth, -100)
    mt, mf = metrics(pred_tiled, t), metrics(pred_full, t)
    out = {}
    for c in CLASSES:
        out[f"{c}_iou"], out[f"{c}_dice"] = mt[c]
        out[f"{c}_dice_full"] = mf[c][1]
    return out


@torch.no_grad()
def predict(model, x, edges, device):
    a = adjacency(edges.astype(np.int64), len(x), device)
    return model(x, a).argmax(1).cpu().numpy()


def border_profile(xy, tile_px, mpp, correct, keep):
    """Доля верных предсказаний по расстоянию до края тайла, в микрометрах."""
    off = xy - (xy // tile_px) * tile_px
    dist = np.minimum(off, tile_px - off).min(1) * mpp
    rows = []
    for lo, hi in zip(BORDER_BINS[:-1], BORDER_BINS[1:]):
        m = keep & (dist >= lo) & (dist < hi)
        if m.sum() >= 50:
            rows.append((lo, min(hi, tile_px * mpp / 2), int(m.sum()), float(correct[m].mean())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", nargs="+", required=True)
    ap.add_argument("--layer1", default="runs/graph/l1_final.pth")
    ap.add_argument("--graph-dir", default="data/processed/graph")
    ap.add_argument("--sides", type=float, nargs="+", default=[16, 24, 32, 48, 64, 128, 256],
                    help="сторона тайла в мкм; 64 = геометрия STHELAR")
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--max-edge-um", type=float, default=50.0)
    ap.add_argument("--min-cells", type=int, default=4)
    ap.add_argument("--out", default="outputs/results/tile_ablation.csv")
    args = ap.parse_args()

    device = get_device()
    model = load_layer1(args.layer1, device)

    rows, profiles = [], []
    for slide in args.slides:
        g = np.load(Path(args.graph_dir) / f"{slide}_graph.npz")
        xy, mpp, full_edges = g["pos"].astype(float), float(g["mpp"]), g["edges"]
        truth = to_binary(g["label"])
        feat = torch.from_numpy(g["feat"].astype(np.float32)).to(device)
        r_px = args.max_edge_um / mpp
        print(f"\n{slide}: клеток {len(truth)}, mpp {mpp:.4f}")
        edge_stats(xy, full_edges, mpp, args.knn, r_px)

        pred_full = predict(model, feat, full_edges, device)
        alive = truth != -100
        rows.append(dict(slide=slide, side_um=np.inf, cells=int(alive.sum()), cells_per_tile=np.nan,
                         **dice_row(pred_full, pred_full, truth, alive)))
        print("  весь срез        " + "  ".join("%s Dice %.3f" % (c, rows[-1][f"{c}_dice"]) for c in CLASSES))

        pred_none = predict(model, feat, np.zeros((2, 0), np.int64), device)
        rows.append(dict(slide=slide, side_um=0.0, cells=int(alive.sum()), cells_per_tile=1.0,
                         **dice_row(pred_none, pred_full, truth, alive)))
        print("  рёбра выключены  " + "  ".join("%s Dice %.3f" % (c, rows[-1][f"{c}_dice"]) for c in CLASSES))

        for side in args.sides:
            tile_px = side / mpp
            edges, keep, per_tile = tiled_edges(xy, tile_px, args.knn, r_px, args.min_cells)
            pred = predict(model, feat, edges, device)
            keep &= alive
            rows.append(dict(slide=slide, side_um=side, cells=int(keep.sum()),
                             cells_per_tile=float(per_tile[keep].mean()) if keep.any() else 0.0,
                             **dice_row(pred, pred_full, truth, keep)))
            r = rows[-1]
            print("  тайл %5.0f мкм   " % side
                  + "  ".join("%s Dice %.3f (полный граф %.3f)"
                              % (c, r[f"{c}_dice"], r[f"{c}_dice_full"]) for c in CLASSES)
                  + "  | клеток на тайл %.0f, учтено %.0f%%" % (r["cells_per_tile"], 100 * keep.mean()))

            ok = (pred == truth) & alive
            for lo, hi, n, acc in border_profile(xy, tile_px, mpp, ok, keep):
                profiles.append(dict(slide=slide, side_um=side, dist_lo=lo, dist_hi=hi,
                                     cells=n, accuracy=acc))
        del feat

    df = pd.DataFrame(rows)
    ensure_dir(Path(args.out).parent)
    df.to_csv(args.out, index=False)
    cols = [f"{c}_dice" for c in CLASSES] + [f"{c}_dice_full" for c in CLASSES]
    print("\n=== среднее по срезам (dice — урезанный граф, dice_full — полный на тех же клетках) ===")
    print(df.groupby("side_um")[cols].mean().round(3).to_string())

    if profiles:
        pr = pd.DataFrame(profiles)
        pr.to_csv(Path(args.out).with_name(Path(args.out).stem + "_border.csv"), index=False)
        print("\n=== точность по расстоянию до края тайла (мкм) ===")
        print(pr.pivot_table(index=["dist_lo", "dist_hi"], columns="side_um",
                             values="accuracy").round(3).to_string())
    print("\nтаблица:", args.out)


if __name__ == "__main__":
    main()
