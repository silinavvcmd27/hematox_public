# Пиксельная истина для сегментации: растеризуем клетки Xenium (tumor/stroma)
# в плотную маску на разрешении H&E (с уменьшением downscale), строим
# проверочный оверлей и список патчей с достаточной разметкой для обучения.
#
# tumor=1, stroma=2, фон/undefined=0 (игнорируется при обучении).
#
# python make_seg_masks.py --slide ovary_prime_he \
#   --cells data/seurat_csv/ovary_prime_he_cells.csv \
#   --he data/raw/ovary_prime/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_he_image.ome.tif

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import ensure_dir
from src.data.seurat_labels import CellTypeMapper

CLS = {"tumor": 1, "stroma": 2}          # undefined -> не красим (0)
COL = {1: (220, 50, 47), 2: (38, 139, 210)}


def he_shape(path):
    import tifffile
    with tifffile.TiffFile(path) as tf:
        s = tf.series[0]; shp = s.shape; axes = s.axes or ""
    if "Y" in axes and "X" in axes:
        return shp[axes.index("Y")], shp[axes.index("X")]
    dims = [d for d in shp if d > 8]
    return dims[0], dims[1]


def small_he(path, max_dim=2000):
    import tifffile
    with tifffile.TiffFile(path) as tf:
        s = tf.series[0]
        try:
            arr = list(s.levels)[-1].asarray()
        except Exception:
            arr = s.asarray()
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    arr = arr[..., :3]
    step = max(1, int(max(arr.shape[:2]) / max_dim))
    return arr[::step, ::step].astype(np.uint8)


def median_nn(coords, sample=5000):
    from sklearn.neighbors import NearestNeighbors
    n = len(coords)
    idx = np.random.choice(n, min(sample, n), replace=False)
    nn = NearestNeighbors(n_neighbors=2).fit(coords)
    d, _ = nn.kneighbors(coords[idx])
    return float(np.median(d[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--he", required=True)
    ap.add_argument("--map", default="config/cell_type_map.yaml")
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--min-label-frac", type=float, default=0.10)
    ap.add_argument("--out-dir", default="data/processed/seg")
    args = ap.parse_args()

    import cv2

    df = pd.read_csv(args.cells)
    mapper = CellTypeMapper(args.map)
    df["zone"] = mapper.map_series(df["cell_type"])
    df = df[df["zone"].isin(CLS)].copy()
    df["cls"] = df["zone"].map(CLS).astype(int)
    coords = df[["x", "y"]].to_numpy(float)
    print(f"{args.slide}: клеток с tumor/stroma: {len(df)} "
          f"({df['zone'].value_counts().to_dict()})")

    H, W = he_shape(args.he)
    ds = args.downscale
    Hd, Wd = H // ds, W // ds
    print(f"H&E {W}x{H} -> маска {Wd}x{Hd} (downscale {ds})")

    r_full = 0.7 * median_nn(coords)
    r = max(1, int(round(r_full / ds)))
    print(f"радиус клетки на маске: {r}px (full {r_full:.1f})")

    mask = np.zeros((Hd, Wd), np.uint8)
    xs = (coords[:, 0] / ds).astype(int)
    ys = (coords[:, 1] / ds).astype(int)
    cl = df["cls"].to_numpy()
    # рисуем строму, потом опухоль (опухоль приоритетна при наложении)
    order = np.argsort(cl)  # 1(tumor) раньше 2(stroma)? нет: сортировка по возр. -> tumor(1) первым
    for i in order[::-1]:   # сначала stroma(2), потом tumor(1) поверх
        cv2.circle(mask, (int(xs[i]), int(ys[i])), r, int(cl[i]), -1)

    out = ensure_dir(args.out_dir)
    np.savez_compressed(out / f"{args.slide}_mask.npz", mask=mask, downscale=ds)
    npx = int((mask == 1).sum()); nsx = int((mask == 2).sum())
    tot = Hd * Wd
    print(f"маска: tumor {100*npx/tot:.1f}% stroma {100*nsx/tot:.1f}% "
          f"(остальное фон)")

    # ---- проверочный оверлей ----
    bg = small_he(args.he)
    mh, mw = bg.shape[:2]
    msmall = cv2.resize(mask, (mw, mh), interpolation=cv2.INTER_NEAREST)
    overlay = bg.copy()
    for c, col in COL.items():
        overlay[msmall == c] = (0.45 * np.array(col) + 0.55 * bg[msmall == c]).astype(np.uint8)
    from PIL import Image
    res_dir = ensure_dir("outputs/results")
    Image.fromarray(np.concatenate([bg, overlay], axis=1)).save(
        res_dir / f"{args.slide}_seg_truth.png")
    print(f"проверка: outputs/results/{args.slide}_seg_truth.png (слева H&E, справа маска)")

    # ---- манифест патчей с разметкой ----
    ps, st = args.patch_size, args.stride
    rows = []
    for y0 in range(0, H - ps + 1, st):
        for x0 in range(0, W - ps + 1, st):
            yd0, yd1 = y0 // ds, (y0 + ps) // ds
            xd0, xd1 = x0 // ds, (x0 + ps) // ds
            sub = mask[yd0:yd1, xd0:xd1]
            lab = (sub > 0).mean()
            if lab >= args.min_label_frac:
                rows.append({"x0": x0, "y0": y0,
                             "frac_tumor": float((sub == 1).mean()),
                             "frac_stroma": float((sub == 2).mean()),
                             "label_frac": float(lab)})
    man = pd.DataFrame(rows)
    man.to_csv(out / f"{args.slide}_patches.csv", index=False)
    print(f"патчей с разметкой (>= {args.min_label_frac}): {len(man)} "
          f"-> {out}/{args.slide}_patches.csv")


if __name__ == "__main__":
    main()
