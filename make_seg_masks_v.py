# Пиксельная истина (Вороной): каждый пиксель ткани = класс ближайшей клетки
# в пределах max_dist, без дырок. tumor=1, stroma=2, фон/undefined=0.
# При нехватке памяти увеличь --downscale (3 или 4).

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils import ensure_dir
from src.data.seurat_labels import CellTypeMapper

CLS = {"tumor": 1, "stroma": 2}
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
    ap.add_argument("--max-dist-factor", type=float, default=1.5)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--min-label-frac", type=float, default=0.10)
    ap.add_argument("--out-dir", default="data/processed/seg")
    args = ap.parse_args()

    import cv2
    from scipy.ndimage import distance_transform_edt

    df = pd.read_csv(args.cells)
    mapper = CellTypeMapper(args.map)
    df["zone"] = mapper.map_series(df["cell_type"])
    df = df[df["zone"].isin(CLS)].copy()
    df["cls"] = df["zone"].map(CLS).astype(int)
    coords = df[["x", "y"]].to_numpy(float)
    print("%s: клеток tumor/stroma: %d %s" % (args.slide, len(df), df["zone"].value_counts().to_dict()))

    H, W = he_shape(args.he)
    ds = args.downscale
    Hd, Wd = H // ds, W // ds
    print("H&E %dx%d -> маска %dx%d (downscale %d)" % (W, H, Wd, Hd, ds))

    nn_ds = median_nn(coords) / ds
    max_dist = max(2.0, args.max_dist_factor * nn_ds)
    print("медианное расстояние ~%.1fpx, заполняем до %.1fpx" % (nn_ds, max_dist))

    seeds = np.zeros((Hd, Wd), np.uint8)
    xs = np.clip((coords[:, 0] / ds).astype(int), 0, Wd - 1)
    ys = np.clip((coords[:, 1] / ds).astype(int), 0, Hd - 1)
    cl = df["cls"].to_numpy()
    order = np.argsort(-cl)
    seeds[ys[order], xs[order]] = cl[order]

    print("строю Вороной...")
    dist, inds = distance_transform_edt(seeds == 0, return_distances=True, return_indices=True)
    mask = seeds[inds[0], inds[1]]
    mask[dist > max_dist] = 0
    del dist, inds

    out = ensure_dir(args.out_dir)
    np.savez_compressed(out / ("%s_mask.npz" % args.slide), mask=mask, downscale=ds)
    tot = Hd * Wd
    print("маска: tumor %.1f%% stroma %.1f%%" % (100*(mask==1).sum()/tot, 100*(mask==2).sum()/tot))

    bg = small_he(args.he)
    msmall = cv2.resize(mask, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = bg.copy()
    for c, col in COL.items():
        overlay[msmall == c] = (0.45 * np.array(col) + 0.55 * bg[msmall == c]).astype(np.uint8)
    from PIL import Image
    res = ensure_dir("outputs/results")
    Image.fromarray(np.concatenate([bg, overlay], 1)).save(res / ("%s_seg_truth.png" % args.slide))
    print("проверка: outputs/results/%s_seg_truth.png" % args.slide)

    ps, st = args.patch_size, args.stride
    rows = []
    for y0 in range(0, H - ps + 1, st):
        for x0 in range(0, W - ps + 1, st):
            sub = mask[y0 // ds:(y0 + ps) // ds, x0 // ds:(x0 + ps) // ds]
            lab = (sub > 0).mean()
            if lab >= args.min_label_frac:
                rows.append({"x0": x0, "y0": y0,
                             "frac_tumor": float((sub == 1).mean()),
                             "frac_stroma": float((sub == 2).mean()),
                             "label_frac": float(lab)})
    pd.DataFrame(rows).to_csv(out / ("%s_patches.csv" % args.slide), index=False)
    print("патчей с разметкой: %d -> %s/%s_patches.csv" % (len(rows), out, args.slide))


if __name__ == "__main__":
    main()
