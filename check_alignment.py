# Проверка совмещения координат Xenium с H&E.
#
# align_xenium_he.py выбирает преобразование по доле клеток внутри кадра, а это
# слабый критерий: неверное преобразование тоже может уложиться в границы.
# Здесь клетки проверяются по ткани — сколько их попало на непустые пиксели.
#
# python -u check_alignment.py --slide ovary_prime_he \
#   --cells data/seurat_csv/ovary_prime_cells.csv \
#   --align data/raw/ovary_prime/..._he_imagealignment.csv \
#   --he    data/raw/ovary_prime/..._he_image.ome.tif

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

XENIUM_PX_UM = 0.2125


def load_matrix(path):
    M = np.loadtxt(path, delimiter=",")
    if M.shape == (3, 3):
        return M[:2, :2], M[:2, 2]
    if M.shape == (2, 3):
        return M[:, :2], M[:, 2]
    raise ValueError(f"ожидал 2x3 или 3x3, получил {M.shape}")


def he_shape(path):
    import tifffile
    with tifffile.TiffFile(path) as tf:
        s = tf.series[0]
        shp, axes = s.shape, (s.axes or "")
    if "Y" in axes and "X" in axes:
        return shp[axes.index("Y")], shp[axes.index("X")]
    dims = [d for d in shp if d > 8]
    return dims[0], dims[1]


def thumbnail(path, max_dim=2500):
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


def candidates(xy_um, A, t):
    Ainv = np.linalg.inv(A)
    px = xy_um / XENIUM_PX_UM
    return {
        "forward_px": (A @ px.T).T + t,
        "forward_um": (A @ xy_um.T).T + t,
        "inverse_px": (Ainv @ (px - t).T).T,
        "inverse_um": (Ainv @ (xy_um - t).T).T,
    }


def on_tissue(xy, tissue, sx, sy, dx=0, dy=0):
    """Доля клеток, попавших на непустой пиксель уменьшенной картинки."""
    Ht, Wt = tissue.shape
    cx = np.round(xy[:, 0] * sx + dx).astype(int)
    cy = np.round(xy[:, 1] * sy + dy).astype(int)
    ok = (cx >= 0) & (cx < Wt) & (cy >= 0) & (cy < Ht)
    if not ok.any():
        return 0.0, 0.0
    return float(tissue[cy[ok], cx[ok]].mean()), float(ok.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--cells", required=True, help="сырые координаты из R, в микрометрах")
    ap.add_argument("--align", required=True)
    ap.add_argument("--he", required=True)
    ap.add_argument("--aligned", help="уже пересчитанный CSV, для сравнения")
    ap.add_argument("--white", type=int, default=220)
    ap.add_argument("--sample", type=int, default=50000)
    args = ap.parse_args()

    H, W = he_shape(args.he)
    thumb = thumbnail(args.he)
    Ht, Wt = thumb.shape[:2]
    sx, sy = Wt / W, Ht / H
    tissue = thumb.mean(-1) < args.white
    print(f"H&E {W}x{H}, эскиз {Wt}x{Ht}, ткани на эскизе {100*tissue.mean():.1f}%")

    df = pd.read_csv(args.cells)
    xy = df[["x", "y"]].to_numpy(float)
    if len(xy) > args.sample:
        xy = xy[np.random.default_rng(0).choice(len(xy), args.sample, replace=False)]
    A, t = load_matrix(args.align)

    print("\nвариант преобразования        в кадре   на ткани")
    best = None
    for name, pts in candidates(xy, A, t).items():
        for flip, tag in ((False, "xy"), (True, "yx")):
            p = pts[:, ::-1] if flip else pts
            frac, inside = on_tissue(p, tissue, sx, sy)
            print(f"  {name:12s} [{tag}]      {inside:6.3f}    {frac:6.3f}")
            if best is None or frac > best[0]:
                best = (frac, name, tag, p)

    frac, name, tag, pts = best
    print(f"\nлучший по ткани: {name} [{tag}], на ткани {frac:.3f}")
    if name != "inverse_px" or tag != "xy":
        print("  ВНИМАНИЕ: align_xenium_he.py по умолчанию берёт inverse_px [xy] — "
              "это не тот вариант")

    # случайные точки внутри кадра дают базовый уровень: с ним и надо сравнивать
    rng = np.random.default_rng(1)
    fake = np.stack([rng.uniform(0, W, 20000), rng.uniform(0, H, 20000)], 1)
    base, _ = on_tissue(fake, tissue, sx, sy)
    print(f"случайные точки по кадру попали бы на ткань в {base:.3f} случаев")

    print("\nподбор сдвига (пиксели эскиза):")
    span = max(4, int(0.06 * max(Ht, Wt)))
    step = max(1, span // 12)
    grid = range(-span, span + 1, step)
    scores = {(dx, dy): on_tissue(pts, tissue, sx, sy, dx, dy)[0] for dx in grid for dy in grid}
    (bdx, bdy), bscore = max(scores.items(), key=lambda kv: kv[1])
    print(f"  без сдвига {frac:.3f} | лучший сдвиг dx={bdx} dy={bdy} -> {bscore:.3f}")
    print(f"  в пикселях исходника: dx={bdx/sx:.0f} dy={bdy/sy:.0f}")
    if bscore - frac < 0.02:
        print("  сдвиг ничего не даёт — по этому признаку совмещение в порядке")
    else:
        print("  сдвиг заметно улучшает попадание — совмещение съехало")

    if args.aligned:
        adf = pd.read_csv(args.aligned)
        axy = adf[["x", "y"]].to_numpy(float)
        if len(axy) > args.sample:
            axy = axy[np.random.default_rng(0).choice(len(axy), args.sample, replace=False)]
        f2, in2 = on_tissue(axy, tissue, sx, sy)
        print(f"\nтекущий {Path(args.aligned).name}: в кадре {in2:.3f}, на ткани {f2:.3f}")

    vis = thumb.copy()
    vis[~tissue] = 245
    cx = np.round(pts[:, 0] * sx).astype(int)
    cy = np.round(pts[:, 1] * sy).astype(int)
    ok = (cx >= 0) & (cx < Wt) & (cy >= 0) & (cy < Ht)
    vis[cy[ok], cx[ok]] = (220, 50, 47)
    out = Path("outputs/results")
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{args.slide}_align_check.png"
    Image.fromarray(np.concatenate([thumb, vis], 1)).save(png)
    print("\nткань и клетки поверх неё:", png)


if __name__ == "__main__":
    main()
