# Локальная доводка выравнивания: своя поправка в каждом блоке сетки.
#
# Жёсткое преобразование (сдвиг+масштаб+поворот) исчерпано в refine_align.py.
# Остаток — нежёсткая деформация среза: ткань местами растянулась между съёмкой
# Xenium и окраской. Здесь слайд делится на блоки, в каждом подбирается свой
# сдвиг, поле сдвигов сглаживается и применяется к клеткам с интерполяцией.
#
#   python refine_align_local.py --cells data/seurat_csv/ovary3_he_cells_fixed.csv \
#     --he data/raw/ovary3/..._he_image.ome.tif \
#     --out data/seurat_csv/ovary3_he_cells_local.csv

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from refine_align import (he_full_shape, load_thumb, tissue_map, cell_map,
                          hit_rate, dice)

Image.MAX_IMAGE_PIXELS = None


def block_shift(tis, xi, yi, rad, step):
    """Сдвиг блока, максимизирующий долю клеток на ткани."""
    h, w = tis.shape
    best = (-1.0, 0, 0)
    for dy in range(-rad, rad + 1, step):
        yy = yi + dy
        for dx in range(-rad, rad + 1, step):
            xx = xi + dx
            ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            if ok.sum() < 0.5 * len(xi):
                continue
            v = float(tis[yy[ok], xx[ok]].mean())
            if v > best[0]:
                best = (v, dx, dy)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="CSV после refine_align.py")
    ap.add_argument("--he", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=6, help="блоков по стороне")
    ap.add_argument("--radius", type=int, default=40, help="поиск, px превью")
    ap.add_argument("--min-cells", type=int, default=300)
    ap.add_argument("--smooth", type=float, default=1.0, help="сглаживание поля сдвигов")
    args = ap.parse_args()

    import cv2
    df = pd.read_csv(args.cells)
    x0, y0 = df["x"].to_numpy(float), df["y"].to_numpy(float)
    H, W = he_full_shape(args.he)
    thumb = load_thumb(args.he)
    h, w = thumb.shape[:2]
    scale = W / float(w)
    tis = tissue_map(thumb)

    xi = np.round(x0 / scale).astype(int)
    yi = np.round(y0 / scale).astype(int)
    hr0 = hit_rate(tis, x0, y0, scale)
    d0 = dice(cell_map(x0, y0, (h, w), scale), tis)
    print("превью %dx%d (downscale %.1f) | клеток %d" % (w, h, scale, len(df)))
    print("до локальной правки: Dice %.3f, попадание %.3f" % (d0, hr0))

    G = args.grid
    sx = np.zeros((G, G), np.float32)
    sy = np.zeros((G, G), np.float32)
    got = np.zeros((G, G), bool)
    bx, by = w / G, h / G

    for gy in range(G):
        for gx in range(G):
            sel = ((xi >= gx * bx) & (xi < (gx + 1) * bx) &
                   (yi >= gy * by) & (yi < (gy + 1) * by))
            n = int(sel.sum())
            if n < args.min_cells:
                continue
            _, dx, dy = block_shift(tis, xi[sel], yi[sel], args.radius, 4)
            v, fdx, fdy = block_shift(tis, xi[sel] + dx, yi[sel] + dy, 3, 1)
            dx, dy = dx + fdx, dy + fdy
            sx[gy, gx], sy[gy, gx], got[gy, gx] = dx, dy, True
            print("  блок (%d,%d): %6d клеток -> сдвиг (%+3d,%+3d), на ткани %.3f"
                  % (gy, gx, n, dx, dy, v))

    if not got.any():
        raise SystemExit("ни один блок не набрал клеток — уменьшите --min-cells")
    sx[~got] = float(np.median(sx[got]))
    sy[~got] = float(np.median(sy[got]))
    if args.smooth:
        sx = cv2.GaussianBlur(sx, (0, 0), args.smooth)
        sy = cv2.GaussianBlur(sy, (0, 0), args.smooth)
    print("поле сдвигов: dx от %+.0f до %+.0f, dy от %+.0f до %+.0f (px превью)"
          % (sx.min(), sx.max(), sy.min(), sy.max()))

    fx = cv2.resize(sx, (w, h), interpolation=cv2.INTER_LINEAR)
    fy = cv2.resize(sy, (w, h), interpolation=cv2.INTER_LINEAR)
    cxi = np.clip(xi, 0, w - 1)
    cyi = np.clip(yi, 0, h - 1)
    xf = x0 + fx[cyi, cxi] * scale
    yf = y0 + fy[cyi, cxi] * scale

    hr1 = hit_rate(tis, xf, yf, scale)
    d1 = dice(cell_map(xf, yf, (h, w), scale), tis)
    print("после локальной правки: Dice %.3f, попадание %.3f" % (d1, hr1))

    out = df.copy()
    out["x"], out["y"] = xf, yf
    inside = (out["x"] >= 0) & (out["x"] < W) & (out["y"] >= 0) & (out["y"] < H)
    out = out[inside].reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print("сохранено: %s (%d клеток)" % (args.out, len(out)))

    before, after = thumb.copy(), thumb.copy()
    for img, xx, yy in ((before, x0, y0), (after, xf, yf)):
        a = np.round(xx / scale).astype(int)
        b = np.round(yy / scale).astype(int)
        ok = (a >= 0) & (a < w) & (b >= 0) & (b < h)
        img[b[ok], a[ok]] = (255, 0, 0)
    gap = np.full((h, 8, 3), 255, np.uint8)
    qc = Path(args.out).with_suffix(".qc.png")
    Image.fromarray(np.concatenate([before, gap, after], 1)).save(qc)
    print("QC (до | после):", qc)


if __name__ == "__main__":
    main()