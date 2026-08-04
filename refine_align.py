# Доводка выравнивания Xenium -> H&E по содержимому изображения.
#
# align_xenium_he.py выбирает вариант трансформации по покрытию (доля клеток
# внутри кадра). Этого мало: клетки могут лежать в кадре, но мимо ткани.
# Здесь сдвиг подбирается кросс-корреляцией карты плотности клеток с картой
# ткани, то есть по тому, что реально видно на срезе.
#
#   python refine_align.py --cells data/processed/ovary3_cells_he.csv \
#     --he data/raw/ovary3/Xenium_V1_Human_Ovary_Cancer_FF_he_image.ome.tif \
#     --out data/processed/ovary3_cells_he_fixed.csv

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def he_full_shape(path):
    import tifffile
    with tifffile.TiffFile(path) as tf:
        s = tf.series[0]
        shp, axes = s.shape, (s.axes or "")
    if "Y" in axes and "X" in axes:
        return shp[axes.index("Y")], shp[axes.index("X")]
    spatial = [d for d in shp if d > 8]
    return spatial[0], spatial[1]


def load_thumb(path, min_long=1200):
    """Самый мелкий уровень пирамиды, у которого длинная сторона >= min_long."""
    import tifffile
    with tifffile.TiffFile(path) as tf:
        s = tf.series[0]
        try:
            levels = list(s.levels)
        except Exception:
            levels = [s]
        pick = levels[0]
        for lv in reversed(levels):
            if max(lv.shape[:2] + lv.shape[-2:]) >= min_long:
                pick = lv
                break
        arr = pick.asarray()
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, -1)
    return arr[..., :3].astype(np.uint8)


def tissue_map(thumb, white=215):
    """Ткань = всё, что темнее фонового стекла. Дырки закрываем."""
    import cv2
    gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    m = (gray < white).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return m.astype(np.float32)


def cell_map(x, y, shape, scale, sigma=2.0):
    """Плотность клеток в координатах превью."""
    import cv2
    h, w = shape
    xi = np.round(x / scale).astype(int)
    yi = np.round(y / scale).astype(int)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    dens = np.zeros((h, w), np.float32)
    np.add.at(dens, (yi[ok], xi[ok]), 1.0)
    dens = cv2.GaussianBlur(dens, (0, 0), sigma)
    return (dens > dens.mean() * 0.5).astype(np.float32)


def best_shift(fixed, moving):
    """Кросс-корреляция по Фурье: сдвиг, при котором карты совпадают лучше всего."""
    F = np.fft.fft2(fixed - fixed.mean())
    M = np.fft.fft2(moving - moving.mean())
    cc = np.real(np.fft.ifft2(F * np.conj(M)))
    dy, dx = np.unravel_index(np.argmax(cc), cc.shape)
    h, w = fixed.shape
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    return int(dx), int(dy)


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2.0 * (a * b).sum() / s) if s else 0.0


def rotate(x, y, deg, cx, cy):
    a = np.deg2rad(deg)
    ca, sa = np.cos(a), np.sin(a)
    dx, dy = x - cx, y - cy
    return cx + dx * ca - dy * sa, cy + dx * sa + dy * ca


def hit_rate(tissue, x, y, scale):
    h, w = tissue.shape
    xi = np.round(x / scale).astype(int)
    yi = np.round(y / scale).astype(int)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    if not ok.any():
        return 0.0
    return float(tissue[yi[ok], xi[ok]].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="CSV после align_xenium_he.py")
    ap.add_argument("--he", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[0.96, 0.98, 0.99, 1.0, 1.01, 1.02, 1.04],
                    help="масштаб по x")
    ap.add_argument("--scales-y", type=float, nargs="+", default=None,
                    help="масштаб по y отдельно (растянуть/сузить); по умолчанию = --scales")
    ap.add_argument("--angles", type=float, nargs="+", default=[0.0],
                    help="проверяемые повороты в градусах")
    ap.add_argument("--white", type=int, default=215)
    args = ap.parse_args()
    scales_y = args.scales_y if args.scales_y else args.scales

    df = pd.read_csv(args.cells)
    x0, y0 = df["x"].to_numpy(float), df["y"].to_numpy(float)
    H, W = he_full_shape(args.he)
    print("H&E полный размер: W=%d H=%d | клеток: %d" % (W, H, len(df)))

    thumb = load_thumb(args.he)
    h, w = thumb.shape[:2]
    scale = W / float(w)
    tis = tissue_map(thumb, args.white)
    print("превью %dx%d (downscale %.1f) | ткань занимает %.1f%%"
          % (w, h, scale, 100 * tis.mean()))
    print("попадание на ткань до правки: %.3f" % hit_rate(tis, x0, y0, scale))

    cx, cy = x0.mean(), y0.mean()
    best = None
    for ang in args.angles:
        xr, yr = rotate(x0, y0, ang, cx, cy) if ang else (x0, y0)
        for sx in args.scales:
            for sy in scales_y:
                xs = cx + (xr - cx) * sx
                ys = cy + (yr - cy) * sy
                cm = cell_map(xs, ys, (h, w), scale)
                dx, dy = best_shift(tis, cm)
                xs2, ys2 = xs + dx * scale, ys + dy * scale
                hr = hit_rate(tis, xs2, ys2, scale)
                d = dice(cell_map(xs2, ys2, (h, w), scale), tis)
                if best is None or d > best[0]:
                    best = (d, hr, ang, sx, sy, dx, dy)
        b = best
        print("  поворот %+5.1f° -> лучший sx=%.2f sy=%.2f, Dice %.3f попадание %.3f"
              % (ang, b[3], b[4], b[0], b[1]))

    d, hr, ang, sx, sy, dx, dy = best
    ddx, ddy = dx * scale, dy * scale
    print("\nлучшее: поворот %+.1f°, масштаб x=%.3f y=%.3f, сдвиг (%+.0f, %+.0f) px"
          % (ang, sx, sy, ddx, ddy))
    print("Dice %.3f | попадание на ткань после правки: %.3f" % (d, hr))

    xr, yr = rotate(x0, y0, ang, cx, cy) if ang else (x0, y0)
    xf = cx + (xr - cx) * sx + ddx
    yf = cy + (yr - cy) * sy + ddy
    out = df.copy()
    out["x"], out["y"] = xf, yf
    inside = (out["x"] >= 0) & (out["x"] < W) & (out["y"] >= 0) & (out["y"] < H)
    out = out[inside].reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print("сохранено: %s (%d клеток, было %d)" % (args.out, len(out), len(df)))

    qc = Path(args.out).with_suffix(".qc.png")
    import cv2
    before, after = thumb.copy(), thumb.copy()
    for img, xx, yy in ((before, x0, y0), (after, xf, yf)):
        xi = np.round(xx / scale).astype(int)
        yi = np.round(yy / scale).astype(int)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        img[yi[ok], xi[ok]] = (255, 0, 0)
    gap = np.full((h, 8, 3), 255, np.uint8)
    Image.fromarray(np.concatenate([before, gap, after], 1)).save(qc)
    print("QC (до | после):", qc)


if __name__ == "__main__":
    main()