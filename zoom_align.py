# Как легли клетки, вблизи. Несколько окон в реальном увеличении: слева чистая
# ткань, справа она же с точками клеток. Эскиз для такого не годится, там пиксель
# это больше сотни микрон, и любое совмещение выглядит одинаково хорошо.
#
#   python zoom_align.py --slide ovary3 \
#     --he data/raw/ovary3/..._he_image.ome.tif \
#     --cells data/seurat_csv/ovary3_he_cells_prime.csv

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
DOT = (0, 200, 0)


def levels(path):
    import tifffile
    with tifffile.TiffFile(path) as tf:
        out = []
        for i, lv in enumerate(tf.series[0].levels):
            dims = sorted((d for d in lv.shape if d > 8), reverse=True)
            out.append((i, dims[0], dims[1]))
    return out


def read_level(path, i):
    import tifffile
    with tifffile.TiffFile(path) as tf:
        arr = tf.series[0].levels[i].asarray()
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    return arr[..., :3].astype(np.uint8)


def pick_level(lvs, max_gb):
    """Самый подробный уровень, который влезает в бюджет памяти."""
    for i, h, w in lvs:
        if h * w * 3 / 1e9 <= max_gb:
            return i
    return lvs[-1][0]


def pick_windows(xy, n, size, seed=0):
    """Окна вокруг случайных клеток, разнесённые друг от друга."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(xy))
    picked = []
    for i in order:
        c = xy[i]
        if all(np.hypot(*(c - p)) > 2 * size for p in picked):
            picked.append(c)
        if len(picked) == n:
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--he", required=True)
    ap.add_argument("--cells", required=True, help="координаты в пикселях H&E")
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--size", type=int, default=400, help="сторона окна в пикселях уровня")
    ap.add_argument("--dot", type=int, default=2)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--max-gb", type=float, default=2.0,
                    help="сколько памяти можно потратить на уровень пирамиды")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lvs = levels(args.he)
    lvl = args.level if args.level is not None else pick_level(lvs, args.max_gb)
    full_h, full_w = lvs[0][1], lvs[0][2]
    _, h, w = lvs[lvl]
    scale = w / full_w
    print(f"уровень {lvl}: {w}x{h}, масштаб {scale:.3f} от полного {full_w}x{full_h}")

    xy = pd.read_csv(args.cells)[["x", "y"]].to_numpy(float) * scale
    print(f"клеток {len(xy)}, окно {args.size} px = {args.size / scale:.0f} px полного")

    img = read_level(args.he, lvl)
    half = args.size // 2
    panels = []
    for cx, cy in pick_windows(xy, args.windows, args.size):
        x0, y0 = int(cx) - half, int(cy) - half
        x0 = max(0, min(x0, w - args.size))
        y0 = max(0, min(y0, h - args.size))
        crop = img[y0:y0 + args.size, x0:x0 + args.size]
        marked = Image.fromarray(crop.copy())
        d = ImageDraw.Draw(marked)
        sel = xy[(xy[:, 0] >= x0) & (xy[:, 0] < x0 + args.size) &
                 (xy[:, 1] >= y0) & (xy[:, 1] < y0 + args.size)]
        for px, py in sel - [x0, y0]:
            d.ellipse([px - args.dot, py - args.dot, px + args.dot, py + args.dot],
                      fill=DOT)
        panels.append(np.concatenate([crop, np.asarray(marked)], 1))
        print(f"  окно {x0},{y0}: клеток {len(sel)}")

    sheet = np.concatenate(panels, 0)
    out = Path(args.out or f"outputs/results/{args.slide}_zoom.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out)
    print("сохранено:", out)


if __name__ == "__main__":
    main()
