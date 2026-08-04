# Перерисовка клеточной карты из сохранённого _cells.npz — мгновенно, без StarDist.
# Крутим прозрачность, границы, режим (территории клеток или точки-ядра).
#
#   python render_cells.py --slide X.svs --npz outputs/results/X_cells.npz --alpha 0.35
#   python render_cells.py --slide X.svs --npz ... --mode dot

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

from seg_infer import SlideReader
from src.utils import (ensure_dir, TUMOR, STROMA_HORMONAL, STROMA_MATRIX,
                       IMMUNE, STROMA, CLASS_NAMES)

Image.MAX_IMAGE_PIXELS = None
COL = {TUMOR: (216, 65, 47), STROMA_HORMONAL: (240, 160, 48),
       STROMA_MATRIX: (46, 134, 193), IMMUNE: (42, 161, 82), STROMA: (128, 128, 128)}
STROMA_ALL = [STROMA_HORMONAL, STROMA_MATRIX, IMMUNE, STROMA]


def fill_territories(he, xy, cls, ox, oy, max_r, alpha, border):
    h, w = he.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.int32), np.arange(h, dtype=np.int32))
    dist, idx = cKDTree(xy - [ox, oy]).query(np.c_[gx.ravel(), gy.ravel()])
    dist = dist.reshape(h, w); idx = idx.reshape(h, w)
    inside = dist < max_r
    clsmap = np.zeros((h, w), np.uint8)
    clsmap[inside] = cls[idx[inside]]
    cmap = np.zeros((h, w, 3), np.uint8)
    for c, col in COL.items():
        cmap[clsmap == c] = col
    ov = he.copy()
    m = clsmap > 0
    ov[m] = (alpha * cmap[m] + (1 - alpha) * he[m]).astype(np.uint8)
    if border:
        lab = np.where(inside, idx, -1)
        bnd = np.zeros((h, w), bool)
        bnd[:, 1:] |= lab[:, 1:] != lab[:, :-1]
        bnd[1:, :] |= lab[1:, :] != lab[:-1, :]
        bnd &= inside
        ov[bnd] = (0.5 * np.array([90, 90, 90]) + 0.5 * ov[bnd]).astype(np.uint8)
    return ov


def dots(he, xy, cls, ox, oy, r):
    im = Image.fromarray(he.copy())
    dr = ImageDraw.Draw(im)
    for (x, y), c in zip(xy, cls):
        a, b = x - ox, y - oy
        dr.ellipse([a - r, b - r, a + r, b + r], fill=COL[int(c)])
    return np.asarray(im)


def legend(canvas, x0, comp):
    img = Image.fromarray(canvas); dr = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    rows = [(TUMOR, "опухоль")] + [(c, CLASS_NAMES[c]) for c in STROMA_ALL]
    y0 = 16
    dr.rectangle([x0 - 12, y0 - 10, x0 + 320, y0 + len(rows) * 32 + 8],
                 fill=(255, 255, 255), outline=(120, 120, 120))
    for i, (c, nm) in enumerate(rows):
        yy = y0 + i * 32
        dr.rectangle([x0, yy, x0 + 22, yy + 22], fill=COL[c], outline=(60, 60, 60))
        dr.text((x0 + 30, yy - 1), f"{nm} — {comp[c]:.1f}%", fill=(20, 20, 20), font=font)
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--target-mpp", type=float, default=0.27)
    ap.add_argument("--mode", choices=["fill", "dot"], default="fill")
    ap.add_argument("--alpha", type=float, default=0.35, help="прозрачность заливки (0.2-0.6)")
    ap.add_argument("--radius-factor", type=float, default=1.15, help="радиус клетки × медиана")
    ap.add_argument("--dot-r", type=int, default=4, help="радиус точки в режиме dot")
    ap.add_argument("--border", action="store_true", help="рисовать границы клеток")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    xy, cls = d["xy"].astype(np.float32), d["cls"]
    sl = SlideReader(args.slide, args.target_mpp, None)
    x0, y0 = xy.min(0); x1, y1 = xy.max(0)
    pad = 40
    rx, ry = int(x0 - pad), int(y0 - pad)
    S = int(max(x1 - x0, y1 - y0) + 2 * pad)
    he = sl.read(rx, ry, S)

    med = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
    if args.mode == "fill":
        ov = fill_territories(he, xy, cls, rx, ry, args.radius_factor * med,
                              args.alpha, args.border)
    else:
        ov = dots(he, xy, cls, rx, ry, args.dot_r)

    comp = {c: 100 * int((cls == c).sum()) / max(len(cls), 1) for c in [TUMOR] + STROMA_ALL}
    canvas = legend(np.concatenate([he, ov], 1), S * 2 - 340, comp)
    out = args.out or str(ensure_dir("outputs/results") /
                          (Path(args.npz).stem.replace("_cells", "") + f"_render_{args.mode}.png"))
    Image.fromarray(canvas).save(out)
    print("сохранено:", out, "| alpha", args.alpha, "| режим", args.mode)


if __name__ == "__main__":
    main()
