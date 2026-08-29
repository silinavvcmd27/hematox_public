# Ручная доводка сдвига между координатами Xenium и H&E.
#
# Показывает не сами клетки, а расхождение: синим ткань без клеток, оранжевым
# клетки мимо ткани. Хорошее совмещение выглядит как почти чистый срез, и сразу
# видно, в какую сторону не хватает.
#
# Перебор вокруг текущей поправки:
#   python nudge_align.py --slide ovary3 --cells data/seurat_csv/ovary3_cells.csv \
#     --align data/raw/ovary3/..._he_imagealignment.csv \
#     --he data/raw/ovary3/..._he_image.ome.tif --span 4000 --step 1000
#
# Одна конкретная поправка, с картинкой:
#   ... --dx -2814 --dy -128

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from align_xenium_he import ORIENTS, orient
from check_alignment import (candidates, cell_mask, coarsen, he_shape, load_matrix,
                             score, thumbnail)

MISS = (38, 139, 210)     # ткань, на которую клетки не легли
OVER = (255, 165, 0)      # клетки за пределами ткани


def parse_range(s):
    lo, hi, step = (float(v) for v in s.split(":"))
    return np.arange(lo, hi + step / 2, step)


def cached_thumb(he, slide):
    path = Path("outputs/results") / f"{slide}_thumb.npy"
    if path.exists():
        return np.load(path)
    arr = thumbnail(he)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


def upscale(mask, k, shape):
    big = np.kron(mask, np.ones((k, k), bool))
    out = np.zeros(shape, bool)
    h, w = big.shape
    out[:h, :w] = big
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--cells", help="сырые координаты в микрометрах")
    ap.add_argument("--aligned", help="готовый csv в пикселях H&E, вместо --cells")
    ap.add_argument("--align", help="матрица, нужна только вместе с --cells")
    ap.add_argument("--he", required=True)
    ap.add_argument("--variant", default="inverse_px")
    ap.add_argument("--orient", default="xy", choices=ORIENTS)
    ap.add_argument("--dx", type=float, default=0.0, help="поправка по x, пиксели H&E")
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--span", type=float,
                    help="перебор вокруг dx и dy на плюс-минус span")
    ap.add_argument("--step", type=float, default=500.0, help="шаг перебора для --span")
    # min:max:step пишется только через знак равенства, иначе argparse принимает
    # минус в начале значения за имя ключа
    ap.add_argument("--sweep-x", help="явный диапазон, --sweep-x=-8000:2000:1000")
    ap.add_argument("--sweep-y")
    ap.add_argument("--overlay", choices=["diff", "cells"], default="diff",
                    help="diff показывает расхождение, cells просто клетки поверх ткани")
    ap.add_argument("--white", type=int, default=220)
    ap.add_argument("--sample", type=int, default=200000)
    args = ap.parse_args()

    H, W = he_shape(args.he)
    thumb = cached_thumb(args.he, args.slide)
    Ht, Wt = thumb.shape[:2]
    sx, sy = Wt / W, Ht / H
    tissue = thumb.mean(-1) < args.white
    k = max(1, int(round(max(Ht, Wt) / 150)))
    tiss = coarsen(tissue, k)

    if not args.aligned and not (args.cells and args.align):
        ap.error("нужен либо --aligned, либо пара --cells и --align")

    xy = pd.read_csv(args.aligned or args.cells)[["x", "y"]].to_numpy(float)
    if len(xy) > args.sample:
        xy = xy[np.random.default_rng(0).choice(len(xy), args.sample, replace=False)]
    if args.aligned:
        pts = xy                                  # уже в пикселях H&E
    else:
        A, t = load_matrix(args.align)
        pts = orient(candidates(xy, A, t)[args.variant], args.orient, W, H)

    def dice_at(dx, dy):
        m, _ = cell_mask(pts, tiss.shape, sx, sy, k, dx * sx, dy * sy)
        return score(m, tiss)[0], m

    xs = ys = None
    if args.span:
        xs = np.arange(args.dx - args.span, args.dx + args.span + args.step / 2, args.step)
        ys = np.arange(args.dy - args.span, args.dy + args.span + args.step / 2, args.step)
    if args.sweep_x:
        xs = parse_range(args.sweep_x)
    if args.sweep_y:
        ys = parse_range(args.sweep_y)

    if xs is not None or ys is not None:
        xs = xs if xs is not None else np.array([args.dx])
        ys = ys if ys is not None else np.array([args.dy])
        print(f"перебор {len(xs)}x{len(ys)} поправок")
        print("     dy: " + " ".join(f"{y:>7.0f}" for y in ys))
        table = {}
        for x in xs:
            row = []
            for y in ys:
                d = dice_at(x, y)[0]
                table[(x, y)] = d
                row.append(f"{d:7.3f}")
            print(f"{x:>7.0f}: " + " ".join(row))
        (bx, by), bd = max(table.items(), key=lambda kv: kv[1])
        print(f"\nлучшее: dx={bx:.0f} dy={by:.0f}, Dice {bd:.3f}")
        if bx in (xs[0], xs[-1]) or by in (ys[0], ys[-1]):
            print("оптимум на краю диапазона, расширьте перебор")
        args.dx, args.dy = bx, by

    d, m = dice_at(args.dx, args.dy)
    _, hit, cover = score(m, tiss)
    print(f"\ndx={args.dx:.0f} dy={args.dy:.0f}: Dice {d:.3f}, "
          f"на ткани {hit:.3f}, покрыто ткани {cover:.3f}")

    vis = thumb.copy()
    if args.overlay == "cells":
        vis[~tissue] = 245
        cx = np.round((pts[:, 0] + args.dx) * sx).astype(int)
        cy = np.round((pts[:, 1] + args.dy) * sy).astype(int)
        ok = (cx >= 0) & (cx < Wt) & (cy >= 0) & (cy < Ht)
        vis[cy[ok], cx[ok]] = (220, 50, 47)
    else:
        vis[upscale(tiss & ~m, k, tissue.shape)] = MISS
        vis[upscale(m & ~tiss, k, tissue.shape)] = OVER
    out = Path("outputs/results")
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{args.slide}_nudge_{args.dx:+.0f}_{args.dy:+.0f}.png"
    Image.fromarray(np.concatenate([thumb, vis], 1)).save(png)
    print("расхождение:", png)


if __name__ == "__main__":
    main()