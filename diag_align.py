# Перебор всех вариантов трансформации Xenium -> H&E с подбором сдвига и масштаба.
#
# align_xenium_he.py выбирает вариант по покрытию (доля клеток в кадре) — этого
# мало, неверный вариант тоже даёт высокое покрытие. Здесь каждый вариант
# оценивается по попаданию клеток на ткань, что нельзя обмануть.
#
#   python diag_align.py --cells data/seurat_csv/ovary3_cells.csv \
#     --align data/raw/ovary3/..._he_imagealignment.csv \
#     --he data/raw/ovary3/..._he_image.ome.tif

import argparse

import numpy as np
import pandas as pd

from align_xenium_he import load_matrix
from refine_align import (he_full_shape, load_thumb, tissue_map, cell_map,
                          best_shift, hit_rate)

XENIUM_PX_UM = 0.2125


def all_variants(xy_um, A, t, px_um):
    Ainv = np.linalg.inv(A)
    px = xy_um / px_um
    base = {
        "forward_px": (A @ px.T).T + t,
        "forward_um": (A @ xy_um.T).T + t,
        "inverse_px": (Ainv @ (px - t).T).T,
        "inverse_um": (Ainv @ (xy_um - t).T).T,
        "raw_px": px,
        "raw_um": xy_um,
    }
    out = {}
    for nm, c in base.items():
        for swap in (False, True):
            d = c[:, ::-1] if swap else c
            for fx in (1, -1):
                for fy in (1, -1):
                    tag = nm + ("_swap" if swap else "")
                    tag += "" if (fx, fy) == (1, 1) else "_flip%s%s" % (
                        "x" if fx < 0 else "", "y" if fy < 0 else "")
                    out[tag] = np.c_[d[:, 0] * fx, d[:, 1] * fy]
    return out


def tissue_extent(tis, scale):
    rows = np.where(tis.any(1))[0]
    cols = np.where(tis.any(0))[0]
    return ((cols[-1] - cols[0] + 1) * scale, (rows[-1] - rows[0] + 1) * scale)


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2.0 * (a * b).sum() / s) if s else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="исходный CSV клеток (мкм)")
    ap.add_argument("--align", required=True)
    ap.add_argument("--he", required=True)
    ap.add_argument("--px-um", type=float, default=XENIUM_PX_UM)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    df = pd.read_csv(args.cells)
    xy = df[["x", "y"]].to_numpy(float)
    A, t = load_matrix(args.align)
    H, W = he_full_shape(args.he)
    thumb = load_thumb(args.he)
    h, w = thumb.shape[:2]
    scale = W / float(w)
    tis = tissue_map(thumb)
    tw, th = tissue_extent(tis, scale)
    print("H&E: W=%d H=%d | превью %dx%d (downscale %.1f)" % (W, H, w, h, scale))
    print("ткань занимает %.1f%% кадра, габарит %.0fx%.0f px" % (100 * tis.mean(), tw, th))
    print("клеток: %d\n" % len(df))

    rows = []
    for name, c in all_variants(xy, A, t, args.px_um).items():
        lo, hi = np.percentile(c, [1, 99], axis=0)   # габарит без выбросов
        ext = hi - lo
        if ext[0] <= 0 or ext[1] <= 0:
            continue
        c = c - lo
        s_auto = float(np.mean([tw / ext[0], th / ext[1]]))
        for s in sorted({1.0, round(s_auto, 4)}):
            cs = c * s
            cm = cell_map(cs[:, 0], cs[:, 1], (h, w), scale)
            if cm.sum() == 0:
                continue
            dx, dy = best_shift(tis, cm)
            cs = cs + np.array([dx * scale, dy * scale])
            cm = cell_map(cs[:, 0], cs[:, 1], (h, w), scale)
            hr = hit_rate(tis, cs[:, 0], cs[:, 1], scale)
            d = dice(cm, tis)
            fill = float(np.mean([ext[0] * s / tw, ext[1] * s / th]))
            rows.append((d, hr, fill, name, s, dx * scale, dy * scale))

    rows.sort(reverse=True)
    print("%-28s %8s %10s %10s %7s %7s %7s"
          % ("вариант", "масштаб", "dx", "dy", "Dice", "попад.", "габарит"))
    for d, hr, fill, name, s, ddx, ddy in rows[:args.top]:
        print("%-28s %8.4f %10.0f %10.0f %7.3f %7.3f %7.2f"
              % (name, s, ddx, ddy, d, hr, fill))
    print("\nDice — совпадение формы (главный критерий); попад. — доля клеток на ткани;")
    print("габарит — размер облака клеток относительно ткани (1.0 = совпадает).")


if __name__ == "__main__":
    main()