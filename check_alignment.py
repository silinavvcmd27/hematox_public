# Проверка совмещения координат Xenium с H&E.
#
# align_xenium_he.py выбирает преобразование по доле клеток внутри кадра, а это
# слабый критерий: неверное преобразование тоже может уложиться в границы.
# Здесь клетки проверяются по ткани.
#
# Считаем именно Dice, а не долю клеток на ткани. Доля односторонняя: облако,
# сжатое в пятно посреди среза, попадает на ткань почти стопроцентно и выигрывает
# у правильного варианта. Dice за такое штрафует, потому что требует ещё и покрыть
# ткань целиком.
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

from align_xenium_he import ORIENTS, orient

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


def coarsen(tissue, k):
    """Маска ткани, огрублённая в k раз. Клетки это отдельные точки, на мелкой
    сетке между ними дырки, и Dice получается заниженным на ровном месте."""
    Ht, Wt = tissue.shape
    h, w = Ht // k, Wt // k
    return tissue[:h * k, :w * k].reshape(h, k, w, k).mean((1, 3)) > 0.5


def cell_mask(xy, shape, sx, sy, k, dx=0, dy=0):
    """Занятые клетками ячейки той же огрублённой сетки и доля клеток в кадре."""
    h, w = shape
    gx = np.floor((xy[:, 0] * sx + dx) / k).astype(int)
    gy = np.floor((xy[:, 1] * sy + dy) / k).astype(int)
    ok = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
    m = np.zeros(shape, bool)
    m[gy[ok], gx[ok]] = True
    return m, float(ok.mean())


def score(cells, tissue):
    """Dice, доля клеток на ткани и доля ткани, покрытой клетками."""
    inter = np.logical_and(cells, tissue).sum()
    if not cells.any():
        return 0.0, 0.0, 0.0
    return (2 * inter / (cells.sum() + tissue.sum()),
            inter / cells.sum(),
            inter / tissue.sum())


def best_shift(pts, grid, sx, sy, k, tiss, xs, ys):
    scored = {(dx, dy): score(cell_mask(pts, grid, sx, sy, k, dx, dy)[0], tiss)[0]
              for dx in xs for dy in ys}
    return max(scored.items(), key=lambda kv: kv[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--cells", required=True, help="сырые координаты из R, в микрометрах")
    ap.add_argument("--align", required=True)
    ap.add_argument("--he", required=True)
    ap.add_argument("--aligned", help="уже пересчитанный CSV, для сравнения")
    ap.add_argument("--white", type=int, default=220)
    ap.add_argument("--sample", type=int, default=200000)
    ap.add_argument("--shift-span", type=float, default=0.12,
                    help="предел перебора сдвига, долей от большей стороны эскиза")
    args = ap.parse_args()

    H, W = he_shape(args.he)
    thumb = thumbnail(args.he)
    Ht, Wt = thumb.shape[:2]
    sx, sy = Wt / W, Ht / H
    tissue = thumb.mean(-1) < args.white

    k = max(1, int(round(max(Ht, Wt) / 150)))
    tiss = coarsen(tissue, k)
    grid = tiss.shape
    print(f"H&E {W}x{H}, эскиз {Wt}x{Ht}, сетка {grid[1]}x{grid[0]} по {k} px")
    print(f"ткани на эскизе {100 * tissue.mean():.1f}%, ячеек с тканью {tiss.sum()}")

    df = pd.read_csv(args.cells)
    xy = df[["x", "y"]].to_numpy(float)
    if len(xy) > args.sample:
        xy = xy[np.random.default_rng(0).choice(len(xy), args.sample, replace=False)]
    A, t = load_matrix(args.align)

    print("\nвариант преобразования           в кадре  на ткани  покрыто  Dice")
    best = None
    for name, pts in candidates(xy, A, t).items():
        for tag in ORIENTS:
            p = orient(pts, tag, W, H)
            m, inside = cell_mask(p, grid, sx, sy, k)
            d, hit, cover = score(m, tiss)
            print(f"  {name:12s} [{tag:7s}]  {inside:6.3f}   {hit:6.3f}   "
                  f"{cover:6.3f}  {d:6.3f}")
            if best is None or d > best[0]:
                best = (d, name, tag, p, cover)

    d, name, tag, pts, cover = best
    print(f"\nлучший: {name} [{tag}], Dice {d:.3f}, покрыто ткани {cover:.3f}")
    if d < 0.8:
        print("  Dice низкий: преобразование не то, либо картинка не от этого запуска")
    if name != "inverse_px" or tag != "xy":
        print("  align_xenium_he.py по умолчанию берёт inverse_px [xy], "
              "тут нужен другой вариант, готовая команда будет ниже")

    # два прохода: грубо по широкой сетке, потом точно вокруг найденного.
    # одним проходом либо долго, либо оптимум упирается в край диапазона
    print("\nподбор сдвига (пиксели эскиза):")
    span = max(6, int(args.shift_span * max(Ht, Wt)))
    step = max(1, span // 12)
    (cdx, cdy), cd = best_shift(pts, grid, sx, sy, k, tiss,
                                range(-span, span + 1, step),
                                range(-span, span + 1, step))
    print(f"  грубо: dx={cdx} dy={cdy} -> {cd:.3f} (шаг {step}, предел {span})")
    if max(abs(cdx), abs(cdy)) > span - step:
        print("  оптимум на краю диапазона, увеличьте --shift-span")

    (bdx, bdy), bd = best_shift(pts, grid, sx, sy, k, tiss,
                                range(cdx - step, cdx + step + 1),
                                range(cdy - step, cdy + step + 1))
    print(f"  точно: dx={bdx} dy={bdy} -> {bd:.3f}")
    sdx, sdy = bdx / sx, bdy / sy
    print(f"  без сдвига {d:.3f}, со сдвигом {bd:.3f}")
    print(f"  в пикселях исходника: dx={sdx:.0f} dy={sdy:.0f}")
    if bd - d < 0.02:
        print("  сдвиг ничего не даёт, по этому признаку совмещение в порядке")
    else:
        print("  сдвиг заметно улучшает Dice, его надо внести в пересчёт:")
        print(f"  python align_xenium_he.py --cells {args.cells} --align {args.align} "
              f"--he {args.he} --out ПУТЬ.csv --variant {name} --orient {tag} "
              f"--shift-x {sdx:.0f} --shift-y {sdy:.0f}")

    # рисуем то, что получится в итоге, а не вариант из таблицы: без поправки
    # картинка выглядит съехавшей ровно на величину найденного сдвига
    shown, what = pts + np.array([sdx, sdy]), "вариант со сдвигом"
    if args.aligned:
        adf = pd.read_csv(args.aligned)
        axy = adf[["x", "y"]].to_numpy(float)
        if len(axy) > args.sample:
            axy = axy[np.random.default_rng(0).choice(len(axy), args.sample, replace=False)]
        m, inside = cell_mask(axy, grid, sx, sy, k)
        d2, hit2, cover2 = score(m, tiss)
        print(f"\nтекущий {Path(args.aligned).name}: в кадре {inside:.3f}, "
              f"на ткани {hit2:.3f}, покрыто {cover2:.3f}, Dice {d2:.3f}")
        shown, what = axy, Path(args.aligned).name

    vis = thumb.copy()
    vis[~tissue] = 245
    cx = np.round(shown[:, 0] * sx).astype(int)
    cy = np.round(shown[:, 1] * sy).astype(int)
    ok = (cx >= 0) & (cx < Wt) & (cy >= 0) & (cy < Ht)
    vis[cy[ok], cx[ok]] = (220, 50, 47)
    out = Path("outputs/results")
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{args.slide}_align_check.png"
    Image.fromarray(np.concatenate([thumb, vis], 1)).save(png)
    print(f"\nна картинке {what}: {png}")


if __name__ == "__main__":
    main()