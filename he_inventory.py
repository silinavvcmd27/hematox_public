# Что за H&E лежат в папках и какому срезу Xenium какой соответствует.
#
# Опознаём по физическому размеру: ширина в пикселях, умноженная на мкм/px, должна
# совпасть с размахом облака клеток из Seurat, которое уже в микрометрах. Совпадение
# с точностью до пары процентов и есть ответ, картинку открывать не нужно.
#
# python he_inventory.py data/raw /mnt/singlecellproject/public_data/Ovarian_cancer \
#     --cells data/seurat_csv

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# снимки DAPI от самого Xenium тоже .ome.tif, но это не H&E и открывать их незачем
NOT_HE = ("morphology", "_mip", "focus")


def find(roots, ext, depth):
    """Обход с ограничением глубины. Рекурсивный rglob по NFS уходит в минуты,
    а нужные файлы лежат на второй-третьей ступени от корня."""
    out = []
    for root in roots:
        p = Path(root)
        if p.is_file():
            out.append(p)
            continue
        for d in range(depth + 1):
            print(f"  ищу в {p}, глубина {d}", flush=True)
            out += p.glob("*/" * d + ext)
    return sorted(set(out))


def he_info(path):
    import tifffile

    with tifffile.TiffFile(str(path)) as tf:
        s = tf.series[0]
        shape, axes = s.shape, (s.axes or "")
        levels = len(list(s.levels))
    if "Y" in axes and "X" in axes:
        H, W = shape[axes.index("Y")], shape[axes.index("X")]
    else:
        dims = [d for d in shape if d > 8]
        H, W = dims[0], dims[1]

    try:
        from slide_mpp import slide_mpp
        mpp = slide_mpp(str(path))
    except Exception as e:
        mpp = None
        print(f"     мкм/px не читается: {e}")
    return {"path": path, "W": W, "H": H, "levels": levels, "mpp": mpp,
            "mm_w": W * mpp / 1000 if mpp else None,
            "mm_h": H * mpp / 1000 if mpp else None}


# как координаты называются в выгрузках, которые ходят по проекту
COORD_COLS = [("x", "y"), ("x_centroid", "y_centroid"), ("x_slide_mm", "y_slide_mm"),
              ("x_um", "y_um"), ("imagecol", "imagerow")]


def cells_info(path):
    head = pd.read_csv(path, nrows=0).columns
    pair = next((p for p in COORD_COLS if p[0] in head and p[1] in head), None)
    if pair is None:
        print(f"  {path.name}: колонок с координатами нет, "
              f"есть {list(head)[:8]}")
        return None
    df = pd.read_csv(path, usecols=list(pair))
    x, y = df[pair[0]].to_numpy(), df[pair[1]].to_numpy()
    to_mm = 1.0 if pair[0].endswith("_mm") else 0.001
    return {"path": path, "n": len(df), "cols": pair,
            "mm_w": (x.max() - x.min()) * to_mm,
            "mm_h": (y.max() - y.min()) * to_mm}


def mismatch(he, cells):
    """Относительное расхождение размеров, с учётом того, что H&E может быть
    повёрнут на 90 градусов относительно кадра Xenium."""
    a = np.array([he["mm_w"], he["mm_h"]])
    b = np.array([cells["mm_w"], cells["mm_h"]])
    return min(np.abs(a - b).max() / b.max(), np.abs(a[::-1] - b).max() / b.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="папки или файлы с H&E")
    ap.add_argument("--cells", nargs="*", default=[],
                    help="папки или csv с координатами клеток в микрометрах")
    ap.add_argument("--depth", type=int, default=3,
                    help="на сколько уровней вглубь спускаться от каждой папки")
    ap.add_argument("--all", action="store_true",
                    help="не пропускать снимки DAPI")
    args = ap.parse_args()

    files = find(args.roots, "*.tif*", args.depth)
    if not args.all:
        files = [f for f in files if not any(s in f.name.lower() for s in NOT_HE)]
    print(f"\nнашлось файлов: {len(files)}\n")

    print("H&E:")
    hes = []
    for f in files:
        print(f"  читаю {f.name}", flush=True)
        info = he_info(f)
        hes.append(info)
        mpp = f"{info['mpp']:.4f}" if info["mpp"] else "?"
        size = (f"{info['mm_w']:.1f}x{info['mm_h']:.1f} мм"
                if info["mpp"] else "размер в мм неизвестен")
        print(f"     {info['W']:>6} x {info['H']:<6} мкм/px {mpp:>7}  {size:>20}  "
              f"уровней {info['levels']}")
        near = sorted(f.parent.glob("*alignment*.csv"))
        print("     матрица:", near[0].name if near else "рядом нет alignment csv")
        print("    ", f)

    if not args.cells:
        return

    print("\nклетки:")
    clouds = [cells_info(f) for f in find(args.cells, "*.csv", args.depth)]
    clouds = [c for c in clouds if c]
    for c in clouds:
        print(f"  {c['n']:>8} клеток  {c['mm_w']:.1f}x{c['mm_h']:.1f} мм  "
              f"по {c['cols'][0]}/{c['cols'][1]}  {c['path']}")

    print("\nпары (расхождение размеров, меньше 0.05 это совпадение):")
    for c in clouds:
        scored = sorted(((mismatch(h, c), h) for h in hes if h["mpp"]),
                        key=lambda kv: kv[0])
        for err, h in scored[:3]:
            print(f"  {c['path'].name:<40} {err:5.3f}  {h['path'].name}")
        print()


if __name__ == "__main__":
    main()