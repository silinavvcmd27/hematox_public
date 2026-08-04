# Картинки для схемы модели: миниатюры, увеличенные фрагменты, сетка токенов,
# граф клеток поверх ткани, маска Вороного.
#
#   python make_figure_assets.py --slide ovary_prime_he \
#     --he data/raw/ovary_prime/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_he_image.ome.tif \
#     --cells data/seurat_csv/ovary_prime_he_cells_aligned.csv

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from src.utils import (ensure_dir, TRAIN_CLASSES, CLASS_NAMES,
                       TUMOR, STROMA_HORMONAL, STROMA_MATRIX, IMMUNE, STROMA)
from src.data.seurat_labels import CellTypeMapper

Image.MAX_IMAGE_PIXELS = None

COL = {
    TUMOR:           (216,  65,  47),
    STROMA_HORMONAL: (240, 160,  48),
    STROMA_MATRIX:   ( 46, 134, 193),
    IMMUNE:          ( 42, 161,  82),
    STROMA:          (128, 128, 128),
}


def thumb(img, max_dim=1400):
    h, w = img.shape[:2]
    s = max(h, w) / max_dim
    if s <= 1:
        return img.copy(), 1.0
    out = np.asarray(Image.fromarray(img).resize(
        (int(w / s), int(h / s)), Image.LANCZOS))
    return out, s


def pick_zoom(cells, cls, W, H, size, step=4):
    """Окно с наибольшим разнообразием классов — там схема выглядит осмысленно."""
    best, bxy = -1.0, (max(0, W // 2 - size // 2), max(0, H // 2 - size // 2))
    xs = np.linspace(0, max(1, W - size), step).astype(int)
    ys = np.linspace(0, max(1, H - size), step).astype(int)
    for y0 in ys:
        for x0 in xs:
            m = ((cells[:, 0] >= x0) & (cells[:, 0] < x0 + size) &
                 (cells[:, 1] >= y0) & (cells[:, 1] < y0 + size))
            n = int(m.sum())
            if n < 200:
                continue
            p = np.array([(cls[m] == c).mean() for c in TRAIN_CLASSES])
            p = p[p > 0]
            ent = float(-(p * np.log(p)).sum())
            score = ent * min(n, 3000)
            if score > best:
                best, bxy = score, (int(x0), int(y0))
    return bxy


def dots(img, xy, cls, r=3, outline=False):
    im = Image.fromarray(img.copy())
    dr = ImageDraw.Draw(im)
    for (x, y), c in zip(xy, cls):
        col = COL.get(int(c))
        if col is None:
            continue
        dr.ellipse([x - r, y - r, x + r, y + r], fill=col,
                   outline=(30, 30, 30) if outline else None)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--he", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--map", default="config/cell_type_map.yaml")
    ap.add_argument("--seg-dir", default="data/processed/seg")
    ap.add_argument("--zoom", type=int, default=2048, help="сторона фрагмента, px")
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--graph-um", type=float, default=50.0)
    ap.add_argument("--mpp", type=float, default=0.2738)
    ap.add_argument("--out-dir", default="outputs/figure")
    args = ap.parse_args()

    out = ensure_dir(args.out_dir)
    df = pd.read_csv(args.cells)
    mapper = CellTypeMapper(args.map)
    df["zone"] = mapper.map_series(df["cell_type"])
    name2cls = {CLASS_NAMES[c]: c for c in TRAIN_CLASSES}
    df["cls"] = df["zone"].map(name2cls)
    df = df.dropna(subset=["cls", "x", "y"])
    xy = df[["x", "y"]].to_numpy(float)
    cls = df["cls"].to_numpy(int)
    print(f"клеток с классом: {len(df)}")

    from src.data.patching import load_image
    print("грузим H&E...")
    img = load_image(args.he)
    H, W = img.shape[:2]
    print(f"{W}x{H}")

    # 1-2. миниатюра целого среза и она же с клетками
    th, s = thumb(img)
    Image.fromarray(th).save(out / "a1_he_thumb.png")
    Image.fromarray(dots(th, xy / s, cls, r=1)).save(out / "a2_cells_thumb.png")
    print("a1_he_thumb.png, a2_cells_thumb.png")

    # 3-4. увеличенный фрагмент
    z = args.zoom
    x0, y0 = pick_zoom(xy, cls, W, H, z)
    crop = img[y0:y0 + z, x0:x0 + z]
    m = ((xy[:, 0] >= x0) & (xy[:, 0] < x0 + z) &
         (xy[:, 1] >= y0) & (xy[:, 1] < y0 + z))
    zxy, zcls = xy[m] - [x0, y0], cls[m]
    Image.fromarray(crop).save(out / "a3_zoom_he.png")
    Image.fromarray(dots(crop, zxy, zcls, r=6, outline=True)).save(out / "a4_zoom_cells.png")
    print(f"a3_zoom_he.png, a4_zoom_cells.png  (окно {x0},{y0} размер {z}, клеток {m.sum()})")

    # 5. маска Вороного на том же фрагменте
    mp = Path(args.seg_dir) / f"{args.slide}_mask.npz"
    if mp.exists():
        d = np.load(mp)
        mask, ds = d["mask"], int(d["downscale"])
        sub = mask[y0 // ds:(y0 + z) // ds, x0 // ds:(x0 + z) // ds]
        rgb = np.full(sub.shape + (3,), 245, np.uint8)
        for c, col in COL.items():
            rgb[sub == c] = col
        Image.fromarray(rgb).resize((z // 2, z // 2), Image.NEAREST).save(
            out / "a5_voronoi.png")
        print("a5_voronoi.png  (истина: класс ближайшей клетки)")
    else:
        print(f"нет {mp} — a5_voronoi.png пропущен")

    # 6. патч с сеткой токенов 14x14
    ps = args.patch
    px0, py0 = x0 + z // 2 - ps // 2, y0 + z // 2 - ps // 2
    patch = img[py0:py0 + ps, px0:px0 + ps]
    big = Image.fromarray(patch).resize((ps * 3, ps * 3), Image.LANCZOS)
    dr = ImageDraw.Draw(big)
    stepg = ps * 3 / 14
    for i in range(1, 14):
        dr.line([(i * stepg, 0), (i * stepg, ps * 3)], fill=(60, 60, 60), width=1)
        dr.line([(0, i * stepg), (ps * 3, i * stepg)], fill=(60, 60, 60), width=1)
    dr.rectangle([0, 0, ps * 3 - 1, ps * 3 - 1], outline=(200, 150, 0), width=4)
    big.save(out / "b1_patch_grid.png")
    print("b1_patch_grid.png  (патч 256 px = 70 мкм, сетка 14x14)")

    # 7. граф клеток поверх фрагмента
    gz = min(z, 900)
    gx0, gy0 = x0 + (z - gz) // 2, y0 + (z - gz) // 2
    gcrop = img[gy0:gy0 + gz, gx0:gx0 + gz]
    mg = ((xy[:, 0] >= gx0) & (xy[:, 0] < gx0 + gz) &
          (xy[:, 1] >= gy0) & (xy[:, 1] < gy0 + gz))
    gxy, gcls = xy[mg] - [gx0, gy0], cls[mg]
    gim = Image.fromarray(gcrop.copy()).resize((gz * 2, gz * 2), Image.LANCZOS)
    dr = ImageDraw.Draw(gim, "RGBA")
    if len(gxy) > 2:
        tree = cKDTree(gxy)
        d, nb = tree.query(gxy, k=min(9, len(gxy)))
        r_px = args.graph_um / args.mpp
        for i in range(len(gxy)):
            for j, dist in zip(nb[i][1:], d[i][1:]):
                if dist <= r_px:
                    dr.line([tuple(gxy[i] * 2), tuple(gxy[j] * 2)],
                            fill=(35, 35, 35, 150), width=2)
    for (x, y), c in zip(gxy * 2, gcls):
        col = COL.get(int(c))
        if col:
            dr.ellipse([x - 9, y - 9, x + 9, y + 9], fill=col + (235,),
                       outline=(25, 25, 25), width=2)
    gim.save(out / "b2_graph_zoom.png")
    print(f"b2_graph_zoom.png  (узлов {len(gxy)}, рёбра до {args.graph_um:.0f} мкм)")

    del img
    print("\nготово ->", out)


if __name__ == "__main__":
    main()