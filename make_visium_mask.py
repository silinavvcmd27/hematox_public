# Visium: споты (tumor/stroma) -> плотная маска Вороного на hires H&E.
# Формат как у Xenium (mask.npz + patches.csv), чтобы seg_extract.py их ел.
#   python make_visium_mask.py --name SP1
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from src.utils import TUMOR, STROMA

Image.MAX_IMAGE_PIXELS = None
VIS = Path("data/processed/visium")
LAB2CLS = {"tumor": TUMOR, "stroma": STROMA}
COL = {0: (245, 245, 245), TUMOR: (220, 50, 47), STROMA: (128, 128, 128)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--min-tissue", type=float, default=0.25,
                    help="мин. доля размеченных пикселей в патче")
    args = ap.parse_args()

    he = np.asarray(Image.open(VIS / f"{args.name}_he.png").convert("RGB"))
    H, W = he.shape[:2]
    df = pd.read_csv(VIS / f"{args.name}_spots.csv").dropna(subset=["label"])
    scale = float(df["hires_scale"].iloc[0])

    sx = df["x_full"].to_numpy() * scale          # горизонталь (столбец)
    sy = df["y_full"].to_numpy() * scale          # вертикаль (строка)
    cls = df["label"].map(LAB2CLS).to_numpy()

    centers = np.c_[sx, sy]
    nn = cKDTree(centers).query(centers, k=2)[0][:, 1]
    max_dist = 1.2 * float(np.median(nn))         # не заливать за пределы ткани

    ys, xs = np.mgrid[0:H, 0:W]
    d, idx = cKDTree(centers).query(np.c_[xs.ravel(), ys.ravel()])
    mask = np.zeros(H * W, np.uint8)
    inside = d < max_dist
    mask[inside] = cls[idx[inside]]
    mask = mask.reshape(H, W)
    np.savez_compressed(VIS / f"{args.name}_mask.npz", mask=mask, downscale=1)

    rows = []
    for y0 in range(0, H - args.patch_size + 1, args.stride):
        for x0 in range(0, W - args.patch_size + 1, args.stride):
            sub = mask[y0:y0 + args.patch_size, x0:x0 + args.patch_size]
            if (sub > 0).mean() >= args.min_tissue:
                rows.append({"x0": x0, "y0": y0,
                             "tumor": float((sub == TUMOR).mean()),
                             "stroma": float((sub == STROMA).mean())})
    pd.DataFrame(rows).to_csv(VIS / f"{args.name}_patches.csv", index=False)
    print(f"{args.name}: {len(rows)} патчей | max_dist {max_dist:.0f}px | "
          f"tumor {int((mask==TUMOR).sum())} / stroma {int((mask==STROMA).sum())} px")

    # QC: H&E | H&E со спотами | маска
    im = Image.fromarray(he.copy())
    dr = ImageDraw.Draw(im)
    r = max(3, int(max_dist / 2))
    for x, y, c in zip(sx, sy, cls):
        dr.ellipse([x - r, y - r, x + r, y + r], fill=COL[int(c)])
    mcol = np.zeros((H, W, 3), np.uint8)
    for c, col in COL.items():
        mcol[mask == c] = col
    qc = np.concatenate([he, np.asarray(im), mcol], 1)
    Image.fromarray(qc).save(VIS / f"{args.name}_qc.png")
    print("QC:", VIS / f"{args.name}_qc.png")


if __name__ == "__main__":
    main()