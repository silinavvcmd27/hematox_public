# Нежёсткая доводка выравнивания по опорным точкам (thin-plate spline).
# Берёт пары соответствий (точка на H&E <- точка на облаке клеток) и плавно
# изгибает координаты всех клеток так, чтобы опорные совпали. Лечит локальную
# деформацию ткани, которую жёсткое преобразование (сдвиг/масштаб/поворот) не берёт.
#
#   python tps_warp.py --cells data/seurat_csv/ovary3_v2_he_cells.csv \
#     --landmarks data/seurat_csv/ovary3_landmarks.csv \
#     --he data/raw/ovary3/..._he_image.ome.tif \
#     --out data/seurat_csv/ovary3_v2_he_tps.csv

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import RBFInterpolator

from refine_align import he_full_shape, load_thumb, tissue_map, cell_map, hit_rate, dice

Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="аффинно выровненный CSV (x,y в px H&E)")
    ap.add_argument("--landmarks", required=True,
                    help="CSV с колонками he_x, he_y, cell_x, cell_y (px полного размера)")
    ap.add_argument("--he", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoothing", type=float, default=0.0,
                    help="сглаживание TPS; 0 = точно через опорные, больше = мягче")
    args = ap.parse_args()

    lm = pd.read_csv(args.landmarks)
    need = {"he_x", "he_y", "cell_x", "cell_y"}
    if not need.issubset(lm.columns):
        raise SystemExit(f"в landmarks нужны колонки {need}, есть {list(lm.columns)}")
    src = lm[["cell_x", "cell_y"]].to_numpy(float)
    dst = lm[["he_x", "he_y"]].to_numpy(float)
    print(f"опорных пар: {len(lm)}")
    if len(lm) < 4:
        raise SystemExit("нужно хотя бы 4 пары, лучше 15-20")

    f = RBFInterpolator(src, dst, kernel="thin_plate_spline", smoothing=args.smoothing)

    df = pd.read_csv(args.cells)
    xy = df[["x", "y"]].to_numpy(float)
    warped = f(xy)

    H, W = he_full_shape(args.he)
    thumb = load_thumb(args.he)
    h, w = thumb.shape[:2]
    scale = W / float(w)
    tis = tissue_map(thumb)
    d0 = dice(cell_map(xy[:, 0], xy[:, 1], (h, w), scale), tis)
    hr0 = hit_rate(tis, xy[:, 0], xy[:, 1], scale)
    d1 = dice(cell_map(warped[:, 0], warped[:, 1], (h, w), scale), tis)
    hr1 = hit_rate(tis, warped[:, 0], warped[:, 1], scale)
    print("до TPS:    Dice %.3f, попадание %.3f" % (d0, hr0))
    print("после TPS: Dice %.3f, попадание %.3f" % (d1, hr1))

    df["x"], df["y"] = warped[:, 0], warped[:, 1]
    inside = (df["x"] >= 0) & (df["x"] < W) & (df["y"] >= 0) & (df["y"] < H)
    df = df[inside].reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("сохранено: %s (%d клеток)" % (args.out, len(df)))

    # QC: клетки поверх ткани до и после
    from PIL import ImageDraw
    def overlay(xx, yy):
        im = Image.fromarray(thumb.copy())
        a = np.round(xx / scale).astype(int)
        b = np.round(yy / scale).astype(int)
        ok = (a >= 0) & (a < w) & (b >= 0) & (b < h)
        arr = np.asarray(im)
        arr = arr.copy()
        arr[b[ok], a[ok]] = (220, 20, 20)
        return arr
    gap = np.full((h, 8, 3), 255, np.uint8)
    qc = np.concatenate([overlay(xy[:, 0], xy[:, 1]), gap,
                         overlay(warped[:, 0], warped[:, 1])], 1)
    qcp = Path(args.out).with_suffix(".qc.png")
    Image.fromarray(qc).save(qcp)
    print("QC (до | после):", qcp)


if __name__ == "__main__":
    main()
