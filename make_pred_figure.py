# Результат слоя 1 для схемы: предсказанный класс каждой клетки поверх ткани.
# Рядом истина — видно, где модель ошибается.
#
#   python make_pred_figure.py --slide ovary2_he \
#     --he data/raw/ovary2/Xenium_Prime_Human_Ovary_FF_he_image.ome.tif \
#     --model runs/graph/l1.pth

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.utils import get_device, ensure_dir, TUMOR
from cell_graph_train import CellGNN, adjacency, to_binary
from make_figure_assets import pick_zoom, dots

Image.MAX_IMAGE_PIXELS = None
PRED_COL = {0: (216, 65, 47), 1: (128, 128, 128)}      # tumor, stroma


def render(crop, xy, cls, path, r=6):
    im = Image.fromarray(crop.copy())
    from PIL import ImageDraw
    dr = ImageDraw.Draw(im)
    for (x, y), c in zip(xy, cls):
        col = PRED_COL.get(int(c))
        if col:
            dr.ellipse([x - r, y - r, x + r, y + r], fill=col, outline=(30, 30, 30))
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--he", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--graph-dir", default="data/processed/graph")
    ap.add_argument("--zoom", type=int, default=2048)
    ap.add_argument("--out-dir", default="outputs/figure")
    args = ap.parse_args()

    out = ensure_dir(args.out_dir)
    g = np.load(Path(args.graph_dir) / f"{args.slide}_graph.npz")
    pos, label, edges = g["pos"], g["label"], g["edges"]
    truth = to_binary(label)
    print(f"{args.slide}: узлов {len(label)}")

    device = get_device()
    ck = torch.load(args.model, map_location=device)
    model = CellGNN(int(ck["in_dim"]), int(ck["hid"]), 2, int(ck["layers"])).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    X = torch.from_numpy(g["feat"].astype(np.float32)).to(device)
    A = adjacency(edges, len(label), device)
    with torch.no_grad():
        pred = model(X, A).argmax(1).cpu().numpy()
    ok = truth != -100
    acc = float((pred[ok] == truth[ok]).mean())
    print(f"совпадение с истиной: {100*acc:.1f}%")

    from src.data.patching import load_image
    print("грузим H&E...")
    img = load_image(args.he)
    H, W = img.shape[:2]

    z = args.zoom
    x0, y0 = pick_zoom(pos, label, W, H, z)
    crop = img[y0:y0 + z, x0:x0 + z]
    m = ((pos[:, 0] >= x0) & (pos[:, 0] < x0 + z) &
         (pos[:, 1] >= y0) & (pos[:, 1] < y0 + z))
    zxy = pos[m] - [x0, y0]
    print(f"окно {x0},{y0} размер {z}, клеток {int(m.sum())}")

    render(crop, zxy, pred[m], out / "c1_layer1_pred.png")
    render(crop, zxy, truth[m], out / "c0_layer1_truth.png")
    dm = dots(crop, zxy, label[m], r=6, outline=True)
    Image.fromarray(dm).save(out / "c2_alltypes_truth.png")
    del img
    print("c0_layer1_truth.png, c1_layer1_pred.png, c2_alltypes_truth.png ->", out)


if __name__ == "__main__":
    main()
