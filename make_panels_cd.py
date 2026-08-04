# Недостающие картинки для схемы:
#   c3_layer2_pred.png — выход модели (5 зон) на обучающем срезе, пара к c2 (истина)
#   d1_tcga_he.png     — фрагмент H&E чужого среза (TCGA), вход панели d
#   d2_zonemap.png     — тот же фрагмент, залитый зонами (клеточные территории), выход
#
# Тот же инференс, что в боевом graph_infer.
#   c3 (обучающий срез, граф уже построен — быстро):
#     python make_panels_cd.py --he data/raw/ovary2/..._he_image.ome.tif \
#         --slide ovary2_he --layer1 runs/graph/l1_final.pth --layer2 runs/graph/l2_final.pth
#   d1/d2 (TCGA, StarDist+UNI на регионе — минуты):
#     python make_panels_cd.py --tcga data/tcga_ov_flat/TCGA-57-1585-...svs \
#         --layer1 runs/graph/l1_final.pth --layer2 runs/graph/l2_final.pth --size 3072

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import torch
from PIL import Image

from src.utils import get_device, ensure_dir, TUMOR
from graph_infer import (load_gnn, classify, scan_block, fill_cells,
                         L2_TO_CLASS, COL)
from cell_graph_train import adjacency
from make_figure_assets import pick_zoom, dots, thumb

Image.MAX_IMAGE_PIXELS = None


def combine(p1, p2):
    """L1 (0 опухоль / 1 строма) + L2-канал -> итоговый класс зоны."""
    cls = np.empty(len(p1), np.uint8)
    cls[p1 == 0] = TUMOR
    sub = np.array([L2_TO_CLASS[int(c)] for c in p2], np.uint8)
    cls[p1 == 1] = sub[p1 == 1]
    return cls


def panel_c3(args, device, l1, l2, out):
    """Предсказание 5 зон на обучающем срезе — то же окно, что у c0-c2."""
    g = np.load(Path(args.graph_dir) / f"{args.slide}_graph.npz")
    pos, label, edges = g["pos"], g["label"], g["edges"]
    X = torch.from_numpy(g["feat"].astype(np.float32)).to(device)
    A = adjacency(edges, len(label), device)
    with torch.no_grad():
        p1 = l1(X, A).argmax(1).cpu().numpy()
        p2 = l2(X, A).argmax(1).cpu().numpy()
    cls = combine(p1, p2)
    print(f"{args.slide}: узлов {len(label)}, опухоль {int((cls==TUMOR).sum())}, "
          f"строма {int((cls!=TUMOR).sum())}")

    from src.data.patching import load_image
    print("грузим H&E обучающего среза...")
    img = load_image(args.he)
    H, W = img.shape[:2]
    z = args.zoom
    x0, y0 = pick_zoom(pos, label, W, H, z)
    crop = img[y0:y0 + z, x0:x0 + z]
    m = ((pos[:, 0] >= x0) & (pos[:, 0] < x0 + z) &
         (pos[:, 1] >= y0) & (pos[:, 1] < y0 + z))
    zxy = pos[m] - [x0, y0]
    Image.fromarray(dots(crop, zxy, cls[m], r=6, outline=True)).save(out / "c3_layer2_pred.png")
    print(f"c3_layer2_pred.png  (окно {x0},{y0} размер {z}, клеток {int(m.sum())})")
    del img


def panel_d(args, device, l1, l2, out):
    """Фрагмент TCGA: чистый H&E + карта зон (те же клеточные территории)."""
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
    from seg_infer import SlideReader, build_uni
    from stain_norm import MacenkoNormalizer

    sd = StarDist2D.from_pretrained("2D_versatile_he")
    uni = build_uni(device)
    norm = MacenkoNormalizer()
    sl = SlideReader(args.tcga, args.target_mpp, None)
    S = min(args.size, sl.W, sl.H)
    rx, ry = args.region if args.region else ((sl.W - S) // 2, (sl.H - S) // 2)
    print(f"TCGA {sl.W}x{sl.H} @ {sl.mpp:.3f} мкм/px | регион x{rx} y{ry} {S}x{S}")
    xy, Fe, _ = scan_block(sl, sd, normalize, uni, norm, device, rx, ry, S, args.patch_size, False)
    print(f"ядер: {len(xy)}")
    if len(xy) < 40:
        raise SystemExit("мало ядер — выбери другой --region")
    cls, med = classify(sl, xy, Fe, l1, l2, device, args.knn, args.max_edge_um)
    he = sl.read(rx, ry, S)
    zone = fill_cells(he, xy, cls, rx, ry, 1.3 * med)
    he_small, _ = thumb(he, args.fig_dim)
    zone_small, _ = thumb(zone, args.fig_dim)
    Image.fromarray(he_small).save(out / "d1_tcga_he.png")
    Image.fromarray(zone_small).save(out / "d2_zonemap.png")
    print("d1_tcga_he.png, d2_zonemap.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--he", help="H&E обучающего среза (для c3)")
    ap.add_argument("--slide", help="имя графа обучающего среза, напр. ovary2_he (для c3)")
    ap.add_argument("--tcga", help="svs TCGA (для d1/d2)")
    ap.add_argument("--layer1", required=True)
    ap.add_argument("--layer2", required=True)
    ap.add_argument("--graph-dir", default="data/processed/graph")
    ap.add_argument("--zoom", type=int, default=2048)
    ap.add_argument("--size", type=int, default=3072, help="сторона региона TCGA")
    ap.add_argument("--region", type=int, nargs=2, default=None)
    ap.add_argument("--target-mpp", type=float, default=0.27)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--max-edge-um", type=float, default=50.0)
    ap.add_argument("--fig-dim", type=int, default=1400, help="макс. сторона фигурных PNG")
    ap.add_argument("--out-dir", default="outputs/figure")
    args = ap.parse_args()

    out = ensure_dir(args.out_dir)
    device = get_device()
    print("device:", device, "| гружу графовые модели...")
    l1 = load_gnn(args.layer1, device)
    l2 = load_gnn(args.layer2, device)

    if args.he and args.slide:
        panel_c3(args, device, l1, l2, out)
    if args.tcga:
        panel_d(args, device, l1, l2, out)
    if not (args.he or args.tcga):
        raise SystemExit("нужен --he+--slide (для c3) и/или --tcga (для d1/d2)")
    print("\nготово ->", out)


if __name__ == "__main__":
    main()
