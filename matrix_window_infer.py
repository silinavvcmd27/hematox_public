# Разметка H&E двумя графовыми слоями: опухоль, строма, матриксная строма.
#
# StarDist находит ядра, признак ядра это токен UNI в его точке. Первый слой
# (cell_graph_train.py) делит ядра на опухоль и строму. Второй
# (matrix_window_train.py) едет окном только по стромальным ядрам первого слоя;
# вероятность ядра это среднее по накрывающим его окнам, от --threshold ядро
# считается матриксным.
#
#   python -u matrix_window_infer.py --slide data/tcga_ov_flat/XXX.svs \
#     --layer1 runs/graph/l1.pth --layer2 outputs/models/matrix_win.pth
#
# Окраска приводится через src/stain.py к тому же эталону, что при извлечении
# обучающих признаков. graph_infer.py берёт старый stain_norm.py с табличным
# эталоном по каждому тайлу, и признаки на применении уезжают по цвету от
# обучающих; поэтому его scan_block получает здесь другой нормализатор.

import argparse
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

from src.utils import get_device, ensure_dir, TUMOR, STROMA_MATRIX, STROMA
from src.stain import slide_normalizer
from seg_infer import SlideReader, build_uni
from graph_infer import load_gnn, scan_block, fill_zones, COL
from cell_graph_train import adjacency
from matrix_window_train import build_model, window_members, pool_matrix, cell_prob

Image.MAX_IMAGE_PIXELS = None
NAMES = {TUMOR: "опухоль", STROMA: "строма", STROMA_MATRIX: "матриксная строма"}


def load_window_model(path, device):
    ck = torch.load(path, map_location=device)
    model = build_model(ck.get("model", "gnn"), int(ck["in_dim"]), int(ck["hid"]),
                        int(ck["layers"])).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


@torch.no_grad()
def classify(xy, feat, mpp, l1, l2, ck, device, knn, max_edge_um, threshold):
    d, nb = cKDTree(xy).query(xy, k=min(knn + 1, len(xy)))
    ok = (d[:, 1:] <= max_edge_um / mpp).ravel()
    src = np.repeat(np.arange(len(xy)), nb.shape[1] - 1)[ok]
    dst = nb[:, 1:].ravel()[ok]
    edges = np.vstack([np.r_[src, dst], np.r_[dst, src]])

    x = torch.from_numpy(feat).to(device)
    a = adjacency(edges, len(xy), device)
    stroma = l1(x, a).argmax(1).cpu().numpy() == 1

    cls = np.where(stroma, STROMA, TUMOR).astype(np.uint8)
    member, n_cells = window_members(xy, stroma, mpp, ck["win_um"], ck["stride_um"],
                                     ck["min_cells"])
    prob = np.full(len(xy), np.nan, np.float32)
    if len(n_cells):
        p_win = torch.sigmoid(l2(x, a, pool_matrix(member, n_cells, device))).cpu().numpy()
        prob = cell_prob(member, p_win)
        cls[stroma & (prob >= threshold)] = STROMA_MATRIX
    return cls, prob, float(np.median(d[:, 1]))


def find_font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    import matplotlib
    return ImageFont.truetype(str(Path(matplotlib.__file__).parent / "mpl-data/fonts/ttf/DejaVuSans.ttf"), size)


def legend(canvas, x0, share):
    img = Image.fromarray(canvas)
    dr = ImageDraw.Draw(img)
    font = find_font(24)
    rows = [TUMOR, STROMA, STROMA_MATRIX]
    dr.rectangle([x0 - 12, 6, x0 + 420, 16 + 34 * len(rows)], fill=(255, 255, 255),
                 outline=(120, 120, 120))
    for i, c in enumerate(rows):
        y = 16 + 34 * i
        dr.rectangle([x0, y, x0 + 24, y + 24], fill=COL[c], outline=(60, 60, 60))
        dr.text((x0 + 34, y), f"{NAMES[c]}  {share[c]:.1f}%", fill=(20, 20, 20), font=font)
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--layer1", required=True, help="опухоль/строма, cell_graph_train.py")
    ap.add_argument("--layer2", required=True, help="окно, matrix_window_train.py")
    ap.add_argument("--stain-ref", default="data/processed/stain_ref_prime.npz")
    ap.add_argument("--mpp", type=float, default=None)
    ap.add_argument("--target-mpp", type=float, default=0.2738)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--max-edge-um", type=float, default=50.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = get_device()
    print("device:", device, "| гружу StarDist, UNI и оба слоя")
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
    sd = StarDist2D.from_pretrained("2D_versatile_he")
    uni = build_uni(device)
    l1 = load_gnn(args.layer1, device)
    l2, ck = load_window_model(args.layer2, device)
    print(f"второй слой {ck.get('model', 'gnn')}, окно {ck['win_um']:.0f} мкм, "
          f"шаг {ck['stride_um']:.0f}, модель с эпохи {ck['best_epoch']}")

    sl = SlideReader(args.slide, args.target_mpp, args.mpp)
    # scan_block из graph_infer.py вызывает norm.transform(tile)
    norm = types.SimpleNamespace(transform=slide_normalizer(args.slide, args.stain_ref))

    bs, ps = args.block, args.patch_size
    starts = [(bx, by) for by in range(0, sl.H, bs) for bx in range(0, sl.W, bs)]
    print(f"{sl.W}x{sl.H} @ {sl.mpp:.3f} мкм/px, блоков {len(starts)}")
    xy_parts, feat_parts = [], []
    for i, (bx, by) in enumerate(starts, 1):
        xy, fe, _ = scan_block(sl, sd, normalize, uni, norm, device, bx, by, bs, ps, False)
        if len(xy):
            xy_parts.append(xy)
            feat_parts.append(fe)
        print(f"  блок {i}/{len(starts)}, ядер {sum(len(p) for p in xy_parts)}   ", end="\r")
    if not xy_parts:
        raise SystemExit("\nядер не найдено")
    xy, feat = np.concatenate(xy_parts), np.concatenate(feat_parts)
    print(f"\nядер всего: {len(xy)}")

    cls, prob, med = classify(xy, feat, sl.mpp, l1, l2, ck, device, args.knn,
                              args.max_edge_um, args.threshold)
    counts = {c: int((cls == c).sum()) for c in NAMES}
    share = {c: 100 * n / len(cls) for c, n in counts.items()}
    n_stroma = counts[STROMA] + counts[STROMA_MATRIX]
    print(f"TSR по ядрам: {n_stroma / len(cls):.3f}")
    if n_stroma:
        print(f"матриксная: {100 * counts[STROMA_MATRIX] / n_stroma:.1f}% стромальных ядер")
        outside = int(((cls != TUMOR) & np.isnan(prob)).sum())
        print(f"стромальных ядер вне окон (меньше {ck['min_cells']} стромальных рядом): "
              f"{outside} ({100 * outside / n_stroma:.1f}%)")

    md = min(2400, max(sl.W, sl.H))
    tw, th = int(sl.W * md / max(sl.W, sl.H)), int(sl.H * md / max(sl.W, sl.H))
    thumb = sl.thumbnail(tw, th)
    sc = sl.W / tw
    ov = fill_zones(thumb, xy, cls, 0, 0, sc, max(2.5, 1.3 * med / sc), 0.55)
    canvas = legend(np.concatenate([thumb, ov], 1), tw + 20, share)

    stem = Path(args.slide).stem
    out = Path(args.out) if args.out else ensure_dir("outputs/results") / f"{stem}_matrix_graph.png"
    Image.fromarray(canvas).save(out)
    np.savez_compressed(out.with_suffix(".npz"), xy=xy, cls=cls, prob=prob.astype(np.float16),
                        mpp=sl.mpp, layer1=str(args.layer1), layer2=str(args.layer2),
                        threshold=args.threshold)
    sl.close()
    print("карта:", out)
    print("ядра и вероятности:", out.with_suffix(".npz"))


if __name__ == "__main__":
    main()
