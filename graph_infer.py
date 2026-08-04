# Полный инференс графовой моделью на обычном H&E (без транскриптомики).
#   StarDist -> токены UNI -> граф клеток -> слой 1 (tumor/stroma) + слой 2
#   (подтип стромы) -> карта зон, TSR, состав стромы.
#
# Область (близкий план, заливка по контурам ядер):
#   python graph_infer.py --slide X.svs --layer1 .. --layer2 .. --size 4096
# Весь срез (обзор зонами):
#   python graph_infer.py --slide X.svs --layer1 .. --layer2 .. --full

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

from src.utils import (get_device, ensure_dir, TUMOR, STROMA_HORMONAL,
                       STROMA_MATRIX, IMMUNE, STROMA, CLASS_NAMES)
from seg_infer import SlideReader, build_uni, _tf, GRID
from stain_norm import MacenkoNormalizer
from cell_graph_train import CellGNN, adjacency

Image.MAX_IMAGE_PIXELS = None

COL = {TUMOR: (216, 65, 47), STROMA_HORMONAL: (240, 160, 48),
       STROMA_MATRIX: (46, 134, 193), IMMUNE: (42, 161, 82), STROMA: (128, 128, 128)}
L2_TO_CLASS = {0: STROMA_HORMONAL, 1: STROMA_MATRIX, 2: IMMUNE, 3: STROMA}
STROMA_ALL = [STROMA_HORMONAL, STROMA_MATRIX, IMMUNE, STROMA]


def load_gnn(path, device):
    ck = torch.load(path, map_location=device)
    m = CellGNN(int(ck["in_dim"]), int(ck["hid"]), len(ck["classes"]),
                int(ck["layers"])).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


@torch.no_grad()
def uni_tokens(uni, patch, device):
    x = _tf(Image.fromarray(patch)).unsqueeze(0).to(device)
    f = uni.forward_features(x)
    if f.shape[1] == GRID * GRID + 1:
        f = f[:, 1:, :]
    elif f.shape[1] != GRID * GRID:
        f = f[:, -GRID * GRID:, :]
    return f[0].reshape(GRID, GRID, -1).cpu().numpy()


def classify(sl, xy, Fe, l1, l2, device, knn, max_edge_um):
    tree = cKDTree(xy)
    d, nb = tree.query(xy, k=min(knn + 1, len(xy)))
    r_px = max_edge_um / sl.mpp
    src = np.repeat(np.arange(len(xy)), nb.shape[1] - 1)
    dst = nb[:, 1:].ravel()
    ok = d[:, 1:].ravel() <= r_px
    edges = np.vstack([np.r_[src[ok], dst[ok]], np.r_[dst[ok], src[ok]]]).astype(np.int64)
    X = torch.from_numpy(Fe).to(device)
    A = adjacency(edges, len(xy), device)
    with torch.no_grad():
        p1 = l1(X, A).argmax(1).cpu().numpy()
        p2 = l2(X, A).argmax(1).cpu().numpy()
    cls = np.empty(len(xy), np.uint8)
    cls[p1 == 0] = TUMOR
    sub = np.array([L2_TO_CLASS[int(c)] for c in p2], np.uint8)
    cls[p1 == 1] = sub[p1 == 1]
    return cls, float(np.median(d[:, 1]))


def scan_block(sl, sd, normalize, uni, norm, device, bx, by, bs, ps, want_polys):
    """Ядра одного блока: (xy, признаки, [полигоны])."""
    feats, coords, polys = [], [], []
    for y in range(by, min(by + bs, sl.H) - ps + 1, ps):
        for x in range(bx, min(bx + bs, sl.W) - ps + 1, ps):
            reg = sl.read(x, y, ps)
            if (reg.mean(-1) > 220).mean() > 0.7:
                continue
            reg = norm.transform(reg)
            _, det = sd.predict_instances(normalize(reg, 1, 99.8, axis=(0, 1)))
            pts = det["points"]
            if len(pts) == 0:
                continue
            tok = uni_tokens(uni, reg, device)
            coord = det["coord"] if want_polys else None
            for i, (r, c) in enumerate(pts):
                ti = min(int(r / ps * GRID), GRID - 1)
                tj = min(int(c / ps * GRID), GRID - 1)
                feats.append(tok[ti, tj])
                coords.append((x + c, y + r))
                if want_polys:
                    ys, xs = coord[i, 0] + y, coord[i, 1] + x
                    polys.append(np.stack([xs, ys], 1).astype(np.float32))
    xy = np.asarray(coords, np.float32) if coords else np.zeros((0, 2), np.float32)
    Fe = np.asarray(feats, np.float32) if feats else np.zeros((0, 1024), np.float32)
    return xy, Fe, polys


def fill_zones(bg, xy, cls, ox, oy, sc, fill_r, alpha=0.5):
    h, w = bg.shape[:2]
    step = 2 if sc > 2 else 1
    hs, ws = h // step, w // step
    gx, gy = np.meshgrid(np.arange(ws) * step, np.arange(hs) * step)
    dist, idx = cKDTree(np.c_[(xy[:, 0] - ox) / sc, (xy[:, 1] - oy) / sc]).query(np.c_[gx.ravel(), gy.ravel()])
    grid = np.zeros(len(idx), np.uint8)
    ins = dist < fill_r
    grid[ins] = cls[idx[ins]]
    grid = grid.reshape(hs, ws)
    cmap = np.zeros((hs, ws, 3), np.uint8)
    for c, col in COL.items():
        cmap[grid == c] = col
    if step > 1:
        cmap = np.asarray(Image.fromarray(cmap).resize((w, h), Image.NEAREST))
    ov = bg.copy()
    mm = cmap.sum(-1) > 0
    ov[mm] = (alpha * cmap[mm] + (1 - alpha) * bg[mm]).astype(np.uint8)
    return ov


def fill_cells(he, xy, cls, ox, oy, max_r, alpha=0.5):
    """Территории клеток: каждый пиксель отходит ближайшему ядру (Вороной),
    граница между клетками проходит посередине. Ядро + цитоплазма, а не только ядро."""
    h, w = he.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.int32), np.arange(h, dtype=np.int32))
    dist, idx = cKDTree(xy - [ox, oy]).query(np.c_[gx.ravel(), gy.ravel()])
    dist = dist.reshape(h, w); idx = idx.reshape(h, w)
    inside = dist < max_r
    lab = np.where(inside, idx, -1)
    clsmap = np.zeros((h, w), np.uint8)
    clsmap[inside] = cls[idx[inside]]
    cmap = np.zeros((h, w, 3), np.uint8)
    for c, col in COL.items():
        cmap[clsmap == c] = col
    bnd = np.zeros((h, w), bool)
    bnd[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    bnd[1:, :] |= lab[1:, :] != lab[:-1, :]
    bnd &= inside
    ov = he.copy()
    m = clsmap > 0
    ov[m] = (alpha * cmap[m] + (1 - alpha) * he[m]).astype(np.uint8)
    ov[bnd] = (55, 55, 55)
    return ov


def legend(canvas, x0, comp=None):
    img = Image.fromarray(canvas)
    dr = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    rows = [(TUMOR, "опухоль")] + [(c, CLASS_NAMES[c]) for c in STROMA_ALL]
    y0 = 16
    dr.rectangle([x0 - 12, y0 - 10, x0 + 320, y0 + len(rows) * 32 + 8],
                 fill=(255, 255, 255), outline=(120, 120, 120))
    for i, (c, nm) in enumerate(rows):
        yy = y0 + i * 32
        dr.rectangle([x0, yy, x0 + 22, yy + 22], fill=COL[c], outline=(60, 60, 60))
        txt = nm + (f" — {comp[c]:.1f}%" if comp and c in comp else "")
        dr.text((x0 + 30, yy - 1), txt, fill=(20, 20, 20), font=font)
    return np.asarray(img)


def stats(cls):
    n_tum = int((cls == TUMOR).sum())
    n_str = int(np.isin(cls, STROMA_ALL).sum())
    print(f"клеток: опухоль {n_tum}, строма {n_str}")
    if n_tum + n_str:
        print(f"TSR = {n_str / (n_tum + n_str):.3f}")
    comp = {c: 100 * int((cls == c).sum()) / max(len(cls), 1) for c in [TUMOR] + STROMA_ALL}
    for c in STROMA_ALL:
        if n_str:
            print(f"  {CLASS_NAMES[c]:16s} {int((cls==c).sum()):6d}  ({100*int((cls==c).sum())/n_str:.1f}% стромы)")
    return comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--layer1", required=True)
    ap.add_argument("--layer2", required=True)
    ap.add_argument("--mpp", type=float, default=None)
    ap.add_argument("--target-mpp", type=float, default=0.27)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--region", type=int, nargs=2, default=None)
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--max-edge-um", type=float, default=50.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = get_device()
    print("device:", device, "| гружу StarDist, UNI, графовые модели...")
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
    sd = StarDist2D.from_pretrained("2D_versatile_he")
    uni = build_uni(device)
    l1 = load_gnn(args.layer1, device)
    l2 = load_gnn(args.layer2, device)
    norm = MacenkoNormalizer()
    sl = SlideReader(args.slide, args.target_mpp, args.mpp)
    ps, stem = args.patch_size, Path(args.slide).stem

    if args.full:
        bs = args.block
        print(f"{sl.W}x{sl.H} @ {sl.mpp:.3f} мкм/px | обзор всего среза, блок {bs}")
        allxy, allF = [], []
        nbx = len(range(0, sl.W, bs)); nby = len(range(0, sl.H, bs)); done = 0
        for by in range(0, sl.H, bs):
            for bx in range(0, sl.W, bs):
                xy, Fe, _ = scan_block(sl, sd, normalize, uni, norm, device, bx, by, bs, ps, False)
                if len(xy):
                    allxy.append(xy); allF.append(Fe)
                done += 1
                print(f"  блок {done}/{nbx*nby}, ядер {sum(len(a) for a in allxy)}   ", end="\r")
        xy = np.concatenate(allxy); Fe = np.concatenate(allF)
        print(f"\nвсего ядер: {len(xy)}")
        cls, med = classify(sl, xy, Fe, l1, l2, device, args.knn, args.max_edge_um)
        comp = stats(cls)
        md = min(2400, max(sl.W, sl.H))
        tw, thh = int(sl.W * md / max(sl.W, sl.H)), int(sl.H * md / max(sl.W, sl.H))
        thumb = sl.thumbnail(tw, thh); sc = sl.W / tw
        ov = fill_zones(thumb, xy, cls, 0, 0, sc, max(2.5, 1.3 * med / sc), 0.55)
        canvas = legend(np.asarray(ov), tw - 340, comp)
        polys = None
    else:
        S = min(args.size, sl.W, sl.H)
        rx, ry = args.region if args.region else ((sl.W - S) // 2, (sl.H - S) // 2)
        print(f"{sl.W}x{sl.H} @ {sl.mpp:.3f} мкм/px | область x{rx} y{ry} {S}x{S}")
        xy, Fe, polys = scan_block(sl, sd, normalize, uni, norm, device, rx, ry, S, ps, False)
        print(f"ядер: {len(xy)}")
        if len(xy) < 20:
            raise SystemExit("мало ядер")
        cls, med = classify(sl, xy, Fe, l1, l2, device, args.knn, args.max_edge_um)
        comp = stats(cls)
        he = sl.read(rx, ry, S)
        ov = fill_cells(he, xy, cls, rx, ry, 1.3 * med)
        canvas = legend(np.concatenate([he, ov], 1), S * 2 - 340, comp)

    np.savez_compressed(ensure_dir("outputs/results") / f"{stem}_cells.npz",
                        xy=xy, cls=cls, mpp=sl.mpp,
                        polys=np.array(polys, dtype=object) if polys else np.array([]))
    out = args.out or str(ensure_dir("outputs/results") /
                          f"{stem}_graph_{'full' if args.full else 'cells'}.png")
    Image.fromarray(canvas).save(out)
    print("карта:", out)


if __name__ == "__main__":
    main()