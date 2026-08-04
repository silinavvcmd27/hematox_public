# Внешний бенчмарк слоя 1 (опухоль/строма) на UBC-OCEAN.
# Маски патолога размечают tumor / stroma / necrosis. Мы гоняем ту же граф-модель,
# что и в боевом инференсе (StarDist -> UNI -> граф клеток -> слой 1), и для каждого
# детектированного ядра берём метку маски в его точке. Считаем confusion и Dice —
# напрямую сопоставимо с внутренними числами (tumor 0.92 / stroma 0.90).
#
#   посмотреть цвета одной маски (подтвердить маппинг):
#     python ubc_bench.py --inspect --mask data/ubc/masks/1234.png
#   какие HGSC-срезы имеют маску:
#     python ubc_bench.py --list --train data/ubc/train.csv --masks data/ubc/masks
#   бенчмарк одного среза:
#     python ubc_bench.py --slide data/ubc/img/1234.png --mask data/ubc/masks/1234.png \
#         --layer1 runs/graph/l1_final.pth --mpp 0.5
#   несколько (каждый --slide/--mask парой, результат усредняется):
#     python ubc_bench.py --pairs data/ubc/pairs.txt --layer1 runs/graph/l1_final.pth --mpp 0.5

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from src.utils import get_device, ensure_dir, TUMOR, STROMA
from seg_infer import build_uni
from stain_norm import MacenkoNormalizer
from cell_graph_train import adjacency
from graph_infer import load_gnn, scan_block

Image.MAX_IMAGE_PIXELS = None


class PngSlide:
    """Лёгкий ридер под обычный PNG (UBC-OCEAN не пирамидальный). Держит нативную
    картинку, а read() отдаёт кроп, пересэмплированный к target_mpp — тот же
    интерфейс (.read/.W/.H/.mpp/.thumbnail), что ждут scan_block и classify."""

    def __init__(self, path, target_mpp, mpp):
        self.img = np.asarray(Image.open(path).convert("RGB"))
        self.nH, self.nW = self.img.shape[:2]
        self.scale = mpp / target_mpp          # рабочих px на 1 нативный px
        self.mpp = target_mpp
        self.W = int(self.nW * self.scale)
        self.H = int(self.nH * self.scale)

    def read(self, x, y, s):
        nx, ny = int(x / self.scale), int(y / self.scale)
        ns = max(1, int(round(s / self.scale)))
        crop = self.img[ny:ny + ns, nx:nx + ns]
        if crop.shape[0] != s or crop.shape[1] != s:
            if crop.size == 0:
                return np.full((s, s, 3), 255, np.uint8)
            crop = np.asarray(Image.fromarray(crop).resize((s, s), Image.BILINEAR))
        return crop

    def thumbnail(self, tw, th):
        return np.asarray(Image.fromarray(self.img).resize((tw, th), Image.BILINEAR))

# цвет маски (RGB) -> класс. ПОДТВЕРДИТЬ через --inspect на реальном файле.
BG, TUM, STR, NEC = 0, 1, 2, 3
MASK_COLORS = {(255, 0, 0): TUM, (0, 255, 0): STR, (0, 0, 255): NEC}
CLSNAME = {TUM: "tumor", STR: "stroma", NEC: "necrosis"}
# цвета для оверлея предсказания
PRED_COL = {TUMOR: (216, 65, 47), STROMA: (120, 120, 120)}


def mask_to_class(rgb):
    """RGB-маску (H,W,3) -> карту классов (H,W) uint8 по MASK_COLORS."""
    cls = np.zeros(rgb.shape[:2], np.uint8)
    for (r, g, b), c in MASK_COLORS.items():
        cls[(rgb[..., 0] == r) & (rgb[..., 1] == g) & (rgb[..., 2] == b)] = c
    return cls


def classify_l1(sl, xy, Fe, l1, device, knn, max_edge_um):
    """Слой 1 по графу: 0=опухоль, 1=строма (как в graph_infer.classify)."""
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
    return p1


def sample_regions(mcls, sl, mask_scale, S, k, min_frac=0.05, tries=400, seed=0):
    """Регионы (в рабочих координатах среза), где в маске есть и опухоль, и строма."""
    rng = np.random.default_rng(seed)
    ys, xs = np.where(mcls > BG)
    if len(xs) == 0:
        return []
    # рамка размеченной области в рабочих координатах
    x0, x1 = xs.min() / mask_scale, xs.max() / mask_scale
    y0, y1 = ys.min() / mask_scale, ys.max() / mask_scale
    out = []
    for _ in range(tries):
        if len(out) >= k:
            break
        rx = int(rng.uniform(x0, max(x0, x1 - S)))
        ry = int(rng.uniform(y0, max(y0, y1 - S)))
        mx0, my0 = int(rx * mask_scale), int(ry * mask_scale)
        mx1, my1 = int((rx + S) * mask_scale), int((ry + S) * mask_scale)
        sub = mcls[my0:my1, mx0:mx1]
        if sub.size == 0:
            continue
        ft = (sub == TUM).mean()
        fs = (sub == STR).mean()
        if ft >= min_frac and fs >= min_frac:
            out.append((rx, ry))
    return out


def overlay(sl, mcls, mask_scale, xy, pred, rx, ry, S, path):
    """H&E с предсказанными классами клеток рядом с кропом маски патолога."""
    he = sl.read(rx, ry, S)
    im = Image.fromarray(he.copy()); dr = ImageDraw.Draw(im)
    for (x, y), p in zip(xy, pred):
        c = TUMOR if p == 0 else STROMA
        a, b = x - rx, y - ry
        dr.ellipse([a - 4, b - 4, a + 4, b + 4], fill=PRED_COL[c])
    mx0, my0 = int(rx * mask_scale), int(ry * mask_scale)
    mcrop = mcls[my0:my0 + int(S * mask_scale), mx0:mx0 + int(S * mask_scale)]
    mrgb = np.zeros((*mcrop.shape, 3), np.uint8)
    mrgb[mcrop == TUM] = PRED_COL[TUMOR]
    mrgb[mcrop == STR] = PRED_COL[STROMA]
    mrgb[mcrop == NEC] = (60, 40, 90)
    mimg = Image.fromarray(mrgb).resize((S, S), Image.NEAREST)
    canvas = Image.new("RGB", (S * 2 + 10, S), (255, 255, 255))
    canvas.paste(im, (0, 0)); canvas.paste(mimg, (S + 10, 0))
    canvas.save(path)


def bench_slide(slide, mask, l1, sd, normalize, uni, norm, device, args):
    sl = PngSlide(slide, args.target_mpp, args.mpp or 0.5)
    rgb = np.asarray(Image.open(mask).convert("RGB"))
    mcls = mask_to_class(rgb)
    mh, mw = rgb.shape[0], rgb.shape[1]
    del rgb  # дальше нужна только карта классов
    mask_scale = mcls.shape[1] / sl.W  # нативных пикселей маски на 1 рабочий пиксель
    stem = Path(slide).stem
    regs = sample_regions(mcls, sl, mask_scale, args.size, args.regions, seed=args.seed)
    print(f"[{stem}] {sl.W}x{sl.H} @ {sl.mpp:.3f} мкм/px | маска {mw}x{mh}"
          f" (scale {mask_scale:.2f}) | регионов с опухолью+стромой: {len(regs)}")
    conf = np.zeros((2, 2), np.int64)  # [pred tum/str][mask tum/str]
    for ri, (rx, ry) in enumerate(regs):
        xy, Fe, _ = scan_block(sl, sd, normalize, uni, norm, device, rx, ry,
                               args.size, args.patch_size, False)
        if len(xy) < 20:
            print(f"  регион {ri}: мало ядер ({len(xy)}), пропуск")
            continue
        pred = classify_l1(sl, xy, Fe, l1, device, args.knn, args.max_edge_um)
        mi = np.clip((xy[:, 1] * mask_scale).astype(int), 0, mcls.shape[0] - 1)
        mj = np.clip((xy[:, 0] * mask_scale).astype(int), 0, mcls.shape[1] - 1)
        mlab = mcls[mi, mj]
        keep = (mlab == TUM) | (mlab == STR)
        pt = (pred[keep] == 1).astype(int)          # 0=опухоль,1=строма
        gt = (mlab[keep] == STR).astype(int)
        for p, g in zip(pt, gt):
            conf[p, g] += 1
        print(f"  регион {ri} x{rx} y{ry}: ядер {len(xy)}, оценено {int(keep.sum())}"
              f" (опухоль {int((gt==0).sum())}, строма {int((gt==1).sum())})")
        if args.save_overlays and ri == 0:
            op = ensure_dir("outputs/ubc") / f"{stem}_r{ri}.png"
            overlay(sl, mcls, mask_scale, xy, pred, rx, ry, args.size, str(op))
            print(f"    оверлей: {op}")
    return conf


def report(conf, tag):
    tt, ts = int(conf[0, 0]), int(conf[0, 1])
    st, ss = int(conf[1, 0]), int(conf[1, 1])
    n = tt + ts + st + ss
    if n == 0:
        print(f"\n=== {tag}: нет оценённых клеток ===")
        return
    dice_t = 2 * tt / (2 * tt + ts + st) if (2 * tt + ts + st) else float("nan")
    dice_s = 2 * ss / (2 * ss + st + ts) if (2 * ss + st + ts) else float("nan")
    acc = (tt + ss) / n
    print(f"\n=== {tag} (по клеткам, N={n}) ===")
    print(f"  confusion  [pred\\mask]   tumor   stroma")
    print(f"    pred tumor            {tt:7d} {ts:7d}")
    print(f"    pred stroma           {st:7d} {ss:7d}")
    print(f"  tumor Dice {dice_t:.3f} | stroma Dice {dice_s:.3f} | accuracy {acc:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide")
    ap.add_argument("--mask")
    ap.add_argument("--pairs", help="файл: строки 'slide.png<TAB>mask.png'")
    ap.add_argument("--layer1", default="runs/graph/l1_final.pth")
    ap.add_argument("--mpp", type=float, default=None, help="нативное разрешение среза (UBC WSI ~0.5)")
    ap.add_argument("--target-mpp", type=float, default=0.27)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--size", type=int, default=2048, help="сторона региона в рабочих пикселях")
    ap.add_argument("--regions", type=int, default=4, help="регионов на срез")
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--max-edge-um", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-overlays", action="store_true")
    # служебные режимы
    ap.add_argument("--inspect", action="store_true", help="вывести уникальные цвета маски")
    ap.add_argument("--list", action="store_true", help="HGSC-срезы с масками")
    ap.add_argument("--train", help="train.csv для --list")
    ap.add_argument("--masks", help="папка масок для --list")
    args = ap.parse_args()

    if args.inspect:
        rgb = np.asarray(Image.open(args.mask).convert("RGB")).reshape(-1, 3)
        cols, cnt = np.unique(rgb, axis=0, return_counts=True)
        order = np.argsort(-cnt)
        print(f"уникальных цветов: {len(cols)} | всего пикселей {len(rgb)}")
        for i in order[:12]:
            c = tuple(int(v) for v in cols[i])
            known = CLSNAME.get(MASK_COLORS.get(c, -1), "—")
            print(f"  RGB {c}  {cnt[i]:12d}  ({100*cnt[i]/len(rgb):5.2f}%)  {known}")
        return

    if args.list:
        masks = {p.stem for p in Path(args.masks).glob("*.png")}
        with open(args.train) as f:
            rows = list(csv.DictReader(f))
        hg = [r for r in rows if r.get("label") == "HGSC" and r["image_id"] in masks]
        print(f"HGSC с маской: {len(hg)}")
        for r in hg:
            print(" ", r["image_id"])
        return

    device = get_device()
    print("device:", device, "| гружу StarDist, UNI, слой 1...")
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
    sd = StarDist2D.from_pretrained("2D_versatile_he")
    uni = build_uni(device)
    l1 = load_gnn(args.layer1, device)
    norm = MacenkoNormalizer()

    if args.pairs:
        pairs = [ln.split("\t")[:2] for ln in Path(args.pairs).read_text().splitlines() if ln.strip()]
    else:
        pairs = [(args.slide, args.mask)]

    total = np.zeros((2, 2), np.int64)
    for slide, mask in pairs:
        conf = bench_slide(slide, mask, l1, sd, normalize, uni, norm, device, args)
        report(conf, Path(slide).stem)
        total += conf
    if len(pairs) > 1:
        report(total, f"ИТОГО по {len(pairs)} срезам")


if __name__ == "__main__":
    main()
