# Картинки для бенчмарка: H&E области, разметка патолога и ответ модели рядом.
#
# Признаки и маски берутся из <набор>_feat.npz, положение каждого фрагмента из
# <набор>_patches.csv, поэтому UNI заново не считается: модель применяется прямо
# к сохранённым признакам. H&E читается из исходного набора и приводится к тому же
# масштабу, что и при подготовке признаков.
#
# Ответ модели показывается только там, где есть разметка: иначе фон и край стекла
# уходят в «прочее» и мешают смотреть (ключ --show-unlabeled возвращает всё).
# Области отбираются так, чтобы в них была и опухоль, и строма, иначе сравнивать
# нечего. Цвета те же, что на картах срезов: опухоль красная, строма серая,
# прочее жёлтое.
#
#   python -u pan_show_bench.py --set ignite_test3 --model outputs/models/pan_l1_deploy3.pth
#   python -u pan_show_bench.py --set bcss3 --n 3 --out outputs/results/pan/bench_bcss.png

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from src.utils import get_device
from seg_infer_svs import ZONE_COL
from render_zones import legend_font
from pan_train import load
from pan_bench import load_model

Image.MAX_IMAGE_PIXELS = None
SRC_MPP = {"bcss": 0.25, "ignite": 0.5}


def roi_image(name, roi, factor, ignite_root, bcss_root):
    if name.startswith("bcss"):
        path = Path(bcss_root) / "rgbs_colorNormalized" / f"{roi}.png"
    else:
        from pan_prep_ignite import index_pngs
        path = index_pngs(ignite_root)[("images", roi)]
    im = Image.open(path)
    im = im if im.mode == "RGB" else im.convert("RGB")
    return im.resize((round(im.width * factor), round(im.height * factor)), Image.BILINEAR)


def overlay(rgb, zones, alpha=0.45):
    out = rgb.astype(np.float32)
    for code, col in ZONE_COL.items():
        m = zones == code
        out[m] = alpha * np.array(col, np.float32) + (1 - alpha) * out[m]
    edge = np.zeros(zones.shape, bool)
    edge[:-1] |= zones[:-1] != zones[1:]
    edge[:, :-1] |= zones[:, :-1] != zones[:, 1:]
    out[edge & (zones > 0)] *= 0.55
    return out.astype(np.uint8)


@torch.no_grad()
def predict(model, codes, X, device, bs=64):
    code_of = torch.from_numpy(codes)
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs].astype(np.float32)).to(device)
        out.append(code_of[model(xb).argmax(1).cpu()].numpy().astype(np.uint8))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="name", required=True, help="ignite_test3, bcss3 и т. п.")
    ap.add_argument("--model", default="outputs/models/pan_l1_deploy3.pth")
    ap.add_argument("--dirs", nargs="+", default=["data/processed/seg", "data/processed/seg_pan"])
    ap.add_argument("--ignite-root", default="data/pan/ignite")
    ap.add_argument("--bcss-root", default="data/pan/bcss")
    ap.add_argument("--n", type=int, default=4, help="сколько областей показать")
    ap.add_argument("--px", type=int, default=560, help="сторона панели на картинке")
    ap.add_argument("--min-frac", type=float, default=0.15,
                    help="минимальная доля опухоли и стромы в области")
    ap.add_argument("--show-unlabeled", dest="mask_unlabeled", action="store_false",
                    help="показывать ответ модели и вне размеченной области")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    X, y = load(args.dirs, args.name)
    meta = None
    for d in args.dirs:
        p = Path(d) / f"{args.name}_patches.csv"
        if p.exists():
            meta = pd.read_csv(p)
            break
    if meta is None or len(meta) != len(X):
        raise SystemExit(f"нет {args.name}_patches.csv нужной длины рядом с признаками")

    per_roi = meta.groupby("roi")[["tumor_frac", "stroma_frac"]].mean()
    good = per_roi[(per_roi.tumor_frac >= args.min_frac) & (per_roi.stroma_frac >= args.min_frac)]
    if good.empty:
        raise SystemExit("нет областей, где есть и опухоль, и строма; ослабьте --min-frac")
    counts = meta.roi.value_counts()
    rois = list(counts[counts.index.isin(good.index)].head(args.n * 3).index)
    rng = np.random.default_rng(args.seed)
    rois = [rois[i] for i in rng.permutation(len(rois))[:args.n]]

    device = get_device()
    model, codes, _ = load_model(args.model, device)
    factor = SRC_MPP["bcss" if args.name.startswith("bcss") else "ignite"] / 0.2738

    panels = []
    for roi in rois:
        rows = meta[meta.roi == roi]
        idx = rows.index.to_numpy()
        img = np.asarray(roi_image(args.name, roi, factor, args.ignite_root, args.bcss_root))
        shape = img.shape[:2]
        truth, pred = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
        answer = predict(model, codes, X[idx], device)
        for k, (_, r) in enumerate(rows.iterrows()):
            y0, x0 = int(r.y0), int(r.x0)
            h, w = min(256, shape[0] - y0), min(256, shape[1] - x0)
            if h <= 0 or w <= 0:
                continue
            for dst, src in ((truth, y[idx[k]]), (pred, answer[k])):
                tile = np.asarray(Image.fromarray(src).resize((256, 256), Image.NEAREST))
                dst[y0:y0 + h, x0:x0 + w] = tile[:h, :w]
        panels.append((roi, img, truth, pred))
        print(f"  {roi}: фрагментов {len(rows)}, размер {shape[1]}x{shape[0]}")

    ps, gap, head, cap = args.px, 12, 34, 26
    font, small = legend_font(24), legend_font(20)
    rows_out = []
    for roi, img, truth, pred in panels:
        if args.mask_unlabeled:
            pred = np.where(truth > 0, pred, 0)
        ims = []
        for pic in (img, overlay(img, truth), overlay(img, pred)):
            tile = Image.fromarray(pic)
            tile.thumbnail((ps, ps), Image.BILINEAR)
            ims.append(tile)
        rows_out.append((roi, ims))

    height = head + sum(cap + ims[0].height + gap for _, ims in rows_out)
    canvas = Image.new("RGB", (3 * ps + 2 * gap, height), "white")
    draw = ImageDraw.Draw(canvas)
    for c, title in enumerate(["H&E", "разметка", "модель"]):
        draw.text((c * (ps + gap) + 6, 6), title, fill=(0, 0, 0), font=font)
    top = head
    for roi, ims in rows_out:
        draw.text((6, top + 3), roi[:44], fill=(90, 90, 90), font=small)
        top += cap
        for c, tile in enumerate(ims):
            canvas.paste(tile, (c * (ps + gap) + (ps - tile.width) // 2, top))
        top += ims[0].height + gap

    out = args.out or f"outputs/results/pan/bench_{args.name}.png"
    canvas.save(out, optimize=True)
    print("сохранено:", out)


if __name__ == "__main__":
    main()
