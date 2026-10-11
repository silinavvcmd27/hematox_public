# Яичник на бенчмарке: H&E, разметка UBC-OCEAN и ответ модели рядом.
#
# В отличие от pan_show_bench.py фрагменты не берутся из готовых признаков: в
# ubc_feat.npz с каждого среза лежит до сотни случайных окон, разбросанных по всей
# площади, и картинка из них выходит шахматной. Здесь на срезе выбирается одно
# сплошное поле, оно режется подряд, кодируется UNI и собирается обратно, поэтому
# карта получается непрерывной.
#
# Поле ищется по ubc_patches.csv. Порога по составу окна нет: стромы в разметке
# UBC всего 11 %, и окон, где сразу много и опухоли, и стромы, попросту не бывает.
# Вместо этого берётся квадрат, внутри которого рядом оказались и опухолевые, и
# стромальные окна, то есть граница опухоли. Срезы подбираются разных гистотипов.
#
# Разметка UBC-OCEAN грубая: красным залита вся опухолевая область целиком, строма
# внутри неё не выделена (у авторов зелёный это «здоровая ткань»). Поэтому строма,
# найденная моделью внутри красного поля, на картинке выглядит как расхождение,
# хотя это и есть то, что мы ищем. В подписи к рисунку это надо сказать прямо.
#
#   python -u pan_show_ubc.py --n 3
#   python -u pan_show_ubc.py --slides 4 66 91 --win-cells 14 --out ovary_bench.png

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.utils import get_device, TUMOR, STROMA
from src.stain import Normalizer, StainStats, fit_stats, sample_pixels
from seg_extract import build_uni, encode
from render_zones import legend_font
from pan_bench import load_model
from pan_prep_ubc import open_slide, decode
from pan_show_bench import overlay, predict

Image.MAX_IMAGE_PIXELS = None


def best_box(rows, side):
    """Центр квадрата, где рядом оказалось больше всего и опухоли, и стромы.

    Оценка это меньшее из двух сумм по окнам квадрата, поэтому поле, где есть
    только опухоль, проигрывает полю с границей. Окон на срезе не больше сотни,
    так что перебор по всем центрам дешёвый."""
    x, y = rows.x0.to_numpy(), rows.y0.to_numpy()
    tf, sf = rows.tumor_frac.to_numpy(), rows.stroma_frac.to_numpy()
    best, cx, cy = -1.0, int(np.median(x)), int(np.median(y))
    for i in range(len(rows)):
        box = (np.abs(x - x[i]) < side / 2) & (np.abs(y - y[i]) < side / 2)
        score = min(tf[box].sum(), sf[box].sum())
        if score > best:
            best, cx, cy = float(score), int(x[i]), int(y[i])
    return best, cx, cy


def pick_slides(meta, n, side):
    """Лучшие срезы по этой оценке, по одному на гистотип, пока хватает гистотипов."""
    scored = []
    for roi, rows in meta.groupby("roi"):
        score, cx, cy = best_box(rows, side)
        scored.append((score, roi, rows.subtype.iloc[0], cx, cy))
    scored.sort(reverse=True)
    chosen, seen = [], set()
    for score, roi, subtype, cx, cy in scored:
        if subtype in seen:
            continue
        seen.add(subtype)
        chosen.append((roi, score, cx, cy))
        if len(chosen) == n:
            return chosen
    for score, roi, subtype, cx, cy in scored:
        if roi not in [c[0] for c in chosen]:
            chosen.append((roi, score, cx, cy))
        if len(chosen) == n:
            break
    return chosen


def window(cx, cy, side, shape):
    """Квадрат вокруг выбранного центра, не вылезающий за срез."""
    x0 = int(np.clip(cx - side // 2, 0, max(shape[1] - side, 0)))
    y0 = int(np.clip(cy - side // 2, 0, max(shape[0] - side, 0)))
    return x0, y0, min(side, shape[1]), min(side, shape[0])


def read_pair(root, roi):
    im = open_slide(Path(root) / "images" / f"{roi}.png")
    img = np.asarray(im if im.mode == "RGB" else im.convert("RGB"))
    mk = Image.open(Path(root) / "masks" / f"{roi}.png")
    mk = mk if mk.mode == "RGB" else mk.convert("RGB")
    if mk.size != (img.shape[1], img.shape[0]):
        mk = mk.resize((img.shape[1], img.shape[0]), Image.NEAREST)
    return img, np.asarray(mk)


def model_map(img, src, ps, uni, model, codes, ref, device, bs):
    gh, gw = img.shape[0] // src, img.shape[1] // src
    if not gh or not gw:
        raise SystemExit(f"поле {img.shape[1]}x{img.shape[0]} px меньше одного окна в {src} px")
    tiles = [np.asarray(Image.fromarray(np.ascontiguousarray(
                 img[y * src:(y + 1) * src, x * src:(x + 1) * src])).resize((ps, ps), Image.BILINEAR))
             for y in range(gh) for x in range(gw)]
    try:
        norm = Normalizer(fit_stats(sample_pixels(np.concatenate([t.reshape(-1, 3) for t in tiles]))), ref)
        tiles = [norm(t) for t in tiles]
    except ValueError as err:
        print(f"    окраска не оценилась ({err}), оставляю как есть")
    answer = predict(model, codes, encode(uni, tiles, device, bs=bs), device)
    out = np.zeros((gh * src, gw * src), np.uint8)
    for k in range(gh * gw):
        y, x = divmod(k, gw)
        out[y * src:(y + 1) * src, x * src:(x + 1) * src] = np.asarray(
            Image.fromarray(answer[k]).resize((src, src), Image.NEAREST))
    return out, gh * src, gw * src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/pan/ubc")
    ap.add_argument("--model", default="outputs/models/pan_l1_deploy3.pth")
    ap.add_argument("--patches", default="data/processed/seg_pan/ubc_patches.csv")
    ap.add_argument("--stain-ref", default="data/processed/stain_ref_prime.npz")
    ap.add_argument("--slides", nargs="+", type=int, help="какие срезы показать, иначе выберу сам")
    ap.add_argument("--n", type=int, default=3, help="сколько срезов показать")
    ap.add_argument("--src-mpp", type=float, default=0.5)
    ap.add_argument("--target-mpp", type=float, default=0.2738)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--win-cells", type=int, default=12, help="окон в стороне поля")
    ap.add_argument("--show-unlabeled", dest="mask_unlabeled", action="store_false",
                    help="показывать ответ модели и вне размеченной области")
    ap.add_argument("--px", type=int, default=560, help="сторона панели на картинке")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out", default="outputs/results/pan/bench_ubc.png")
    args = ap.parse_args()

    meta = pd.read_csv(args.patches)
    src = round(args.patch_size * args.target_mpp / args.src_mpp)
    side = args.win_cells * src
    subtype = dict(zip(meta.roi, meta.subtype))
    if args.slides:
        missing = [r for r in args.slides if not (meta.roi == r).any()]
        if missing:
            raise SystemExit(f"нет в {args.patches}: {missing}")
        chosen = [(roi, *best_box(meta[meta.roi == roi], side)) for roi in args.slides]
    else:
        chosen = pick_slides(meta, args.n, side)
    print("срезы:", ", ".join(f"{roi} ({subtype.get(roi, '?')}, оценка {s:.1f})"
                              for roi, s, _, _ in chosen))

    device = get_device()
    model, codes, _ = load_model(args.model, device)
    ref = StainStats.load(args.stain_ref)
    uni = build_uni(device)

    panels = []
    for roi, _, cx, cy in chosen:
        img, mask = read_pair(args.root, roi)
        x0, y0, w, h = window(cx, cy, side, img.shape[:2])
        img, mask = img[y0:y0 + h, x0:x0 + w], mask[y0:y0 + h, x0:x0 + w]
        pred, ph, pw = model_map(img, src, args.patch_size, uni, model, codes, ref, device, args.bs)
        img, truth = img[:ph, :pw], decode(mask[:ph, :pw])
        if args.mask_unlabeled:
            pred = np.where(truth > 0, pred, 0)
        panels.append((roi, img, truth, pred))
        print(f"  срез {roi} ({subtype.get(roi, '?')}): поле {pw}x{ph} px от ({x0}, {y0}), "
              f"в разметке опухоли {(truth == TUMOR).mean():.0%} и стромы {(truth == STROMA).mean():.0%}, "
              f"у модели {(pred == TUMOR).mean():.0%} и {(pred == STROMA).mean():.0%}")

    ps, gap, head, cap = args.px, 12, 34, 26
    font, small = legend_font(24), legend_font(20)
    rows_out = []
    for roi, img, truth, pred in panels:
        ims = []
        for pic in (img, overlay(img, truth), overlay(img, pred)):
            tile = Image.fromarray(pic)
            tile.thumbnail((ps, ps), Image.BILINEAR)
            ims.append(tile)
        rows_out.append((f"срез {roi}, {subtype.get(roi, '?')}", ims))

    canvas = Image.new("RGB", (3 * ps + 2 * gap,
                               head + sum(cap + ims[0].height + gap for _, ims in rows_out)), "white")
    draw = ImageDraw.Draw(canvas)
    for c, title in enumerate(["H&E", "разметка", "модель"]):
        draw.text((c * (ps + gap) + 6, 6), title, fill=(0, 0, 0), font=font)
    top = head
    for title, ims in rows_out:
        draw.text((6, top + 3), title, fill=(90, 90, 90), font=small)
        top += cap
        for c, tile in enumerate(ims):
            canvas.paste(tile, (c * (ps + gap) + (ps - tile.width) // 2, top))
        top += ims[0].height + gap

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print("сохранено:", args.out)


if __name__ == "__main__":
    main()
