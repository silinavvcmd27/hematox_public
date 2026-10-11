# На что смотрит модель. Два способа, оба на уже посчитанных данных.
#
# --mode occlusion: карта важности по токенам. Токен за токеном заменяется на
#   средний токен фрагмента, и смотрится, насколько меняется ответ. Получается
#   решётка 14 на 14: видно, опирается ли модель на сам участок или на соседние.
#   UNI не запускается, работает только декодер, поэтому считается за секунды.
#   Рядом печатается доля изменения, которая приходится на участок под самим
#   токеном: чем она меньше, тем сильнее модель опирается на окружение.
#
# --mode distort: чего стоит искажение картинки. Области набора режутся заново,
#   к каждому фрагменту применяется искажение, фрагменты кодируются UNI и
#   считаются те же метрики, что в бенчмарке. Это и отвечает на вопрос, на что
#   модель опирается: на цвет, на текстуру или на форму. Здесь UNI гоняется по
#   каждому искажению, так что считается минутами, а не секундами.
#
# Базовый прогон «без искажений» считается тем же кодом, что и остальные, и может
# слегка отличаться от числа из pan_bench.py: окраска оценивается по области, а
# не по пациенту. Сравнивать надо искажения с базой внутри этой таблицы.
#
#   python -u pan_explain.py --mode occlusion --set ignite_test3 --n 3
#   python -u pan_explain.py --mode distort --set ignite_test3 --rois 12

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter

from src.utils import get_device, ensure_dir
from src.stain import Normalizer, StainStats, fit_stats, sample_pixels
from seg_extract import build_uni, encode
from render_zones import legend_font
from pan_train import load
from pan_bench import load_model, patch_counts, summarize
from pan_show_bench import roi_image, overlay, SRC_MPP

Image.MAX_IMAGE_PIXELS = None
GRID = 14          # токенов в стороне фрагмента
CELL = 112 // GRID  # сколько точек карты приходится на один токен


def meta_of(dirs, name):
    for d in dirs:
        p = Path(d) / f"{name}_patches.csv"
        if p.exists():
            return pd.read_csv(p)
    raise SystemExit(f"нет {name}_patches.csv в {dirs}")


def heat(v):
    """Белый, жёлтый, тёмно-красный: нули не мешают смотреть на ткань."""
    stops = np.array([[255, 255, 255], [250, 200, 60], [150, 20, 20]], np.float32)
    v = np.clip(v, 0, 1) * (len(stops) - 1)
    lo = np.clip(v.astype(int), 0, len(stops) - 2)
    f = (v - lo)[..., None]
    return (stops[lo] * (1 - f) + stops[lo + 1] * f).astype(np.uint8)


@torch.no_grad()
def occlusion(model, x, device, bs=64):
    """Изменение ответа при выключении каждого токена: решётка 14 на 14 и доля местного."""
    base = F.softmax(model(torch.from_numpy(x[None].astype(np.float32)).to(device)), 1)[0]
    mean_token = x.mean(0)
    variants = np.repeat(x[None], GRID * GRID, 0)
    for t in range(GRID * GRID):
        variants[t, t] = mean_token
    total, local = np.zeros(GRID * GRID), np.zeros(GRID * GRID)
    for i in range(0, len(variants), bs):
        p = F.softmax(model(torch.from_numpy(variants[i:i + bs].astype(np.float32)).to(device)), 1)
        diff = (p - base[None]).abs().sum(1)
        for k in range(len(diff)):
            t = i + k
            r, c = divmod(t, GRID)
            d = diff[k]
            total[t] = float(d.sum())
            local[t] = float(d[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL].sum())
    return total.reshape(GRID, GRID), float(local.sum() / max(total.sum(), 1e-9)), base.argmax(0).cpu().numpy()


def run_occlusion(args, model, codes, device):
    X, y = load(args.dirs, args.name)
    meta = meta_of(args.dirs, args.name)
    if len(meta) != len(X):
        raise SystemExit("таблица фрагментов не той длины")
    good = meta[(meta.tumor_frac >= args.min_frac) & (meta.stroma_frac >= args.min_frac)]
    if good.empty:
        raise SystemExit("нет фрагментов, где есть и опухоль, и строма")
    idx = list(good.sample(min(args.n, len(good)), random_state=args.seed).index)

    factor = SRC_MPP["bcss" if args.name.startswith("bcss") else "ignite"] / 0.2738
    ps, gap, head, cap = args.px, 12, 34, 26
    font, small = legend_font(24), legend_font(19)
    rows, cache = [], {}
    for i in idx:
        r = meta.loc[i]
        grid, share, chan = occlusion(model, X[i], device)
        if r.roi not in cache:
            cache[r.roi] = np.asarray(roi_image(args.name, r.roi, factor, args.ignite_root, args.bcss_root))
        img = cache[r.roi][int(r.y0):int(r.y0) + 256, int(r.x0):int(r.x0) + 256]
        if img.shape[:2] != (256, 256):
            print(f"  {r.roi} фрагмент {i}: у края области, пропускаю")
            continue
        pred = codes[chan].astype(np.uint8)
        pred = np.asarray(Image.fromarray(pred).resize((256, 256), Image.NEAREST))
        g = grid / max(grid.max(), 1e-9)
        hm = np.asarray(Image.fromarray(heat(g)).resize((256, 256), Image.NEAREST))
        rows.append((f"{r.roi}, важность под самим токеном {share:.0%}", [img, overlay(img, pred), hm]))
        print(f"  {r.roi} фрагмент {i}: на участок под токеном приходится {share:.1%} изменения, "
              f"разброс важности {grid.min():.3f}..{grid.max():.3f}")

    canvas = Image.new("RGB", (3 * ps + 2 * gap, head + len(rows) * (cap + ps + gap)), "white")
    draw = ImageDraw.Draw(canvas)
    for c, t in enumerate(["фрагмент", "ответ модели", "важность токенов"]):
        draw.text((c * (ps + gap) + 6, 6), t, fill=(0, 0, 0), font=font)
    top = head
    for title, pics in rows:
        draw.text((6, top + 3), title, fill=(90, 90, 90), font=small)
        top += cap
        for c, pic in enumerate(pics):
            canvas.paste(Image.fromarray(pic).resize((ps, ps), Image.BILINEAR), (c * (ps + gap), top))
        top += ps + gap
    out = args.out or f"outputs/results/pan/explain_occlusion_{args.name}.png"
    ensure_dir(Path(out).parent)
    canvas.save(out, optimize=True)
    print("сохранено:", out)


def distortions(names):
    grey = lambda a: np.repeat(np.asarray(Image.fromarray(a).convert("L"))[..., None], 3, 2)
    blur = lambda a, r: np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(r)))
    all_of = {
        "без искажений": (lambda a: a, False),
        "без нормализации окраски": (lambda a: a, True),
        "обесцвечено": (grey, False),
        "размыто, 1 px": (lambda a: blur(a, 1), False),
        "размыто, 2 px": (lambda a: blur(a, 2), False),
        "повёрнуто на 90": (lambda a: np.rot90(a).copy(), False),
        "отражено": (lambda a: a[:, ::-1].copy(), False),
    }
    return {k: v for k, v in all_of.items() if not names or k in names}


def run_distort(args, model, codes, device):
    X, y = load(args.dirs, args.name)
    meta = meta_of(args.dirs, args.name)
    if len(meta) != len(X):
        raise SystemExit("таблица фрагментов не той длины")
    counts = meta.roi.value_counts()
    rois = list(counts.index[:args.rois])
    factor = SRC_MPP["bcss" if args.name.startswith("bcss") else "ignite"] / 0.2738
    ref = StainStats.load(args.stain_ref)
    uni = build_uni(device)
    jobs = distortions(args.only)
    print(f"{args.name}: областей {len(rois)}, фрагментов {int(counts[rois].sum())}, "
          f"искажений {len(jobs)}")

    table = []
    for label, (fn, raw) in jobs.items():
        parts = []
        for roi in rois:
            rows = meta[meta.roi == roi]
            img = np.asarray(roi_image(args.name, roi, factor, args.ignite_root, args.bcss_root))
            tiles, labs = [], []
            for i, r in rows.iterrows():
                y0, x0 = int(r.y0), int(r.x0)
                t = img[y0:y0 + 256, x0:x0 + 256]
                if t.shape[:2] != (256, 256):
                    continue
                tiles.append(fn(t))
                labs.append(np.rot90(y[i]).copy() if label.startswith("повёр")
                            else y[i][:, ::-1].copy() if label == "отражено" else y[i])
            if not tiles:
                continue
            if not raw:
                try:
                    stats = fit_stats(sample_pixels(np.concatenate([t.reshape(-1, 3) for t in tiles])))
                    norm = Normalizer(stats, ref)
                    tiles = [norm(t) for t in tiles]
                except ValueError as err:
                    print(f"    {roi}: окраска не оценилась ({err})")
            feats = encode(uni, tiles, device, bs=args.bs)
            parts.append(patch_counts(model, codes, feats, np.stack(labs), device))
        c = pd.concat(parts, ignore_index=True)
        s = summarize(c)
        table.append({"искажение": label, "фрагментов": len(c),
                      "опухоль": round(s["tumor_dice"], 3), "строма": round(s["stroma_dice"], 3),
                      "доля стромы": round(s["pred_stroma_share"], 3)})
        print(f"\n{label}: опухоль {s['tumor_dice']:.3f}  строма {s['stroma_dice']:.3f}  "
              f"доля стромы {s['true_stroma_share']:.3f} / {s['pred_stroma_share']:.3f}")

    df = pd.DataFrame(table)
    base = df[df["искажение"] == "без искажений"]
    if not base.empty:
        df["потеря опухоли"] = (float(base["опухоль"].iloc[0]) - df["опухоль"]).round(3)
    out = args.out or f"outputs/results/pan/explain_distort_{args.name}.csv"
    ensure_dir(Path(out).parent)
    df.to_csv(out, index=False)
    print("\n" + df.to_string(index=False))
    print("таблица:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["occlusion", "distort"], required=True)
    ap.add_argument("--set", dest="name", default="ignite_test3")
    ap.add_argument("--model", default="outputs/models/pan_l1_deploy3.pth")
    ap.add_argument("--dirs", nargs="+", default=["data/processed/seg", "data/processed/seg_pan"])
    ap.add_argument("--ignite-root", default="data/pan/ignite")
    ap.add_argument("--bcss-root", default="data/pan/bcss")
    ap.add_argument("--stain-ref", default="data/processed/stain_ref_prime.npz")
    ap.add_argument("--n", type=int, default=3, help="сколько фрагментов показать (occlusion)")
    ap.add_argument("--rois", type=int, default=12, help="сколько областей взять (distort)")
    ap.add_argument("--only", nargs="+", help="считать только эти искажения")
    ap.add_argument("--min-frac", type=float, default=0.25)
    ap.add_argument("--px", type=int, default=420)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = get_device()
    model, codes, _ = load_model(args.model, device)
    (run_occlusion if args.mode == "occlusion" else run_distort)(args, model, codes, device)


if __name__ == "__main__":
    main()
