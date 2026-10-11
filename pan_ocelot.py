# Перенос на шесть органов OCELOT: мочевой пузырь, эндометрий, голова и шея, почка,
# простата, желудок. Модель на этих органах не обучалась.
#
# Разметка там не наша: размечена опухолевая ОБЛАСТЬ (Cancer Area) целиком, вместе с
# лежащей внутри стромой, плюс фон (прочая ткань) и неразмеченное. Поэтому сравнивать
# их область с нашим классом опухоли нельзя, получится заниженный Dice не по нашей вине.
# Сравниваем правильно:
#
#   Dice опухолевого ложа   их область против нашей суммы опухоли и стромы
#   утечка опухоли          доля наших опухолевых пикселей, попавшая в фон
#   пропуск области         доля их области, где мы не увидели ни опухоли, ни стромы
#   TSR внутри области      описательно, эталона для него в наборе нет
#
# Тканевые картинки раздаются уменьшенными вчетверо (0.8 мкм на пиксель), тянуть их до
# 0.2738 вверх нельзя. Поэтому изображение берётся из парного клеточного патча, он в
# родном разрешении, а кусок маски вырезается из тканевой по координатам центра из
# metadata.json.
#
#   python pan_ocelot.py --root data/pan/ocelot
#   python pan_ocelot.py --root data/pan/ocelot --limit 20

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.utils import get_device, ensure_dir, TUMOR
from src.stain import Normalizer, StainStats, fit_stats, sample_pixels
from pan_train import STROMA_CODES, OTHER
from pan_bench import load_model
from seg_extract import build_uni, encode

Image.MAX_IMAGE_PIXELS = None
BG, CA, UNK = 1, 2, 255
MAP = 112


def mask_for_patch(tissue_png, meta, cell_px, tissue_px):
    """Кусок тканевой маски, соответствующий клеточному патчу.

    Тканевая картинка покрывает вчетверо большее поле и лежит уменьшенной, поэтому
    клеточному патчу в ней отвечает квадрат в cell_px/4 пикселей вокруг центра,
    заданного относительными координатами patch_x_offset и patch_y_offset.
    """
    side = tissue_px * cell_px // 4096
    cx = int(meta["patch_x_offset"] * tissue_px)
    cy = int(meta["patch_y_offset"] * tissue_px)
    x0 = min(max(cx - side // 2, 0), tissue_px - side)
    y0 = min(max(cy - side // 2, 0), tissue_px - side)
    return tissue_png[y0:y0 + side, x0:x0 + side]


def fragments(img, mask, size, min_labeled, white_max):
    """Нарезка на фрагменты вместе с маской того же места."""
    for y in range(0, img.shape[0] - size + 1, size):
        for x in range(0, img.shape[1] - size + 1, size):
            tile = img[y:y + size, x:x + size]
            if (tile.mean(-1) > 220).mean() > white_max:
                continue
            m = mask[y:y + size, x:x + size]
            if (m != UNK).mean() < min_labeled:
                continue
            yield tile, np.asarray(Image.fromarray(m).resize((MAP, MAP), Image.NEAREST))


@torch.no_grad()
def predict(model, codes, feats, device, bs):
    out = []
    for i in range(0, len(feats), bs):
        x = torch.from_numpy(feats[i:i + bs].astype(np.float32)).to(device)
        out.append(codes[model(x).argmax(1).cpu().numpy()])
    return np.concatenate(out)


def counts(pred, mask):
    """Счётчики пикселей по клеткам таблицы: что предсказано внутри области и в фоне."""
    tum = pred == TUMOR
    stroma = np.isin(pred, STROMA_CODES)
    other = pred == OTHER
    inside, outside = mask == CA, mask == BG
    return {
        "область": int(inside.sum()),
        "фон": int(outside.sum()),
        "опухоль_в_области": int((tum & inside).sum()),
        "строма_в_области": int((stroma & inside).sum()),
        "прочее_в_области": int((other & inside).sum()),
        "опухоль_в_фоне": int((tum & outside).sum()),
        "строма_в_фоне": int((stroma & outside).sum()),
        "прочее_в_фоне": int((other & outside).sum()),
    }


def summarize(c):
    bed_in = c["опухоль_в_области"] + c["строма_в_области"]
    bed_out = c["опухоль_в_фоне"] + c["строма_в_фоне"]
    bed_pred = bed_in + bed_out
    tumor_all = c["опухоль_в_области"] + c["опухоль_в_фоне"]
    stroma_in = c["строма_в_области"]
    return {
        "область_млн_px": round(c["область"] / 1e6, 2),
        "dice_ложа": round(2 * bed_in / (bed_pred + c["область"]), 3) if bed_pred + c["область"] else np.nan,
        "утечка_опухоли": round(c["опухоль_в_фоне"] / tumor_all, 3) if tumor_all else np.nan,
        "пропуск_области": round(c["прочее_в_области"] / c["область"], 3) if c["область"] else np.nan,
        "TSR_в_области": round(stroma_in / bed_in, 3) if bed_in else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/pan/ocelot")
    ap.add_argument("--model", default="outputs/models/pan_l1_deploy3.pth")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--stain-ref", default="data/processed/stain_ref_prime.npz")
    ap.add_argument("--target-mpp", type=float, default=0.2738)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--min-labeled", type=float, default=0.5,
                    help="какая доля фрагмента должна быть размечена, а не Unknown")
    ap.add_argument("--white-max", type=float, default=0.85)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="сколько образцов взять, 0 это все")
    ap.add_argument("--out", default="outputs/results/pan/ocelot.csv")
    args = ap.parse_args()

    root = Path(args.root)
    meta_all = json.loads((root / "metadata.json").read_text())["sample_pairs"]
    ref = StainStats.load(args.stain_ref)
    device = get_device()
    model, codes, _ = load_model(args.model, device)
    uni = build_uni(device)

    rows = []
    done = 0
    for split in args.splits:
        cells = sorted((root / "images" / split / "cell").glob("*.jpg"))
        for path in cells:
            uid = path.stem
            meta = meta_all.get(uid)
            if meta is None:
                continue
            tissue = np.asarray(Image.open(root / "annotations" / split / "tissue" / f"{uid}.png"))
            img = np.asarray(Image.open(path).convert("RGB"))
            mask = mask_for_patch(tissue, meta, img.shape[0], tissue.shape[0])

            scale = float(meta.get("mpp_x", 0.2)) / args.target_mpp
            side = int(round(img.shape[0] * scale))
            img = np.asarray(Image.fromarray(img).resize((side, side), Image.BILINEAR))
            mask = np.asarray(Image.fromarray(mask).resize((side, side), Image.NEAREST))

            cut = list(fragments(img, mask, args.patch_size, args.min_labeled, args.white_max))
            if not cut:
                continue
            tiles, masks = [c[0] for c in cut], [c[1] for c in cut]
            try:
                stats = fit_stats(sample_pixels(np.concatenate([t.reshape(-1, 3) for t in tiles])))
                norm = Normalizer(stats, ref)
                tiles = [norm(t) for t in tiles]
            except ValueError as err:
                print(f"  {uid}: окраска не оценилась ({err}), беру как есть")

            feats = encode(uni, tiles, device, bs=args.bs)
            pred = predict(model, codes, feats, device, args.bs)
            row = counts(pred, np.stack(masks))
            row.update(орган=meta["organ"], образец=uid, split=split, фрагментов=len(tiles))
            rows.append(row)

            done += 1
            if done % 25 == 0:
                print(f"  обработано {done}")
            if args.limit and done >= args.limit:
                break
        if args.limit and done >= args.limit:
            break

    if not rows:
        raise SystemExit("ни одного образца не обработано, проверьте --root")

    table = pd.DataFrame(rows)
    num = [c for c in table.columns if c not in ("орган", "образец", "split")]
    by_organ = table.groupby("орган")[num].sum()
    report = pd.DataFrame({o: summarize(r) for o, r in by_organ.iterrows()}).T
    report.insert(0, "образцов", table.groupby("орган").size())
    total = summarize(table[num].sum())

    print("\n" + report.to_string())
    print("\nвсего:", ", ".join(f"{k} {v}" for k, v in total.items()))

    ensure_dir(Path(args.out).parent)
    table.to_csv(args.out, index=False)
    report.to_csv(str(args.out).replace(".csv", "_по_органам.csv"))
    print("\nпо образцам:", args.out)


if __name__ == "__main__":
    main()
