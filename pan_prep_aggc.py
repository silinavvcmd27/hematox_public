# AGGC2022 (предстательная железа) в наш формат признаков.
#
# На каждый срез папка с бинарными масками по классам: Stroma_Mask, Normal_Mask,
# G3_Mask, G4_Mask, G5_Mask (последние два есть не везде). Раскладка прямая:
#
#   Gleason 3, 4, 5                опухоль
#   Stroma                         строма
#   Normal (доброкачественный      прочее (нормальная паренхима органа)
#   эпителий)
#   неразмеченное                  игнор
#
# Простата важна тем, что её строма богата гладкой мышцей и выглядит иначе всего, что
# модель видела раньше. На незнакомых органах простата у нас вторая с конца по качеству
# (Dice ложа 0,527 на OCELOT), так что этот источник целится в измеренную слабость.
#
# Маски записаны факсовым сжатием CCITT, которое tifffile не распаковывает, поэтому
# читаются через PIL, а если не выйдет, через OpenCV.
#
# Разрешения в метаданных файлов нет, организаторы заявляют 20x, поэтому исходный
# масштаб принят 0,5 мкм на пиксель. Если окажется иначе, поменять ключом --src-mpp:
# ошибка здесь сдвинет размер поля, и модель увидит ткань не в том увеличении.
# Для проверки есть --dump: сохраняет несколько готовых фрагментов картинками, их
# можно сравнить глазом с фрагментами IGNITE.
#
#   python -u pan_prep_aggc.py --split Train --name aggc_proba --per-slide 40 --dump 6
#   python -u pan_prep_aggc.py --split Train --name aggc_train --per-slide 300

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.utils import get_device, ensure_dir, TUMOR, STROMA
from src.stain import Normalizer, StainStats, fit_stats, sample_pixels
from pan_train import OTHER
from seg_extract import build_uni, encode, OUT

Image.MAX_IMAGE_PIXELS = None
TO_CLASS = {"Stroma": STROMA, "Normal": OTHER, "G3": TUMOR, "G4": TUMOR, "G5": TUMOR}


def read_mask(path):
    """Бинарная маска в CCITT. PIL обычно справляется, иначе пробуем OpenCV."""
    try:
        return np.asarray(Image.open(path)).astype(bool)
    except Exception as err:
        import cv2
        m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise SystemExit(f"не прочиталась маска {path}: {err}")
        return m.astype(bool)


def label_map(folder):
    """Одна карта кодов из набора бинарных масок. Опухоль пишется последней, поэтому
    при перекрытии побеждает она."""
    masks = sorted(folder.glob("*_Mask.tif"))
    if not masks:
        return None
    out = None
    for cls in (STROMA, OTHER, TUMOR):
        for path in masks:
            if TO_CLASS.get(path.name.replace("_Mask.tif", "")) != cls:
                continue
            m = read_mask(path)
            if out is None:
                out = np.zeros(m.shape, np.uint8)
            out[m] = cls
            del m
    return out


def slide_patches(slide, labels, args, rng):
    import openslide
    img = openslide.OpenSlide(str(slide))
    w, h = img.dimensions
    if (h, w) != labels.shape:
        print(f"  {slide.stem}: размеры не совпали, картинка {(h, w)}, маска {labels.shape}")
        return []
    step = int(round(args.patch_size * args.target_mpp / args.src_mpp))

    spots = [(y, x)
             for y in range(0, h - step + 1, step)
             for x in range(0, w - step + 1, step)
             if (labels[y:y + step, x:x + step] > 0).mean() >= args.min_label_frac]
    if len(spots) > args.per_slide:
        spots = [spots[i] for i in rng.choice(len(spots), args.per_slide, replace=False)]

    out = []
    for y, x in spots:
        tile = np.asarray(img.read_region((x, y), 0, (step, step)).convert("RGB"))
        if (tile.mean(-1) > 220).mean() > args.white_max:
            continue
        tile = np.asarray(Image.fromarray(tile).resize((args.patch_size, args.patch_size),
                                                       Image.BILINEAR))
        m = Image.fromarray(labels[y:y + step, x:x + step])
        out.append((tile, np.asarray(m.resize((OUT, OUT), Image.NEAREST)), y, x))
    img.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/pan/aggc")
    ap.add_argument("--split", default="Train", choices=["Train", "Test"])
    ap.add_argument("--subset", default="Subset2", help="Subset2 это биопсии")
    ap.add_argument("--name", required=True)
    ap.add_argument("--src-mpp", type=float, default=0.5)
    ap.add_argument("--target-mpp", type=float, default=0.2738)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--per-slide", type=int, default=300)
    ap.add_argument("--min-label-frac", type=float, default=0.10)
    ap.add_argument("--white-max", type=float, default=0.85)
    ap.add_argument("--stain-ref", default="data/processed/stain_ref_prime.npz")
    ap.add_argument("--dump", type=int, default=0, help="сохранить столько фрагментов картинками")
    ap.add_argument("--slides", type=int, default=0, help="взять только первые N срезов")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out-dir", default="data/processed/seg_pan")
    args = ap.parse_args()

    base = Path(args.root) / args.split.lower()
    ann = base / f"{args.subset}_{args.split}_annotation" / args.split
    slides = sorted(base.glob(f"{args.subset}_{args.split}_*.tiff"))
    if args.slides:
        slides = slides[:args.slides]
    if not slides:
        raise SystemExit(f"нет срезов {args.subset}_{args.split}_*.tiff в {base}")
    step = int(round(args.patch_size * args.target_mpp / args.src_mpp))
    print(f"{args.name}: срезов {len(slides)}, разметка в {ann}")
    print(f"поле {args.patch_size * args.target_mpp:.0f} мкм, в исходных пикселях {step}")

    ref = StainStats.load(args.stain_ref)
    rng = np.random.default_rng(0)
    tiles, masks, rows = [], [], []
    for slide in slides:
        folder = ann / slide.stem
        if not folder.exists():
            print(f"  {slide.stem}: нет папки с разметкой, пропускаю")
            continue
        labels = label_map(folder)
        if labels is None:
            print(f"  {slide.stem}: масок не нашлось, пропускаю")
            continue
        cut = slide_patches(slide, labels, args, rng)
        del labels
        if not cut:
            continue
        pool = [t for t, _, _, _ in cut]
        try:
            stats = fit_stats(sample_pixels(np.concatenate([t.reshape(-1, 3) for t in pool])))
            norm = Normalizer(stats, ref)
        except ValueError as err:
            print(f"  {slide.stem}: окраска не оценилась ({err}), оставляю как есть")
            norm = None
        for tile, m, y, x in cut:
            tiles.append(norm(tile) if norm else tile)
            masks.append(m)
            rows.append({"roi": slide.stem, "slide": slide.stem, "patient": slide.stem,
                         "source": "aggc", "scanner": "unknown", "organ": "prostate",
                         "x0": int(x), "y0": int(y),
                         "tumor_frac": round(float((m == TUMOR).mean()), 3),
                         "stroma_frac": round(float((m == STROMA).mean()), 3),
                         "other_frac": round(float((m == OTHER).mean()), 3)})
        print(f"  {slide.stem}: патчей {len(cut)}, всего {len(tiles)}")

    y = np.stack(masks)
    labeled = y > 0
    print(f"\nпатчей {len(tiles)}, размечено {labeled.mean():.1%} пикселей; "
          f"среди размеченного опухоль {(y[labeled] == TUMOR).mean():.1%}, "
          f"строма {(y[labeled] == STROMA).mean():.1%}, прочее {(y[labeled] == OTHER).mean():.1%}")

    if args.dump:
        folder = ensure_dir(Path("outputs/results/pan") / f"{args.name}_examples")
        for i in range(min(args.dump, len(tiles))):
            Image.fromarray(tiles[i]).save(folder / f"{i:02d}.png")
        print("примеры фрагментов:", folder)

    device = get_device()
    print("device:", device, "| кодируем через UNI...")
    X = encode(build_uni(device), tiles, device, bs=args.bs)

    out = ensure_dir(args.out_dir)
    np.savez_compressed(out / f"{args.name}_feat.npz", X=X, y=y, patch_px=args.patch_size,
                        patch_um=args.patch_size * args.target_mpp, stain_ref=str(args.stain_ref))
    pd.DataFrame(rows).to_csv(out / f"{args.name}_patches.csv", index=False)
    print(f"\nсохранено: {out / f'{args.name}_feat.npz'}  X={X.shape} y={y.shape}")


if __name__ == "__main__":
    main()
