# TIGER (рак молочной железы) в наш формат признаков.
#
# Подмножество WSIROIS: области с пиксельной разметкой по семи классам. Коды маски:
# 1 инвазивная опухоль, 2 опухоль-ассоциированная строма, 3 карцинома in situ,
# 4 здоровые железы, 5 некроз, 6 воспалённая строма, 7 прочее, 0 вне области.
#
# Раскладка по нашим классам: 1 в опухоль; 2 и 6 в строму (иммунные у нас в строме);
# 4 и 5 в прочее; 3 в игнор, потому что in situ это спорный случай и своего класса у
# нас для него нет; 0 в игнор.
#
# Класс 7 (rest) это сборная солянка: жир, сосуды, кожа, кровь. По смыслу это наше
# прочее, но определения могут не совпасть, поэтому он управляется ключом --rest.
# Рядом с готовой картой в файл кладутся исходные коды TIGER (y_raw), так что любую
# другую раскладку можно собрать потом, не гоняя UNI заново.
#
# Папка tissue-bcss это те же области, из которых сделан BCSS, наш внешний бенчмарк.
# В обучение идёт только tissue-cells (RUMC и JB), иначе бенчмарк перестанет быть
# внешним.
#
#   python -u pan_prep_tiger.py --root data/pan/tiger/roi-level-annotations --name tiger_cells
#   python -u pan_prep_tiger.py --root ... --subset tissue-bcss --name tiger_bcss

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.utils import get_device, ensure_dir, TUMOR, STROMA
from src.stain import Normalizer, StainStats, fit_stats, sample_pixels
from pan_train import OTHER
from seg_extract import build_uni, encode, OUT

Image.MAX_IMAGE_PIXELS = None
SLIDE = re.compile(r"^(.*?)_\[")


def code_lut(rest):
    lut = np.zeros(256, np.uint8)
    lut[1] = TUMOR
    lut[2] = lut[6] = STROMA
    lut[4] = lut[5] = OTHER
    if rest == "other":
        lut[7] = OTHER
    return lut


def slide_of(path):
    m = SLIDE.match(path.name)
    return m.group(1) if m else path.stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="папка roi-level-annotations")
    ap.add_argument("--subset", default="tissue-cells", choices=["tissue-cells", "tissue-bcss"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--rest", default="other", choices=["other", "ignore"],
                    help="куда отнести класс 7")
    ap.add_argument("--stain-ref", default="data/processed/stain_ref_prime.npz")
    ap.add_argument("--src-mpp", type=float, default=0.5)
    ap.add_argument("--target-mpp", type=float, default=0.2738)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--min-label-frac", type=float, default=0.10)
    ap.add_argument("--max-per-roi", type=int, default=8,
                    help="сколько патчей брать с области; областей почти две тысячи, "
                         "без ограничения UNI будет считать неделю")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out-dir", default="data/processed/seg_pan")
    args = ap.parse_args()

    sub = Path(args.root) / args.subset
    images = sorted((sub / "images").glob("*.png"))
    if not images:
        raise SystemExit(f"нет картинок в {sub / 'images'}")
    by_slide = {}
    for p in images:
        by_slide.setdefault(slide_of(p), []).append(p)
    print(f"{args.name}: областей {len(images)}, срезов {len(by_slide)}, класс 7 в {args.rest}")

    lut = code_lut(args.rest)
    ref = StainStats.load(args.stain_ref)
    rng = np.random.default_rng(args.seed)
    factor = args.src_mpp / args.target_mpp
    ps = args.patch_size
    tiles, masks, raws, rows = [], [], [], []

    for slide, paths in by_slide.items():
        opened = [Image.open(p).convert("RGB") for p in paths]
        try:
            stats = fit_stats(sample_pixels(np.concatenate([np.asarray(im).reshape(-1, 3)
                                                            for im in opened])))
            norm = Normalizer(stats, ref)
        except ValueError as err:
            print(f"  срез {slide}: окраска не оценилась ({err}), оставляю как есть")
            norm = None

        for im, path in zip(opened, paths):
            size = (round(im.width * factor), round(im.height * factor))
            img = np.asarray(im.resize(size, Image.BILINEAR))
            codes = np.asarray(Image.open(sub / "masks" / path.name).resize(size, Image.NEAREST))
            if codes.ndim != 2:
                raise SystemExit(f"маска {path.name} не одноканальная ({codes.shape})")
            lab = lut[codes]

            spots = [(y0, x0)
                     for y0 in range(0, size[1] - ps + 1, ps)
                     for x0 in range(0, size[0] - ps + 1, ps)
                     if (lab[y0:y0 + ps, x0:x0 + ps] > 0).mean() >= args.min_label_frac]
            if len(spots) > args.max_per_roi:
                spots = [spots[i] for i in rng.choice(len(spots), args.max_per_roi, replace=False)]
            for y0, x0 in spots:
                m = np.ascontiguousarray(lab[y0:y0 + ps, x0:x0 + ps])
                raw = np.ascontiguousarray(codes[y0:y0 + ps, x0:x0 + ps])
                tile = np.ascontiguousarray(img[y0:y0 + ps, x0:x0 + ps])
                tiles.append(norm(tile) if norm else tile)
                masks.append(np.asarray(Image.fromarray(m).resize((OUT, OUT), Image.NEAREST)))
                raws.append(np.asarray(Image.fromarray(raw).resize((OUT, OUT), Image.NEAREST)))
                rows.append({"roi": path.stem, "slide": slide, "patient": slide,
                             "source": "tcga" if slide.upper().startswith("TCGA") else "tiger",
                             "scanner": "unknown", "organ": "breast",
                             "x0": x0, "y0": y0,
                             "tumor_frac": round(float((m == TUMOR).mean()), 3),
                             "stroma_frac": round(float((m == STROMA).mean()), 3),
                             "other_frac": round(float((m == OTHER).mean()), 3)})
        print(f"  срез {slide}: областей {len(paths)}, патчей всего {len(tiles)}", end="\r")

    y = np.stack(masks)
    labeled = y > 0
    print(f"\nпатчей {len(tiles)}, размечено {labeled.mean():.1%} пикселей; "
          f"среди размеченного опухоль {(y[labeled] == TUMOR).mean():.1%}, "
          f"строма {(y[labeled] == STROMA).mean():.1%}, прочее {(y[labeled] == OTHER).mean():.1%}")

    device = get_device()
    print("device:", device, "| кодируем через UNI...")
    X = encode(build_uni(device), tiles, device, bs=args.bs)

    out = ensure_dir(args.out_dir)
    np.savez_compressed(out / f"{args.name}_feat.npz", X=X, y=y, y_raw=np.stack(raws),
                        patch_px=ps, patch_um=ps * args.target_mpp,
                        stain_ref=str(args.stain_ref), rest=args.rest)
    pd.DataFrame(rows).to_csv(out / f"{args.name}_patches.csv", index=False)
    print(f"\nсохранено: {out / f'{args.name}_feat.npz'}  X={X.shape} y={y.shape}")


if __name__ == "__main__":
    main()
