# IGNITE (немелкоклеточный рак лёгкого) в наш формат признаков: опухоль и строма.
#
# 408 областей H&E от 82 пациентов: Radboudumc и TCGA, три сканера, резекции и
# биопсии, первичные опухоли лёгкого и метастазы в мозг, печень, надпочечник и
# кость. Всё при 0.5 мкм/px, маска это карта кодов из he_label_map.json. В опухоль
# идёт Tumor epithelium, в строму Stroma, Inflammation и Macrophages (иммунные у
# нас в строме), остальные 13 классов в игнор.
#
# Область увеличивается до 0.2738 мкм/px и режется на патчи 256 px без
# перекрытия; берутся патчи, где размечено хотя бы 10%, как в make_seg_masks_v.py.
# Окраска оценивается по всем областям пациента вместе, как по одному срезу.
#
#   python -u pan_prep_ignite.py --root data/pan/ignite --split test --name ignite_test
#   python -u pan_prep_ignite.py --root data/pan/ignite --split train --name ignite_train

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.utils import get_device, ensure_dir, TUMOR, STROMA
from src.stain import Normalizer, StainStats, fit_stats, sample_pixels
from alt_encode import build_uni, encode
from seg_extract import OUT

Image.MAX_IMAGE_PIXELS = None
TO_TUMOR = ("Tumor epithelium",)
TO_STROMA = ("Stroma", "Inflammation", "Macrophages")


def code_lut(label_map):
    lut = np.zeros(256, np.uint8)
    for name, code in label_map.items():
        if name in TO_TUMOR:
            lut[code] = TUMOR
        elif name in TO_STROMA:
            lut[code] = STROMA
    return lut


def index_pngs(root):
    """{('images' | 'annotations', имя области): путь} для H&E, где бы ни распаковался архив."""
    found = {}
    for p in Path(root).rglob("*.png"):
        if p.parent.name == "he":
            found[(p.parent.parent.name, p.stem)] = p
    return found


def read_codes(path, size, n_codes):
    codes = np.asarray(Image.open(path).resize(size, Image.NEAREST))
    if codes.ndim != 2:
        raise SystemExit(f"маска {path.name} не одноканальная ({codes.shape}), разбор надо менять")
    # в выложенных масках код записан со знаком минус и уложен в байт: опухоль (2)
    # лежит как 254, строма (4) как 252, неразмеченное 0 так и осталось 0. В
    # некоторых масках попадаются и прямые коды (фон 1 рядом с фоном 255), поэтому
    # переворачиваются только значения из верхнего края байта, пересечься они не могут
    negative = codes >= 256 - n_codes + 1
    codes = np.where(negative, 256 - codes.astype(np.int16), codes).astype(np.uint8)
    if codes.max() >= n_codes:
        raise SystemExit(f"в маске {path.name} значения вне карты кодов: {np.unique(codes)[:10]}")
    return codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="папка, куда распакованы images.zip и annotations.zip")
    ap.add_argument("--split", choices=["train", "test", "all"], required=True)
    ap.add_argument("--name", required=True, help="имя набора, станет <name>_feat.npz")
    ap.add_argument("--stain-ref", default="data/processed/stain_ref_prime.npz")
    ap.add_argument("--src-mpp", type=float, default=0.5)
    ap.add_argument("--target-mpp", type=float, default=0.2738)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--min-label-frac", type=float, default=0.10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out-dir", default="data/processed/seg_pan")
    args = ap.parse_args()

    root = Path(args.root)
    label_map = json.loads((root / "he_label_map.json").read_text())
    lut = code_lut(label_map)
    n_codes = max(label_map.values()) + 1
    meta = pd.read_csv(root / "data_overview.csv")
    meta = meta[meta["task"] == "he_tissue_segmentation"]
    if args.split != "all":
        meta = meta[meta["split"] == args.split]
    files = index_pngs(root)
    missing = [n for n in meta["name"] if ("images", n) not in files or ("annotations", n) not in files]
    if missing:
        raise SystemExit(f"не нашлось {len(missing)} областей, например {missing[:3]}: "
                         "распакованы ли оба архива внутрь --root?")
    print(f"{args.name}: областей {len(meta)}, пациентов {meta['patient_id'].nunique()}",
          meta["organ"].value_counts().to_dict())

    ref = StainStats.load(args.stain_ref)
    factor = args.src_mpp / args.target_mpp
    ps = args.patch_size
    tiles, masks, rows = [], [], []
    class_px = np.zeros(n_codes, np.int64)
    for patient, group in meta.groupby("patient_id"):
        images = [Image.open(files[("images", n)]).convert("RGB") for n in group["name"]]
        try:
            stats = fit_stats(sample_pixels(np.concatenate([np.asarray(im).reshape(-1, 3)
                                                            for im in images])))
            norm = Normalizer(stats, ref)
        except ValueError as err:
            # на маленькой биопсии красителей может не хватить для оценки
            print(f"  пациент {patient}: окраска не оценилась ({err}), оставляю как есть")
            norm = None

        for im, r in zip(images, group.itertuples(index=False)):
            size = (round(im.width * factor), round(im.height * factor))
            img = np.asarray(im.resize(size, Image.BILINEAR))
            codes = read_codes(files[("annotations", r.name)], size, n_codes)
            class_px += np.bincount(codes.ravel(), minlength=n_codes)
            lab = lut[codes]
            for y0 in range(0, size[1] - ps + 1, ps):
                for x0 in range(0, size[0] - ps + 1, ps):
                    m = np.ascontiguousarray(lab[y0:y0 + ps, x0:x0 + ps])
                    if (m > 0).mean() < args.min_label_frac:
                        continue
                    tile = np.ascontiguousarray(img[y0:y0 + ps, x0:x0 + ps])
                    tiles.append(norm(tile) if norm else tile)
                    masks.append(np.asarray(Image.fromarray(m).resize((OUT, OUT), Image.NEAREST)))
                    rows.append({"roi": r.name, "patient": patient, "source": r.source,
                                 "scanner": r.scanner, "organ": r.organ,
                                 "subtype": r.histological_subtype, "specimen": r.specimen_type,
                                 "x0": x0, "y0": y0, "tumor_frac": round(float((m == TUMOR).mean()), 3),
                                 "stroma_frac": round(float((m == STROMA).mean()), 3)})
        print(f"  пациент {patient}: областей {len(group)}, патчей всего {len(tiles)}", end="\r")

    names = {code: name for name, code in label_map.items()}
    share = {names[c]: round(float(n / class_px.sum()), 3) for c, n in enumerate(class_px) if n}
    print("\nклассы по площади:", share)
    if not tiles:
        raise SystemExit("ни одного патча с опухолью или стромой, проверьте карту кодов")

    y = np.stack(masks)
    labeled = y > 0
    print(f"патчей {len(tiles)}, размечено {labeled.mean():.1%} пикселей, "
          f"строма среди размеченного {(y[labeled] == STROMA).mean():.1%}")

    device = get_device()
    print("device:", device, "| кодируем через UNI...")
    X = encode(build_uni(device), tiles, device, bs=args.bs)

    out = ensure_dir(args.out_dir)
    np.savez_compressed(out / f"{args.name}_feat.npz", X=X, y=y, patch_px=ps,
                        patch_um=ps * args.target_mpp, stain_ref=str(args.stain_ref))
    pd.DataFrame(rows).to_csv(out / f"{args.name}_patches.csv", index=False)
    print(f"\nсохранено: {out / f'{args.name}_feat.npz'}  X={X.shape} y={y.shape}")


if __name__ == "__main__":
    main()
