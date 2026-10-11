# BCSS (молочная железа, TCGA-BRCA) в наш формат признаков. Только для проверки:
# молочной железы не видела ни одна наша модель, в обучение набор не идёт.
#
# 151 область из 151 среза, по одной на пациентку, 21 центр TCGA, 0.25 мкм/px.
# Маска это серый PNG с кодами из meta/gtruth_codes.tsv. В опухоль идёт tumor, в
# строму stroma и иммунные (как лимфоциты NCT-CRC и Inflammation в IGNITE), DCIS,
# сосуды, некроз и остальное в игнор. Картинки на Drive авторы уже нормализовали,
# наша нормализация поверх оценивается по каждой области отдельно.
#
# Область уменьшается до 0.2738 мкм/px и режется на патчи 256 px без перекрытия,
# берутся патчи, где размечено хотя бы 10%. Из крупных областей не больше
# --max-per-roi случайных патчей, иначе кодирование на CPU растянется на полдня;
# TSR области считается по тем же патчам и у разметки, и у модели.
#
#   python -u pan_prep_bcss.py --root data/pan/bcss --name bcss

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.utils import get_device, ensure_dir, TUMOR, STROMA
from src.stain import Normalizer, StainStats, fit_stats, sample_pixels
from seg_extract import build_uni, encode, OUT

Image.MAX_IMAGE_PIXELS = None
LABELS = ("outside_roi", "tumor", "stroma", "lymphocytic_infiltrate", "necrosis_or_debris",
          "glandular_secretions", "blood", "exclude", "metaplasia_NOS", "fat", "plasma_cells",
          "other_immune_infiltrate", "mucoid_material", "normal_acinus_or_duct", "lymphatics",
          "undetermined", "nerve", "skin_adnexa", "blood_vessel", "angioinvasion", "dcis", "other")
TO_TUMOR = ("tumor",)
TO_STROMA = ("stroma",)
IMMUNE = ("lymphocytic_infiltrate", "plasma_cells", "other_immune_infiltrate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/pan/bcss")
    ap.add_argument("--name", default="bcss", help="имя набора, станет <name>_feat.npz")
    ap.add_argument("--no-lym", action="store_true", help="иммунные в игнор, а не в строму")
    ap.add_argument("--stain-ref", default="data/processed/stain_ref_prime.npz")
    ap.add_argument("--src-mpp", type=float, default=0.25)
    ap.add_argument("--target-mpp", type=float, default=0.2738)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--min-label-frac", type=float, default=0.10)
    ap.add_argument("--max-per-roi", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out-dir", default="data/processed/seg_pan")
    args = ap.parse_args()

    root = Path(args.root)
    lut = np.zeros(256, np.uint8)
    lut[[LABELS.index(n) for n in TO_TUMOR]] = TUMOR
    lut[[LABELS.index(n) for n in TO_STROMA + (() if args.no_lym else IMMUNE)]] = STROMA

    rois = sorted((root / "masks").glob("*.png"))
    missing = [p.name for p in rois if not (root / "rgbs_colorNormalized" / p.name).exists()]
    if not rois or missing:
        raise SystemExit(f"масок {len(rois)}, из них без картинки {len(missing)} {missing[:2]}: "
                         "докачайте pan_get_bcss.py")
    print(f"{args.name}: областей {len(rois)}")

    ref = StainStats.load(args.stain_ref)
    factor = args.src_mpp / args.target_mpp
    ps = args.patch_size
    rng = np.random.default_rng(args.seed)
    tiles, masks, rows = [], [], []
    class_px = np.zeros(256, np.int64)
    for k, mpath in enumerate(rois, 1):
        im = Image.open(root / "rgbs_colorNormalized" / mpath.name).convert("RGB")
        size = (round(im.width * factor), round(im.height * factor))
        img = np.asarray(im.resize(size, Image.BILINEAR))
        codes = np.asarray(Image.open(mpath).resize(size, Image.NEAREST))
        if codes.ndim != 2 or codes.max() >= len(LABELS):
            raise SystemExit(f"маска {mpath.name}: форма {codes.shape}, коды {np.unique(codes)[:10]}, "
                             "разбор надо менять")
        class_px += np.bincount(codes.ravel(), minlength=256)
        lab = lut[codes]
        try:
            norm = Normalizer(fit_stats(sample_pixels(img.reshape(-1, 3))), ref)
        except ValueError as err:
            print(f"\n  {mpath.stem}: окраска не оценилась ({err}), оставляю как есть")
            norm = None

        spots = [(y0, x0) for y0 in range(0, size[1] - ps + 1, ps)
                 for x0 in range(0, size[0] - ps + 1, ps)
                 if (lab[y0:y0 + ps, x0:x0 + ps] > 0).mean() >= args.min_label_frac]
        if len(spots) > args.max_per_roi:
            keep = np.sort(rng.choice(len(spots), args.max_per_roi, replace=False))
            spots = [spots[i] for i in keep]
        for y0, x0 in spots:
            m = np.ascontiguousarray(lab[y0:y0 + ps, x0:x0 + ps])
            tile = np.ascontiguousarray(img[y0:y0 + ps, x0:x0 + ps])
            tiles.append(norm(tile) if norm else tile)
            masks.append(np.asarray(Image.fromarray(m).resize((OUT, OUT), Image.NEAREST)))
            # в имени области штрихкод TCGA: пациентка это первые 12 знаков, центр 6-й и 7-й
            rows.append({"roi": mpath.stem, "patient": mpath.stem[:12], "source": mpath.stem[5:7],
                         "organ": "breast", "x0": x0, "y0": y0,
                         "tumor_frac": round(float((m == TUMOR).mean()), 3),
                         "stroma_frac": round(float((m == STROMA).mean()), 3)})
        print(f"  областей {k}/{len(rois)}, патчей {len(tiles)}", end="\r")

    share = {LABELS[c]: round(float(n / class_px[1:].sum()), 3)
             for c, n in enumerate(class_px) if c and n}
    print("\nклассы по площади области:", share)
    if not tiles:
        raise SystemExit("ни одного патча с опухолью или стромой, проверьте коды масок")

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
