# NCT-CRC (Kather) в наш формат признаков для первого слоя: опухоль и строма.
#
# По умолчанию берутся три класса набора: TUM это опухоль (код 1), STR и LYM это
# строма (код 5, иммунные в нашем TSR входят в строму). Детрит, слизь, жир и
# нормальную слизистую можно взять отдельным набором как прочую ткань (код 7) для
# модели с третьим выходом; фон и мышца не берутся никогда. Класс один на весь
# патч, поэтому маска патча залита одним кодом.
#
# Патч 224 px при 0.5 мкм/px увеличивается до 0.2738 мкм/px, из центра берётся
# 256 px (70 мкм, как у наших срезов), окраска оценивается по пулу патчей опухоли,
# стромы и лимфоцитов (--stain-classes) и приводится к нашему эталону, дальше UNI.
# Пул один и тот же при любых --classes: на жире или слизи Macenko оценивается
# плохо, а прочее должно нормализоваться так же, как основной набор.
#
#   обучающая часть, по 4000 патчей на класс:
#     python -u pan_prep_crc.py --root data/pan/crc/NCT-CRC-HE-100K-NONORM --name nct_crc --per-class 4000
#   отложенная часть для бенчмарка, целиком:
#     python -u pan_prep_crc.py --root data/pan/crc/CRC-VAL-HE-7K --name nct_crc_val
#   прочее:
#     python -u pan_prep_crc.py --root data/pan/crc/NCT-CRC-HE-100K-NONORM --name nct_crc_other \
#       --classes DEB MUC ADI NORM --per-class 2000

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.utils import get_device, ensure_dir, TUMOR, STROMA
from src.stain import Normalizer, StainStats, fit_stats, sample_pixels
from seg_extract import build_uni, encode, OUT
from pan_eval_crc import to_model_scale
from pan_train import OTHER

CODES = {"TUM": TUMOR, "STR": STROMA, "LYM": STROMA,
         "DEB": OTHER, "MUC": OTHER, "ADI": OTHER, "NORM": OTHER}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="папка с подпапками классов ADI ... TUM")
    ap.add_argument("--name", required=True, help="имя набора, станет <name>_feat.npz")
    ap.add_argument("--classes", nargs="+", default=["TUM", "STR", "LYM"], choices=list(CODES))
    ap.add_argument("--stain-classes", nargs="+", default=["TUM", "STR", "LYM"])
    ap.add_argument("--per-class", type=int, default=0, help="сколько патчей класса брать, 0 все")
    ap.add_argument("--no-lym", action="store_true", help="лимфоциты не считать стромой")
    ap.add_argument("--stain-ref", default="data/processed/stain_ref_prime.npz")
    ap.add_argument("--src-mpp", type=float, default=0.5)
    ap.add_argument("--target-mpp", type=float, default=0.2738)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data/processed/seg_pan")
    args = ap.parse_args()

    root = Path(args.root)
    classes = [c for c in args.classes if not (args.no_lym and c == "LYM")]
    rng = np.random.default_rng(args.seed)
    files = []
    for cls in classes:
        paths = sorted((root / cls).glob("*.tif"))
        if not paths:
            raise SystemExit(f"в {root}/{cls} нет .tif")
        if args.per_class and len(paths) > args.per_class:
            paths = sorted(rng.choice(paths, args.per_class, replace=False))
        files += [(cls, CODES[cls], p) for p in paths]
    print(f"{args.name}: патчей {len(files)}",
          pd.Series([c for c, _, _ in files]).value_counts().to_dict())

    stain_paths = [p for c in args.stain_classes for p in sorted((root / c).glob("*.tif"))]
    pick = rng.choice(len(stain_paths), min(500, len(stain_paths)), replace=False)
    pool = np.stack([np.asarray(Image.open(stain_paths[i]).convert("RGB")) for i in pick])
    norm = Normalizer(fit_stats(sample_pixels(pool)), StainStats.load(args.stain_ref))

    tiles = [norm(to_model_scale(Image.open(p).convert("RGB"), args.src_mpp,
                                 args.target_mpp, args.patch_size)) for _, _, p in files]
    y = np.stack([np.full((OUT, OUT), code, np.uint8) for _, code, _ in files])

    device = get_device()
    print("device:", device, "| кодируем через UNI...")
    X = encode(build_uni(device), tiles, device, bs=args.bs)

    out = ensure_dir(args.out_dir)
    np.savez_compressed(out / f"{args.name}_feat.npz", X=X, y=y, patch_px=args.patch_size,
                        patch_um=args.patch_size * args.target_mpp, stain_ref=str(args.stain_ref))
    pd.DataFrame({"file": [p.name for _, _, p in files],
                  "class": [c for c, _, _ in files],
                  "code": [code for _, code, _ in files]}).to_csv(out / f"{args.name}_patches.csv", index=False)
    print(f"\nсохранено: {out / f'{args.name}_feat.npz'}  X={X.shape} y={y.shape}")


if __name__ == "__main__":
    main()
