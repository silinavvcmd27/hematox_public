# Как нынешняя сегментация делит опухоль и строму на чужой опухоли: CRC-VAL-HE-7K.
#
# Набор это 7 180 патчей 224 px при 0.5 мкм/px от 50 пациентов с колоректальным
# раком (Kather et al. 2019), класс один на весь патч. Модель училась на
# 0.2738 мкм/px и поле 70 мкм, поэтому патч увеличивается до этого масштаба и из
# центра берётся 256 px. Гормональная и матриксная сливаются в «строму», класс
# патча это то, что занимает больше половины его площади, как в TSR.
#
#   unzip -q data/pan/crc/CRC-VAL-HE-7K.zip -d data/pan/crc
#   python -u pan_eval_crc.py --root data/pan/crc/CRC-VAL-HE-7K --model outputs/models/seg4_deploy.pth
#
# Патчи набора авторы уже нормализовали по Macenko к своему эталону. Мы оцениваем
# окраску по пулу патчей как по одному срезу и приводим к нашему эталону, как на
# обучении; --no-stain выключает это для сравнения.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.utils import get_device, ensure_dir
from src.stain import Normalizer, StainStats, fit_stats, sample_pixels
from seg_infer_svs import build_uni, load_decoder, run_batch

TUMOR = 1
STROMA_CODES = (2, 3, 5)
# как класс набора считается в нашей схеме; остальные классы только в отчёт
TRUTH = {"TUM": "tumor", "STR": "stroma", "LYM": "stroma"}


def to_model_scale(img, src_mpp, dst_mpp, size):
    scale = src_mpp / dst_mpp
    side = int(round(img.width * scale))
    img = img.resize((side, side), Image.BILINEAR)
    off = (side - size) // 2
    return np.asarray(img.crop((off, off, off + size, off + size)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="папка с подпапками классов ADI ... TUM")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stain-ref", default="data/processed/stain_ref_prime.npz")
    ap.add_argument("--no-stain", action="store_true")
    ap.add_argument("--src-mpp", type=float, default=0.5)
    ap.add_argument("--target-mpp", type=float, default=0.2738)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--per-class", type=int, default=0, help="сколько патчей класса брать, 0 все")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/results/pan/crc_val_seg4.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    files = []
    for d in sorted(p for p in Path(args.root).iterdir() if p.is_dir()):
        paths = sorted(d.glob("*.tif"))
        if args.per_class and len(paths) > args.per_class:
            paths = list(rng.choice(paths, args.per_class, replace=False))
        files += [(d.name, p) for p in paths]
    print(f"патчей {len(files)}:", pd.Series([c for c, _ in files]).value_counts().to_dict())

    norm = None
    if not args.no_stain:
        pick = rng.choice(len(files), min(500, len(files)), replace=False)
        pool = np.stack([np.asarray(Image.open(files[i][1]).convert("RGB")) for i in pick])
        norm = Normalizer(fit_stats(sample_pixels(pool)), StainStats.load(args.stain_ref))
        print("окраска набора оценена по", len(pick), "патчам и приводится к", args.stain_ref)

    device = get_device()
    uni = build_uni(device)
    dec, _, ch2cls = load_decoder(args.model, device)
    ch2cls = np.array(ch2cls)
    tumor_ch = ch2cls == TUMOR
    stroma_ch = np.isin(ch2cls, STROMA_CODES)

    rows = []
    for a in range(0, len(files), args.bs):
        chunk = files[a:a + args.bs]
        tiles = [to_model_scale(Image.open(p).convert("RGB"), args.src_mpp, args.target_mpp,
                                args.patch_size) for _, p in chunk]
        probs = run_batch(uni, dec, tiles, device, norm)
        winner = probs.argmax(1)
        for (cls, path), pr, win in zip(chunk, probs, winner):
            rows.append({"file": path.name, "class": cls,
                         "tumor_frac": float(tumor_ch[win].mean()),
                         "stroma_frac": float(stroma_ch[win].mean()),
                         "p_tumor": float(pr[tumor_ch].sum(0).mean()),
                         "p_stroma": float(pr[stroma_ch].sum(0).mean())})
        print(f"  {min(a + args.bs, len(files))}/{len(files)}", end="\r")

    res = pd.DataFrame(rows)
    res["pred"] = np.where(res["tumor_frac"] > 0.5, "tumor", "stroma")
    out = Path(args.out)
    ensure_dir(out.parent)
    res.to_csv(out, index=False)

    print("\nкуда уходит каждый класс набора (доля патчей):")
    by_class = res.groupby("class").agg(n=("pred", "size"),
                                        tumor=("pred", lambda s: (s == "tumor").mean()),
                                        tumor_area=("tumor_frac", "mean"))
    print(by_class.round(3).to_string())

    scored = res[res["class"].isin(TRUTH)].copy()
    scored["truth"] = scored["class"].map(TRUTH)
    for nm in ("tumor", "stroma"):
        p, t = scored["pred"] == nm, scored["truth"] == nm
        dice = 2 * (p & t).sum() / max(p.sum() + t.sum(), 1)
        print(f"{nm:7s} Dice {dice:.3f}  чувствительность {(p & t).sum() / max(t.sum(), 1):.3f}")
    recall = [((scored["pred"] == nm) & (scored["truth"] == nm)).sum() / max((scored["truth"] == nm).sum(), 1)
              for nm in ("tumor", "stroma")]
    print(f"сбалансированная точность {np.mean(recall):.3f} (TUM против STR и LYM)")
    print("по патчам:", out)


if __name__ == "__main__":
    main()
