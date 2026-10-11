# Цветовые признаки вместо UNI, на тех же фрагментах.
#
# Нужны, чтобы ответить на вопрос «а почему не просто порог по цвету»: без этой
# линии сравнение базовых моделей показывает только вклад головы, а не вклад
# кодировщика. Фрагменты берутся те же самые, по координатам из <набор>_patches.csv,
# нарезка и нормализация окраски повторяют pan_prep_ignite.py, метки копируются из
# готовых признаков, поэтому строки совпадают один к одному.
#
# На каждую клетку сетки 14 на 14 (те же 16 пикселей входа UNI, то есть около 5 мкм)
# считается девять чисел: средняя и стандартная концентрация гематоксилина, то же для
# эозина, средняя оптическая плотность, доля тёмных пикселей и средние R, G, B.
#
# Результат ложится рядом как <набор>_color_feat.npz, так что дальше его читает
# pan_baselines.py без изменений:
#
#   python -u pan_color_feats.py --set ignite_train3 --root data/pan/ignite
#   python -u pan_color_feats.py --set ignite_test3  --root data/pan/ignite
#   python -u pan_baselines.py --other --kinds linear mlp --lr 1e-2 \
#     --dirs data/processed/seg_color --train ignite_train3_color --val ignite_test3_color \
#     --out outputs/results/pan/baselines_color.csv

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.utils import ensure_dir
from src.stain import Normalizer, StainStats, fit_stats, sample_pixels, to_od
from seg_extract import OUT

Image.MAX_IMAGE_PIXELS = None
GRID = 14
DARK_OD = 0.5
FEATURES = ("hema_mean", "hema_std", "eosin_mean", "eosin_std",
            "od_mean", "dark_frac", "r_mean", "g_mean", "b_mean")


def patches_table(dirs, name):
    """Таблица фрагментов набора; если в ней нет координат, берём её у базового набора."""
    for d in dirs:
        p = Path(d) / f"{name}_patches.csv"
        if p.exists():
            t = pd.read_csv(p)
            if {"x0", "y0", "roi"} <= set(t.columns):
                return t
            base = name.rstrip("3")
            for d2 in dirs:
                q = Path(d2) / f"{base}_patches.csv"
                if q.exists():
                    t2 = pd.read_csv(q)
                    if len(t2) == len(t):
                        print(f"координат в {p.name} нет, беру их из {q.name}")
                        return t2
            raise SystemExit(f"в {p} нет колонок x0, y0, roi")
    raise SystemExit(f"нет {name}_patches.csv ни в одной из папок {dirs}")


def labels(dirs, name):
    for d in dirs:
        p = Path(d) / f"{name}_feat.npz"
        if p.exists():
            return np.load(p)["y"]
    raise SystemExit(f"нет {name}_feat.npz ни в одной из папок {dirs}")


def cell_features(tile, he):
    """Девять чисел на каждую клетку сетки 14 на 14."""
    small = np.asarray(Image.fromarray(tile).resize((GRID * 16, GRID * 16), Image.BILINEAR))
    od = to_od(small)
    conc = np.linalg.lstsq(he, od.T, rcond=None)[0].T          # концентрации H и E
    px = np.concatenate([conc,
                         np.linalg.norm(od, axis=1, keepdims=True),
                         (np.linalg.norm(od, axis=1, keepdims=True) > DARK_OD).astype(np.float64),
                         small.reshape(-1, 3) / 255.0], axis=1)
    cells = px.reshape(GRID, 16, GRID, 16, px.shape[1]).transpose(0, 2, 1, 3, 4)
    cells = cells.reshape(GRID * GRID, 16 * 16, px.shape[1])
    m, s = cells.mean(1), cells.std(1)
    return np.stack([m[:, 0], s[:, 0], m[:, 1], s[:, 1],
                     m[:, 2], m[:, 3], m[:, 4], m[:, 5], m[:, 6]], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="имя набора, например ignite_test3")
    ap.add_argument("--root", required=True, help="папка набора, например data/pan/ignite")
    ap.add_argument("--dirs", nargs="+", default=["data/processed/seg", "data/processed/seg_pan"])
    ap.add_argument("--stain-ref", default="data/processed/stain_ref_prime.npz")
    ap.add_argument("--src-mpp", type=float, default=0.5)
    ap.add_argument("--target-mpp", type=float, default=0.2738)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--out-dir", default="data/processed/seg_color")
    args = ap.parse_args()

    tab = patches_table(args.dirs, args.set)
    y = labels(args.dirs, args.set)
    if len(tab) != len(y):
        raise SystemExit(f"строк в таблице {len(tab)}, а масок {len(y)}: наборы разъехались")
    pics = {p.stem: p for p in Path(args.root).glob("**/images/**/*.png")}
    missing = sorted(set(tab["roi"]) - set(pics))
    if missing:
        raise SystemExit(f"не нашлись картинки областей: {missing[:5]}")
    # окраска оценивается по всем областям пациента разом, как в подготовке данных;
    # если пациент в таблице не записан, оцениваем по области
    key = "patient" if "patient" in tab.columns else "roi"
    print(f"{args.set}: фрагментов {len(tab)}, областей {tab['roi'].nunique()}, "
          f"окраска по {tab[key].nunique()} группам ({key})")

    ref = StainStats.load(args.stain_ref)
    factor, ps = args.src_mpp / args.target_mpp, args.patch_size
    X = np.zeros((len(tab), GRID * GRID, len(FEATURES)), np.float32)
    done = 0
    for patient, group in tab.groupby(key):
        rois = {r: Image.open(pics[r]).convert("RGB") for r in group["roi"].unique()}
        try:
            stats = fit_stats(sample_pixels(np.concatenate(
                [np.asarray(im).reshape(-1, 3) for im in rois.values()])))
            norm = Normalizer(stats, ref)
        except ValueError as err:
            print(f"  пациент {patient}: окраска не оценилась ({err}), оставляю как есть")
            norm = None

        for roi, im in rois.items():
            size = (round(im.width * factor), round(im.height * factor))
            img = np.asarray(im.resize(size, Image.BILINEAR))
            rows = group[group["roi"] == roi]
            for i, x0, y0 in zip(rows.index, rows["x0"], rows["y0"]):
                tile = np.ascontiguousarray(img[y0:y0 + ps, x0:x0 + ps])
                X[i] = cell_features(norm(tile) if norm else tile, ref.he)
                done += 1
            print(f"\r  {done}/{len(tab)}", end="")
    print()

    out = ensure_dir(args.out_dir)
    np.savez_compressed(out / f"{args.set}_color_feat.npz", X=X, y=y,
                        features=np.array(FEATURES), patch_px=ps, grid=GRID, mask_px=OUT)
    tab.to_csv(out / f"{args.set}_color_patches.csv", index=False)
    print(f"сохранено: {out / f'{args.set}_color_feat.npz'}  X={X.shape} y={y.shape}")
    print("средние по признакам: " + ", ".join(
        f"{nm} {v:.3f}" for nm, v in zip(FEATURES, X.reshape(-1, len(FEATURES)).mean(0))))


if __name__ == "__main__":
    main()
