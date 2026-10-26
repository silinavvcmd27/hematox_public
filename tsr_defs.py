# Четыре определения TSR по готовым трёхклассовым картам зон.
#
#   TSR              вся ткань среза, как в pan_collect_tsr.py
#   TSR_бед          только полоса вокруг опухоли: строма дальше --dist-um от опухолевой
#                    массы в подсчёт не входит. Из знаменателя уходят сальник, брюшина и
#                    остатки нормального яичника, остаётся реактивная строма.
#   TSR_поле_опухоль круг радиусом --radius-mm с наибольшей площадью опухоли. Отбор идёт
#                    по опухоли, а меряется строма, поэтому оценка смещена вниз; вариант
#                    оставлен для сопоставления с van Wagensveld.
#   TSR_поле_строма  методика патолога: внутри опухолевой массы поле с наибольшей долей
#                    стромы. Буквальный максимум заменён квантилью --quantile, потому что
#                    отбор идёт по той же величине, которая меряется, и на большом стекле
#                    максимум тем выше, чем больше положений окна.
#
# Прежние tumor_bed.py и tsr_variants.py писались под карты seg4, где строма разложена на
# гормональную и матриксную (коды 2 и 3). В картах deploy3 строма это код 5, поэтому коды
# берутся из pan_train, а не задаются числами.
#
# Расстояния и поля считаются по сетке ячеек в --cell-mm: полноразмерный EDT на карте в
# десятки миллионов точек съедает память, а точность в полклетки тут не нужна.
#
# Колонки совпадают с pan_collect_tsr.py, так что таблицу можно отдать в tsr_outcome.py.
#
#   python tsr_defs.py
#   python tsr_defs.py --dist-um 150 --radius-mm 1.1

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve
from scipy.stats import spearmanr

from src.utils import TUMOR
from pan_train import STROMA_CODES, OTHER
from pan_collect_tsr import tsr, halves
from collect_tsr import slide_key

DEFS = ["TSR", "TSR_бед", "TSR_поле_опухоль", "TSR_поле_строма"]


def cell_counts(cls, mpp, cell_mm):
    """Число точек опухоли и стромы в каждой ячейке сетки."""
    step = max(1, int(round(cell_mm * 1000 / mpp)))
    h = cls.shape[0] // step * step
    w = cls.shape[1] // step * step
    if not h or not w:
        return None, None
    blocks = cls[:h, :w].reshape(h // step, step, w // step, step)
    return (blocks == TUMOR).sum((1, 3)), np.isin(blocks, STROMA_CODES).sum((1, 3))


def share_of(tumor, stroma):
    t, s = float(np.sum(tumor)), float(np.sum(stroma))
    return round(s / (t + s), 3) if t + s else None


def bed_tsr(tumor, stroma, cell_mm, dist_um):
    solid = tumor > stroma
    if not solid.any():
        return None, None
    near = distance_transform_edt(~solid, sampling=cell_mm * 1000) <= dist_um
    return share_of(tumor[near], stroma[near]), round(float(near.mean()), 3)


def field_tsr(tumor, stroma, cell_mm, radius_mm, min_cover, quantile):
    r = int(round(radius_mm / cell_mm))
    if min(tumor.shape) < 2 * r + 1:
        return None, None
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    disc = (x * x + y * y <= r * r).astype(np.float64)
    # В круге целое число точек карты, поэтому свёртка округляется до целых. В float32 без
    # округления FFT давал вдали от ткани шум до 0,6 точки, условие total > 0 пропускало
    # такие пустые круги, и их «доля стромы» (шум на шум, вплоть до минус миллионов)
    # попадала в квантиль поля по строме (найдено 28.10.2026, tsr_field_check.py).
    tum = np.rint(fftconvolve(tumor.astype(np.float64), disc, mode="same"))
    stro = np.rint(fftconvolve(stroma.astype(np.float64), disc, mode="same"))
    total = tum + stro
    ok = total > 0
    share = np.where(ok, stro / np.where(ok, total, 1), np.nan)

    i = int(np.argmax(tum))
    by_tumor = round(float(share.flat[i]), 3) if ok.flat[i] else None
    inside = ok & (tum >= min_cover * total)
    by_stroma = round(float(np.quantile(share[inside], quantile)), 3) if inside.any() else None
    return by_tumor, by_stroma


def slide_row(path, args):
    d = np.load(path)
    cls, mpp = d["cls"], float(d["mpp"])
    counts = {"опухоль": int((cls == TUMOR).sum()),
              "строма": int(np.isin(cls, STROMA_CODES).sum()),
              "прочее": int((cls == OTHER).sum())}
    tissue = sum(counts.values())
    if not tissue:
        return None
    left, right = halves(cls)
    row = {"slide": path.name.replace("_seg_map.npz", ""),
           "площадь_мм2": round(tissue * mpp * mpp / 1e6, 1)}
    for name, n in counts.items():
        row[f"{name}_%"] = round(100 * n / tissue, 1)
    row["TSR"] = tsr(cls)
    row["расхождение_половин"] = (round(abs(left - right), 3)
                                  if left is not None and right is not None else None)

    tumor, stroma = cell_counts(cls, mpp, args.cell_mm)
    if tumor is None:
        return row
    row["TSR_бед"], row["доля_полосы"] = bed_tsr(tumor, stroma, args.cell_mm, args.dist_um)
    row["TSR_поле_опухоль"], row["TSR_поле_строма"] = field_tsr(
        tumor, stroma, args.cell_mm, args.radius_mm, args.min_cover, args.quantile)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/results/pan/tcga_ov_deploy3")
    ap.add_argument("--cell-mm", type=float, default=0.05)
    ap.add_argument("--dist-um", type=float, default=250.0)
    ap.add_argument("--radius-mm", type=float, default=1.6)
    ap.add_argument("--min-cover", type=float, default=0.3,
                    help="какую долю ткани поля должна занимать опухоль, чтобы поле "
                         "считалось лежащим в опухолевой массе")
    ap.add_argument("--quantile", type=float, default=0.9)
    ap.add_argument("--keep-duplicates", action="store_true")
    ap.add_argument("--out", default=None, help="по умолчанию tsr_defs.csv рядом с картами")
    args = ap.parse_args()

    rows = []
    for path in sorted(Path(args.dir).glob("*_seg_map.npz")):
        row = slide_row(path, args)
        if row:
            rows.append(row)
            print(f"  {row['slide']}: срез {row['TSR']}, полоса {row.get('TSR_бед')}, "
                  f"поле по опухоли {row.get('TSR_поле_опухоль')}, "
                  f"поле по строме {row.get('TSR_поле_строма')}")
    if not rows:
        raise SystemExit(f"в {args.dir} нет карт *_seg_map.npz")

    table = pd.DataFrame(rows)
    if not args.keep_duplicates:
        table = (table.assign(key=table.slide.map(slide_key))
                      .drop_duplicates("key").drop(columns="key"))

    print(f"\nсрезов {len(table)}")
    print(table[DEFS].describe().loc[["50%", "25%", "75%", "min", "max"]].round(3).to_string())
    print("\nсогласие определений (ранговая корреляция)")
    for i, a in enumerate(DEFS):
        for b in DEFS[i + 1:]:
            ok = table[[a, b]].dropna()
            print(f"  {a} и {b}: ро {spearmanr(ok[a], ok[b])[0]:.3f} по {len(ok)} срезам")

    out = args.out or str(Path(args.dir) / "tsr_defs.csv")
    table.to_csv(out, index=False)
    print("\nтаблица:", out)


if __name__ == "__main__":
    main()
