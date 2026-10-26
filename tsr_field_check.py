# Проверка поля по строме из tsr_defs.py: две возможные слабости.
#
# 1) Свёртка через FFT в float32 даёт вдали от ткани не ровный ноль, а шум. Условие
#    total > 0 в field_tsr пропускает часть таких положений, и их «доля стромы» (шум,
#    делённый на шум) попадает в квантиль. Выше 0,70 она не бывает по построению, ниже
#    бывает сколь угодно, но часть шумовых долей ложится и между 0 и 0,70, так что квантиль
#    сдвигается в любую сторону. После исправления tsr_defs.py 28.10 (свёртка в float64 с
#    округлением) варианты «как сейчас» и «без шума» должны совпасть.
# 2) Нет требования к заполнению круга тканью: круг, почти целиком лежащий на стекле,
#    считается наравне с полным.
#
# Для каждого среза три варианта квантили: как сейчас (сам field_tsr), без шумовых
# положений (в круге хотя бы одна точка опухоли или стромы) и только по кругам, которые
# опухоль и строма заполняют хотя бы на --fill. Исход не используется, это проверка
# измерения, а не анализ.
#
#   python tsr_field_check.py --dir outputs/results/pan/tcga_ov_deploy5_raw
#   python tsr_field_check.py --dir outputs/results/pan/ptrc_deploy5_raw

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from scipy.stats import spearmanr

from tsr_defs import cell_counts, field_tsr


def варианты(tumor, stroma, step, args):
    r = int(round(args.radius_mm / args.cell_mm))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    disc = (x * x + y * y <= r * r).astype(np.float32)
    tum = fftconvolve(tumor.astype(np.float32), disc, mode="same")
    stro = fftconvolve(stroma.astype(np.float32), disc, mode="same")
    total = tum + stro
    ёмкость = float(disc.sum()) * step * step

    def квантиль(ok):
        share = np.where(ok, stro / np.where(ok, total, 1), np.nan)
        inside = ok & (tum >= args.min_cover * total)
        if not inside.any():
            return np.nan, 0, np.nan
        return (float(np.quantile(share[inside], args.quantile)), int(inside.sum()),
                float(share[inside].min()))

    сейчас, n_сейчас, мин_сейчас = квантиль(total > 0)
    чисто, n_чисто, _ = квантиль(total > 0.5)
    полные, n_полные, _ = квантиль(total >= args.fill * ёмкость)
    return {"сейчас": сейчас, "без_шума": чисто, "полные_круги": полные,
            "положений": n_сейчас, "шумовых": n_сейчас - n_чисто, "полных": n_полные,
            "наименьшая_доля_сейчас": мин_сейчас,
            "шум_вне_ткани": float(np.abs(total[total < 0.5]).max()) if (total < 0.5).any()
            else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/results/pan/tcga_ov_deploy5_raw")
    ap.add_argument("--cell-mm", type=float, default=0.05)
    ap.add_argument("--radius-mm", type=float, default=1.6)
    ap.add_argument("--min-cover", type=float, default=0.3)
    ap.add_argument("--quantile", type=float, default=0.9)
    ap.add_argument("--fill", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="по умолчанию field_check.csv рядом с картами")
    args = ap.parse_args()

    строки = []
    файлы = sorted(Path(args.dir).glob("*_seg_map.npz"))
    for k, путь in enumerate(файлы, 1):
        d = np.load(путь)
        cls, mpp = d["cls"], float(d["mpp"])
        tumor, stroma = cell_counts(cls, mpp, args.cell_mm)
        if tumor is None:
            continue
        step = max(1, int(round(args.cell_mm * 1000 / mpp)))
        _, официальное = field_tsr(tumor, stroma, args.cell_mm, args.radius_mm,
                                   args.min_cover, args.quantile)
        r = варианты(tumor, stroma, step, args)
        строки.append({"slide": путь.name.replace("_seg_map.npz", ""),
                       "TSR_поле_строма": официальное, **r})
        print(f"  {k}/{len(файлы)}", end="\r")

    t = pd.DataFrame(строки)
    расхождение = (t.сейчас.round(3) - t.TSR_поле_строма).abs().max()
    print(f"\nсрезов {len(t)}; пересчёт «как сейчас» совпадает с tsr_defs.py до "
          f"{расхождение:.3f}")
    без_шума = (t.без_шума.round(3) - t.TSR_поле_строма).abs()
    print(f"tsr_defs.py против варианта без шума: наибольшее расхождение {без_шума.max():.3f}"
          " (после исправления должно быть 0)")
    print(f"шум свёртки float32 вне ткани: медиана наибольшего {t.шум_вне_ткани.median():.2g}, "
          f"наибольший {t.шум_вне_ткани.max():.2g}")
    print(f"шумовых положений среди допустимых: медиана {(t.шумовых / t.положений).median():.1%}, "
          f"наибольшая доля {(t.шумовых / t.положений).max():.1%}; срезов, где они есть, "
          f"{int((t.шумовых > 0).sum())}")
    print(f"наименьшая «доля стромы» среди допустимых положений: медиана "
          f"{t.наименьшая_доля_сейчас.median():.3f}, минимум {t.наименьшая_доля_сейчас.min():.3g}")
    for кол, имя in (("без_шума", "без шумовых положений"),
                     ("полные_круги", f"только круги, заполненные на {args.fill:.0%}")):
        ok = t[["TSR_поле_строма", кол]].dropna()
        разн = ok[кол] - ok.TSR_поле_строма
        print(f"\n{имя} против tsr_defs.py: срезов {len(ok)}, сдвиг медиана {разн.median():+.4f}, "
              f"от {разн.min():+.3f} до {разн.max():+.3f}, |сдвиг| > 0,02 у "
              f"{int((разн.abs() > 0.02).sum())}; ро {spearmanr(ok.TSR_поле_строма, ok[кол])[0]:.3f}; "
              f"медиана {ok[кол].median():.3f} против {ok.TSR_поле_строма.median():.3f}")
        print(t.reindex(разн.abs().sort_values(ascending=False).index).head(5)
              [["slide", "TSR_поле_строма", кол, "шумовых", "положений"]].to_string(index=False))
    путь = args.out or str(Path(args.dir) / "field_check.csv")
    t.to_csv(путь, index=False)
    print("\nтаблица:", путь)


if __name__ == "__main__":
    main()
