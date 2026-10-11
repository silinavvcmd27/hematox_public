# Что дала новая модель на когорте TCGA-OV: сравнение с прежним подсчётом и связь TSR
# с выживаемостью.
#
# Первая часть сводит старую когорту seg4 и новую deploy3 по срезам. Интересует не
# столько сдвиг уровня, сколько сохранился ли порядок срезов.
#
# Вторая часть берёт клиническую таблицу и делит пациенток по TSR. Повторные препараты
# одной пациентки схлопываются, остаётся срез с наибольшей площадью ткани. Лог-ранг
# считается руками, чтобы не тащить lifelines; если он всё же стоит, дополнительно
# считается Кокс с поправкой на возраст и стадию.
#
# Уровень TSR сильно зависит от источника TCGA (Краскел-Уоллис p 0.0025), поэтому
# кроме обычного лог-ранга считается стратифицированный: сравнение идёт только внутри
# источника. Когорта маленькая, 37 пациенток, так что это разведка, а не доказательство.
#
#   python tsr_outcome.py
#   python tsr_outcome.py --split 0.5

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr, kruskal


def logrank_terms(days, event, high):
    """Наблюдённые и ожидаемые события в группе высокого TSR плюс дисперсия."""
    obs = exp = var = 0.0
    for t in np.unique(days[event == 1]):
        at_risk = days >= t
        n = at_risk.sum()
        n_high = (at_risk & high).sum()
        d = ((days == t) & (event == 1)).sum()
        obs += ((days == t) & (event == 1) & high).sum()
        exp += d * n_high / n
        if n > 1:
            var += d * (n_high / n) * (1 - n_high / n) * (n - d) / (n - 1)
    return obs, exp, var


def logrank(days, event, high):
    obs, exp, var = logrank_terms(days, event, high)
    stat = (obs - exp) ** 2 / var if var else np.nan
    return stat, chi2.sf(stat, 1), obs, exp


def logrank_stratified(days, event, high, stratum):
    obs = exp = var = 0.0
    used = []
    for name in np.unique(stratum):
        m = stratum == name
        if m.sum() < 4 or high[m].all() or not high[m].any():
            continue
        o, e, v = logrank_terms(days[m], event[m], high[m])
        obs, exp, var = obs + o, exp + e, var + v
        used.append(f"{name} (n={m.sum()})")
    if not used:
        return None
    stat = (obs - exp) ** 2 / var if var else np.nan
    return stat, chi2.sf(stat, 1), obs, exp, used


def km_median(days, event):
    """Медиана выживаемости по Каплану-Мейеру, None если кривая не опустилась до 0,5."""
    s = 1.0
    for t in np.unique(days[event == 1]):
        n = (days >= t).sum()
        s *= 1 - ((days == t) & (event == 1)).sum() / n
        if s <= 0.5:
            return float(t)
    return None


def source_of(slides):
    return slides.str.split("-").str[1]


def compare_cohorts(new, old):
    both = new.merge(old, on="slide", suffixes=("_новая", "_старая"))
    if both.empty:
        print("старая и новая когорты не сошлись по именам срезов")
        return
    d = both.TSR_новая - both.TSR_старая
    print(f"сошлось срезов: {len(both)}")
    print(f"TSR медиана: старая {both.TSR_старая.median():.3f}, новая {both.TSR_новая.median():.3f}")
    print(f"сдвиг: медиана {d.median():+.3f}, размах {d.min():+.3f}..{d.max():+.3f}")
    print(f"ранговая корреляция порядка срезов: {spearmanr(both.TSR_новая, both.TSR_старая)[0]:.3f}")
    rich = both[both["прочее_%"] >= 15]
    poor = both[both["прочее_%"] < 15]
    print(f"сдвиг там, где прочего много (от 15 %, n={len(rich)}): медиана {(rich.TSR_новая - rich.TSR_старая).median():+.3f}")
    print(f"сдвиг там, где прочего мало (n={len(poor)}): медиана {(poor.TSR_новая - poor.TSR_старая).median():+.3f}\n")
    worst = both.reindex(d.abs().sort_values(ascending=False).index).head(5)
    print("сильнее всего разошлись:")
    print(worst[["slide", "TSR_старая", "TSR_новая", "прочее_%"]].to_string(index=False))


def by_source(t):
    src = source_of(t.slide)
    groups = [g.TSR.values for _, g in t.groupby(src) if len(g) >= 3]
    print("\nмедиана TSR по источникам (где срезов от трёх):")
    for name, g in t.groupby(src):
        if len(g) >= 3:
            print(f"  TSS {name}: n={len(g)}, TSR {g.TSR.median():.3f}, "
                  f"расхождение половин {g.расхождение_половин.median():.3f}")
    if len(groups) > 1:
        print(f"  Краскел-Уоллис по источникам: p = {kruskal(*groups)[1]:.4f}")


def outcome(t, split):
    per_patient = (t.sort_values("площадь_мм2").groupby("пациент").last().reset_index()
                    .dropna(subset=["дней_наблюдения", "событие"]))
    days = per_patient.дней_наблюдения.to_numpy(float)
    event = per_patient.событие.to_numpy(int)
    cut = per_patient.TSR.median() if split is None else split
    high = (per_patient.TSR > cut).to_numpy()

    print(f"\nпациенток {len(per_patient)}, событий {event.sum()}, порог TSR {cut:.3f}")
    print(f"  высокий TSR: n={high.sum()}, медиана выживаемости {km_median(days[high], event[high])}")
    print(f"  низкий  TSR: n={(~high).sum()}, медиана выживаемости {km_median(days[~high], event[~high])}")
    stat, p, obs, exp = logrank(days, event, high)
    print(f"  лог-ранг: хи-квадрат {stat:.2f}, p = {p:.3f} (событий при высоком TSR "
          f"{obs:.0f}, ожидалось {exp:.1f})")

    strat = logrank_stratified(days, event, high, source_of(per_patient.slide).to_numpy())
    if strat:
        stat, p, obs, exp, used = strat
        print(f"  с поправкой на источник: хи-квадрат {stat:.2f}, p = {p:.3f} "
              f"({obs:.0f} против {exp:.1f}), источники: {', '.join(used)}")

    try:
        from lifelines import CoxPHFitter
    except ImportError:
        print("  lifelines не установлен, Кокс пропущен")
        return per_patient
    fit = per_patient[["дней_наблюдения", "событие", "TSR", "возраст"]].dropna().copy()
    fit["стадия_IV"] = per_patient.стадия.fillna("").str.contains("IV").astype(int)
    print()
    CoxPHFitter().fit(fit, "дней_наблюдения", "событие").print_summary(decimals=3)
    return per_patient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default="outputs/results/pan/tcga_ov_deploy3/cohort.csv")
    ap.add_argument("--old", default="outputs/results/tcga_cohort.csv")
    ap.add_argument("--clinical", default="outputs/results/cohort_clinical.csv")
    ap.add_argument("--split", type=float, default=None,
                    help="порог TSR; по умолчанию медиана когорты")
    ap.add_argument("--tsr-col", default="TSR",
                    help="какое определение TSR брать: TSR, TSR_бед или TSR_поле "
                         "(колонки из tsr_defs.csv)")
    ap.add_argument("--out", default="outputs/results/pan/tsr_outcome.csv")
    args = ap.parse_args()

    new = pd.read_csv(args.new)
    if args.tsr_col != "TSR":
        if args.tsr_col not in new:
            raise SystemExit(f"в {args.new} нет колонки {args.tsr_col}")
        new = new.drop(columns="TSR").rename(columns={args.tsr_col: "TSR"})
        print(f"определение TSR: {args.tsr_col}")
    new = new.dropna(subset=["TSR"])
    old = pd.read_csv(args.old)[["slide", "TSR", "гормональная_в_строме_%"]]
    compare_cohorts(new, old)
    by_source(new)

    clinical = pd.read_csv(args.clinical).drop(columns=["TSR", "площадь_мм2", "опухоль_%"],
                                               errors="ignore")
    joined = new.merge(clinical, on="slide", how="inner")
    if joined.empty:
        print("\nклиническая таблица не сошлась с новой когортой по именам срезов")
        return
    per_patient = outcome(joined, args.split)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    per_patient.to_csv(args.out, index=False)
    print("\nтаблица по пациенткам:", args.out)


if __name__ == "__main__":
    main()
