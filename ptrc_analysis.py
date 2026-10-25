# Анализ PTRC-HGSOC строго по ptrc_plan.md, зафиксированному до получения данных.
# Всё, чего нет в плане, здесь не считается; разведку, если понадобится, писать отдельно
# и так и подписывать.
#
# На входе таблица срезов из ptrc_prepare.py и доли стромы по четырём определениям из
# tsr_defs.py, посчитанные по несглаженным картам (pan_resmooth.py).
#
# Порядок, как в плане:
#   отбор: без неоадъювантной химиотерапии; срез выбрасывается, если опухоль и строма
#          вместе меньше 50 % ткани или опухоли меньше 5 %
#   единица: пациентка; доля стромы опухоли это среднее по оставшимся срезам блока; из
#          двух опухолей берётся опухоль яичника, иначе образец с меньшим номером
#   основная проверка: логистическая регрессия «рефрактерная против чувствительной» по
#          доле стромы всей ткани на межквартильный размах с поправкой на центр и место
#          (яичник против прочих)
#   дополнительные: 1) три других определения, Бонферрони на 3; 2) верхние 41/192
#          пациенток против остальных, Фишер; 3) AUC с бутстрепом по пациенткам;
#          4) устойчивость: все опухоли с учётом пациентки, с неоадъювантными, с поправкой
#          на стадию и возраст
#   разведка (уточнение плана от 26.10, до подсчёта долей стромы): вся ткань отдельно
#          внутри яичника и внутри сальника, с поправкой на центр и на IQR всей основной
#          выборки; взаимодействие доли стромы с признаком «яичник»
#
# Логистическая регрессия считается здесь же (Ньютон-Рафсон), чтобы не зависеть от
# statsmodels; для анализа по опухолям ошибки кластеризуются по пациентке (сэндвич).
#
#   python -u ptrc_analysis.py

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, norm

ОПРЕДЕЛЕНИЯ = {"TSR": "вся ткань", "TSR_бед": "полоса 250 мкм",
               "TSR_поле_строма": "поле по строме", "TSR_поле_опухоль": "поле по опухоли"}
ДОЛЯ_LOU = 41 / 192
МЕСТА = {"OV": "яичник или труба", "OM": "сальник"}


def logit(X, y, cluster=None, iters=100):
    """Коэффициенты и стандартные ошибки без свободного члена."""
    X = np.column_stack([np.ones(len(y)), X]).astype(float)
    y = np.asarray(y, float)
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ b))
    Hinv = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    if cluster is None:
        cov = Hinv
    else:
        U = pd.DataFrame(X * (y - p)[:, None]).groupby(np.asarray(cluster)).sum().to_numpy()
        cov = Hinv @ (U.T @ U) @ Hinv
    return b[1:], np.sqrt(np.diag(cov))[1:]


def модель(t, кол, поправки, cluster=None, iqr=None):
    """Отношение шансов рефрактерности на межквартильный размах доли стромы."""
    d = t.dropna(subset=[кол]).copy()
    if iqr is None:
        iqr = d[кол].quantile(0.75) - d[кол].quantile(0.25)
    X = [d[кол] / iqr]
    for c in поправки:
        if c == "центр":
            X.append(pd.get_dummies(d.центр, prefix="центр", drop_first=True, dtype=float))
        elif d[c].nunique() > 1:
            X.append(d[c].astype(float))
    X = pd.concat(X, axis=1)
    b, se = logit(X.to_numpy(), d.рефрактерная, None if cluster is None else d[cluster])
    z = b[0] / se[0]
    return {"OR на IQR": np.exp(b[0]), "ниж": np.exp(b[0] - 1.96 * se[0]),
            "верх": np.exp(b[0] + 1.96 * se[0]), "p": 2 * norm.sf(abs(z)),
            "IQR": iqr, "n": len(d), "рефрактерных": int(d.рефрактерная.sum())}


def строка(имя, r, бонферрони=1):
    p = min(1.0, r["p"] * бонферрони)
    подпись = f", p с поправкой {p:.3f}" if бонферрони > 1 else ""
    print(f"  {имя}: OR {r['OR на IQR']:.2f} [{r['ниж']:.2f}; {r['верх']:.2f}], "
          f"p = {r['p']:.4f}{подпись} (IQR {r['IQR']:.3f}, n {r['n']}, "
          f"рефрактерных {r['рефрактерных']})")


def auc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg):
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def по_опухолям(slides, defs):
    """Среднее по срезам блока после правила качества; одна строка на опухоль."""
    t = slides.merge(defs, on="slide", how="inner")
    плохой = (t["опухоль_%"] + t["строма_%"] < 50) | (t["опухоль_%"] < 5)
    t["исключён"] = плохой
    print(f"\nсрезов с картой {len(t)}, исключено правилом качества {int(плохой.sum())}")
    print("  исключённые по ответу:", t[плохой].ответ.value_counts().to_dict(),
          "| оставшиеся:", t[~плохой].ответ.value_counts().to_dict())
    good = t[~плохой]
    агрегаты = {c: (c, "mean") for c in ОПРЕДЕЛЕНИЯ}
    агрегаты["срезов"] = ("slide", "size")
    опухоли = (good.groupby("образец").agg(**агрегаты).reset_index()
                   .merge(t.drop_duplicates("образец")[["образец", "пациент", "центр", "ответ",
                                                        "неоадъювант", "место", "стадия",
                                                        "возраст"]], on="образец"))
    print(f"опухолей с хотя бы одним годным срезом {len(опухоли)} из "
          f"{t.образец.nunique()}; по двум срезам {int((опухоли.срезов == 2).sum())}")
    опухоли["рефрактерная"] = (опухоли.ответ == "refractory").astype(int)
    опухоли["яичник"] = (опухоли.место == "OV").astype(int)
    опухоли["стадия_IV"] = опухоли.стадия.astype(str).str.strip().eq("IV").astype(int)
    опухоли["возраст"] = pd.to_numeric(опухоли.возраст.astype(str).str.replace("+", "",
                                                                              regex=False),
                                       errors="coerce")
    return опухоли


def по_пациенткам(опухоли):
    """Опухоль яичника, если она есть, иначе образец с меньшим номером."""
    o = опухоли.assign(номер=pd.to_numeric(опухоли.образец, errors="coerce"))
    o = o.sort_values(["пациент", "яичник", "номер"], ascending=[True, False, True])
    return o.drop_duplicates("пациент").drop(columns="номер")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", default="data/ptrc/slides.csv")
    ap.add_argument("--defs", default="outputs/results/pan/ptrc_deploy5_raw/tsr_defs.csv")
    ap.add_argument("--out", default="outputs/results/pan/ptrc_analysis")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    slides = pd.read_csv(args.slides, dtype={"образец": str, "пациент": str})
    defs = pd.read_csv(args.defs)
    print(f"срезов в таблице {len(slides)}, с долями стромы {len(defs)}")
    нет = sorted(set(slides.slide) - set(defs.slide))
    if нет:
        print(f"нет карты для {len(нет)} срезов, например {нет[:3]}")

    опухоли = по_опухолям(slides, defs)
    все = по_пациенткам(опухоли)
    основа = все[~все.неоадъювант]
    print(f"\nпациенток всего {len(все)}, без неоадъюванта {len(основа)}: "
          f"{основа.ответ.value_counts().to_dict()}")
    print("по центрам:", основа.groupby("центр").ответ.value_counts().unstack().to_dict("index"))
    print("\nмедиана доли стромы [IQR] по группам:")
    for кол, имя in ОПРЕДЕЛЕНИЯ.items():
        части = []
        for отв in ("refractory", "sensitive"):
            v = основа.loc[основа.ответ == отв, кол].dropna()
            части.append(f"{отв} {v.median():.3f} [{v.quantile(.25):.3f}; {v.quantile(.75):.3f}]")
        print(f"  {имя}: " + ", ".join(части))

    поправки = ["центр", "яичник"]
    print("\n== основная проверка: вся ткань, поправка на центр и место")
    print("  (OR больше 1: чем больше стромы, тем чаще рефрактерность, как у Lou et al.)")
    строка("вся ткань", модель(основа, "TSR", поправки))

    print("\n== дополнительная 1: другие определения, Бонферрони на 3")
    for кол in ("TSR_бед", "TSR_поле_строма", "TSR_поле_опухоль"):
        строка(ОПРЕДЕЛЕНИЯ[кол], модель(основа, кол, поправки), бонферрони=3)

    print(f"\n== дополнительная 2: схема Lou, верхние {ДОЛЯ_LOU:.1%} против остальных")
    d = основа.dropna(subset=["TSR"])
    порог = d.TSR.quantile(1 - ДОЛЯ_LOU)
    высокая = d.TSR >= порог
    a = int((высокая & (d.рефрактерная == 1)).sum())
    b = int((высокая & (d.рефрактерная == 0)).sum())
    c = int((~высокая & (d.рефрактерная == 1)).sum())
    e = int((~высокая & (d.рефрактерная == 0)).sum())
    odds, p = fisher_exact([[a, b], [c, e]])
    print(f"  порог доли стромы {порог:.3f}; рефрактерных {a} из {a + b} при высокой и "
          f"{c} из {c + e} при низкой; OR {odds:.2f}, Фишер p = {p:.4f}")

    print("\n== дополнительная 3: AUC доли стромы (выше у рефрактерных, если больше 0,5)")
    rng = np.random.default_rng(args.seed)
    for кол, имя in ОПРЕДЕЛЕНИЯ.items():
        d = основа.dropna(subset=[кол])
        s, y = d[кол].to_numpy(), d.рефрактерная.to_numpy()
        boot = []
        for _ in range(args.boot):
            i = rng.integers(0, len(d), len(d))
            boot.append(auc(s[i], y[i]))
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        print(f"  {имя}: AUC {auc(s, y):.3f} [{lo:.3f}; {hi:.3f}]")

    print("\n== дополнительная 4: устойчивость основной проверки")
    без_неоадъюванта = опухоли[~опухоли.неоадъювант]
    строка("все опухоли, ошибки по пациентке",
           модель(без_неоадъюванта, "TSR", поправки, cluster="пациент"))
    строка("с неоадъювантными", модель(все, "TSR", поправки))
    строка("с неоадъювантными и поправкой на неоадъювант",
           модель(все.assign(неоадъювант_=все.неоадъювант.astype(int)), "TSR",
                  поправки + ["неоадъювант_"]))
    строка("поправка ещё на стадию IV и возраст",
           модель(основа.dropna(subset=["возраст"]), "TSR", поправки + ["стадия_IV", "возраст"]))

    print("\n== разведка (уточнение плана от 26.10): внутри одного места взятия, вся ткань")
    print("  поправки на число проверок нет, p приводится как есть")
    d = основа.dropna(subset=["TSR"])
    медианы = d.groupby("место").TSR.agg(["median", "size"])
    print("  медиана доли стромы по месту:",
          ", ".join(f"{м} {r['median']:.3f} (n {int(r['size'])})" for м, r in медианы.iterrows()))
    iqr_всех = d.TSR.quantile(0.75) - d.TSR.quantile(0.25)
    for место, имя in МЕСТА.items():
        строка(имя, модель(d[d.место == место], "TSR", ["центр"], iqr=iqr_всех))
    x = (d.TSR / iqr_всех).rename("TSR")
    X = pd.concat([x, d.яичник.astype(float), (x * d.яичник).rename("TSR_x_яичник"),
                   pd.get_dummies(d.центр, prefix="центр", drop_first=True, dtype=float)], axis=1)
    b, se = logit(X.to_numpy(), d.рефрактерная)
    print(f"  взаимодействие доли стромы с местом (яичник против прочих): отношение OR "
          f"{np.exp(b[2]):.2f} [{np.exp(b[2] - 1.96 * se[2]):.2f}; "
          f"{np.exp(b[2] + 1.96 * se[2]):.2f}], p = {2 * norm.sf(abs(b[2] / se[2])):.4f}")

    основа.to_csv(out / "ptrc_patients.csv", index=False)
    опухоли.to_csv(out / "ptrc_tumours.csv", index=False)
    print("\nтаблицы:", out / "ptrc_patients.csv", out / "ptrc_tumours.csv")


if __name__ == "__main__":
    main()
