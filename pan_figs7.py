# Те же графики, что в работах о доле стромы при раке яичника, на наших данных.
#
# Новых проверок здесь нет. Всё нарисованное уже посчитано по заранее названному плану
# (tsr_outcome.py, ptrc_analysis.py), рисунки только показывают это так, как результаты
# подают Lou et al. и другие: кривые выживаемости, отношения рисков, доля рефрактерных.
# Исключение одно, перебор порога (fig_cutoff_scan). Он показывает, как при свободном
# выборе порога находится «значимая» граница, и рядом с наивным p стоит p с поправкой на
# перебор по перестановкам. Это иллюстрация, а не проверка.
#
#   fig_km_all       Каплан–Мейер: 4 определения × общая и без прогрессирования, по медиане
#   fig_forest_os    отношения рисков смерти: по медиане, верхние 17,5 % (схема Lou),
#   fig_forest_pfs   на IQR без поправок и с поправкой на возраст и стадию; то же для
#                    прогрессирования
#   fig_cutoff_scan  p лог-ранга при каждом пороге от 15-го до 85-го процентиля, вся ткань
#   fig_ptrc         PTRC: доля стромы по ответу, ROC, доля рефрактерных по схеме Lou
#
#   python pan_figs7.py

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from pan_figs import ЦВЕТА, СЕРЫЙ, оформить, сохранить
from pan_figs3 import ТЕКСТ, выживаемость
from pan_figs4 import ОПРЕДЕЛЕНИЯ
from pan_figs5 import ДОЛЯ_LOU, исходы, кокс, лес, число
from ptrc_analysis import ДОЛЯ_LOU as ДОЛЯ_LOU_PTRC, auc, модель
from tsr_outcome import logrank

КОНЕЧНЫЕ = {"общая": "общая выживаемость", "безрецидивная": "без прогрессирования"}
РЕФРАКТЕРНЫЕ, ЧУВСТВИТЕЛЬНЫЕ = ЦВЕТА[5], ЦВЕТА[4]


def км_все(каталог, out):
    fig, axes = plt.subplots(4, 2, figsize=(10, 13))
    for строка, (кол, (имя, _)) in zip(axes, ОПРЕДЕЛЕНИЯ.items()):
        for ax, (ep, конечная) in zip(строка, КОНЕЧНЫЕ.items()):
            t = исходы(каталог, кол, ep)
            выживаемость(ax, t, f"{имя}: {конечная}")
            t["высокая"] = (t.TSR > t.TSR.median()).astype(int)
            s, n, событий = кокс(t, ["высокая"])
            r = s.loc["высокая"]
            ax.text(0.03, 0.05, f"HR {число(r.hr)} [{число(r.lo)}; {число(r.hi)}]",
                    transform=ax.transAxes, fontsize=9.5, color=ТЕКСТ)
            print(f"  {имя}, {конечная}: n {n}, событий {событий}, HR выше медианы "
                  f"{r.hr:.2f} [{r.lo:.2f}; {r.hi:.2f}], p {r.p:.3f}")
    fig.tight_layout(h_pad=2.5, w_pad=2)
    сохранить(fig, out, "fig_km_all")


def лес_tcga(каталог, out):
    for ep, конечная in КОНЕЧНЫЕ.items():
        строки = []
        if ep == "общая":
            строки.append(("Lou et al., TCGA: верхние 17,5 %", 1.91, 0.96, 3.78, ТЕКСТ, True))
        for кол, (имя, цвет) in ОПРЕДЕЛЕНИЯ.items():
            t = исходы(каталог, кол, ep)
            iqr = t.TSR.quantile(0.75) - t.TSR.quantile(0.25)
            t = t.assign(медиана=(t.TSR > t.TSR.median()).astype(int),
                         lou=(t.TSR >= t.TSR.quantile(1 - ДОЛЯ_LOU)).astype(int),
                         TSR_IQR=t.TSR / iqr)
            for подпись, столбцы in (("выше медианы", ["медиана"]),
                                     ("верхние 17,5 %", ["lou"]),
                                     ("на IQR", ["TSR_IQR"]),
                                     ("на IQR, возраст и стадия", ["TSR_IQR", "возраст",
                                                                   "стадия_IV"])):
                s, n, _ = кокс(t, столбцы)
                r = s.loc[столбцы[0]]
                строки.append((f"{имя}: {подпись}", r.hr, r.lo, r.hi, цвет, False))
                print(f"  {конечная}, {имя}, {подпись}: HR {r.hr:.2f} [{r.lo:.2f}; "
                      f"{r.hi:.2f}], p {r.p:.3f}, n {n}")
        s, _, _ = кокс(исходы(каталог, "TSR", ep), ["стадия_IV", "возраст"])
        r = s.loc["стадия_IV"]
        строки.append(("для сравнения: стадия IV", r.hr, r.lo, r.hi, ТЕКСТ, True))

        fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(строки) + 1.2))
        лес(ax, строки, "отношение рисков (лог. шкала)", (0.2, 6))
        ax.tick_params(axis="y", labelsize=11)
        ax.set_title(конечная, fontsize=14, color=ТЕКСТ, loc="left")
        сохранить(fig, out, "fig_forest_os" if ep == "общая" else "fig_forest_pfs")


def перебор_порога(каталог, out, перестановок):
    доли = np.arange(0.15, 0.851, 0.02)
    rng = np.random.default_rng(20261027)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, (ep, конечная) in zip(axes, КОНЕЧНЫЕ.items()):
        t = исходы(каталог, "TSR", ep)
        дни, событие = t.дней_наблюдения.to_numpy(float), t.событие.to_numpy(int)
        x = t.TSR.to_numpy(float)

        def все_p(v):
            return np.array([logrank(дни, событие, v > c)[1] for c in np.quantile(v, доли)])

        p = все_p(x)
        i = int(np.nanargmin(p))
        наименьшие = np.array([np.nanmin(все_p(rng.permutation(x))) for _ in range(перестановок)])
        p_попр = (1 + (наименьшие <= p[i]).sum()) / (1 + перестановок)
        print(f"  {конечная}: наименьшее p {p[i]:.4f} при пороге на {100 * доли[i]:.0f}-м "
              f"процентиле, с поправкой на перебор {p_попр:.3f} ({перестановок} перестановок)")

        ax.plot(100 * доли, -np.log10(p), color=ЦВЕТА[0], linewidth=2)
        ax.plot(100 * доли[i], -np.log10(p[i]), "o", color=ЦВЕТА[1], markersize=8)
        ax.axhline(-np.log10(0.05), color=СЕРЫЙ, linewidth=1, linestyle="--")
        ax.text(16, -np.log10(0.05) * 1.03, "p = 0,05", fontsize=9.5, color=СЕРЫЙ, va="bottom")
        ax.text(0.98, 0.96, f"наименьшее p {число(p[i], 3)}\nс поправкой на перебор "
                f"{число(p_попр, 2)}", transform=ax.transAxes, ha="right", va="top",
                fontsize=10, color=ТЕКСТ)
        оформить(ax, "−log₁₀ p лог-ранга")
        ax.set_xlabel("порог: процентиль доли стромы, вся ткань", fontsize=10, color=ТЕКСТ)
        ax.set_ylim(0, max(2.0, -np.log10(np.nanmin(p)) * 1.25))
        ax.set_title(конечная, fontsize=11, color=ТЕКСТ, loc="left")
    fig.tight_layout(w_pad=3)
    сохранить(fig, out, "fig_cutoff_scan")


def roc(score, y):
    порядок = np.argsort(-score)
    y = y[порядок]
    return (np.r_[0, np.cumsum(1 - y) / (1 - y).sum()], np.r_[0, np.cumsum(y) / y.sum()])


def птрц(путь, out):
    d = pd.read_csv(путь)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.7, 1, 0.8]})

    ax = axes[0]
    разброс = np.random.default_rng(0)
    линии = dict(color=СЕРЫЙ, linewidth=1)
    for k, (кол, (имя, _)) in enumerate(ОПРЕДЕЛЕНИЯ.items()):
        r = модель(d, кол, ["центр", "яичник"])
        p = r["p"] if кол == "TSR" else min(1.0, 3 * r["p"])
        for j, (флаг, цвет) in enumerate(((1, РЕФРАКТЕРНЫЕ), (0, ЧУВСТВИТЕЛЬНЫЕ))):
            v = d.loc[d.рефрактерная == флаг, кол].dropna().to_numpy()
            x = 3 * k + j
            ax.boxplot(v, positions=[x], widths=0.6, showfliers=False, manage_ticks=False,
                       medianprops=dict(color=ТЕКСТ, linewidth=2), boxprops=линии,
                       whiskerprops=линии, capprops=линии)
            ax.scatter(x + разброс.uniform(-0.2, 0.2, len(v)), v, s=12, color=цвет,
                       alpha=0.8, linewidths=0, zorder=3)
        ax.text(3 * k + 0.5, 1.02, f"OR {число(r['OR на IQR'])}\np {число(p, 3)}"
                + ("" if кол == "TSR" else "*"), ha="center", va="bottom", fontsize=9.5,
                color=ТЕКСТ)
    ax.set_xticks([3 * k + 0.5 for k in range(len(ОПРЕДЕЛЕНИЯ))],
                  [имя for имя, _ in ОПРЕДЕЛЕНИЯ.values()])
    ax.set_ylim(0, 1.18)
    оформить(ax, "доля стромы")
    ax.tick_params(axis="x", labelsize=10)
    ax.scatter([], [], color=РЕФРАКТЕРНЫЕ, label=f"рефрактерные (n = {int(d.рефрактерная.sum())})")
    ax.scatter([], [], color=ЧУВСТВИТЕЛЬНЫЕ,
               label=f"чувствительные (n = {int((d.рефрактерная == 0).sum())})")
    # под осью: у поля по опухоли точки лежат у самого нуля, внутри легенда их закроет
    ax.legend(frameon=False, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.1),
              ncol=2, labelcolor=ТЕКСТ)
    ax.set_title("а) доля стромы по ответу; OR на IQR с поправкой на центр и место\n"
                 "* p с поправкой Бонферрони на 3", fontsize=10.5, color=ТЕКСТ, loc="left")

    ax = axes[1]
    for кол, (имя, цвет) in ОПРЕДЕЛЕНИЯ.items():
        m = d[кол].notna()
        s, y = d.loc[m, кол].to_numpy(float), d.loc[m, "рефрактерная"].to_numpy(int)
        fpr, tpr = roc(s, y)
        ax.plot(fpr, tpr, color=цвет, linewidth=2, label=f"{имя}, AUC {число(auc(s, y))}")
    ax.plot([0, 1], [0, 1], color=СЕРЫЙ, linewidth=1, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    оформить(ax, "чувствительность")
    ax.set_xlabel("1 − специфичность", fontsize=10, color=ТЕКСТ)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=ТЕКСТ)
    ax.set_title("б) ROC: рефрактерность по доле стромы", fontsize=10.5, color=ТЕКСТ, loc="left")

    ax = axes[2]
    v = d.dropna(subset=["TSR"])
    высокая = v.TSR >= v.TSR.quantile(1 - ДОЛЯ_LOU_PTRC)
    a, b = int(v.рефрактерная[высокая].sum()), int(высокая.sum())
    c, e = int(v.рефрактерная[~высокая].sum()), int((~высокая).sum())
    _, p = fisher_exact([[a, b - a], [c, e - c]])
    for x, (k, n, цвет) in enumerate(((a, b, ЦВЕТА[0]), (c, e, ЦВЕТА[1]))):
        ax.bar(x, 100 * k / n, width=0.6, color=цвет)
        ax.text(x, 100 * k / n + 1.5, f"{k} из {n}\n{100 * k / n:.0f} %".replace(".", ","),
                ha="center", va="bottom", fontsize=10, color=ТЕКСТ)
    ax.set_xticks([0, 1], [f"верхние\n{100 * ДОЛЯ_LOU_PTRC:.0f} %", "остальные"])
    ax.set_ylim(0, 75)
    оформить(ax, "рефрактерных, %")
    ax.set_title(f"в) схема Lou, вся ткань\nФишер p = {число(p, 2)}", fontsize=10.5,
                 color=ТЕКСТ, loc="left")
    print(f"  PTRC, схема Lou: {a} из {b} против {c} из {e}, Фишер p {p:.3f}")

    fig.tight_layout(w_pad=2.5)
    сохранить(fig, out, "fig_ptrc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/results/pan/tcga_ov_deploy5_raw")
    ap.add_argument("--ptrc", default="outputs/results/pan/ptrc_analysis/ptrc_patients.csv")
    ap.add_argument("--out", default="outputs/results/pan/figs_literature")
    ap.add_argument("--perm", type=int, default=1000, help="перестановок для перебора порога")
    args = ap.parse_args()
    каталог, out = Path(args.dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white",
                         "axes.facecolor": "white"})
    шаги = [("кривые выживаемости", lambda: км_все(каталог, out)),
            ("отношения рисков", lambda: лес_tcga(каталог, out)),
            ("перебор порога", lambda: перебор_порога(каталог, out, args.perm)),
            ("PTRC", lambda: птрц(args.ptrc, out))]
    for имя, шаг in шаги:
        print(f"\n== {имя}")
        try:
            шаг()
        except Exception as e:
            print(f"ОШИБКА в шаге «{имя}»: {type(e).__name__}: {e}")
    print(f"\nвсё в {out}")


if __name__ == "__main__":
    main()
