# Рисунки для доклада на семинаре лаборатории: то, чего нет среди рисунков диплома.
# Размер каждого рисунка равен месту на слайде, подписи 14 пт и крупнее, чтобы читались
# с проектора.
#
#   slide_cox       модель Кокса: доля стромы, стадия, возраст (слайд 21)
#   slide_subtypes  доля стромы и индекс активации стромы по подтипам TCGA (слайд 24)
#   slide_lou       схема Lou et al. на наших срезах рядом с литературой (слайд 25)
#   slide_layer2    различимость подтипов стромы внутри среза и на чужом срезе (слайд 27)
#   slide_emul20    доля стромы при имитации скана 20× против исходной разметки (слайд 30)
#   slide_km        кривые выживаемости крупнее, чем на рисунке 19 (слайд 20)
#
# Пересчитываются только модели Кокса по готовым таблицам outcome_*.csv, остальное читается
# из таблиц, которые уже написали tsr_subtype.py, tsr_activation.py и pan_collect_tsr.py.
# Числа второго слоя взяты из вывода stroma_probe.py (таблица 12 диплома), признаки заново
# не считаются. В ту же папку копируются пример выхода hematox-tsr и карты пробного прогона
# PTRC, чтобы всё для деки скачать одной папкой.
#
#   python pan_figs5.py

import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from pan_figs import ЦВЕТА, СЕРЫЙ, запятая, оформить, сохранить
from pan_figs3 import ВЫСОКАЯ, МЕСЯЦ, НИЗКАЯ, ТЕКСТ, кривая
from pan_figs4 import ОПРЕДЕЛЕНИЯ
from tsr_outcome import logrank

ШРИФТ = 14
# место под рисунок справа от текстовой колонки слайда и вся ширина слайда, дюймы
ПРАВАЯ = (7.45, 4.9)
ШИРОКАЯ = (12.1, 4.0)
ОРАНЖЕВЫЙ = ЦВЕТА[5]
ДОЛЯ_LOU = 18 / 103
ПОДТИПЫ = {"Mesenchymal": "мезенхимальный", "Immunoreactive": "иммунореактивный",
           "Differentiated": "дифференцированный", "Proliferative": "пролиферативный"}

# AUC логистической регрессии на признаках UNI, гормональная строма против матриксной:
# внутри среза, перенос на чужой срез как есть, с центрированием признаков по срезу,
# с центрированием и без трёх главных компонент (stroma_probe.py, 28.09)
ВТОРОЙ_СЛОЙ = {"ovary_prime": (0.714, 0.547, 0.576, 0.488),
               "ovary2": (0.933, 0.680, 0.675, 0.640),
               "ovary3": (0.589, 0.524, 0.543, 0.535)}
УСЛОВИЯ = [("внутри своего среза", "o", ЦВЕТА[2]), ("на чужом срезе как есть", "o", СЕРЫЙ),
           ("с центрированием по срезу", "s", "#9a9a9a"),
           ("центрирование и без 3 главных компонент", "^", "#c4c4c4")]


def крупно(ax, подпись_x="", подпись_y=None):
    оформить(ax, подпись_y)
    ax.tick_params(labelsize=ШРИФТ)
    ax.set_xlabel(подпись_x, fontsize=ШРИФТ, color=ТЕКСТ)
    if подпись_y:
        ax.set_ylabel(подпись_y, fontsize=ШРИФТ, color=ТЕКСТ)


def число(v, знаков=2):
    return f"{v:.{знаков}f}".replace(".", ",").replace("-", "−")


def исходы(каталог, кол, ep):
    """Таблица исхода по одному определению доли стромы, стадия IV против I–III."""
    t = pd.read_csv(каталог / f"outcome_все_{кол}_{ep}.csv")
    if "возраст" in t:
        t["возраст"] = pd.to_numeric(t.возраст, errors="coerce")
    if "стадия" in t:
        # как в tsr_outcome.py: неизвестная стадия идёт в I–III, пациентка остаётся в модели
        t["стадия_IV"] = t.стадия.fillna("").astype(str).str.contains("IV").astype(int)
    return t.dropna(subset=["TSR", "дней_наблюдения", "событие"])


def кокс(t, столбцы):
    from lifelines import CoxPHFitter
    d = t[["дней_наблюдения", "событие"] + столбцы].dropna()
    s = CoxPHFitter().fit(d, "дней_наблюдения", "событие").summary
    s = s.rename(columns={"exp(coef)": "hr", "exp(coef) lower 95%": "lo",
                          "exp(coef) upper 95%": "hi"})
    return s[["hr", "lo", "hi", "p"]], len(d), int(d.событие.sum())


def лес(ax, строки, подпись_x, пределы):
    """Отношения рисков с интервалами; строки: подпись, hr, низ, верх, цвет, жирно."""
    y = np.arange(len(строки))[::-1]
    for yi, (_, hr, lo, hi, цвет, жирно) in zip(y, строки):
        ax.plot([lo, hi], [yi, yi], color=цвет, linewidth=2.6, solid_capstyle="butt")
        ax.plot(hr, yi, "s", color=цвет, markersize=10)
        ax.text(1.03, yi, f"{число(hr)} [{число(lo)}; {число(hi)}]",
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=ШРИФТ,
                color=ТЕКСТ, fontweight="bold" if жирно else "normal")
    ax.axvline(1, color=СЕРЫЙ, linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlim(*пределы)
    тики = [v for v in (0.25, 0.5, 1, 2, 4) if пределы[0] <= v <= пределы[1]]
    ax.set_xticks(тики, [запятая(v) for v in тики])
    ax.minorticks_off()
    ax.set_yticks(y, [s[0] for s in строки])
    ax.set_ylim(-0.6, len(строки) - 0.4)
    крупно(ax, подпись_x)
    ax.grid(axis="y", linewidth=0)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.6)
    ax.tick_params(axis="y", length=0)


def рисунок_кокс(каталог, out):
    строки = []
    for ep, подпись in (("общая", "общая выживаемость"), ("безрецидивная", "без прогрессирования")):
        t = исходы(каталог, "TSR", ep)
        t["TSR_IQR"] = t.TSR / (t.TSR.quantile(0.75) - t.TSR.quantile(0.25))
        s, n, событий = кокс(t, ["TSR_IQR", "возраст", "стадия_IV"])
        print(f"Кокс, {подпись}: пациенток {n}, событий {событий}")
        print(s.round(4).to_string())
        r = s.loc["TSR_IQR"]
        строки.append((f"доля стромы\n{подпись}", r.hr, r.lo, r.hi, "#9a9a9a", False))
        if ep == "общая":
            общая = s
    for имя, кол in (("стадия IV против I–III", "стадия_IV"), ("возраст, на год", "возраст")):
        r = общая.loc[кол]
        строки.append((f"{имя}\nобщая выживаемость", r.hr, r.lo, r.hi, ТЕКСТ, кол == "стадия_IV"))
    print("в дипломе: TSR 0,85 [0,59; 1,23] и 0,87 [0,62; 1,21], стадия 2,87 [1,72; 4,78], "
          "возраст 1,04 [1,017; 1,063]")

    fig, ax = plt.subplots(figsize=(ПРАВАЯ[0] - 2.2, ПРАВАЯ[1]))
    лес(ax, строки, "отношение рисков (лог. шкала)", (0.4, 6))
    ax.set_title("поправка на возраст и стадию;\nдоля стромы на межквартильный размах",
                 fontsize=12, color=СЕРЫЙ, loc="left")
    сохранить(fig, out, "slide_cox")


def рисунок_подтипы(каталог, out):
    t = pd.read_csv(каталог / "tsr_subtype.csv")[["пациент", "подтип", "TSR_бед"]]
    акт = pd.read_csv(каталог / "tsr_activation.csv")[["пациент", "активация"]]
    t = t.merge(акт, on="пациент", how="left")
    t = t[t.подтип.isin(list(ПОДТИПЫ))]
    порядок = list(ПОДТИПЫ)
    y = np.arange(len(порядок))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=ПРАВАЯ, sharey=True)
    for k, (ax, кол, подпись) in enumerate(zip(axes, ("TSR_бед", "активация"),
                                             ("доля стромы,\nполоса 250 мкм",
                                              "индекс активации\nстромы по РНК"))):
        группы = [t.loc[t.подтип == п, кол].dropna().to_numpy() for п in порядок]
        разброс = np.random.default_rng(0)
        линии = dict(color=СЕРЫЙ, linewidth=1)
        for yi, п, g in zip(y, порядок, группы):
            ax.boxplot(g, positions=[yi], vert=False, widths=0.55, showfliers=False,
                       manage_ticks=False, medianprops=dict(color=ТЕКСТ, linewidth=2),
                       boxprops=линии, whiskerprops=линии, capprops=линии)
            ax.scatter(g, yi + разброс.uniform(-0.18, 0.18, len(g)), s=20, linewidths=0,
                       color=ОРАНЖЕВЫЙ if п == "Mesenchymal" else ЦВЕТА[0], alpha=0.85, zorder=3)
        мез = t.loc[t.подтип == "Mesenchymal", кол].dropna()
        прочие = t.loc[t.подтип != "Mesenchymal", кол].dropna()
        p = mannwhitneyu(мез, прочие)[1]
        print(f"{подпись.replace(chr(10), ' ')}: медианы "
              + ", ".join(f"{ПОДТИПЫ[п]} {np.median(g):.3f} (n={len(g)})"
                          for п, g in zip(порядок, группы))
              + f"; мезенхимальный против прочих p = {p:.4f}")
        if кол == "активация":
            ax.axvline(0, color=СЕРЫЙ, linewidth=1)
        else:
            ax.set_yticks(y, [f"{ПОДТИПЫ[п]}\n(n = {len(g)})" for п, g in zip(порядок, группы)])
        крупно(ax, подпись)
        ax.grid(axis="y", linewidth=0)
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.6)
        ax.tick_params(axis="y", length=0)
        ax.set_title("p < 0,001" if p < 0.001 else f"p = {число(p, 3)}", fontsize=13,
                     color=СЕРЫЙ, loc="right")
    fig.tight_layout(w_pad=1.5)
    сохранить(fig, out, "slide_subtypes")


def рисунок_lou(каталог, out):
    # оранжевый уже занят полем по опухоли, литература чёрным и серым
    строки = [("Lou et al., TCGA\n103 пациентки", 1.91, 0.96, 3.78, ТЕКСТ, True)]
    for кол, (имя, цвет) in ОПРЕДЕЛЕНИЯ.items():
        t = исходы(каталог, кол, "общая")
        t["высокая"] = (t.TSR >= t.TSR.quantile(1 - ДОЛЯ_LOU)).astype(int)
        s, n, _ = кокс(t, ["высокая"])
        r = s.loc["высокая"]
        print(f"схема Lou, {имя}: высоких {t.высокая.sum()} из {n}, HR {r.hr:.2f} "
              f"[{r.lo:.2f}; {r.hi:.2f}]")
        строки.append((f"наши срезы, {имя}", r.hr, r.lo, r.hi, цвет, False))
    строки.append(("STAR, с поправками\nAggarwal et al., 2026", 1.5, 0.82, 2.75, "#9a9a9a", False))

    fig, ax = plt.subplots(figsize=(ПРАВАЯ[0] - 2.2, ПРАВАЯ[1]))
    лес(ax, строки, "отношение рисков смерти (лог. шкала)", (0.25, 5))
    ax.set_title("высокая доля стромы против низкой,\nобщая выживаемость", fontsize=12,
                 color=СЕРЫЙ, loc="left")
    сохранить(fig, out, "slide_lou")


def рисунок_второй_слой(out):
    срезы = list(ВТОРОЙ_СЛОЙ)
    y = np.arange(len(срезы))[::-1]
    fig, ax = plt.subplots(figsize=ПРАВАЯ)
    for yi, срез in zip(y, срезы):
        v = ВТОРОЙ_СЛОЙ[срез]
        ax.plot([min(v), max(v)], [yi, yi], color="#e0e0e0", linewidth=3, zorder=1)
        for j, (_, маркер, цвет) in enumerate(УСЛОВИЯ):
            ax.plot(v[j], yi, маркер, color=цвет, markersize=13 if j == 0 else 11, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.8)
        ax.text(v[0], yi + 0.22, число(v[0], 3), ha="center", va="bottom", fontsize=ШРИФТ,
                color=ТЕКСТ, fontweight="bold")
    ax.axvline(0.5, color=СЕРЫЙ, linewidth=1, linestyle="--")
    ax.text(0.505, len(срезы) - 0.45, "угадывание", fontsize=12, color=СЕРЫЙ, va="top")
    for имя, маркер, цвет in УСЛОВИЯ:
        ax.plot([], [], маркер, color=цвет, markersize=10, label=имя)
    ax.legend(frameon=False, fontsize=12, loc="lower right", labelcolor=ТЕКСТ)
    ax.set_xlim(0.45, 1.0)
    ax.set_yticks(y, срезы)
    ax.set_ylim(-1.6, len(срезы) - 0.3)
    крупно(ax, "AUC: гормональная строма против матриксной")
    ax.grid(axis="y", linewidth=0)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.6)
    ax.tick_params(axis="y", length=0)
    сохранить(fig, out, "slide_layer2")


def рисунок_20x(сорок, двадцать, out):
    a = pd.read_csv(сорок / "cohort.csv")
    b = pd.read_csv(двадцать / "cohort.csv")
    t = a.merge(b, on="slide", suffixes=("_40", "_20"))
    сдвиг = t.TSR_20 - t.TSR_40
    ро = spearmanr(t.TSR_40, t.TSR_20)[0]
    print(f"20×: срезов {len(t)}, ро {ро:.3f}, медиана сдвига {сдвиг.median():+.3f}, "
          f"наибольший {сдвиг.abs().max():.3f}")

    fig, ax = plt.subplots(figsize=(5.4, ПРАВАЯ[1]))
    ax.plot([0, 1], [0, 1], color=СЕРЫЙ, linewidth=1, linestyle="--")
    ax.scatter(t.TSR_40, t.TSR_20, s=46, color=ЦВЕТА[0], edgecolors="white", linewidths=0.6,
               zorder=3)
    ax.text(0.04, 0.96, f"ρ = {число(ро, 3)}, n = {len(t)}\nсдвиг медианы "
            f"{'+' if сдвиг.median() >= 0 else ''}{число(сдвиг.median(), 3)}\n"
            f"наибольший {число(сдвиг.abs().max(), 3)}", transform=ax.transAxes, ha="left",
            va="top", fontsize=ШРИФТ, color=ТЕКСТ)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    крупно(ax, "доля стромы, исходный скан 40×", "доля стромы, имитация 20×")
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.6)
    сохранить(fig, out, "slide_emul20")


def рисунок_км(каталог, out):
    fig, axes = plt.subplots(1, 2, figsize=ШИРОКАЯ)
    for ax, ep, заголовок in zip(axes, ("общая", "безрецидивная"),
                                 ("Общая выживаемость", "Выживаемость без прогрессирования")):
        t = pd.read_csv(каталог / f"outcome_все_TSR_{ep}.csv")
        дни = t.дней_наблюдения.to_numpy(float) / МЕСЯЦ
        событие = t.событие.to_numpy(int)
        высокая = (t.TSR > t.TSR.median()).to_numpy()
        _, p, _, _ = logrank(дни, событие, высокая)
        for маска, цвет, где in ((высокая, ВЫСОКАЯ, "выше"), (~высокая, НИЗКАЯ, "ниже")):
            x, s = кривая(дни[маска], событие[маска])
            ax.step(x, s, where="post", color=цвет, linewidth=2.4,
                    label=f"строма {где} медианы (n = {маска.sum()})")
            выбыли = дни[маска & (событие == 0)]
            ax.plot(выбыли, s[np.searchsorted(x, выбыли, side="right") - 1], "|", color=цвет,
                    markersize=8, markeredgewidth=1.4)
        ax.set_ylim(0, 1.03)
        ax.set_xlim(0, None)
        крупно(ax, "месяцы", "доля без события")
        ax.grid(axis="x", linewidth=0)
        ax.text(0.97, 0.97, f"лог-ранг p = {число(p)}", transform=ax.transAxes, ha="right",
                va="top", fontsize=ШРИФТ, color=ТЕКСТ)
        ax.legend(frameon=False, fontsize=13, loc="upper right", bbox_to_anchor=(1, 0.88),
                  labelcolor=ТЕКСТ)
        ax.set_title(заголовок, fontsize=15, color=ТЕКСТ, loc="left")
    fig.tight_layout(w_pad=3)
    сохранить(fig, out, "slide_km")


def собрать(out, hematox_tsr, пилот):
    пример = Path(hematox_tsr).expanduser() / "docs" / "example_overlay.jpg"
    if пример.exists():
        shutil.copy(пример, out / "hematox_tsr_example.jpg")
        print("скопирован:", пример)
    else:
        print("нет файла:", пример)
    карты = sorted(Path(пилот).glob("*_seg_map.png"))
    for p in карты:
        shutil.copy(p, out / f"ptrc_pilot_{p.name}")
    print(f"карт пробного прогона PTRC скопировано: {len(карты)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/results/pan/tcga_ov_deploy5_raw")
    ap.add_argument("--emul", default="outputs/results/pan/tcga_ov_emul20_raw")
    ap.add_argument("--pilot", default="outputs/results/pan/ptrc_pilot")
    ap.add_argument("--hematox-tsr", default="~/hematox-tsr")
    ap.add_argument("--out", default="outputs/results/pan/figs_slides")
    args = ap.parse_args()
    каталог, out = Path(args.dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white",
                         "axes.facecolor": "white"})
    # каждый рисунок отдельно: если для одного нет таблицы, остальные всё равно сохранятся
    шаги = [("Кокс", lambda: рисунок_кокс(каталог, out)),
            ("подтипы", lambda: рисунок_подтипы(каталог, out)),
            ("схема Lou", lambda: рисунок_lou(каталог, out)),
            ("второй слой", lambda: рисунок_второй_слой(out)),
            ("20×", lambda: рисунок_20x(каталог, Path(args.emul), out)),
            ("кривые выживаемости", lambda: рисунок_км(каталог, out)),
            ("пример и карты PTRC", lambda: собрать(out, args.hematox_tsr, args.pilot))]
    for имя, шаг in шаги:
        print(f"\n== {имя}")
        try:
            шаг()
        except Exception as e:
            print(f"ОШИБКА в шаге «{имя}»: {type(e).__name__}: {e}")
    print(f"\nвсё в {out}")


if __name__ == "__main__":
    main()
