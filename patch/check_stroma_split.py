"""Проверка: делятся ли стромальные клетки Xenium на гормональную и матриксную.

Это первый шаг перед любым обучением на 5 классов. Если по маркерным генам
два типа стромы не разделяются, то и учить их по H&E нечему — метки будут
шумом, а модель выучит случайность.

ДВА РЕЖИМА.

1. По подтипам (основной, начинайте с него). Читает таблицы средней
   экспрессии по подтипам сразу со всех срезов и предлагает готовое
   разнесение подтипов по группам — то, что потом идёт в
   config/cell_type_map.yaml. Решение принимается по цифрам, а не по
   названию подтипа: в трёх ваших срезах словари названий разные, и
   гормональные популяции в них называются по-разному.

       python check_stroma_split.py --subtype-expr data/seurat_csv/*_subtype_expr.csv

   Выход:
       stroma_split_subtypes.csv   балл каждого подтипа и предложенная группа
       stroma_split_subtypes.png   подтипы в осях «стероидогенез / матрикс»
       stroma_split_suggested.yaml готовый блок groups: скопировать в
                                   config/cell_type_map.yaml после проверки

2. По отдельным клеткам (как было). Отвечает на вопрос, разделяются ли
   стромальные клетки внутри среза, и рисует, где какая строма лежит.

       python check_stroma_split.py --cells data/seurat_csv/ovary2_he_cells.csv \
           --expr data/seurat_csv/ovary2_he_expr.csv --out outputs/results/stroma_split

ВАЖНО: скрипт ничего не решает за вас. Он печатает баллы и предложение,
а вы смотрите картинку и правите yaml руками, если предложение спорное.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Маркеры стероидогенеза: ферменты синтеза стероидов + факторы
# гранулёзо/теко-клеточной линии.
STEROID = ["STAR", "CYP11A1", "CYP17A1", "CYP19A1", "HSD3B2", "HSD17B1",
           "FOXL2", "NR5A1", "INHA", "INHBA", "AMH", "GATA4"]

# Маркеры матриксной стромы: коллагены, активированные фибробласты,
# ремоделирование ECM.
MATRIX = ["COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL6A3", "FAP", "POSTN",
          "THY1", "LUM", "DCN", "FN1", "SPARC", "TAGLN", "ACTA2", "PDGFRB",
          "MMP2", "MMP11", "TIMP1"]


def score(expr, genes, label):
    """Средняя нормированная экспрессия набора генов на клетку.

    Каждый ген приводится к z-оценке по всем клеткам, потом усредняется. Так
    один высокоэкспрессируемый ген (COL1A1) не заглушает остальные.
    """
    present = [g for g in genes if g in expr.columns]
    missing = [g for g in genes if g not in expr.columns]
    print(f"  {label}: найдено {len(present)}/{len(genes)} генов в панели")
    if missing:
        print(f"    нет в панели: {', '.join(missing)}")
    if not present:
        return None, present
    X = expr[present].to_numpy(float)
    X = np.log1p(X)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    return ((X - mu) / sd).mean(1), present


# Подтипы, которые вообще не строма: их не раскладываем по двум типам.
# \b — граница слова. Без неё короткие куски ловят чужие слова:
# "nk" находится внутри "Unknown", "dc" внутри "adcy". Проверено на тесте.
NOT_STROMA = r"cancer|tumou?r cell|epi_area|epithel|myeloid|macroph|lymph|" \
             r"\bT cells|\bB cells|\bNK\b|plasma|mast cell|\bDC\b|\bcDC\b|" \
             r"\bpDC\b|endotheli|\bEC cells|tip cells|doublet|delete|other cells"

# Похоже на строму по названию. Перицитов и гладкую мышцу включаем: они
# стромальные, но по маркерам должны уйти в матриксную группу.
IS_STROMA = r"fibro|\bCAFs?\b|\bNFs?\b|pericyt|perycit|\bpery\b|\bSMC\b|" \
            r"granulosa|theca|lutein|stroma|mesothel|^Hs$"

# Насколько балл стероидогенеза должен превышать матриксный, чтобы отнести
# подтип к гормональной строме. Разница считается в z-оценках внутри среза,
# и при явном разделении она получается около 2. Порог 0.5 — это заведомо
# больше шума, но заметно меньше настоящего сигнала. Всё, что попало в
# коридор от -0.5 до +0.5, помечается «спорно» и решается глазами: около
# нуля различие неотличимо от случайного.
MARGIN = 0.5
# На сколько экспрессия стероидогенных генов должна превышать их фон в
# нестромальных клетках того же среза. Страховка от того, что z-оценка
# относительная: без неё в срезе из одних матриксных подтипов «победитель»
# всё равно объявляется гормональным. Единицы — как в данных (log-нормировка).
MIN_LIFT = 0.5
MIN_CELLS = 30      # подтипы мельче — в отдельную группу, они ненадёжны


def subtype_mode(paths, out):
    """Разнесение подтипов по группам сразу по всем срезам."""
    import yaml

    frames = []
    for p in paths:
        t = pd.read_csv(p)
        t["slide"] = Path(p).name.replace("_subtype_expr.csv", "")
        frames.append(t)
    tab = pd.concat(frames, ignore_index=True)
    print("срезов: %d, строк-подтипов: %d" % (tab.slide.nunique(), len(tab)))

    gene_cols = [c for c in tab.columns
                 if c not in ("cell_subtype", "cell_type", "n_cells", "slide")]
    print("генов в таблице: %d" % len(gene_cols))

    raw = tab[gene_cols].copy()      # сырые значения нужны для проверки ниже

    # z-оценка каждого гена считается ВНУТРИ среза: уровни экспрессии между
    # прогонами Xenium различаются, и общая нормировка смешала бы разницу
    # между срезами с разницей между подтипами.
    for g in gene_cols:
        tab[g] = tab.groupby("slide")[g].transform(
            lambda v: (v - v.mean()) / (v.std() if v.std() > 0 else 1.0))

    st_genes = [g for g in STEROID if g in gene_cols]
    mx_genes = [g for g in MATRIX if g in gene_cols]
    print("стероидогенез: %d/%d генов %s" % (len(st_genes), len(STEROID),
                                             st_genes))
    print("матрикс: %d/%d генов %s" % (len(mx_genes), len(MATRIX), mx_genes))
    if not st_genes:
        raise SystemExit(
            "в панели нет ни одного гена стероидогенеза — разделить строму "
            "по этим данным нельзя. Варианты в конце README шага 6.")

    tab["score_steroid"] = tab[st_genes].mean(1)
    tab["score_matrix"] = tab[mx_genes].mean(1) if mx_genes else 0.0
    tab["diff"] = tab.score_steroid - tab.score_matrix

    # Проверка в абсолютных величинах. z-оценка относительная: она всегда
    # кого-то ставит выше остальных. Если в срезе ВСЕ подтипы матриксные,
    # самый «менее матриксный» получит положительную разницу и будет
    # объявлен гормональным на пустом месте.
    #
    # Опора берётся внутри среза и не зависит от его общего уровня: фоновая
    # экспрессия стероидогенных генов в НЕстромальных клетках (опухоль,
    # иммунные). У настоящей гормональной стромы STAR и CYP11A1 заметно выше
    # этого фона, у матриксной — на уровне фона. Подъём считается в тех же
    # единицах, в которых лежат данные (log-нормировка Seurat).
    tab["raw_steroid"] = raw[st_genes].mean(1)

    # Строма определяется по названию подтипа ИЛИ по названию основного типа.
    # Только по подтипу нельзя: у подтипа может быть произвольное рабочее имя
    # («Stroma1», «Hs», номер кластера), и он потерялся бы молча.
    name = tab.cell_subtype.astype(str)
    main = tab.cell_type.astype(str)
    hit = lambda v, p: v.str.contains(p, case=False, regex=True)
    tab["kind"] = np.where(
        hit(name, NOT_STROMA) | hit(main, NOT_STROMA), "не строма",
        np.where(hit(name, IS_STROMA) | hit(main, IS_STROMA),
                 "строма", "неясно"))
    n_unclear = int((tab.kind == "неясно").sum())
    if n_unclear:
        print("\nне поняла, строма или нет (%d) — проверьте руками:" % n_unclear)
        for _, r in tab[tab.kind == "неясно"].iterrows():
            print("   %-30s тип: %-20s клеток %d"
                  % (str(r.cell_subtype)[:30], str(r.cell_type)[:20], r.n_cells))

    # фон: медиана по нестромальным подтипам среза
    base = (tab[tab.kind == "не строма"].groupby("slide")["raw_steroid"]
            .median())
    tab["lift"] = tab.raw_steroid - tab.slide.map(base)
    no_ref = sorted(set(tab.slide) - set(base.index))
    if no_ref:
        print("\nВНИМАНИЕ: в срезах %s нет нестромальных подтипов, сравнить "
              "не с чем.\n  Проверка на абсолютный уровень для них пропущена, "
              "решение только по относительной разнице —\n  посмотрите эти "
              "подтипы особенно внимательно." % ", ".join(no_ref))

    def decide(r):
        if r.kind != "строма":
            return "-"
        if r.n_cells < MIN_CELLS:
            return "мало клеток"
        if r["diff"] > MARGIN:
            # разница есть, но над фоном подъёма нет — значит, это просто
            # самый «менее матриксный» подтип, а не гормональная строма
            if pd.notna(r.lift) and r.lift < MIN_LIFT:
                return "спорно"
            return "stroma_hormonal"
        if r["diff"] < -MARGIN:
            return "stroma_matrix"
        return "спорно"

    tab["предложение"] = tab.apply(decide, axis=1)

    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["slide", "cell_subtype", "cell_type", "n_cells", "score_steroid",
            "score_matrix", "diff", "raw_steroid", "lift", "kind",
            "предложение"]
    tab[cols].sort_values(["slide", "diff"], ascending=[True, False]).to_csv(
        "%s_subtypes.csv" % out, index=False)

    print("\n%-34s %-12s %6s %7s  %s" % ("подтип", "срез", "клеток", "разн.",
                                          "предложение"))
    for _, r in tab[tab.kind == "строма"].sort_values(
            "diff", ascending=False).iterrows():
        print("%-34s %-12s %6d %7.2f  %s" % (
            str(r.cell_subtype)[:34], str(r.slide)[:12], r.n_cells, r["diff"],
            r["предложение"]))

    # --- заготовка для cell_type_map.yaml ---
    groups = {"stroma_hormonal": [], "stroma_matrix": [], "требует решения": []}
    for _, r in tab.iterrows():
        if r["предложение"] in ("stroma_hormonal", "stroma_matrix"):
            groups[r["предложение"]].append(str(r.cell_subtype))
        elif r["предложение"] in ("спорно", "мало клеток"):
            groups["требует решения"].append(
                "%s  # %s, клеток %d, разница %.2f"
                % (r.cell_subtype, r["предложение"], r.n_cells, r["diff"]))
    for k in groups:
        groups[k] = sorted(set(groups[k]))

    ypath = Path("%s_suggested.yaml" % out)
    with open(ypath, "w", encoding="utf-8") as f:
        f.write("# Предложение скрипта, НЕ готовая конфигурация.\n"
                "# Посмотрите %s_subtypes.png, поправьте и перенесите\n"
                "# в config/cell_type_map.yaml.\n"
                "# Порог отнесения: разница баллов больше %.2f.\n\n"
                % (out.name, MARGIN))
        yaml.safe_dump({"groups": groups}, f, allow_unicode=True,
                       sort_keys=False, default_flow_style=False)

    n_h = len(groups["stroma_hormonal"])
    n_m = len(groups["stroma_matrix"])
    print("\nгормональных подтипов: %d, матриксных: %d, спорных: %d"
          % (n_h, n_m, len(groups["требует решения"])))
    if n_h == 0:
        print("ВЫВОД: ни один подтип не прошёл по стероидогенезу. Гормональной "
              "стромы в этих данных не видно — либо генов нет в панели, либо "
              "популяции нет. Пятый класс учить нельзя.")
    elif sum(tab.loc[tab["предложение"] == "stroma_hormonal", "n_cells"]) < 500:
        print("ВНИМАНИЕ: гормональных клеток меньше 500 суммарно. Класс будет "
              "крайне редким, модель почти наверняка его не выучит.")
    else:
        print("ВЫВОД: разделение есть, можно строить метки.")

    # --- рисунок ---
    # Столбики, а не облако точек: решение принимается по одной величине —
    # разнице баллов, и на столбиках видно, кто близко к порогу. На точечном
    # графике подписи 18 подтипов налезают друг на друга и читать нечего.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = tab[tab.kind == "строма"].sort_values("diff").reset_index(drop=True)
    if len(d) == 0:
        print("стромальных подтипов не нашлось — рисовать нечего")
        return
    col = {"stroma_hormonal": "#2f7d4f", "stroma_matrix": "#1f6fb4",
           "спорно": "#c8891f", "мало клеток": "#999999"}
    h = max(2.6, 0.30 * len(d) + 1.4)
    fig, ax = plt.subplots(figsize=(8.2, h))
    y = np.arange(len(d))
    ax.barh(y, d["diff"], height=0.68,
            color=[col.get(p, "#cccccc") for p in d["предложение"]],
            edgecolor="0.3", linewidth=0.5)
    ax.axvline(0, color="0.3", lw=0.9)
    for m in (-MARGIN, MARGIN):
        ax.axvline(m, color="#b5482a", lw=0.9, ls="--")
    ax.axvspan(-MARGIN, MARGIN, color="#b5482a", alpha=0.07, lw=0)

    ax.set_yticks(y)
    ax.set_yticklabels(["%s  (%s, n=%d)" % (str(r.cell_subtype)[:30], r.slide,
                                            r.n_cells)
                        for _, r in d.iterrows()], fontsize=7)
    ax.set_xlabel("балл стероидогенеза минус балл матрикса (z-оценки внутри среза)")
    ax.set_title("Разнесение подтипов стромы. Полоса у нуля — спорная зона "
                 "(порог %.2f)" % MARGIN, fontsize=9, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.01)

    seen, handles = [], []
    from matplotlib.patches import Patch
    for p in d["предложение"]:
        if p not in seen:
            seen.append(p)
            handles.append(Patch(fc=col.get(p, "#cccccc"), ec="0.3", lw=0.5,
                                 label=p))
    ax.legend(handles=handles, frameon=False, fontsize=7,
              loc="lower right" if d["diff"].iloc[-1] > 0 else "upper right")
    fig.tight_layout()
    fig.savefig("%s_subtypes.png" % out, dpi=200)
    print("\nзаписано:\n  %s_subtypes.csv\n  %s_subtypes.png\n  %s"
          % (out, out, ypath))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtype-expr", nargs="*", default=None,
                    help="таблицы <срез>_subtype_expr.csv (режим по подтипам)")
    ap.add_argument("--cells")
    ap.add_argument("--expr")
    ap.add_argument("--out", default="outputs/results/stroma_split")
    ap.add_argument("--stroma-types", nargs="*", default=None,
                    help="какие cell_type считать стромой; по умолчанию всё, "
                         "что содержит fibro/CAF/NF/pery/SMC/granulosa/theca")
    args = ap.parse_args()

    if args.subtype_expr:
        subtype_mode([p for p in args.subtype_expr], Path(args.out))
        return
    if not (args.cells and args.expr):
        raise SystemExit("нужен либо --subtype-expr, либо пара --cells/--expr")

    cells = pd.read_csv(args.cells)
    expr = pd.read_csv(args.expr)
    if "cell_id" not in expr.columns:
        raise SystemExit("в --expr нет столбца cell_id")
    df = cells.merge(expr, on="cell_id", how="inner")
    print(f"клеток совмещено: {len(df)} из {len(cells)}")
    if len(df) == 0:
        raise SystemExit("cell_id не совпали между --cells и --expr")

    if args.stroma_types:
        keep = df.cell_type.isin(args.stroma_types)
    else:
        pat = r"fibro|CAF|NF|pery|peri|SMC|granulosa|theca|lutein|stroma"
        keep = df.cell_type.str.contains(pat, case=False, na=False)
    st = df[keep].copy()
    print(f"стромальных клеток: {len(st)}")
    print("по типам:", st.cell_type.value_counts().to_dict())
    if len(st) < 100:
        raise SystemExit("слишком мало стромальных клеток для оценки")

    gene_cols = [c for c in expr.columns if c != "cell_id"]
    print(f"генов в панели: {len(gene_cols)}")
    s_ster, g_ster = score(st[gene_cols], STEROID, "стероидогенез")
    s_matr, g_matr = score(st[gene_cols], MATRIX, "матрикс")
    if s_ster is None or s_matr is None:
        raise SystemExit("не хватает маркерных генов в панели Xenium — "
                         "разделить строму по этим данным нельзя")

    st["score_steroid"] = s_ster
    st["score_matrix"] = s_matr
    st["stroma_class"] = np.where(s_ster > s_matr, "stroma_hormonal", "stroma_matrix")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    st[["cell_id", "x", "y", "cell_type", "score_steroid", "score_matrix",
        "stroma_class"]].to_csv(f"{out}_scores.csv", index=False)

    # --- главный вопрос: разделяются ли клетки на две группы? ---
    d = s_ster - s_matr
    frac_h = float((st.stroma_class == "stroma_hormonal").mean())
    # бимодальность: сравниваем разброс внутри двух групп с общим разбросом.
    # Если деление осмысленно, внутри групп разброс заметно меньше.
    within = np.sqrt(((d[d > 0].std() ** 2) * (d > 0).sum() +
                      (d[d <= 0].std() ** 2) * (d <= 0).sum()) / len(d))
    ratio = within / d.std() if d.std() > 0 else np.nan
    print()
    print(f"доля гормональной стромы: {frac_h:.1%}")
    print(f"разброс внутри групп / общий разброс: {ratio:.3f}")
    if ratio < 0.75 and 0.05 < frac_h < 0.95:
        print("ВЫВОД: два типа стромы различимы, метки строить можно")
    else:
        print("ВЫВОД: чёткого разделения нет. Варианты: (1) взять другие "
              "маркеры, (2) кластеризовать строму без учителя и посмотреть, "
              "какие кластеры получились, (3) отказаться от 4-го класса "
              "и остаться на tumor/stroma")

    # --- рисунок ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    ax1.hist(d, bins=60, color="#6a8caf")
    ax1.axvline(0, color="#b5482a", lw=1.2, ls="--")
    ax1.set_xlabel("балл стероидогенеза минус балл матрикса")
    ax1.set_ylabel("клеток")
    ax1.set_title("Разделяется ли строма на два типа")
    col = {"stroma_hormonal": "#2f7d4f", "stroma_matrix": "#1f6fb4"}
    for cls, sub in st.groupby("stroma_class"):
        ax2.scatter(sub.x, sub.y, s=0.6, c=col[cls], label=cls, linewidths=0)
    ax2.set_aspect("equal")
    ax2.invert_yaxis()
    ax2.legend(markerscale=12, frameon=False, fontsize=8)
    ax2.set_title("Где какая строма лежит на срезе")
    ax2.set_xticks([]); ax2.set_yticks([])
    fig.tight_layout()
    fig.savefig(f"{out}.png", dpi=200)
    print(f"\nзаписано: {out}_scores.csv и {out}.png")


if __name__ == "__main__":
    main()
