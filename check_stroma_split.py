"""Проверка: делятся ли стромальные клетки Xenium на гормональную и матриксную.

Это первый шаг перед любым обучением на 4 класса. Если по маркерным генам
два типа стромы не разделяются, то и учить их по H&E нечему — метки будут
шумом, а модель выучит случайность.

Запуск:
    python check_stroma_split.py --cells data/seurat_csv/ovary2_he_cells.csv \
        --expr data/seurat_csv/ovary2_he_expr.csv --out outputs/results/stroma_split

Вход:
    --cells  cell_id, x, y, cell_type            (уже есть, из R/export_seurat.R)
    --expr   cell_id + столбцы генов              (новое, см. R/export_seurat.R)

Выход:
    stroma_split_scores.csv   на каждую клетку: балл стероидогенеза и балл матрикса
    stroma_split.png          две панели: распределение баллов и карта среза
    и печатает ответ на главный вопрос: разделяются или нет
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--expr", required=True)
    ap.add_argument("--out", default="outputs/results/stroma_split")
    ap.add_argument("--stroma-types", nargs="*", default=None,
                    help="какие cell_type считать стромой; по умолчанию всё, "
                         "что содержит fibro/CAF/NF/pery/SMC/granulosa/theca")
    args = ap.parse_args()

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
