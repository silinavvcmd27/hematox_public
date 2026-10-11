# Из чего на самом деле состоит группа «гормональная строма» и где эти клетки
# лежат относительно опухоли. Считается по исходной аннотации Xenium, ни модель,
# ни собранные графы не участвуют.
#
# Повод: в config/cell_type_map.yaml в stroma_hormonal вместе с C7+ CAF записаны
# Granulosa cells, Theca cells, Luteinized cells, Steroidogenic cells и четыре
# вида NFs. То есть класс с самого начала смешивает CAF с нормальной тканью
# яичника, и непонятно, в какой пропорции.
#
# Скрипт разбирает группы по исходным подтипам, считает для каждого подтипа
# расстояние до ближайшей опухолевой клетки и отдельно проверяет, лежат ли два
# типа настоящих CAF вперемешку или участками.
#
#   python caf_context.py --slides ovary2_he ovary_prime_he ovary3_he

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

X_NAMES = ["x_centroid", "aligned_x", "he_x", "x_he", "imagecol", "x"]
Y_NAMES = ["y_centroid", "aligned_y", "he_y", "y_he", "imagerow", "y"]


def load_map(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    flat = {}
    for section in ("subtype_groups", "groups"):
        flat[section] = {name.lower(): zone
                         for zone, names in cfg.get(section, {}).items()
                         for name in names}
    return flat["subtype_groups"], flat["groups"]


def assign(df, sub_map, top_map):
    """Зона и та метка, по которой она получена. Подтип имеет приоритет, как в
    cell_graph_build.py: у двух срезов из трёх дробные типы живут в cell_subtype,
    а cell_type там у всех фибробластов одинаковый.

    Допущение: если подтип есть, но в карте его нет, пробуем cell_type. Настоящий
    mapper может в этом случае сразу ставить undefined, стоит сверить по числу
    undefined в выводе."""
    ctype = df["cell_type"].astype(str)
    csub = df["cell_subtype"].astype(str) if "cell_subtype" in df.columns else None

    zone = pd.Series("undefined", index=df.index)
    used = ctype.copy()
    if csub is not None:
        hit = csub.str.lower().map(sub_map)
        zone = zone.mask(hit.notna(), hit)
        used = used.mask(hit.notna(), csub)
    fallback = ctype.str.lower().map(top_map)
    take = (zone == "undefined") & fallback.notna()
    zone = zone.mask(take, fallback)
    used = used.mask(take, ctype)
    return zone, used


def kind(name):
    """CAF, нормальный фибробласт или прочая нормальная ткань, по названию
    подтипа. CYP11A1+ fibroblasts и Granulosa cells попадают в «прочее»
    намеренно: это не CAF, но и не обычный фибробласт, пусть считаются отдельно."""
    low = name.lower()
    if "caf" in low:
        return "CAF"
    if "nf" in low.replace("nfs", "nf"):
        return "NFs"
    return "прочее"


def coords_um(df, mpp, x_col, y_col):
    x = x_col or next((c for c in X_NAMES if c in df.columns), None)
    y = y_col or next((c for c in Y_NAMES if c in df.columns), None)
    if x not in df.columns or y not in df.columns:
        raise SystemExit(f"не нашла колонки координат, есть: {list(df.columns)[:25]}")
    return x, y, df[[x, y]].to_numpy(float) * mpp


def slide_mpp(slide, graph_dir, fallback):
    """mpp лежит в собранном графе. Файл большой, но npz распакует только
    нужный массив, читать 500 МБ ради одного числа не придётся."""
    path = Path(graph_dir) / f"{slide}_graph.npz"
    if path.exists():
        return float(np.load(path)["mpp"])
    return fallback


def report_group(zone_name, sub, dist, is_tumor_present):
    here = sub.index
    total = len(here)
    print(f"\nразбор группы {zone_name}: {total} клеток")
    print("  %-34s %8s %7s %12s" % ("подтип", "клеток", "доля", "до опухоли"))
    for name, n in sub.value_counts().items():
        med = np.median(dist[sub.to_numpy() == name]) if is_tumor_present else float("nan")
        print("  %-34s %8d %6.1f%% %11.0f" % (str(name)[:34], n, 100 * n / total, med))
    by_kind = sub.map(kind).value_counts()
    parts = ", ".join(f"{k} {100 * v / total:.1f}%" for k, v in by_kind.items())
    print(f"  итого: {parts}")


def mixing(xy, lab, k, shuffles, rng):
    """Доля соседей своего типа среди k ближайших. Сравнивается с перемешанными
    метками при тех же координатах: если совпадает, типы лежат вперемешку."""
    if len(xy) <= k:
        return None, None
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    idx = idx[:, 1:]
    same = float((lab[idx] == lab[:, None]).mean())
    trials = []
    for _ in range(shuffles):
        shuffled = rng.permutation(lab)
        trials.append((shuffled[idx] == shuffled[:, None]).mean())
    return same, float(np.mean(trials))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", nargs="+", required=True)
    ap.add_argument("--cells", nargs="+", default=None,
                    help="пути к csv в том же порядке, что и срезы; без этого "
                         "берётся первый подходящий по маске, а их в папке много")
    ap.add_argument("--cells-dir", default="data/seurat_csv")
    ap.add_argument("--graph-dir", default="data/processed/graph")
    ap.add_argument("--map", default="config/cell_type_map.yaml")
    ap.add_argument("--mpp", type=float, default=0.2125,
                    help="если рядом нет собранного графа")
    ap.add_argument("--x-col", default=None)
    ap.add_argument("--y-col", default=None)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--shuffles", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sub_map, top_map = load_map(args.map)
    rng = np.random.default_rng(args.seed)

    if args.cells and len(args.cells) != len(args.slides):
        raise SystemExit("--cells должен идти в том же числе, что и --slides")

    for i, slide in enumerate(args.slides):
        if args.cells:
            found = [Path(args.cells[i])]
        else:
            found = sorted(Path(args.cells_dir).glob(f"{slide}_cells*.csv"))
        if not found or not found[0].exists():
            print(f"\n{slide}: не нашла csv в {args.cells_dir}")
            continue
        df = pd.read_csv(found[0])
        mpp = slide_mpp(slide, args.graph_dir, args.mpp)
        x_col, y_col, xy = coords_um(df, mpp, args.x_col, args.y_col)

        zone, used = assign(df, sub_map, top_map)
        span = (xy.max(0) - xy.min(0)) / 1000
        print(f"\n=== {slide} ===")
        print(f"{found[0].name}, клеток {len(df)}, координаты {x_col}/{y_col}, "
              f"mpp {mpp:.4f}")
        # проверка единиц: срез должен получиться около сантиметра, если вышли
        # доли миллиметра или метры, значит координаты были не в пикселях
        print(f"поле {span[0]:.1f} x {span[1]:.1f} мм")
        print("зоны:", ", ".join(f"{z} {n}" for z, n in zone.value_counts().items()))

        tumor = xy[(zone == "tumor").to_numpy()]
        if len(tumor):
            dist = cKDTree(tumor).query(xy, k=1)[0]
        else:
            dist = np.full(len(xy), np.nan)
            print("опухолевых клеток нет, расстояния не считаю")

        for zone_name in ("stroma_hormonal", "stroma_matrix"):
            mask = (zone == zone_name).to_numpy()
            if mask.sum():
                report_group(zone_name, used[mask], dist[mask], len(tumor) > 0)

        # перемешанность считаем только по настоящим CAF: нормальная ткань в
        # этой проверке участвовать не должна, иначе меряем не то
        caf = ((zone.isin(["stroma_hormonal", "stroma_matrix"])).to_numpy()
               & (used.map(kind) == "CAF").to_numpy())
        if caf.sum() > args.knn:
            lab = (zone[caf] == "stroma_hormonal").to_numpy().astype(np.int8)
            same, chance = mixing(xy[caf], lab, args.knn, args.shuffles, rng)
            print(f"\nтолько CAF ({caf.sum()} клеток), соседей своего типа среди "
                  f"{args.knn}: {100*same:.1f}%, при перемешанных метках "
                  f"{100*chance:.1f}%")
            if len(tumor):
                for name, want in (("гормональные CAF", 1), ("матриксные CAF", 0)):
                    d = dist[caf][lab == want]
                    if len(d):
                        q1, med, q3 = np.percentile(d, [25, 50, 75])
                        print(f"  {name:18s} n {len(d):7d}  до опухоли "
                              f"{med:6.0f} мкм ({q1:.0f}..{q3:.0f})")


if __name__ == "__main__":
    main()
