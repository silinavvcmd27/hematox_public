# Профиль стромы по расстоянию до ближайшей опухоли. Отвечает на два вопроса,
# которые иначе не разделить:
#
#   1. Различие между источниками (TSS 23 против 42 и 57) сидит снаружи опухоли
#      или внутри неё? Если снаружи, значит на этих стёклах просто больше
#      непухолевой ткани, и TSR по всему срезу меряет состав препарата.
#   2. Гормональная строма это CAF рядом с опухолью или остатки нормального
#      яичника? У нормальной ткани яичника строма тоже гормональная, и если она
#      лежит отдельными полями вдали от опухоли, наш показатель частично меряет
#      её, а не микроокружение.
#
# Расстояние считается по сетке ячеек: ячейка считается опухолевой, если опухоли
# в ней больше, чем стромы. Состав при этом берётся из полного разрешения, то
# есть квантуется только расстояние.
#
#   python stroma_distance.py
#   python stroma_distance.py --cell-mm 0.1 --out outputs/results/stroma_distance.csv

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import kruskal

from collect_tsr import slide_key
from tsr_variants import cell_counts

TUMOR, HORMONAL, MATRIX = 1, 2, 3

EDGES = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
BANDS = ["внутри", "0.05", "0.1", "0.25", "0.5", "1", "2", ">4"]
NEAR = 0.25   # что считаем перитуморальной стромой
FAR = 1.0     # дальше этого опухоль на состав уже не влияет


def profile(cls, mpp, cell_mm):
    counts, _ = cell_counts(cls, mpp, cell_mm)
    if counts is None:
        return None
    tumor, horm, matrix = counts[TUMOR], counts[HORMONAL], counts[MATRIX]
    solid_tumor = tumor > horm + matrix
    if not solid_tumor.any():
        return None

    dist = distance_transform_edt(~solid_tumor, sampling=cell_mm)
    band = np.digitize(dist, EDGES)

    horm_band = np.array([horm[band == i].sum() for i in range(len(BANDS))], float)
    stroma_band = horm_band + np.array([matrix[band == i].sum() for i in range(len(BANDS))], float)
    return horm_band, stroma_band


def share(horm, stroma):
    return 100 * horm.sum() / stroma.sum() if stroma.sum() else None


def slide_row(path, cell_mm):
    d = np.load(path)
    cls, mpp = d["cls"], float(d["mpp"])
    made = profile(cls, mpp, cell_mm)
    if made is None:
        return None
    horm_band, stroma_band = made
    if not stroma_band.sum():
        return None

    near = np.array(EDGES + [np.inf]) <= NEAR
    far = np.array([0] + EDGES) >= FAR
    row = {"slide": path.name.replace("_seg_map.npz", ""),
           "площадь_мм2": round(float((cls > 0).sum()) * mpp * mpp / 1e6, 1),
           "строма_до_0.25мм_%": round(100 * stroma_band[near].sum() / stroma_band.sum(), 1),
           "гормональная_рядом_%": share(horm_band[near], stroma_band[near]),
           "гормональная_вдали_%": share(horm_band[far], stroma_band[far])}
    for name, value in zip(BANDS, 100 * stroma_band / stroma_band.sum()):
        row[f"полоса_{name}_%"] = round(float(value), 1)
    for key in ("гормональная_рядом_%", "гормональная_вдали_%"):
        row[key] = round(row[key], 1) if row[key] is not None else None
    return row


def by_site(rows, field):
    sites = {}
    for r in rows:
        if r[field] is not None:
            sites.setdefault(r["slide"].split("-")[1], []).append(r[field])
    return sites


def report(rows):
    print("\nпрофиль стромы по расстоянию до опухоли, медиана доли стромы в полосе, %")
    print("  %-5s %4s  %s" % ("TSS", "n", "".join("%8s" % b for b in BANDS)))
    groups = {}
    for r in rows:
        groups.setdefault(r["slide"].split("-")[1], []).append(r)
    for site in sorted(groups, key=lambda s: -len(groups[s])):
        group = groups[site]
        line = "".join("%8.1f" % np.median([r[f"полоса_{b}_%"] for r in group]) for b in BANDS)
        print(f"  {site:<5s} {len(group):4d}  {line}")

    print(f"\nдоля стромы ближе {NEAR} мм к опухоли и состав рядом против вдали:")
    print("  %-5s %4s %12s %12s %12s" % ("TSS", "n", "рядом,%", "горм.рядом", "горм.вдали"))
    for site in sorted(groups, key=lambda s: -len(groups[s])):
        group = groups[site]
        near = np.median([r["строма_до_0.25мм_%"] for r in group])
        hn = [r["гормональная_рядом_%"] for r in group if r["гормональная_рядом_%"] is not None]
        hf = [r["гормональная_вдали_%"] for r in group if r["гормональная_вдали_%"] is not None]
        print("  %-5s %4d %12.1f %12s %12s"
              % (site, len(group), near,
                 "%.1f" % np.median(hn) if hn else "нет",
                 "%.1f" % np.median(hf) if hf else "нет"))

    big = [s for s, v in by_site(rows, "строма_до_0.25мм_%").items() if len(v) >= 5]
    if len(big) >= 2:
        sets = [by_site(rows, "строма_до_0.25мм_%")[s] for s in big]
        h, p = kruskal(*sets)
        print(f"\nдоля перитуморальной стромы, Краскел-Уоллис по {', '.join(big)}: "
              f"H {h:.2f}  p {p:.4f}")

    paired = [(r["гормональная_рядом_%"], r["гормональная_вдали_%"]) for r in rows
              if r["гормональная_рядом_%"] is not None and r["гормональная_вдали_%"] is not None]
    if paired:
        diff = np.array([a - b for a, b in paired])
        print(f"\nгормональная рядом минус вдали: медиана {np.median(diff):+.1f} пункта, "
              f"выше вдали у {int((diff < 0).sum())} срезов из {len(diff)}")
        print("  если гормональной больше вдали от опухоли, это довод за нормальную ткань")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/results")
    ap.add_argument("--model", default="outputs/models/seg3_deploy.pth")
    ap.add_argument("--out", default="outputs/results/stroma_distance.csv")
    ap.add_argument("--all", action="store_true", help="брать и устаревшие карты")
    ap.add_argument("--cell-mm", type=float, default=0.05,
                    help="шаг сетки, до него квантуется расстояние")
    args = ap.parse_args()

    model_time = Path(args.model).stat().st_mtime
    rows, stale, seen = [], 0, {}
    for path in sorted(Path(args.dir).glob("*_seg_map.npz")):
        if not args.all and path.stat().st_mtime < model_time:
            stale += 1
            continue
        key = slide_key(path.name)
        if key in seen:
            continue
        print(f"  {path.name[:46]}", end="\r")
        row = slide_row(path, args.cell_mm)
        if row:
            seen[key] = path.name
            rows.append(row)

    if stale:
        print(f"пропущено устаревших карт: {stale}")
    if not rows:
        raise SystemExit("нечего считать")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"срезов: {len(rows)}\n")
    print("%-46s %8s %10s %10s" % ("срез", "рядом,%", "горм.рядом", "горм.вдали"))
    for r in rows:
        shown = ["%.1f" % r[k] if r[k] is not None else "нет"
                 for k in ("гормональная_рядом_%", "гормональная_вдали_%")]
        print("%-46s %8.1f %10s %10s"
              % (r["slide"][:46], r["строма_до_0.25мм_%"], *shown))

    report(rows)
    print("\nтаблица:", args.out)


if __name__ == "__main__":
    main()
