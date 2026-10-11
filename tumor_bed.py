# Отсечение нормальной ткани по расстоянию до опухоли.
#
# Модель обучена на трёх классах и сказать «не знаю» не может: класс отказа мы
# пробовали, он ухудшил все три класса сразу. Поэтому норму отделяем после
# разметки, геометрически: строма дальше заданного расстояния от опухоли это не
# CAF, а остаточная нормальная ткань (яичник, труба, мезентерий).
#
# Порог взят из измерения по клеткам Xenium, а не подобран по картинке: на ovary2,
# где нормальной ткани нет вовсе, настоящие гормональные CAF лежат в 44..217 мкм
# от опухоли (медиана 101), матриксные в 16..63 (медиана 29). Значит 250 мкм
# накрывает обе популяции с запасом. Это определение, а не находка: утверждать по
# карте, что два типа CAF по-разному удалены от опухоли, после такой обрезки уже
# нельзя, для этого есть caf_context.py по эталонным меткам.
#
# На выходе карта с тем же именем плюс суффикс, чтобы прежняя осталась.
#
#   python tumor_bed.py --dir outputs/results
#   python tumor_bed.py --dir outputs/results --dist-um 150 --suffix _bed150

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from tsr_variants import cell_counts

TUMOR, HORMONAL, MATRIX, NORMAL = 1, 2, 3, 7


def reclassify(cls, mpp, cell_mm, dist_um):
    """Строма дальше dist_um от опухоли становится нормой. Расстояние считается
    по сетке ячеек: до пикселя его разрешать незачем, а полноразмерный EDT на
    карте в десятки миллионов пикселей съедает память."""
    counts, step = cell_counts(cls, mpp, cell_mm)
    if counts is None:
        return None, None
    solid_tumor = counts[TUMOR] > counts[HORMONAL] + counts[MATRIX]
    if not solid_tumor.any():
        return None, None

    far = distance_transform_edt(~solid_tumor, sampling=cell_mm * 1000) > dist_um
    far_full = np.repeat(np.repeat(far, step, 0), step, 1)
    # сетка строилась по обрезанной до кратности карте, хвост справа и снизу
    # достраиваем краем, иначе маска не сойдётся по форме
    pad = [(0, cls.shape[i] - far_full.shape[i]) for i in (0, 1)]
    far_full = np.pad(far_full, pad, mode="edge")

    out = cls.copy()
    out[far_full & np.isin(cls, (HORMONAL, MATRIX))] = NORMAL
    return out, far_full


def shares(cls, px_mm2):
    counts = {c: int((cls == c).sum()) for c in (TUMOR, HORMONAL, MATRIX, NORMAL)}
    tissue = counts[TUMOR] + counts[HORMONAL] + counts[MATRIX]
    stroma = counts[HORMONAL] + counts[MATRIX]
    return {
        "TSR": round(stroma / tissue, 3) if tissue else None,
        "гормональная_в_строме_%": round(100 * counts[HORMONAL] / stroma, 1) if stroma else None,
        "норма_мм2": round(counts[NORMAL] * px_mm2, 1),
        "ткань_мм2": round(tissue * px_mm2, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/results")
    ap.add_argument("--dist-um", type=float, default=250.0,
                    help="дальше этого расстояния от опухоли строма считается нормой")
    ap.add_argument("--cell-mm", type=float, default=0.05)
    ap.add_argument("--suffix", default="_bed")
    ap.add_argument("--out", default="outputs/results/tumor_bed.csv")
    args = ap.parse_args()

    rows = []
    for path in sorted(Path(args.dir).glob("*_seg_map.npz")):
        if args.suffix in path.name:
            continue
        d = np.load(path)
        cls, mpp = d["cls"], float(d["mpp"])
        px_mm2 = mpp * mpp / 1e6
        out, _ = reclassify(cls, mpp, args.cell_mm, args.dist_um)
        if out is None:
            print(f"{path.name[:46]}: опухоли нет, пропускаю")
            continue

        name = path.name.replace("_seg_map.npz", "")
        before, after = shares(cls, px_mm2), shares(out, px_mm2)
        dst = path.with_name(f"{name}{args.suffix}_seg_map.npz")
        np.savez_compressed(dst, cls=out, mpp=mpp)

        row = {"срез": name,
               "TSR_до": before["TSR"], "TSR_после": after["TSR"],
               "гормональная_до_%": before["гормональная_в_строме_%"],
               "гормональная_после_%": after["гормональная_в_строме_%"],
               "норма_мм2": after["норма_мм2"], "ткань_до_мм2": before["ткань_мм2"]}
        rows.append(row)
        print("%-46s TSR %.3f -> %.3f   горм. %5.1f -> %5.1f   норма %6.1f мм2"
              % (name[:46], row["TSR_до"], row["TSR_после"],
                 row["гормональная_до_%"] or 0, row["гормональная_после_%"] or 0,
                 row["норма_мм2"]))

    if not rows:
        raise SystemExit("нечего обрабатывать")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    moved = np.array([r["норма_мм2"] / r["ткань_до_мм2"] for r in rows], float)
    print(f"\nсрезов {len(rows)}, в норму ушло медианно {100 * np.median(moved):.1f}% ткани "
          f"(размах {100 * moved.min():.1f}..{100 * moved.max():.1f})")
    print("таблица:", args.out)


if __name__ == "__main__":
    main()
