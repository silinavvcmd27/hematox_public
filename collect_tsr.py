# Сводка по когорте: проходит по картам зон и складывает TSR и состав стромы
# в одну таблицу.
#
# Карты, посчитанные не текущей моделью, пропускаются: до нормализации окраски
# и до трёхклассовой схемы числа несопоставимы, а перепутать их в общей таблице
# слишком легко.
#
#   python collect_tsr.py
#   python collect_tsr.py --dir outputs/results --out outputs/results/cohort.csv

import argparse
import csv
from pathlib import Path

import numpy as np

from seg_infer_svs import ZONE_NAMES

TUMOR, HORMONAL, MATRIX = 1, 2, 3


def slide_key(name):
    """Пациент плюс номер препарата. Один и тот же срез встречается в папке под
    разными именами, а вот DX1 и DX2 одного пациента это действительно разные
    препараты, и объединять их нельзя."""
    parts = name.split("-")
    dx = next((p[:3] for p in parts if p.upper().startswith("DX")), "DX?")
    return "-".join(parts[:3]) + "/" + dx.upper()


def hormonal_share(cls):
    horm = int((cls == HORMONAL).sum())
    stroma = horm + int((cls == MATRIX).sum())
    return 100 * horm / stroma if stroma else None


def halves(cls):
    """Состав стромы в левой и правой половинах. Делим не пополам по ширине, а по
    массе ткани, иначе на срезах с пустым краем в одной половине почти ничего нет.
    Это опора для сравнения: насколько показатель гуляет внутри одного среза."""
    per_col = (cls > 0).sum(0).cumsum()
    if not per_col[-1]:
        return None, None
    mid = int(np.searchsorted(per_col, per_col[-1] / 2))
    return hormonal_share(cls[:, :mid]), hormonal_share(cls[:, mid:])


def slide_row(path):
    d = np.load(path)
    cls, mpp = d["cls"], float(d["mpp"])
    counts = {c: int((cls == c).sum()) for c in (TUMOR, HORMONAL, MATRIX)}
    tissue = sum(counts.values())
    if not tissue:
        return None
    left, right = halves(cls)

    stroma = counts[HORMONAL] + counts[MATRIX]
    px_mm2 = mpp * mpp / 1e6
    return {
        "slide": path.name.replace("_seg_map.npz", ""),
        "площадь_мм2": round(tissue * px_mm2, 1),
        "опухоль_%": round(100 * counts[TUMOR] / tissue, 1),
        "гормональная_%": round(100 * counts[HORMONAL] / tissue, 1),
        "матриксная_%": round(100 * counts[MATRIX] / tissue, 1),
        "TSR": round(stroma / tissue, 3),
        "гормональная_в_строме_%": round(100 * counts[HORMONAL] / stroma, 1) if stroma else None,
        "левая_половина_%": round(left, 1) if left is not None else None,
        "правая_половина_%": round(right, 1) if right is not None else None,
        "расхождение_половин": (round(abs(left - right), 1)
                                if left is not None and right is not None else None),
    }


def summarize(rows, field):
    vals = np.array([r[field] for r in rows if r[field] is not None], float)
    if not len(vals):
        return
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    print(f"  {field:24s} медиана {med:6.3f}  квартили {q1:.3f}..{q3:.3f}  "
          f"размах {vals.min():.3f}..{vals.max():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/results")
    ap.add_argument("--model", default="outputs/models/seg3_deploy.pth",
                    help="карты старше этого файла считаем устаревшими")
    ap.add_argument("--out", default="outputs/results/tcga_cohort.csv")
    ap.add_argument("--all", action="store_true", help="брать и устаревшие карты")
    args = ap.parse_args()

    model_time = Path(args.model).stat().st_mtime
    rows, stale = [], 0
    for path in sorted(Path(args.dir).glob("*_seg_map.npz")):
        if not args.all and path.stat().st_mtime < model_time:
            stale += 1
            continue
        row = slide_row(path)
        if row:
            rows.append(row)

    seen, unique, dups = {}, [], []
    for row in rows:
        key = slide_key(row["slide"])
        if key in seen:
            dups.append((row["slide"], seen[key]))
            continue
        seen[key] = row["slide"]
        unique.append(row)
    rows = unique

    if stale:
        print(f"пропущено устаревших карт: {stale}")
    for name, kept in dups:
        print(f"дубликат {name[:46]} (оставлен {kept[:46]})")
    if not rows:
        raise SystemExit("нечего собирать")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"срезов в таблице: {len(rows)}")
    for r in rows:
        print("  %-46s TSR %.3f  горм. в строме %5.1f%%  %6.1f мм2"
              % (r["slide"][:46], r["TSR"], r["гормональная_в_строме_%"] or 0,
                 r["площадь_мм2"]))

    print("\nпо когорте:")
    for field in ("TSR", "гормональная_в_строме_%", "опухоль_%",
                  "расхождение_половин"):
        summarize(rows, field)
    print("  расхождение половин это разброс внутри среза, с ним и надо "
          "сравнивать размах по когорте")
    print("\nтаблица:", args.out)


if __name__ == "__main__":
    main()