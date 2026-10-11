# Сводка по когорте для моделей первого слоя: TSR и доли классов по картам зон.
# Опухоль это код 1, строма коды 2, 3 и 5, прочее код 7 (некроз, кровь, слизь,
# жир) в TSR не входит и считается отдельно. Старый collect_tsr.py остаётся для
# карт seg4, где строма разделена на гормональную и матриксную.
#
# Повторные препараты одного пациента отбрасываются, как и раньше: DX1 и DX2 это
# разные препараты, а вот один и тот же срез под двумя именами считать дважды нельзя.
#
#   python pan_collect_tsr.py --dir outputs/results/pan/tcga_ov_deploy3

import argparse
import csv
from pathlib import Path

import numpy as np

from src.utils import TUMOR
from pan_train import STROMA_CODES, OTHER
from collect_tsr import slide_key, summarize


def tsr(cls):
    tum = int((cls == TUMOR).sum())
    stroma = int(np.isin(cls, STROMA_CODES).sum())
    return stroma / (tum + stroma) if tum + stroma else None


def halves(cls):
    """TSR в левой и правой половинах, поделённых по массе ткани, а не по ширине:
    показывает, насколько показатель гуляет внутри одного среза."""
    per_col = (cls > 0).sum(0).cumsum()
    if not per_col[-1]:
        return None, None
    mid = int(np.searchsorted(per_col, per_col[-1] / 2))
    return tsr(cls[:, :mid]), tsr(cls[:, mid:])


def slide_row(path):
    d = np.load(path)
    cls, mpp = d["cls"], float(d["mpp"])
    counts = {"опухоль": int((cls == TUMOR).sum()),
              "строма": int(np.isin(cls, STROMA_CODES).sum()),
              "прочее": int((cls == OTHER).sum())}
    tissue = sum(counts.values())
    if not tissue:
        return None
    left, right = halves(cls)
    px_mm2 = mpp * mpp / 1e6
    row = {"slide": path.name.replace("_seg_map.npz", ""),
           "площадь_мм2": round(tissue * px_mm2, 1)}
    for name, n in counts.items():
        row[f"{name}_%"] = round(100 * n / tissue, 1)
    row["TSR"] = round(tsr(cls), 3)
    row["TSR_левая"] = round(left, 3) if left is not None else None
    row["TSR_правая"] = round(right, 3) if right is not None else None
    row["расхождение_половин"] = (round(abs(left - right), 3)
                                  if left is not None and right is not None else None)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/results/pan/tcga_ov_deploy3")
    ap.add_argument("--out", default=None, help="по умолчанию cohort.csv рядом с картами")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="не отбрасывать повторные препараты одного пациента")
    args = ap.parse_args()

    rows = [r for r in (slide_row(p) for p in sorted(Path(args.dir).glob("*_seg_map.npz"))) if r]
    if not rows:
        raise SystemExit(f"в {args.dir} нет карт *_seg_map.npz")
    if not args.keep_duplicates:
        seen, unique = {}, []
        for row in rows:
            key = slide_key(row["slide"])
            if key in seen:
                print(f"  {row['slide']}: тот же препарат, что {seen[key]}, пропускаю")
                continue
            seen[key] = row["slide"]
            unique.append(row)
        rows = unique

    out = Path(args.out or Path(args.dir) / "cohort.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"срезов {len(rows)}, таблица: {out}")
    for field in ("TSR", "опухоль_%", "строма_%", "прочее_%", "расхождение_половин"):
        summarize(rows, field)


if __name__ == "__main__":
    main()
