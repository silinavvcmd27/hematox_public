# Пространственная неоднородность состава стромы: считаем долю гормональной
# в блоках по всему срезу и смотрим на разброс между блоками.
#
# Деление пополам показало, что у одних срезов состав ровный, а у других идёт
# градиент через всё стекло. Сетка превращает это наблюдение в число.
#
#   python stroma_heterogeneity.py --block-mm 2

import argparse
import csv
from pathlib import Path

import numpy as np

HORMONAL, MATRIX = 2, 3


def block_shares(cls, mpp, block_mm, min_stroma):
    step = max(1, int(round(block_mm * 1000 / mpp)))
    shares = []
    for y in range(0, cls.shape[0] - step + 1, step):
        for x in range(0, cls.shape[1] - step + 1, step):
            b = cls[y:y + step, x:x + step]
            horm = int((b == HORMONAL).sum())
            stroma = horm + int((b == MATRIX).sum())
            if stroma >= min_stroma:
                shares.append(100 * horm / stroma)
    return np.array(shares)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/results")
    ap.add_argument("--model", default="outputs/models/seg3_deploy.pth")
    ap.add_argument("--block-mm", type=float, default=2.0)
    ap.add_argument("--min-stroma", type=int, default=5000,
                    help="минимум стромальных пикселей в блоке, иначе блок пропускаем")
    ap.add_argument("--out", default="outputs/results/stroma_heterogeneity.csv")
    args = ap.parse_args()

    model_time = Path(args.model).stat().st_mtime
    rows = []
    for path in sorted(Path(args.dir).glob("*_seg_map.npz")):
        if path.stat().st_mtime < model_time:
            continue
        d = np.load(path)
        shares = block_shares(d["cls"], float(d["mpp"]), args.block_mm, args.min_stroma)
        if len(shares) < 3:
            print(f"  {path.name[:46]}: блоков {len(shares)}, слишком мало")
            continue
        q1, med, q3 = np.percentile(shares, [25, 50, 75])
        rows.append({
            "slide": path.name.replace("_seg_map.npz", ""),
            "блоков": len(shares),
            "медиана_%": round(med, 1),
            "квартильный_размах": round(q3 - q1, 1),
            "СКО": round(float(shares.std()), 1),
            "мин_%": round(float(shares.min()), 1),
            "макс_%": round(float(shares.max()), 1),
        })

    if not rows:
        raise SystemExit("нечего считать")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"блок {args.block_mm} мм, срезов {len(rows)}\n")
    print("%-46s %6s %8s %8s %8s" % ("срез", "блоков", "медиана", "разброс", "СКО"))
    for r in sorted(rows, key=lambda x: x["квартильный_размах"]):
        print("%-46s %6d %8.1f %8.1f %8.1f"
              % (r["slide"][:46], r["блоков"], r["медиана_%"],
                 r["квартильный_размах"], r["СКО"]))

    spread = np.array([r["квартильный_размах"] for r in rows], float)
    print("\nразброс внутри среза: медиана %.1f, размах %.1f..%.1f"
          % (np.median(spread), spread.min(), spread.max()))
    print("таблица:", args.out)


if __name__ == "__main__":
    main()
