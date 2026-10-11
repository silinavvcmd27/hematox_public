# Варианты подсчёта TSR по уже готовым картам зон. Кроме нашего среднего по всему
# срезу считаем принятые в литературе:
#
#   MI       круг радиусом 1.6 мм с наибольшей площадью опухоли. Это эмуляция
#            поля зрения 10x из van Wagensveld 2023 (HGSOC, 340 пациенток).
#   vanPelt  тот же круг, стромально богатое поле внутри опухолевой массы
#            (van Pelt 2018), взятое как квантиль 0.9 по всем таким полям.
#   vanPelt_max  то же, но буквальным максимумом, как написано в протоколе.
#
# Максимум приходится заменять квантилем, потому что здесь поле отбирается по
# строме и строма же измеряется. Отбор и измеряемая величина совпадают, значит
# оценка смещена вверх тем сильнее, чем больше стекло и чем больше на нём
# положений окна. У патолога этого нет: он просматривает десяток полей, а не
# тысячи. Квантиль от числа кандидатов не зависит. У MI такой беды нет изначально,
# там поле выбирается по опухоли, а меряется строма.
#
# Все они считаются внутри области измерения, а не по всему стеклу. Этим они и
# отличаются от нашего TSR: непухолевая ткань, лежащая отдельно от опухоли, в них
# не попадает.
#
#   python tsr_variants.py
#   python tsr_variants.py --radius-mm 1.1 --out outputs/results/tsr_variants.csv

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import spearmanr

from collect_tsr import slide_key

TUMOR, HORMONAL, MATRIX = 1, 2, 3

# подпись, колонка с TSR, колонка с составом стромы
VARIANTS = [("срез", "TSR_срез", "гормональная_срез_%"),
            ("MI", "TSR_MI", "гормональная_MI_%"),
            ("vanPelt", "TSR_vanPelt", "гормональная_vanPelt_%"),
            ("vanPelt_max", "TSR_vanPelt_max", "гормональная_vanPelt_max_%")]


def cell_counts(cls, mpp, cell_mm):
    """Сворачиваем карту в сетку ячеек по cell_mm, в каждой ячейке число пикселей
    каждого класса. Окно потом ходит по этой сетке: поле зрения шириной три
    миллиметра разрешать до пикселя незачем, а считается это в тысячи раз быстрее."""
    step = max(1, int(round(cell_mm * 1000 / mpp)))
    h = cls.shape[0] // step * step
    w = cls.shape[1] // step * step
    if not h or not w:
        return None, step
    blocks = cls[:h, :w].reshape(h // step, step, w // step, step)
    return {c: (blocks == c).sum((1, 3)).astype(np.float32)
            for c in (TUMOR, HORMONAL, MATRIX)}, step


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r).astype(np.float32)


def edge_sectors(r):
    """Четыре сектора внешнего кольца поля зрения. Требование протокола, чтобы
    опухоль была по всем четырём краям кадра, на круглом поле означает, что она
    есть в каждом из них."""
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    d2 = x * x + y * y
    ring = (d2 <= r * r) & (d2 >= (0.6 * r) ** 2)
    return [(ring & (x >= abs(y))).astype(np.float32),
            (ring & (-x >= abs(y))).astype(np.float32),
            (ring & (y >= abs(x))).astype(np.float32),
            (ring & (-y >= abs(x))).astype(np.float32)]


def sweep(counts, kernel):
    """Суммы по окну для каждого класса. Отрицательные хвосты от БПФ срезаем,
    иначе на пустых углах вылезают доли вида -0.0001."""
    return {c: np.clip(fftconvolve(a, kernel, mode="same"), 0, None)
            for c, a in counts.items()}


def shares(window):
    tumor, horm, matrix = window[TUMOR], window[HORMONAL], window[MATRIX]
    tissue = tumor + horm + matrix
    stroma = horm + matrix
    with np.errstate(invalid="ignore", divide="ignore"):
        tsr = np.where(tissue > 0, stroma / tissue, 0)
        horm_share = np.where(stroma > 0, 100 * horm / stroma, np.nan)
    return tissue, tsr, horm_share


def pick_field(tsr, allowed, q):
    """Положение окна, представляющее квантиль q распределения TSR по всем
    допустимым полям. При q = 1 это обычный максимум."""
    inside = np.flatnonzero(allowed)
    target = np.quantile(tsr.flat[inside], q)
    return inside[np.argmin(np.abs(tsr.flat[inside] - target))]


def field_variants(cls, mpp, cell_mm, radius_mm, min_cover, edge_tumor, quantile):
    counts, step = cell_counts(cls, mpp, cell_mm)
    if counts is None:
        return None

    r = max(1, int(round(radius_mm / cell_mm)))
    kernel = disk(r)
    capacity = kernel.sum() * step * step

    window = sweep(counts, kernel)
    tissue, tsr, horm_share = shares(window)
    cover = tissue / capacity

    ok = cover >= min_cover
    if not ok.any():
        # мелкий срез, целое поле зрения на него не ложится: берём самое полное
        ok = cover >= cover.max() * 0.99

    # поле выбирается по площади опухоли во всём круге, а не по её доле среди
    # ткани: иначе побеждает круг, наполовину висящий за краем среза, где ткани
    # мало, зато вся она опухолевая
    mi = np.argmax(window[TUMOR])
    row = {"TSR_MI": float(tsr.flat[mi]),
           "гормональная_MI_%": float(horm_share.flat[mi]),
           "покрытие_MI": float(cover.flat[mi])}

    # «опухоль по всем четырём краям» значит, что поле лежит внутри опухолевой
    # массы. Одного опухолевого пикселя в секторе для этого мало: почти чисто
    # стромальное поле, которого опухоль едва касается, проходит такую проверку,
    # а дальше максимум по всем положениям окна вытаскивает именно его
    total = counts[TUMOR] + counts[HORMONAL] + counts[MATRIX]
    four_sides = np.ones(tsr.shape, bool)
    for sector in edge_sectors(r):
        sector_area = sector.sum() * step * step
        sector_tumor = np.clip(fftconvolve(counts[TUMOR], sector, mode="same"), 0, None)
        sector_tissue = np.clip(fftconvolve(total, sector, mode="same"), 0, None)
        four_sides &= ((sector_tissue >= 0.1 * sector_area)
                       & (sector_tumor >= edge_tumor * sector_tissue))
    allowed = ok & four_sides
    row["четыре_стороны"] = int(allowed.any())
    if not allowed.any():
        allowed = ok

    for label, q in (("vanPelt", quantile), ("vanPelt_max", 1.0)):
        i = pick_field(tsr, allowed, q)
        row[f"TSR_{label}"] = float(tsr.flat[i])
        row[f"гормональная_{label}_%"] = float(horm_share.flat[i])
    return row


def slide_row(path, cell_mm, radius_mm, min_cover, edge_tumor, quantile):
    d = np.load(path)
    cls, mpp = d["cls"], float(d["mpp"])
    counts = {c: int((cls == c).sum()) for c in (TUMOR, HORMONAL, MATRIX)}
    tissue = sum(counts.values())
    if not tissue:
        return None
    fields = field_variants(cls, mpp, cell_mm, radius_mm, min_cover, edge_tumor, quantile)
    if fields is None:
        return None

    stroma = counts[HORMONAL] + counts[MATRIX]
    row = {"slide": path.name.replace("_seg_map.npz", ""),
           "площадь_мм2": round(tissue * mpp * mpp / 1e6, 1),
           "TSR_срез": round(stroma / tissue, 3),
           "гормональная_срез_%": round(100 * counts[HORMONAL] / stroma, 1) if stroma else None}
    for key, value in fields.items():
        if key.startswith("гормональная"):
            row[key] = round(value, 1)
        elif key.startswith(("TSR", "покрытие")):
            row[key] = round(value, 3)
        else:
            row[key] = value
    # протокол требует округления до десятков процентов, и от этого зависит, в
    # какую сторону от порога 50 попадёт пограничный срез
    row["vanPelt_10%"] = int(round(row["TSR_vanPelt"] * 10) * 10)
    row["строма_высокая"] = int(row["vanPelt_10%"] > 50)
    return row


def spearman(x, y):
    pairs = [(a, b) for a, b in zip(x, y)
             if a is not None and b is not None and not np.isnan(b)]
    if len(pairs) < 4:
        return None, None
    return spearmanr(*zip(*pairs))


def report(rows):
    area = [r["площадь_мм2"] for r in rows]
    print("\n%-9s %8s %8s %8s   %s"
          % ("вариант", "медиана", "нижн.кв", "верх.кв", "связь с площадью среза"))
    for label, tsr_key, _ in VARIANTS:
        vals = np.array([r[tsr_key] for r in rows], float)
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        rho, p = spearman(area, [r[tsr_key] for r in rows])
        print("%-9s %8.3f %8.3f %8.3f   rho %+.3f  p %.4f" % (label, med, q1, q3, rho, p))

    print("\nсвязь TSR с составом стромы внутри того же варианта:")
    for label, tsr_key, horm_key in VARIANTS:
        rho, p = spearman([r[tsr_key] for r in rows], [r[horm_key] for r in rows])
        print(f"  {label:9s} rho {rho:+.3f}  p {p:.4f}")

    sites = {}
    for r in rows:
        sites.setdefault(r["slide"].split("-")[1], []).append(r)
    print("\nпо источникам (TSS), медианы TSR:")
    print("  %-5s %4s  %s" % ("TSS", "n", "  ".join("%9s" % l for l, _, _ in VARIANTS)))
    for site in sorted(sites, key=lambda s: -len(sites[s])):
        group = sites[site]
        meds = "  ".join("%9.3f" % np.median([r[k] for r in group]) for _, k, _ in VARIANTS)
        print(f"  {site:<5s} {len(group):4d}  {meds}")

    high = sum(r["строма_высокая"] for r in rows)
    print(f"\nстромально богатых по протоколу (vanPelt, порог 50%): {high} из {len(rows)}")
    # 0.18 это оптимальный порог для MI у van Wagensveld, подобранный на их
    # когорте. К нам он не переносится, но показывает, куда смотреть
    print("для сравнения, выше 0.18 по MI: %d из %d"
          % (sum(r["TSR_MI"] > 0.18 for r in rows), len(rows)))
    lost = [r["slide"] for r in rows if not r["четыре_стороны"]]
    if lost:
        print(f"без поля с опухолью по всем четырём краям: {len(lost)}")
        for name in lost:
            print("   ", name[:46])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/results")
    ap.add_argument("--model", default="outputs/models/seg3_deploy.pth",
                    help="карты старше этого файла считаем устаревшими")
    ap.add_argument("--out", default="outputs/results/tsr_variants.csv")
    ap.add_argument("--all", action="store_true", help="брать и устаревшие карты")
    ap.add_argument("--radius-mm", type=float, default=1.6,
                    help="радиус поля зрения; 1.6 соответствует 10x у van Wagensveld")
    ap.add_argument("--cell-mm", type=float, default=0.1, help="шаг сетки, по которой ходит окно")
    ap.add_argument("--min-cover", type=float, default=0.5,
                    help="минимальная доля ткани в поле зрения")
    ap.add_argument("--edge-tumor", type=float, default=0.10,
                    help="минимальная доля опухоли в каждом из четырёх секторов края")
    ap.add_argument("--quantile", type=float, default=0.9,
                    help="квантиль по стромально богатым полям вместо максимума")
    args = ap.parse_args()

    model_time = Path(args.model).stat().st_mtime
    rows, stale, seen = [], 0, {}
    for path in sorted(Path(args.dir).glob("*_seg_map.npz")):
        if not args.all and path.stat().st_mtime < model_time:
            stale += 1
            continue
        key = slide_key(path.name)
        if key in seen:
            print(f"дубликат {path.name[:46]} (оставлен {seen[key][:46]})")
            continue
        print(f"  {path.name[:46]}", end="\r")
        row = slide_row(path, args.cell_mm, args.radius_mm, args.min_cover,
                        args.edge_tumor, args.quantile)
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
    print("%-46s %7s %7s %8s %8s" % ("срез", "срез", "MI", "vanPelt", "vP_max"))
    for r in rows:
        flag = "" if r["четыре_стороны"] else "  (поля с опухолью по краям нет)"
        print("%-46s %7.3f %7.3f %8.3f %8.3f%s"
              % (r["slide"][:46], r["TSR_срез"], r["TSR_MI"], r["TSR_vanPelt"],
                 r["TSR_vanPelt_max"], flag))

    report(rows)
    print("\nтаблица:", args.out)


if __name__ == "__main__":
    main()
