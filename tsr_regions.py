# TSR в измерительной области, по методике OTSR (Hacking et al., PMC10680933).
#
# Считает долю стромы тремя способами:
#   whole  — по всему срезу, как считалось раньше
#   MI     — в круге радиусом 1.6 мм (поле зрения 10x) с наибольшей долей опухоли
#   WT     — в ложе опухоли: ткань, где локальная плотность опухоли выше порога
#
# Вход — карта классов из seg_infer_svs.py: 0 фон, 1 опухоль, 2 строма.
#
# python tsr_regions.py --map outputs/results/demo_tcga_1582.npz

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from src.utils import TUMOR, STROMA_HORMONAL, STROMA_MATRIX


def box_mean(a, out_h, out_w):
    """Средняя доля по ячейкам грубой сетки. Замена cv2.resize(INTER_AREA).

    OpenCV тянулся в проект целиком (~60 МБ) ради двух изменений размера.
    """
    h, w = a.shape
    ys = np.linspace(0, h, out_h + 1).astype(int)
    xs = np.linspace(0, w, out_w + 1).astype(int)
    cs = np.zeros((h + 1, w + 1), np.float64)
    cs[1:, 1:] = a.cumsum(0).cumsum(1)
    Y0, Y1 = ys[:-1, None], ys[1:, None]
    X0, X1 = xs[None, :-1], xs[None, 1:]
    s = cs[Y1, X1] - cs[Y0, X1] - cs[Y1, X0] + cs[Y0, X0]
    n = np.maximum((Y1 - Y0) * (X1 - X0), 1)
    return (s / n).astype(np.float32)


def resize_nearest(a, out_h, out_w):
    """Ближайший сосед. Замена cv2.resize(INTER_NEAREST)."""
    yi = (np.arange(out_h) * a.shape[0] / out_h).astype(int).clip(0, a.shape[0] - 1)
    xi = (np.arange(out_w) * a.shape[1] / out_w).astype(int).clip(0, a.shape[1] - 1)
    return a[yi][:, xi]


CSV_FIELDS = ["срез", "область", "TSR", "площадь_мм2", "гормональная_мм2",
              "матриксная_мм2", "wt_threshold", "mi_radius_mm", "work_mpp"]


def write_rows(path, new_rows):
    """Записать строки, заменив прежние с тем же ключом (срез, область, параметры).

    БЫЛО: режим "a" без ключа. В outputs/results/tsr_regions.csv накопилось
    шесть строк на один срез — дубли от прогонов с разными параметрами, и какая
    строка актуальная, по файлу понять нельзя.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def key(r):
        return (str(r.get("срез")), str(r.get("область")),
                str(r.get("wt_threshold")), str(r.get("mi_radius_mm")),
                str(r.get("work_mpp")))

    old = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            old = list(csv.DictReader(fh))
    keys = {key(r) for r in new_rows}
    kept = [r for r in old if key(r) not in keys]
    if len(kept) < len(old):
        print(f"  заменено прежних строк: {len(old) - len(kept)}")

    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in kept + new_rows:
            w.writerow(r)
    tmp.replace(path)      # атомарно: обрыв не оставит обрезанный файл
    print(f"\nзаписано: {path} ({len(kept) + len(new_rows)} строк)")


def disc(radius_px):
    r = int(round(radius_px))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r).astype(np.float32)


def tsr(stroma_px, tumor_px, min_px=50):
    """Доля стромы среди опухоли и стромы. NA, если считать не на чем.

    БЫЛО: при tumor_px == 0 возвращалось ровно 1.0, и срез без опухоли попадал
    в группу худшего прогноза. Это худший вид ошибки: она выглядит как
    результат. TSR без опухоли не определён.

    min_px — минимум размеченных ячеек, ниже которого доля случайна.
    """
    total = stroma_px + tumor_px
    if total < min_px or tumor_px == 0:
        return float("nan")
    return float(stroma_px / total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, help="npz из seg_infer_svs.py")
    ap.add_argument("--mi-radius-mm", type=float, default=1.6,
                    help="радиус поля зрения 10x")
    ap.add_argument("--mi-min-tissue", type=float, default=2 / 3,
                    help="какую долю круга должна занимать неfоновая ткань")
    ap.add_argument("--wt-radius-mm", type=float, default=0.5,
                    help="окно, в котором считается локальная плотность опухоли")
    ap.add_argument("--wt-threshold", type=float, default=0.10,
                    help="порог плотности опухоли для ложа")
    ap.add_argument("--wt-min-tissue", type=float, default=0.5,
                    help="какую долю ячейки должна занимать ткань, чтобы войти в ложе")
    ap.add_argument("--work-mpp", type=float, default=16.0,
                    help="мкм/px, на которых ищутся области")
    ap.add_argument("--out-csv", default="outputs/results/tsr_regions.csv")
    args = ap.parse_args()


    d = np.load(args.map)
    cls = d["cls"]
    mpp = float(d["mpp"])
    H, W = cls.shape
    px_mm2 = (mpp / 1000) ** 2
    print(f"карта {W}x{H}, {mpp:.2f} мкм/px, срез {W*mpp/1000:.1f} x {H*mpp/1000:.1f} мм")

    tumor_full = (cls == TUMOR)
    horm_full = (cls == STROMA_HORMONAL)
    matr_full = (cls == STROMA_MATRIX)
    stroma_full = horm_full | matr_full
    print(f"опухоль {tumor_full.sum()*px_mm2:.1f} мм², "
          f"строма {stroma_full.sum()*px_mm2:.1f} мм² "
          f"(гормональная {horm_full.sum()*px_mm2:.1f}, "
          f"матриксная {matr_full.sum()*px_mm2:.1f} мм²)")

    # области ищем на грубой сетке: точность до пары микрометров тут не нужна,
    # а свёртка с кругом в 800 пикселей на полном разрешении неподъёмна
    f = args.work_mpp / mpp
    Ww, Hw = max(1, int(W / f)), max(1, int(H / f))
    tum_w = box_mean(tumor_full.astype(np.float32), Hw, Ww)
    str_w = box_mean(stroma_full.astype(np.float32), Hw, Ww)
    horm_w = box_mean(horm_full.astype(np.float32), Hw, Ww)
    matr_w = box_mean(matr_full.astype(np.float32), Hw, Ww)
    tis_w = tum_w + str_w   # доля неfоновых пикселей в ячейке
    print(f"рабочая сетка {Ww}x{Hw} при {args.work_mpp:.0f} мкм/px")

    rows = [{"область": "весь срез",
             "TSR": tsr(stroma_full.sum(), tumor_full.sum()),
             "площадь_мм2": (tumor_full.sum() + stroma_full.sum()) * px_mm2}]

    # --- MI: круг с максимальной долей опухоли ---
    r_mi = args.mi_radius_mm * 1000 / args.work_mpp
    k = disc(r_mi)
    n_k = float(k.sum())
    t_sum = fftconvolve(tum_w, k, mode="same")
    s_sum = fftconvolve(str_w, k, mode="same")
    tissue_sum = t_sum + s_sum
    frac_tumor = np.divide(t_sum, tissue_sum, out=np.zeros_like(t_sum), where=tissue_sum > 0)
    frac_tumor[tissue_sum / n_k < args.mi_min_tissue] = -1.0

    mi_center = None
    if frac_tumor.max() < 0:
        print("\nMI: ни одно поле зрения не набрало нужной доли ткани")
    else:
        iy, ix = np.unravel_index(np.argmax(frac_tumor), frac_tumor.shape)
        mi_center = (int(ix), int(iy))
        cy, cx = int(iy * f), int(ix * f)
        r_full = int(round(args.mi_radius_mm * 1000 / mpp))
        y0, y1 = max(0, cy - r_full), min(H, cy + r_full + 1)
        x0, x1 = max(0, cx - r_full), min(W, cx + r_full + 1)
        yy, xx = np.ogrid[y0 - cy:y1 - cy, x0 - cx:x1 - cx]
        mi_mask = np.zeros_like(cls, bool)
        mi_mask[y0:y1, x0:x1] = (xx * xx + yy * yy <= r_full * r_full)

        t_mi = int((tumor_full & mi_mask).sum())
        s_mi = int((stroma_full & mi_mask).sum())
        print(f"\nMI: центр {cx*mpp/1000:.1f}, {cy*mpp/1000:.1f} мм | "
              f"опухоли в круге {100*frac_tumor[iy, ix]:.1f}%")
        rows.append({"область": f"MI, круг {args.mi_radius_mm} мм",
                     "TSR": tsr(s_mi, t_mi), "площадь_мм2": (t_mi + s_mi) * px_mm2})

    # --- WT: ложе опухоли по локальной плотности ---
    r_wt = args.wt_radius_mm * 1000 / args.work_mpp
    kd = disc(r_wt)
    t2 = fftconvolve(tum_w, kd, mode="same")
    s2 = fftconvolve(str_w, kd, mode="same")
    tis2 = t2 + s2
    dens = np.divide(t2, tis2, out=np.zeros_like(t2), where=tis2 > 0)
    # сглаживание размывает плотность за край среза, поэтому отдельно требуем,
    # чтобы сама ячейка была тканью, иначе ложе вылезает на пустое поле
    bed_w = (dens >= args.wt_threshold) & (tis_w >= args.wt_min_tissue)

    bed_full = resize_nearest(bed_w.astype(np.uint8), H, W) > 0
    t_wt = int((tumor_full & bed_full).sum())
    s_wt = int((stroma_full & bed_full).sum())
    print(f"WT: ложе занимает {100*bed_w.mean():.1f}% кадра, "
          f"{(t_wt + s_wt) * px_mm2:.1f} мм² ткани")
    rows.append({"область": f"WT, порог {args.wt_threshold}",
                 "TSR": tsr(s_wt, t_wt), "площадь_мм2": (t_wt + s_wt) * px_mm2})

    print(f"\n{'область':24s} {'TSR':>7s} {'площадь':>11s}  вывод")
    for r in rows:
        verdict = "богатый стромой" if r["TSR"] >= 0.5 else "бедный стромой"
        print(f"{r['область']:24s} {100*r['TSR']:6.1f}% {r['площадь_мм2']:9.1f} мм²  {verdict}")

    name = Path(args.map).stem
    for r in rows:
        r.update({"срез": name,
                  "TSR": round(r["TSR"], 4) if r["TSR"] == r["TSR"] else "",
                  "площадь_мм2": round(r["площадь_мм2"], 1),
                  "wt_threshold": args.wt_threshold,
                  "mi_radius_mm": args.mi_radius_mm,
                  "work_mpp": args.work_mpp})
    write_rows(args.out_csv, rows)

    vis = np.full((Hw, Ww, 3), 245, np.uint8)
    vis[str_w > tum_w] = (38, 139, 210)
    vis[tum_w > str_w] = (220, 50, 47)
    vis[tis_w < 0.15] = 245
    # Контуры рисуем matplotlib: OpenCV нужен был только здесь.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(Ww, 200) / 100, max(Hw, 200) / 100), dpi=100)
    ax.imshow(vis)
    ax.contour(bed_w.astype(float), levels=[0.5], colors="#148d3c", linewidths=1.2)
    if mi_center is not None:
        ax.add_patch(plt.Circle(mi_center, r_mi, fill=False, ec="#141414", lw=1.2))
    ax.set_axis_off()
    png = Path(args.out_csv).parent / f"{name}_tsr_regions.png"
    fig.savefig(png, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("области:", png, "— зелёный контур WT, чёрный круг MI")


if __name__ == "__main__":
    main()