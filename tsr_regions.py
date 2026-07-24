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


def disc(radius_px):
    r = int(round(radius_px))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r).astype(np.float32)


def tsr(stroma_px, tumor_px):
    total = stroma_px + tumor_px
    return float(stroma_px / total) if total else float("nan")


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

    import cv2

    d = np.load(args.map)
    cls = d["cls"]
    mpp = float(d["mpp"])
    H, W = cls.shape
    px_mm2 = (mpp / 1000) ** 2
    print(f"карта {W}x{H}, {mpp:.2f} мкм/px, срез {W*mpp/1000:.1f} x {H*mpp/1000:.1f} мм")

    tumor_full = (cls == 1)
    stroma_full = (cls == 2)
    print(f"опухоль {tumor_full.sum()*px_mm2:.1f} мм², "
          f"строма {stroma_full.sum()*px_mm2:.1f} мм²")

    # области ищем на грубой сетке: точность до пары микрометров тут не нужна,
    # а свёртка с кругом в 800 пикселей на полном разрешении неподъёмна
    f = args.work_mpp / mpp
    Ww, Hw = max(1, int(W / f)), max(1, int(H / f))
    tum_w = cv2.resize(tumor_full.astype(np.float32), (Ww, Hw), interpolation=cv2.INTER_AREA)
    str_w = cv2.resize(stroma_full.astype(np.float32), (Ww, Hw), interpolation=cv2.INTER_AREA)
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

    bed_full = cv2.resize(bed_w.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
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
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    first = not out_csv.exists()
    with open(out_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["срез", "область", "TSR", "площадь_мм2"])
        if first:
            w.writeheader()
        for r in rows:
            w.writerow({"срез": name, "область": r["область"],
                        "TSR": round(r["TSR"], 4),
                        "площадь_мм2": round(r["площадь_мм2"], 1)})
    print("\nдописано в", out_csv)

    vis = np.full((Hw, Ww, 3), 245, np.uint8)
    vis[str_w > tum_w] = (38, 139, 210)
    vis[tum_w > str_w] = (220, 50, 47)
    vis[tis_w < 0.15] = 245
    cnts, _ = cv2.findContours(bed_w.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (20, 130, 60), 2)
    if mi_center is not None:
        cv2.circle(vis, mi_center, int(r_mi), (20, 20, 20), 2)
    png = out_csv.parent / f"{name}_tsr_regions.png"
    cv2.imwrite(str(png), vis[:, :, ::-1])
    print("области:", png, "— зелёный контур WT, чёрный круг MI")


if __name__ == "__main__":
    main()