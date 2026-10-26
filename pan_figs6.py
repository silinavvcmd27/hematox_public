# Рисунок: какую область среза берёт каждое из четырёх определений TSR.
#
# Четыре строки, по одной на определение. Слева H&E, справа карта модели, и на обоих
# выделена одна и та же область подсчёта: внутри неё изображение как есть, снаружи бледно,
# поля обведены. Области строятся тем же способом, что в tsr_defs.py (сетка ячеек 50 мкм,
# полоса 250 мкм, круг радиусом 1,6 мм, квантиль 0,9), а числа в заголовках считаются
# функциями самого tsr_defs.py, так что рисунок и таблица не могут разойтись.
#
# Поле по строме в tsr_defs.py это не одно место, а квантиль 0,9 доли стромы по всем
# положениям круга, где опухоль занимает не меньше 30 % ткани поля. Из-за этого условия
# доля стромы в таком поле не бывает выше 0,70. На рисунке показано положение с долей
# стромы в пределах 0,01 от квантили и с наибольшим количеством ткани в круге.
#
#   python pan_figs6.py --list                   # срезы, где определения сильнее расходятся
#   python pan_figs6.py --slide TCGA-23-1022

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve

from pan_collect_tsr import tsr
from pan_figs import сохранить
from pan_train import OTHER, STROMA_CODES
from src.utils import TUMOR
from tsr_defs import bed_tsr, cell_counts, field_tsr

ЦВЕТ_ОПУХОЛИ = (232, 105, 159)
ЦВЕТ_СТРОМЫ = (86, 180, 233)
ЦВЕТ_ПРОЧЕГО = (127, 194, 62)
ЧЕРНИЛА = "#1e2230"
ТЕКСТ = "#222222"
БЛЕДНОСТЬ_КАРТЫ = 0.22    # сколько цвета остаётся у карты вне области
БЛЕДНОСТЬ_HE = 0.30       # то же для H&E, чуть больше, иначе не видно строения


def число(v):
    return "—" if v is None else f"{v:.3f}".replace(".", ",")


def раскрасить(c):
    out = np.full(c.shape + (3,), 255, np.uint8)
    out[c == TUMOR] = ЦВЕТ_ОПУХОЛИ
    out[np.isin(c, STROMA_CODES)] = ЦВЕТ_СТРОМЫ
    out[c == OTHER] = ЦВЕТ_ПРОЧЕГО
    return out


def бледно(rgb, сколько):
    return (255 - (255 - rgb.astype(np.float32)) * сколько).astype(np.uint8)


def круг(форма, ci, cj, r):
    yy, xx = np.ogrid[:форма[0], :форма[1]]
    return (yy - ci) ** 2 + (xx - cj) ** 2 <= r * r


def поля(tumor, stroma, r, min_cover, quantile, ёмкость, допуск=0.01):
    """Центры двух полей в ячейках, как их выбирает field_tsr.

    Квантиль это число, а не место: доля стромы, близкая к ней, бывает во многих
    положениях круга, в том числе на краю среза, где круг почти пуст. Для рисунка из
    положений с долей в пределах допуска от квантили берётся то, где ткани больше всего.
    Возвращается и заполнение круга тканью, опухолью и стромой (ёмкость это число точек
    карты в полном круге)."""
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    disc = (x * x + y * y <= r * r).astype(np.float32)
    tum = fftconvolve(tumor.astype(np.float32), disc, mode="same")
    stro = fftconvolve(stroma.astype(np.float32), disc, mode="same")
    total = tum + stro
    ok = total > 0
    share = np.where(ok, stro / np.where(ok, total, 1), np.nan)
    по_опухоли = np.unravel_index(int(np.argmax(tum)), tum.shape)
    inside = ok & (tum >= min_cover * total)
    q = np.quantile(share[inside], quantile)
    отклонение = np.where(inside, np.abs(share - q), np.inf)
    близкие = отклонение <= max(допуск, отклонение.min() + 1e-9)
    по_строме = np.unravel_index(int(np.argmax(np.where(близкие, total, -1))), tum.shape)
    return (по_опухоли, по_строме, float(share[по_строме]),
            float(total[по_строме] / ёмкость), float(total[по_опухоли] / ёмкость))


def кандидаты(каталог, n=10):
    t = pd.read_csv(Path(каталог) / "tsr_defs.csv")
    t = t[(t["опухоль_%"] >= 15) & (t["доля_полосы"] < 0.9)].copy()
    t["разброс"] = t[["TSR", "TSR_бед", "TSR_поле_опухоль", "TSR_поле_строма"]].std(axis=1)
    t = t.sort_values("разброс", ascending=False).head(n)
    print(t[["slide", "опухоль_%", "строма_%", "доля_полосы", "TSR", "TSR_бед",
             "TSR_поле_строма", "TSR_поле_опухоль"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/results/pan/tcga_ov_deploy5_raw")
    ap.add_argument("--svs-dir", default="data/tcga_ov_flat")
    ap.add_argument("--slide", default="TCGA-23-1029")
    ap.add_argument("--out", default="outputs/results/pan/figs")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--cell-mm", type=float, default=0.05)
    ap.add_argument("--dist-um", type=float, default=250.0)
    ap.add_argument("--radius-mm", type=float, default=1.6)
    ap.add_argument("--min-cover", type=float, default=0.3)
    ap.add_argument("--quantile", type=float, default=0.9)
    ap.add_argument("--width", type=int, default=1400, help="ширина панели в точках")
    args = ap.parse_args()

    if args.list:
        кандидаты(args.dir)
        return
    файлы = glob.glob(f"{args.dir}/{args.slide}*_seg_map.npz")
    срезы = glob.glob(f"{args.svs_dir}/{args.slide}*.svs")
    if not файлы or not срезы:
        raise SystemExit(f"не нашлись карта ({len(файлы)}) или срез ({len(срезы)}) {args.slide}")
    d = np.load(файлы[0])
    cls, mpp = d["cls"], float(d["mpp"])

    # Числа берутся у tsr_defs.py, области строятся рядом по тем же правилам.
    tumor, stroma = cell_counts(cls, mpp, args.cell_mm)
    step = max(1, int(round(args.cell_mm * 1000 / mpp)))
    r = int(round(args.radius_mm / args.cell_mm))
    v_вся = tsr(cls)
    v_полоса, _ = bed_tsr(tumor, stroma, args.cell_mm, args.dist_um)
    v_опухоль, v_строма = field_tsr(tumor, stroma, args.cell_mm, args.radius_mm,
                                    args.min_cover, args.quantile)
    solid = tumor > stroma
    near = distance_transform_edt(~solid, sampling=args.cell_mm * 1000) <= args.dist_um
    ёмкость = float(np.sum(круг((2 * r + 1, 2 * r + 1), r, r, r))) * step * step
    ц_опухоль, ц_строма, доля_в_круге, заполнение_с, заполнение_о = поля(
        tumor, stroma, r, args.min_cover, args.quantile, ёмкость)
    print(f"{Path(файлы[0]).name}: вся ткань {число(v_вся)}, полоса {число(v_полоса)}, "
          f"поле по строме {число(v_строма)} (на рисунке круг с {число(доля_в_круге)}, "
          f"опухоль и строма занимают {100 * заполнение_с:.0f} % круга), "
          f"поле по опухоли {число(v_опухоль)} ({100 * заполнение_о:.0f} % круга)")

    # Панели в уменьшенном виде: точка экрана берёт класс ближайшей точки карты, а
    # принадлежность к области у той ячейки сетки, в которую эта точка попадает.
    f = max(1, int(np.ceil(max(cls.shape) / args.width)))
    ys, xs = np.arange(0, cls.shape[0], f), np.arange(0, cls.shape[1], f)
    c = cls[np.ix_(ys, xs)]
    ткань = c > 0
    iy = np.where(ys // step < tumor.shape[0])[0]
    ix = np.where(xs // step < tumor.shape[1])[0]

    def на_экран(маска):
        out = np.zeros(c.shape, bool)
        out[np.ix_(iy, ix)] = маска[np.ix_(ys[iy] // step, xs[ix] // step)]
        return out

    # Карта покрывает срез целиком, поэтому миниатюра H&E просто приводится к её размеру,
    # как в pan_figure_assets.py.
    import openslide
    sl = openslide.OpenSlide(срезы[0])
    he = sl.get_thumbnail((c.shape[1], c.shape[0])).convert("RGB")
    he = np.asarray(he.resize((c.shape[1], c.shape[0]), Image.LANCZOS))

    круг_строма = на_экран(круг(tumor.shape, *ц_строма, r))
    круг_опухоль = на_экран(круг(tumor.shape, *ц_опухоль, r))
    # Полоса без обводки: её граница идёт и по стеклу в пределах 250 мкм от опухоли, и
    # вокруг каждого выпавшего островка стромы, и обводка превращается в шум. Что вошло,
    # видно по яркости, а сколько ткани вошло, написано в заголовке.
    полоса = ткань & на_экран(near)
    в_полосе = 100 * полоса.sum() / ткань.sum()
    панели = [
        ("а", "Вся ткань среза", v_вся, ткань, None),
        ("б", f"Перитуморальная полоса {args.dist_um:.0f} мкм, "
              f"{в_полосе:.0f} % ткани", v_полоса, полоса, None),
        ("в", f"Поле с наибольшей стромой внутри опухоли (опухоли в поле не меньше "
              f"{100 * args.min_cover:.0f} %)", v_строма, ткань & круг_строма, круг_строма),
        ("г", "Поле с наибольшей опухолью", v_опухоль, ткань & круг_опухоль, круг_опухоль),
    ]

    карта = раскрасить(c)
    высота = c.shape[0] / c.shape[1]
    fig, axes = plt.subplots(4, 2, figsize=(10, 20 * высота + 2.6))
    for строка, (буква, имя, v, область, контур) in zip(axes, панели):
        слева = np.where(область[..., None], he, бледно(he, БЛЕДНОСТЬ_HE))
        справа = np.where(область[..., None], карта, бледно(карта, БЛЕДНОСТЬ_КАРТЫ))
        справа[~ткань] = 255
        for ax, img in zip(строка, (слева, справа)):
            ax.imshow(img, interpolation="nearest")
            if контур is not None:
                ax.contour(контур.astype(float), levels=[0.5], colors=[ЧЕРНИЛА],
                           linewidths=1.2)
            ax.set_axis_off()
        строка[0].set_title(f"{буква}) {имя}: TSR {число(v)}", fontsize=10.5, color=ТЕКСТ,
                            loc="left")
    axes[0, 0].text(0.5, 1.22, "H&E", transform=axes[0, 0].transAxes, ha="center",
                    fontsize=11, color=ТЕКСТ, weight="bold")
    axes[0, 1].text(0.5, 1.22, "Разметка модели", transform=axes[0, 1].transAxes,
                    ha="center", fontsize=11, color=ТЕКСТ, weight="bold")

    # масштаб на первой панели H&E
    мм = 2
    длина = мм * 1000 / (mpp * f)
    ax = axes[0, 0]
    y0, x0 = c.shape[0] * 0.96, c.shape[1] * 0.03
    ax.plot([x0, x0 + длина], [y0, y0], color=ЧЕРНИЛА, linewidth=2.5)
    ax.text(x0 + длина / 2, y0 - c.shape[0] * 0.02, f"{мм} мм", ha="center", va="bottom",
            fontsize=9, color=ТЕКСТ)

    легенда = [Patch(color=np.array(ЦВЕТ_ОПУХОЛИ) / 255, label="опухоль"),
               Patch(color=np.array(ЦВЕТ_СТРОМЫ) / 255, label="строма"),
               Patch(color=np.array(ЦВЕТ_ПРОЧЕГО) / 255, label="прочее, в TSR не входит"),
               Patch(color=бледно(np.array([ЦВЕТ_СТРОМЫ], np.uint8), БЛЕДНОСТЬ_КАРТЫ)[0] / 255,
                     label="бледно: вне области подсчёта")]
    fig.legend(handles=легенда, loc="lower center", ncol=4, frameon=False, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    имя = f"fig_tsr_regions_{args.slide[:12]}"
    сохранить(fig, Path(args.out), имя)


if __name__ == "__main__":
    main()
