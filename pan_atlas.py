# Атлас около 30 срезов из TCGA-OV и PTRC-HGSOC: на каждой странице H&E, карта модели,
# зоны подсчёта и доля стромы при всех четырёх определениях.
#
# «Интересные» срезы отбираются правилом, а не на глаз, и причина отбора пишется на
# странице и в таблице-указателе:
#   TCGA-OV  срезы рисунков 18 и 19 диплома; по одному на 5, 25, 50, 75 и 95-м процентиле
#            доли стромы; четыре, где определения расходятся сильнее всего; два с наибольшей
#            долей прочей ткани; все, кого исключило правило качества
#   PTRC     типичный срез (доля стромы ближе всего к медиане группы) для яичника и сальника
#            у рефрактерных и у чувствительных и для кишки у рефрактерных; четыре с наибольшим
#            расхождением определений; самая низкая и самая высокая доля стромы; по одному
#            исключённому на каждую причину; один после неоадъюванта; срез с лимфоидными
#            скоплениями из проверки карт
# Из одного блока (у TCGA из одной пациентки) берётся не больше одного среза. Ответ на
# платину на страницах PTRC показан: анализ по плану уже сделан, атлас описательный.
#
# Зоны и числа строятся как в pan_figs6.py: числа дают функции tsr_defs.py, круг поля по
# строме рисуется там, где доля стромы в пределах 0,01 от квантили и ткани больше всего.
#
#   python pan_atlas.py

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve

from pan_collect_tsr import tsr
from pan_figs import ЦВЕТА
from pan_figs6 import (ЦВЕТ_ОПУХОЛИ, ЦВЕТ_ПРОЧЕГО, ЦВЕТ_СТРОМЫ, ЧЕРНИЛА, ТЕКСТ, бледно, круг,
                       раскрасить, число)
from tsr_defs import bed_tsr, cell_counts, field_tsr

ОПРЕДЕЛЕНИЯ = ["TSR", "TSR_бед", "TSR_поле_строма", "TSR_поле_опухоль"]
ПОЛЕ_СТРОМА, ПОЛЕ_ОПУХОЛЬ = ЦВЕТА[4], ЦВЕТА[5]
МЕСТА = {"OV": "яичник или труба", "OM": "сальник", "GI": "кишка", "PT": "брюшина",
         "UT": "матка", "LN": "лимфоузел"}
ОТВЕТ = {"refractory": "рефрактерная", "sensitive": "чувствительная"}


def запятая(v, знаков=1):
    return f"{v:.{знаков}f}".replace(".", ",")


def причина_исключения(r):
    if r["опухоль_%"] < 5:
        return "опухоль < 5 %"
    if r["опухоль_%"] + r["строма_%"] < 50:
        return "опухоль и строма < 50 %"
    return ""


def добавить(выбор, t, idx, причина):
    """Срез idx, если из его блока ещё ничего не взято."""
    r = t.loc[idx]
    if r.блок in {v["блок"] for v in выбор}:
        return 0
    выбор.append({**r.to_dict(), "причина": причина})
    return 1


def по_порядку(выбор, t, порядок, причина, n):
    взято = 0
    for idx in порядок:
        if взято == n:
            break
        взято += добавить(выбор, t, idx, причина)


def разброс(годные, t):
    g = годные[годные["опухоль_%"] >= 15]
    return g[ОПРЕДЕЛЕНИЯ].std(axis=1).sort_values(ascending=False).index


def отбор_tcga(t):
    t = t.assign(блок=t.slide.str[:12], исключён=t.apply(причина_исключения, axis=1))
    годные = t[t.исключён == ""]
    выбор = []
    for имя, подпись in (("TCGA-23-1029", "рисунок 18 диплома"),
                         ("TCGA-23-2084", "рисунок 19 диплома")):
        по_порядку(выбор, t, t.index[t.slide.str.startswith(имя)], подпись, 1)
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        цель = годные.TSR.quantile(q)
        по_порядку(выбор, t, (годные.TSR - цель).abs().sort_values().index,
                   f"уровень доли стромы: {100 * q:.0f}-й процентиль когорты", 1)
    по_порядку(выбор, t, разброс(годные, t), "определения расходятся сильнее всего", 4)
    по_порядку(выбор, t, годные["прочее_%"].sort_values(ascending=False).index,
               "больше всего прочей ткани", 2)
    for idx in t.index[t.исключён != ""]:
        добавить(выбор, t, idx, f"исключён правилом качества: {t.at[idx, 'исключён']}")
    return выбор


def отбор_ptrc(t, slides):
    t = t.merge(slides, on="slide", how="inner")
    t = t.assign(блок=t.образец.astype(str), исключён=t.apply(причина_исключения, axis=1))
    годные = t[(t.исключён == "") & ~t.неоадъювант]
    выбор = []
    for место, ответ in (("OV", "refractory"), ("OV", "sensitive"), ("OM", "refractory"),
                         ("OM", "sensitive"), ("GI", "refractory")):
        g = годные[(годные.место == место) & (годные.ответ == ответ)]
        по_порядку(выбор, t, (g.TSR - g.TSR.median()).abs().sort_values().index,
                   f"типичный срез: {МЕСТА[место]}, {ОТВЕТ[ответ]}", 1)
    по_порядку(выбор, t, разброс(годные, t), "определения расходятся сильнее всего", 4)
    по_порядку(выбор, t, годные.TSR.sort_values().index, "самая низкая доля стромы", 1)
    по_порядку(выбор, t, годные.TSR.sort_values(ascending=False).index,
               "самая высокая доля стромы", 1)
    for причина in ("опухоль < 5 %", "опухоль и строма < 50 %"):
        g = t[(t.исключён == причина) & ~t.неоадъювант]
        по_порядку(выбор, t, g["площадь_мм2"].sort_values(ascending=False).index,
                   f"исключён правилом качества: {причина}", 1)
    нео = t[(t.исключён == "") & t.неоадъювант]
    по_порядку(выбор, t, (нео.TSR - нео.TSR.median()).abs().sort_values().index,
               "после неоадъювантной химиотерапии", 1)
    по_порядку(выбор, t, t.index[t.slide.str.startswith("1009423_171044")],
               "лимфоидные скопления отнесены к строме (проверка карт)", 1)
    return выбор


def центры(tumor, stroma, r, min_cover, quantile, допуск=0.01):
    """Центры полей, как в pan_figs6.поля, но без ошибки, если поля по строме нет."""
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    disc = (x * x + y * y <= r * r).astype(np.float32)
    tum = fftconvolve(tumor.astype(np.float32), disc, mode="same")
    stro = fftconvolve(stroma.astype(np.float32), disc, mode="same")
    total = tum + stro
    ok = total > 0.5
    share = np.where(ok, stro / np.where(ok, total, 1), np.nan)
    по_опухоли = np.unravel_index(int(np.argmax(tum)), tum.shape) if tum.max() > 0.5 else None
    inside = ok & (tum >= min_cover * total)
    if not inside.any():
        return по_опухоли, None
    q = np.quantile(share[inside], quantile)
    отклонение = np.where(inside, np.abs(share - q), np.inf)
    близкие = отклонение <= max(допуск, отклонение.min() + 1e-9)
    return по_опухоли, np.unravel_index(int(np.argmax(np.where(близкие, total, -1))), tum.shape)


def страница(pdf, out, номер, r, когорта, npz, svs, args):
    d = np.load(npz)
    cls, mpp = d["cls"], float(d["mpp"])
    tumor, stroma = cell_counts(cls, mpp, args.cell_mm)
    step = max(1, int(round(args.cell_mm * 1000 / mpp)))
    rr = int(round(args.radius_mm / args.cell_mm))
    v = {"TSR": tsr(cls)}
    v["TSR_бед"], _ = bed_tsr(tumor, stroma, args.cell_mm, args.dist_um)
    v["TSR_поле_опухоль"], v["TSR_поле_строма"] = field_tsr(
        tumor, stroma, args.cell_mm, args.radius_mm, args.min_cover, args.quantile)
    solid = tumor > stroma
    near = (distance_transform_edt(~solid, sampling=args.cell_mm * 1000) <= args.dist_um
            if solid.any() else np.zeros_like(solid))
    ц_опухоль, ц_строма = центры(tumor, stroma, rr, args.min_cover, args.quantile)

    f = max(1, int(np.ceil(max(cls.shape) / args.width)))
    ys, xs = np.arange(0, cls.shape[0], f), np.arange(0, cls.shape[1], f)
    c = cls[np.ix_(ys, xs)]
    ткань = c > 0
    iy = np.where(ys // step < tumor.shape[0])[0]
    ix = np.where(xs // step < tumor.shape[1])[0]

    def на_экран(маска):
        out_ = np.zeros(c.shape, bool)
        out_[np.ix_(iy, ix)] = маска[np.ix_(ys[iy] // step, xs[ix] // step)]
        return out_

    import openslide
    sl = openslide.OpenSlide(svs)
    he = sl.get_thumbnail((c.shape[1], c.shape[0])).convert("RGB")
    he = np.asarray(he.resize((c.shape[1], c.shape[0]), Image.LANCZOS))
    полоса = ткань & на_экран(near)
    в_полосе = 100 * полоса.sum() / max(1, ткань.sum())

    fig = plt.figure(figsize=(11.69, 8.27))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.2], hspace=0.12, wspace=0.04,
                          left=0.02, right=0.98, top=0.88, bottom=0.03)
    оси = [fig.add_subplot(gs[0, j]) for j in range(3)]
    карта = раскрасить(c)
    зоны = np.where(полоса[..., None], he, бледно(he, 0.3))
    for ax, img, имя in zip(оси, (he, карта, зоны), ("H&E", "разметка модели",
                                                     "зоны подсчёта")):
        ax.imshow(img, interpolation="nearest")
        ax.set_axis_off()
        ax.set_title(имя, fontsize=10.5, color=ТЕКСТ, loc="left")
    for центр, цвет in ((ц_строма, ПОЛЕ_СТРОМА), (ц_опухоль, ПОЛЕ_ОПУХОЛЬ)):
        if центр is None:
            continue
        m = на_экран(круг(tumor.shape, *центр, rr)).astype(float)
        for ax in оси[1:]:
            ax.contour(m, levels=[0.5], colors=[цвет], linewidths=1.8)
    длина = 2000 / (mpp * f)
    y0, x0 = c.shape[0] * 0.97, c.shape[1] * 0.03
    оси[0].plot([x0, x0 + длина], [y0, y0], color=ЧЕРНИЛА, linewidth=2.5)
    оси[0].text(x0 + длина / 2, y0 - c.shape[0] * 0.02, "2 мм", ha="center", va="bottom",
                fontsize=8.5, color=ТЕКСТ)

    if когорта == "TCGA-OV":
        имя = r["slide"][:12]
        клиника = f"учреждение (TSS) {r['slide'][5:7]}"
    else:
        имя = f"{r['slide'].replace('_ImageActual', '')}, пациентка {r['пациент']}"
        клиника = (f"место: {МЕСТА.get(r['место'], 'нет данных')}; "
                   f"ответ на платину: {ОТВЕТ.get(r['ответ'], r['ответ'])}; центр {r['центр']}; "
                   f"неоадъювант: {'да' if r['неоадъювант'] else 'нет'}")
    fig.text(0.02, 0.955, f"{номер}. {имя}", fontsize=14, color=ТЕКСТ, weight="bold")
    fig.text(0.02, 0.92, f"{когорта} · отобран: {r['причина']}", fontsize=10.5, color=ТЕКСТ)

    ax = fig.add_subplot(gs[1, 0:2])
    ax.axis("off")
    строки = [["вся ткань среза", число(v["TSR"]), "опухоль и строма на всём срезе"],
              ["перитуморальная полоса 250 мкм", число(v["TSR_бед"]),
               f"{в_полосе:.0f} % ткани: яркая часть панели зон"],
              ["поле с наибольшей стромой", число(v["TSR_поле_строма"]),
               "бирюзовый круг 1,6 мм; опухоли в поле не меньше 30 %"],
              ["поле с наибольшей опухолью", число(v["TSR_поле_опухоль"]),
               "оранжевый круг 1,6 мм"]]
    таблица = ax.table(cellText=строки, colLabels=["определение", "TSR", "что считается"],
                       cellLoc="left", colLoc="left", colWidths=[0.3, 0.09, 0.61],
                       bbox=[0, 0.34, 1, 0.66])
    таблица.auto_set_font_size(False)
    таблица.set_fontsize(9.5)
    for (i, _), ячейка in таблица.get_celld().items():
        ячейка.set_edgecolor("#dddddd")
        if i == 0:
            ячейка.set_text_props(weight="bold")
    исключён = r["исключён"]
    ax.text(0, 0.2, f"состав ткани: опухоль {запятая(r['опухоль_%'])} %, строма "
            f"{запятая(r['строма_%'])} %, прочее {запятая(r['прочее_%'])} %; площадь "
            f"{запятая(r['площадь_мм2'])} мм²; правило качества: "
            + ("проходит" if not исключён else f"не проходит ({исключён})"),
            transform=ax.transAxes, fontsize=9.5, color=ТЕКСТ, va="top")
    ax.text(0, 0.07, клиника, transform=ax.transAxes, fontsize=9.5, color=ТЕКСТ, va="top")

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.legend(handles=[Patch(color=np.array(ЦВЕТ_ОПУХОЛИ) / 255, label="опухоль"),
                       Patch(color=np.array(ЦВЕТ_СТРОМЫ) / 255, label="строма"),
                       Patch(color=np.array(ЦВЕТ_ПРОЧЕГО) / 255, label="прочее, в TSR не входит"),
                       Patch(color=бледно(np.array([[200, 120, 170]], np.uint8), 0.3)[0] / 255,
                             label="бледно: вне полосы 250 мкм"),
                       Line2D([], [], color=ПОЛЕ_СТРОМА, linewidth=2, label="поле по строме"),
                       Line2D([], [], color=ПОЛЕ_ОПУХОЛЬ, linewidth=2, label="поле по опухоли")],
              loc="upper left", frameon=False, fontsize=9.5)

    pdf.savefig(fig)
    fig.savefig(out / f"{номер:02d}_{r['slide'][:20]}.png", dpi=130, facecolor="white")
    plt.close(fig)
    return {"номер": номер, "когорта": когорта, "slide": r["slide"], "причина": r["причина"],
            **{k: v[k] for k in ОПРЕДЕЛЕНИЯ}, "в_полосе_%": round(в_полосе, 1),
            "опухоль_%": r["опухоль_%"], "строма_%": r["строма_%"], "прочее_%": r["прочее_%"],
            "исключён": исключён, "место": r.get("место", ""), "ответ": r.get("ответ", "")}


def найти(шаблоны):
    for ш in шаблоны:
        найдено = glob.glob(ш)
        if найдено:
            return найдено[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcga-dir", default="outputs/results/pan/tcga_ov_deploy5_raw")
    ap.add_argument("--tcga-svs", default="data/tcga_ov_flat")
    ap.add_argument("--ptrc-dir", default="outputs/results/pan/ptrc_deploy5_raw")
    ap.add_argument("--ptrc-svs", nargs="+", default=["data/ptrc/svs_1", "data/ptrc/svs_2"])
    ap.add_argument("--ptrc-slides", default="data/ptrc/slides.csv")
    ap.add_argument("--out", default="outputs/results/pan/atlas")
    ap.add_argument("--width", type=int, default=900, help="ширина панели в точках")
    ap.add_argument("--cell-mm", type=float, default=0.05)
    ap.add_argument("--dist-um", type=float, default=250.0)
    ap.add_argument("--radius-mm", type=float, default=1.6)
    ap.add_argument("--min-cover", type=float, default=0.3)
    ap.add_argument("--quantile", type=float, default=0.9)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    tcga = отбор_tcga(pd.read_csv(Path(args.tcga_dir) / "tsr_defs.csv"))
    slides = pd.read_csv(args.ptrc_slides, dtype={"образец": str, "пациент": str})
    ptrc = отбор_ptrc(pd.read_csv(Path(args.ptrc_dir) / "tsr_defs.csv"), slides)
    print(f"отобрано: TCGA-OV {len(tcga)}, PTRC-HGSOC {len(ptrc)}")

    указатель = []
    with PdfPages(out / "tsr_atlas.pdf") as pdf:
        номер = 0
        for когорта, выбор, карты, срезы in (("TCGA-OV", tcga, args.tcga_dir, [args.tcga_svs]),
                                             ("PTRC-HGSOC", ptrc, args.ptrc_dir, args.ptrc_svs)):
            for r in выбор:
                npz = найти([f"{карты}/{r['slide']}_seg_map.npz"])
                svs = найти([f"{d}/{r['slide']}*.svs" for d in срезы])
                if not npz or not svs:
                    print(f"  пропущен {r['slide']}: нет {'карты' if not npz else 'среза'}")
                    continue
                номер += 1
                try:
                    указатель.append(страница(pdf, out, номер, r, когорта, npz, svs, args))
                    print(f"  {номер:2d}. {r['slide'][:30]:30s} {r['причина']}")
                except Exception as e:
                    print(f"  ОШИБКА на {r['slide']}: {type(e).__name__}: {e}")
    pd.DataFrame(указатель).to_csv(out / "tsr_atlas_index.csv", index=False, encoding="utf-8-sig")
    print(f"\nатлас: {out / 'tsr_atlas.pdf'}, указатель: {out / 'tsr_atlas_index.csv'}")


if __name__ == "__main__":
    main()
