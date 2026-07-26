#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Текстовое описание среза по карте зон.

    python describe_slide.py --map outputs/results/ovary3_seg_map.npz
    python describe_slide.py --map ...npz --json outputs/results/ovary3_desc.json

На входе .npz, который пишет seg_infer.py: массив cls с номерами классов и
мкм/px. Ничего не обучается и не запускается заново — только счёт по карте,
поэтому работает за секунды и не требует ни GPU, ни весов модели.

На выходе связный текст и, если попросить, те же числа в JSON. Текст собран из
шаблонов: каждое утверждение опирается на посчитанное число, а порог, по
которому выбрана формулировка, назван прямо в тексте. Так проверяемо, откуда
взялось слово «очаговая» или «преобладает».

Чего скрипт НЕ делает: не ставит диагноз, не сравнивает с нормой, не даёт
прогноза. Это описание того, что на карте, — не заключение.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils import (BACKGROUND, IGNORE, TUMOR, STROMA_HORMONAL,  # noqa: E402
                       STROMA_MATRIX, VESSELS_IMMUNE, CLASS_NAMES,
                       STROMA_FOR_TSR)

# Пороги. Собраны в одном месте, чтобы их было видно и можно было менять,
# не разыскивая по коду. Ни один из них не «стандарт из литературы» — это
# границы словесных формулировок, а не клинические отсечки.
MIN_ISLAND_MM2 = 0.05     # мельче — считаем шумом сегментации, не очагом
SOLID_FRAC = 0.70         # доля опухоли в наибольшем очаге -> «солидная»
FOCAL_FRAC = 0.30         # ниже -> «диффузная/рассыпная»
DOMINANT = 0.65           # доля одного типа стромы -> «преобладает»
BALANCED = 0.55           # ниже -> «сопоставимы»
NEAR_UM = 50.0            # ближе этого к опухоли -> «перитуморальная»
UNLABELED_WARN = 0.10     # доля не размеченного, выше которой предупреждаем
GRADIENT_RATIO = 1.3      # во сколько раз доля должна отличаться центр/край


def load_map(path):
    d = np.load(path)
    cls = d["cls"]
    mpp = float(d["mpp"])
    meta = {k: d[k].item() if d[k].ndim == 0 else d[k].tolist()
            for k in d.files if k not in ("cls",)}
    return cls, mpp, meta


def composition(cls, px_mm2):
    """Площади классов и доли от размеченной ткани.

    Знаменатель — ткань БЕЗ фона и БЕЗ не размеченного. Считать долю от всего
    кадра нельзя: она зависит от того, сколько стекла попало в скан.
    """
    tissue = np.isin(cls, (TUMOR, STROMA_HORMONAL, STROMA_MATRIX,
                           VESSELS_IMMUNE))
    n_tissue = int(tissue.sum())
    out = {}
    for c in (TUMOR, STROMA_HORMONAL, STROMA_MATRIX, VESSELS_IMMUNE):
        n = int((cls == c).sum())
        out[CLASS_NAMES[c]] = {
            "площадь_мм2": round(n * px_mm2, 3),
            "доля_ткани": round(n / n_tissue, 4) if n_tissue else None,
        }
    return out, tissue, n_tissue


def tsr_value(cls, min_px=50):
    """Доля стромы среди опухоль+строма. Сосуды и иммунитет не в знаменателе.

    Возвращает None вместо числа, когда считать нечего: без опухоли отношение
    равно единице механически, а не потому что срез весь стромальный.
    """
    n_t = int((cls == TUMOR).sum())
    n_s = int(np.isin(cls, STROMA_FOR_TSR).sum())
    if n_t + n_s < min_px or n_t == 0:
        return None
    return n_s / (n_t + n_s)


def islands(cls, px_mm2, min_mm2=MIN_ISLAND_MM2):
    """Опухолевые очаги как связные области. Мелочь отбрасываем.

    Связность — по 8 соседям: диагональный контакт двух пикселей опухоли это
    один очаг, а не два. При 4-связности тонкая косая тяжа рассыпалась бы на
    десятки «очагов».
    """
    lab, n = ndi.label(cls == TUMOR, structure=np.ones((3, 3), bool))
    if n == 0:
        return {"очагов": 0, "площади_мм2": [], "доля_в_наибольшем": None}
    sizes = np.bincount(lab.ravel())[1:] * px_mm2
    keep = np.sort(sizes[sizes >= min_mm2])[::-1]
    if keep.size == 0:
        return {"очагов": 0, "площади_мм2": [], "доля_в_наибольшем": None,
                "отброшено_мелких": int(n)}
    return {
        "очагов": int(keep.size),
        "отброшено_мелких": int(n - keep.size),
        "наибольший_мм2": round(float(keep[0]), 3),
        "медиана_мм2": round(float(np.median(keep)), 3),
        "доля_в_наибольшем": round(float(keep[0] / keep.sum()), 3),
        "площади_мм2": [round(float(v), 3) for v in keep[:10]],
    }


def stroma_geometry(cls, mpp, near_um=NEAR_UM):
    """Насколько каждый тип стромы прилегает к опухоли.

    distance_transform_edt считает расстояние до ближайшего нуля, поэтому
    подаём инверсию маски опухоли: на выходе в каждом пикселе — расстояние до
    опухоли в пикселях. Умножаем на мкм/px.

    Это отвечает на вопрос, который для гормональной стромы содержательный:
    лежит она вплотную к опухоли (перитуморальная реакция) или отдельными
    полями вдали от неё (сохранившаяся строма яичника).
    """
    tum = cls == TUMOR
    if not tum.any():
        return {}
    dist_um = ndi.distance_transform_edt(~tum) * mpp
    out = {}
    for c in STROMA_FOR_TSR:
        m = cls == c
        if not m.any():
            out[CLASS_NAMES[c]] = None
            continue
        d = dist_um[m]
        out[CLASS_NAMES[c]] = {
            "медиана_расстояния_мкм": round(float(np.median(d)), 1),
            "доля_ближе_%d_мкм" % int(near_um): round(float((d < near_um).mean()), 3),
        }
    return out


def center_periphery(cls, tissue):
    """Доля опухоли во внутренней половине ткани против внешней.

    «Внутренняя» — не геометрический центр кадра: срез бывает подковой или
    несколькими кусками. Берём расстояние до края ткани и делим по медиане,
    так что обе половины по площади равны и сравнивать доли корректно.
    """
    if not tissue.any():
        return None
    d = ndi.distance_transform_edt(tissue)
    dv = d[tissue]
    thr = float(np.median(dv))
    if thr <= 0:
        return None
    inner = tissue & (d > thr)
    outer = tissue & (d <= thr)
    if not inner.any() or not outer.any():
        return None
    fi = float((cls[inner] == TUMOR).mean())
    fo = float((cls[outer] == TUMOR).mean())
    return {"доля_опухоли_внутри": round(fi, 3),
            "доля_опухоли_снаружи": round(fo, 3),
            "порог_глубины_px": round(thr, 1)}


def pct(x):
    return "—" if x is None else "%.0f %%" % (100 * x)


def describe(cls, mpp, meta, name):
    """Собрать числа и текст. Возвращает (словарь_чисел, текст)."""
    H, W = cls.shape
    px_mm2 = (mpp / 1000.0) ** 2
    comp, tissue, n_tissue = composition(cls, px_mm2)
    n_unlab = int((cls == IGNORE).sum())
    n_frame = H * W
    tsr = tsr_value(cls)
    isl = islands(cls, px_mm2)
    geo = stroma_geometry(cls, mpp)
    cp = center_periphery(cls, tissue)

    hor = comp[CLASS_NAMES[STROMA_HORMONAL]]["площадь_мм2"]
    mat = comp[CLASS_NAMES[STROMA_MATRIX]]["площадь_мм2"]
    stroma_tot = hor + mat
    hor_frac = hor / stroma_tot if stroma_tot else None

    nums = {
        "срез": name,
        "мкм_на_px": round(mpp, 3),
        "размер_мм": [round(W * mpp / 1000, 1), round(H * mpp / 1000, 1)],
        "площадь_ткани_мм2": round(n_tissue * px_mm2, 2),
        "доля_не_размечено": round(n_unlab / n_frame, 4),
        "состав": comp,
        "TSR": None if tsr is None else round(tsr, 3),
        "очаги_опухоли": isl,
        "строма_относительно_опухоли": geo,
        "центр_периферия": cp,
        "доля_гормональной_среди_стромы": None if hor_frac is None else round(hor_frac, 3),
        "параметры_прогона": meta,
    }

    L = []
    A = L.append
    A("Срез %s" % name)
    A("=" * (6 + len(name)))
    A("")
    A("Карта %d x %d при %.2f мкм/px, это %.1f x %.1f мм. Ткани %.1f мм²."
      % (W, H, mpp, W * mpp / 1000, H * mpp / 1000, n_tissue * px_mm2))

    if n_unlab / n_frame > UNLABELED_WARN:
        A("ВНИМАНИЕ: %s кадра модель не разметила — уверенность ниже порога. "
          "Всё, что ниже, посчитано по оставшейся части, и доли могут быть "
          "смещены." % pct(n_unlab / n_frame))

    A("")
    A("Состав ткани")
    A("------------")
    for k, v in comp.items():
        A("  %-22s %7.2f мм²   %s" % (k, v["площадь_мм2"], pct(v["доля_ткани"])))
    A("  (доли от площади ткани; фон и не размеченное в знаменатель не входят)")

    A("")
    A("Соотношение опухоль/строма")
    A("--------------------------")
    if tsr is None:
        A("  TSR не определён: опухоли на срезе не найдено. Отношение в этом "
          "случае равнялось бы единице механически и смысла не имеет.")
    else:
        A("  TSR = %.2f — строма занимает %s площади «опухоль + строма»."
          % (tsr, pct(tsr)))
        A("  Сосуды и иммунные клетки в знаменатель не входят: это отдельная "
          "зона, а не строма.")

    A("")
    A("Как расположена опухоль")
    A("-----------------------")
    if isl["очагов"] == 0:
        A("  Очагов опухоли крупнее %.2f мм² нет." % MIN_ISLAND_MM2)
    else:
        A("  Очагов: %d (крупнее %.2f мм²; мельче отброшено: %d)."
          % (isl["очагов"], MIN_ISLAND_MM2, isl.get("отброшено_мелких", 0)))
        A("  Наибольший %.2f мм², медианный %.2f мм². В наибольшем %s всей "
          "опухолевой площади."
          % (isl["наибольший_мм2"], isl["медиана_мм2"],
             pct(isl["доля_в_наибольшем"])))
        f = isl["доля_в_наибольшем"]
        if f >= SOLID_FRAC:
            A("  Рост солидный: один массив держит больше %s опухоли (порог %s)."
              % (pct(SOLID_FRAC), pct(SOLID_FRAC)))
        elif f >= FOCAL_FRAC:
            A("  Рост очаговый: несколько сопоставимых узлов, ни один не "
              "доминирует (доля наибольшего между %s и %s)."
              % (pct(FOCAL_FRAC), pct(SOLID_FRAC)))
        else:
            A("  Рост рассыпной: опухоль раздроблена на множество мелких "
              "очагов (доля наибольшего ниже %s)." % pct(FOCAL_FRAC))
    if cp:
        r = cp["доля_опухоли_внутри"] / max(cp["доля_опухоли_снаружи"], 1e-9)
        if r >= GRADIENT_RATIO:
            A("  Опухоль смещена вглубь: %s внутренней половины ткани против "
              "%s наружной." % (pct(cp["доля_опухоли_внутри"]),
                                pct(cp["доля_опухоли_снаружи"])))
        elif r <= 1 / GRADIENT_RATIO:
            A("  Опухоль смещена к краю: %s наружной половины против %s "
              "внутренней." % (pct(cp["доля_опухоли_снаружи"]),
                               pct(cp["доля_опухоли_внутри"])))
        else:
            A("  По глубине распределена ровно: %s внутри, %s снаружи "
              "(различие меньше чем в %.1f раза)."
              % (pct(cp["доля_опухоли_внутри"]),
                 pct(cp["доля_опухоли_снаружи"]), GRADIENT_RATIO))

    A("")
    A("Строма: два типа")
    A("----------------")
    if stroma_tot == 0:
        A("  Стромы на срезе не найдено.")
    else:
        A("  Гормональная %.2f мм² (%s стромы), матриксная %.2f мм² (%s)."
          % (hor, pct(hor_frac), mat, pct(1 - hor_frac)))
        if hor_frac >= DOMINANT:
            A("  Преобладает гормональная (порог %s)." % pct(DOMINANT))
        elif hor_frac <= 1 - DOMINANT:
            A("  Преобладает матриксная (порог %s)." % pct(DOMINANT))
        elif abs(hor_frac - 0.5) <= BALANCED - 0.5:
            A("  Типы сопоставимы по площади.")
        else:
            A("  Небольшой перевес %s."
              % ("гормональной" if hor_frac > 0.5 else "матриксной"))
        for c in STROMA_FOR_TSR:
            g = geo.get(CLASS_NAMES[c])
            if not g:
                continue
            key = "доля_ближе_%d_мкм" % int(NEAR_UM)
            A("  %-22s медиана до опухоли %6.0f мкм, вплотную (<%d мкм) %s"
              % (CLASS_NAMES[c], g["медиана_расстояния_мкм"], int(NEAR_UM),
                 pct(g[key])))
        gh = geo.get(CLASS_NAMES[STROMA_HORMONAL])
        gm = geo.get(CLASS_NAMES[STROMA_MATRIX])
        if gh and gm:
            dh = gh["медиана_расстояния_мкм"]
            dm = gm["медиана_расстояния_мкм"]
            if dh > dm * GRADIENT_RATIO:
                A("  Гормональная строма лежит дальше от опухоли, чем "
                  "матриксная (%.0f против %.0f мкм) — картина сохранившейся "
                  "стромы яичника, оттеснённой опухолью." % (dh, dm))
            elif dm > dh * GRADIENT_RATIO:
                A("  Гормональная строма прилегает к опухоли ближе, чем "
                  "матриксная (%.0f против %.0f мкм)." % (dh, dm))
            else:
                A("  Оба типа стромы удалены от опухоли одинаково "
                  "(%.0f и %.0f мкм)." % (dh, dm))

    A("")
    A("Оговорки")
    A("--------")
    A("  Это описание карты зон, а не заключение по препарату. Метки зон "
      "получены моделью, обученной на срезах с транскриптомикой; на новом "
      "материале её точность не проверена.")
    if meta.get("stain_norm") in (None, "none"):
        A("  Нормализация окраски не применялась. Для срезов из другой "
          "лаборатории доли зон могут быть систематически смещены.")
    A("  Пороги словесных формулировок заданы в начале файла describe_slide.py "
      "и клиническими отсечками не являются.")

    return nums, "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--map", required=True, help="npz от seg_infer.py")
    ap.add_argument("--json", help="куда сложить числа (по умолчанию рядом)")
    ap.add_argument("--txt", help="куда сложить текст (по умолчанию рядом)")
    ap.add_argument("--quiet", action="store_true", help="не печатать текст")
    args = ap.parse_args()

    cls, mpp, meta = load_map(args.map)
    name = Path(args.map).stem.replace("_seg_map", "")
    nums, text = describe(cls, mpp, meta, name)

    txt = Path(args.txt or Path(args.map).with_name(name + "_описание.txt"))
    txt.write_text(text + "\n", encoding="utf-8")
    js = Path(args.json or Path(args.map).with_name(name + "_описание.json"))
    js.write_text(json.dumps(nums, ensure_ascii=False, indent=2),
                  encoding="utf-8")

    if not args.quiet:
        print(text)
    print("\nтекст: %s\nчисла: %s" % (txt, js))


if __name__ == "__main__":
    main()
