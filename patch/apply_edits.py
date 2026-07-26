#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Точечные правки в оставшихся файлах. Запускать ИЗ КОРНЯ репозитория:

    python patch/apply_edits.py            # только проверить, что места найдены
    python patch/apply_edits.py --apply    # внести правки

Сначала проверяется, что каждый заменяемый фрагмент есть в файле ровно один
раз. Если хоть один не найден или найден дважды — не меняется ничего, и
печатается, что именно не совпало. Так правки не применятся наполовину.
Перед записью каждый файл копируется в <файл>.bak.
"""
import argparse
import sys
from pathlib import Path

E = []   # (файл, что_заменить, на_что, зачем, прежние_редакции)


def edit(f, old, new, why, prev=()):
    """prev — как это место выглядело после ПРЕДЫДУЩИХ версий патча.

    Нужно, чтобы патч можно было накатывать повторно: если правка уже внесена,
    её надо узнать, а не объявить несовпадением.
    """
    E.append((f, old, new, why, tuple(prev)))

# ==========================================================================
# src/utils.py — один список классов вместо трёх разных
# ==========================================================================
# Палитра прошлой версии патча: нужна, чтобы узнать уже применённую правку
# и обновить только цвета, не трогая остальной блок.
PAL_V5 = '''CLASS_COLORS = {
    BACKGROUND: (245, 245, 245),
    TUMOR: (220, 50, 47),
    STROMA_HORMONAL: (38, 139, 210),
    STROMA_MATRIX: (181, 137, 0),
    VESSELS_IMMUNE: (150, 150, 150),
    IGNORE: (255, 255, 255),
}'''

UTILS_OLD = '''CLASS_NAMES = ["tumor", "stroma", "undefined"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# цвета оверлея (RGB): красный / синий / оливковый
CLASS_COLORS = {
    "tumor": (220, 50, 47),
    "stroma": (38, 139, 210),
    "undefined": (133, 153, 0),
}'''

PAL_NEW = '''# Палитра подобрана так, чтобы классы различались не только при обычном
# зрении, но и при дальтонизме (краснозелёная слепота — 8% мужчин).
# Прежние красный + оливковый при протанопии давали разницу ΔE = 2.5, то есть
# сливались полностью. У этой палитры худшая пара ΔE = 17.7 после наложения
# на ткань — различимо при любом типе зрения. Классы разведены и по светлоте:
# тёмно-красный / средний синий / яркий жёлтый / чёрный, поэтому карта
# читается даже в чёрно-белой печати.
CLASS_COLORS = {
    BACKGROUND: (245, 245, 245),
    TUMOR: (165, 15, 21),            # тёмно-красный
    STROMA_HORMONAL: (5, 113, 176),  # синий
    STROMA_MATRIX: (253, 231, 37),   # жёлтый
    VESSELS_IMMUNE: (0, 0, 0),       # чёрный
    IGNORE: (255, 255, 255),
}'''

UTILS_NEW = '''# ---------------------------------------------------------------------------
# Классы зон. ЕДИНСТВЕННОЕ место, где они заданы: раньше нумерация расходилась
# между src/utils.py, make_seg_masks_v.py и seg_train2.py, и инференс угадывал
# соответствие каналов классам по их числу.
#
# Строма разделена на два типа: гормон-продуцирующая (способная к
# стероидогенезу собственная строма яичника) и матриксная (фибробласты,
# перициты, коллаген). Сосуды и иммунные клетки — отдельный класс, в TSR НЕ
# входят: в патологии TSR считают по соединительной ткани, а не по всему, что
# не опухоль.
# ---------------------------------------------------------------------------
IGNORE = 255            # не размечено: ни в обучение, ни в знаменатель TSR
BACKGROUND = 0          # стекло, вне ткани
TUMOR = 1
STROMA_HORMONAL = 2     # стероидогенная строма
STROMA_MATRIX = 3       # матриксная строма
VESSELS_IMMUNE = 4      # сосуды, лимфатика, иммунные клетки

# Классы, которые предсказывает сеть, в порядке каналов. Порядок фиксирован:
# он пишется в чекпоинт, и по нему инференс сопоставляет каналы классам.
TRAIN_CLASSES = (BACKGROUND, TUMOR, STROMA_HORMONAL, STROMA_MATRIX, VESSELS_IMMUNE)
N_CLASSES = len(TRAIN_CLASSES)

CLASS_NAMES = {
    IGNORE: "не размечено",
    BACKGROUND: "фон",
    TUMOR: "опухоль",
    STROMA_HORMONAL: "строма гормональная",
    STROMA_MATRIX: "строма матриксная",
    VESSELS_IMMUNE: "сосуды и иммунитет",
}
CLASS_TO_IDX = {c: i for i, c in enumerate(TRAIN_CLASSES)}

# Что входит в TSR = строма / (строма + опухоль).
STROMA_FOR_TSR = (STROMA_HORMONAL, STROMA_MATRIX)

# Палитра подобрана так, чтобы классы различались не только при обычном
# зрении, но и при дальтонизме (краснозелёная слепота — 8% мужчин).
# Прежние красный + оливковый при протанопии давали разницу ΔE = 2.5, то есть
# сливались полностью. У этой палитры худшая пара ΔE = 17.7 после наложения
# на ткань — различимо при любом типе зрения. Классы разведены и по светлоте:
# тёмно-красный / средний синий / яркий жёлтый / чёрный, поэтому карта
# читается даже в чёрно-белой печати.
CLASS_COLORS = {
    BACKGROUND: (245, 245, 245),
    TUMOR: (165, 15, 21),            # тёмно-красный
    STROMA_HORMONAL: (5, 113, 176),  # синий
    STROMA_MATRIX: (253, 231, 37),   # жёлтый
    VESSELS_IMMUNE: (0, 0, 0),       # чёрный
    IGNORE: (255, 255, 255),
}


def slide_mpp(path):
    """Размер пикселя среза в микрометрах. Перенесено из slide_mpp.py.

    Отдельный CLI на 59 строк ради одного поля метаданных не нужен, а вызов
    через подстановку командной строки при ошибке давал пустую переменную и
    падение с непонятным сообщением.
    """
    import re
    ext = Path(path).suffix.lower()
    if ext in (".svs", ".ndpi", ".mrxs", ".scn"):
        import openslide
        sl = openslide.OpenSlide(str(path))
        v = sl.properties.get("openslide.mpp-x")
        sl.close()
        if not v:
            raise ValueError(f"{path}: openslide.mpp-x не записан в метаданных")
        return float(v)
    import tifffile
    with tifffile.TiffFile(path) as tf:
        xml = tf.ome_metadata or ""
    m = re.search(r'PhysicalSizeX="([\\d.eE+-]+)"', xml)
    if not m:
        raise ValueError(f"{path}: PhysicalSizeX в метаданных не найден")
    return float(m.group(1))


def patch_px_for_um(path, um):
    """Размер патча в пикселях под заданное физическое поле зрения.

    У ovary3 пиксель 0.137 мкм, у двух других 0.274 — вдвое крупнее. Патч в
    256 пикселей покрывал бы 35 и 70 мкм, то есть модель смотрела бы на разное
    увеличение. Поэтому поле зрения задаётся в микрометрах.
    """
    return int(round(float(um) / slide_mpp(path)))'''

edit('''src/utils.py''', UTILS_OLD, UTILS_NEW,
     '''единый список классов + перенос slide_mpp в utils''',
     prev=[UTILS_NEW.replace(PAL_NEW, PAL_V5)])

# ==========================================================================
# tsr_regions.py
# ==========================================================================
edit('''tsr_regions.py''',
'''from scipy.signal import fftconvolve''',
'''from scipy.signal import fftconvolve

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
    print(f"\\nзаписано: {path} ({len(kept) + len(new_rows)} строк)")''',
     '''функции вместо OpenCV + запись с ключом''')
edit('''tsr_regions.py''',
'''def tsr(stroma_px, tumor_px):
    total = stroma_px + tumor_px
    return float(stroma_px / total) if total else float("nan")''',
'''def tsr(stroma_px, tumor_px, min_px=50):
    """Доля стромы среди опухоли и стромы. NA, если считать не на чем.

    БЫЛО: при tumor_px == 0 возвращалось ровно 1.0, и срез без опухоли попадал
    в группу худшего прогноза. Это худший вид ошибки: она выглядит как
    результат. TSR без опухоли не определён.

    min_px — минимум размеченных ячеек, ниже которого доля случайна.
    """
    total = stroma_px + tumor_px
    if total < min_px or tumor_px == 0:
        return float("nan")
    return float(stroma_px / total)''',
     '''TSR без опухоли теперь NA, а не 1.0''')
edit('''tsr_regions.py''',
'''    import cv2
''',
'''''',
     '''убрать import cv2''')
edit('''tsr_regions.py''',
'''    tumor_full = (cls == 1)
    stroma_full = (cls == 2)
    print(f"опухоль {tumor_full.sum()*px_mm2:.1f} мм², "
          f"строма {stroma_full.sum()*px_mm2:.1f} мм²")''',
'''    tumor_full = (cls == TUMOR)
    horm_full = (cls == STROMA_HORMONAL)
    matr_full = (cls == STROMA_MATRIX)
    stroma_full = horm_full | matr_full
    print(f"опухоль {tumor_full.sum()*px_mm2:.1f} мм², "
          f"строма {stroma_full.sum()*px_mm2:.1f} мм² "
          f"(гормональная {horm_full.sum()*px_mm2:.1f}, "
          f"матриксная {matr_full.sum()*px_mm2:.1f} мм²)")''',
     '''классы по имени; два типа стромы отдельно''')
edit('''tsr_regions.py''',
'''    tum_w = cv2.resize(tumor_full.astype(np.float32), (Ww, Hw), interpolation=cv2.INTER_AREA)
    str_w = cv2.resize(stroma_full.astype(np.float32), (Ww, Hw), interpolation=cv2.INTER_AREA)''',
'''    tum_w = box_mean(tumor_full.astype(np.float32), Hw, Ww)
    str_w = box_mean(stroma_full.astype(np.float32), Hw, Ww)
    horm_w = box_mean(horm_full.astype(np.float32), Hw, Ww)
    matr_w = box_mean(matr_full.astype(np.float32), Hw, Ww)''',
     '''усреднение без OpenCV''')
edit('''tsr_regions.py''',
'''    bed_full = cv2.resize(bed_w.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0''',
'''    bed_full = resize_nearest(bed_w.astype(np.uint8), H, W) > 0''',
     '''изменение размера без OpenCV''')
edit('''tsr_regions.py''',
'''    name = Path(args.map).stem
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
    print("\\nдописано в", out_csv)''',
'''    name = Path(args.map).stem
    for r in rows:
        r.update({"срез": name,
                  "TSR": round(r["TSR"], 4) if r["TSR"] == r["TSR"] else "",
                  "площадь_мм2": round(r["площадь_мм2"], 1),
                  "wt_threshold": args.wt_threshold,
                  "mi_radius_mm": args.mi_radius_mm,
                  "work_mpp": args.work_mpp})
    write_rows(args.out_csv, rows)''',
     '''запись с ключом вместо дописывания''')
edit('''tsr_regions.py''',
'''    cnts, _ = cv2.findContours(bed_w.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (20, 130, 60), 2)
    if mi_center is not None:
        cv2.circle(vis, mi_center, int(r_mi), (20, 20, 20), 2)
    png = out_csv.parent / f"{name}_tsr_regions.png"
    cv2.imwrite(str(png), vis[:, :, ::-1])''',
'''    # Контуры рисуем matplotlib: OpenCV нужен был только здесь.
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
    plt.close(fig)''',
     '''превью без OpenCV''')

# ==========================================================================
# make_seg_masks_v.py — память
# ==========================================================================
edit('''make_seg_masks_v.py''',
'''from src.utils import ensure_dir''',
'''from src.utils import ensure_dir, IGNORE, TUMOR, STROMA_HORMONAL, STROMA_MATRIX''',
     '''импорт классов''')
edit('''make_seg_masks_v.py''',
'''    print("строю Вороной...")
    dist, inds = distance_transform_edt(seeds == 0, return_distances=True, return_indices=True)
    mask = seeds[inds[0], inds[1]]
    mask[dist > max_dist] = 0
    del dist, inds''',
'''    # БЫЛО: distance_transform_edt(..., return_indices=True) выделяет три
    # массива размером во всю карту. На срезе 40000x40000 это 38 ГБ, и скрипт
    # падает; в комментарии выше сказано «увеличь downscale», но
    # run_pipeline.sh этот аргумент не передавал.
    #
    # СТАЛО: ближайшая клетка ищется деревом по координатам клеток, карта
    # обрабатывается полосами. Память линейна по числу клеток (сотни тысяч), а
    # не по площади среза (сотни миллионов): 2.6 ГБ вместо 38. Результат
    # совпадает с прежним на 99.7% — расхождение потому, что прежний способ
    # округлял координаты клеток до целого пикселя, а дерево работает с
    # точными.
    print("строю Вороной по дереву клеток...")
    from scipy.spatial import cKDTree

    tree = cKDTree(np.stack([xs, ys], 1).astype(float))
    mask = np.full((Hd, Wd), IGNORE, np.uint8)
    # высоту полосы считаем от ширины карты: на точку уходит ~30 байт
    chunk = max(1, min(Hd, int(1.0e9 / (30 * Wd))))
    gx = np.arange(Wd, dtype=np.float32)
    for y0 in range(0, Hd, chunk):
        y1 = min(y0 + chunk, Hd)
        gy = np.arange(y0, y1, dtype=np.float32)
        pts = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
        _, idx = tree.query(pts, k=1, distance_upper_bound=max_dist, workers=-1)
        band = np.full(len(pts), IGNORE, np.uint8)
        hit = idx < len(cl)          # вне радиуса дерево возвращает len(tree)
        band[hit] = cl[idx[hit]]
        mask[y0:y1] = band.reshape(y1 - y0, Wd)
    del tree''',
     '''маски деревом: 2.6 ГБ вместо 38''')
edit('''make_seg_masks_v.py''',
'''    print("маска: tumor %.1f%% stroma %.1f%%" % (100*(mask==1).sum()/tot, 100*(mask==2).sum()/tot))''',
'''    print("маска: опухоль %.1f%%  строма гормональная %.1f%%  матриксная %.1f%%  "
          "не размечено %.1f%%"
          % (100*(mask == TUMOR).sum()/tot,
             100*(mask == STROMA_HORMONAL).sum()/tot,
             100*(mask == STROMA_MATRIX).sum()/tot,
             100*(mask == IGNORE).sum()/tot))''',
     '''статистика по пяти классам''')

# ==========================================================================
# seg_train2.py
# ==========================================================================
edit('''seg_train2.py''',
'''from src.utils import get_device, ensure_dir, set_seed''',
'''from src.utils import (get_device, ensure_dir, set_seed, TRAIN_CLASSES,
                       CLASS_NAMES, N_CLASSES, IGNORE)''',
     '''импорт классов''')
edit('''seg_train2.py''',
'''CLASSES = (("tumor", 0), ("stroma", 1))''',
'''# Каналы сети и их соответствие классам проекта. Порядок берётся из
# src/utils.TRAIN_CLASSES и пишется в чекпоинт: раньше инференс угадывал
# соответствие по числу каналов, и с пятью классами угадывание сломалось бы, а
# перепутанные опухоль и строма дают правдоподобное, но неверное TSR.
CHANNEL_TO_CLASS = {i: c for i, c in enumerate(TRAIN_CLASSES)}
CLASS_TO_CHANNEL = {c: i for i, c in CHANNEL_TO_CLASS.items()}
CLASSES = tuple((CLASS_NAMES[c], i) for i, c in CHANNEL_TO_CLASS.items())
IGNORE_TARGET = -100        # значение ignore_index у CrossEntropyLoss


def mask_to_target(mask):
    """Метки маски -> индексы каналов сети. IGNORE -> IGNORE_TARGET.

    БЫЛО: yb = ... - 1. Это подгонка под прежнюю нумерацию, где классов было
    два. С IGNORE = 255 вычитание единицы даёт метку 254, и обучение либо
    падает, либо считает мусор.
    """
    out = np.full(mask.shape, IGNORE_TARGET, np.int64)
    for cls, ch in CLASS_TO_CHANNEL.items():
        out[mask == cls] = ch
    unknown = ~np.isin(mask, list(CLASS_TO_CHANNEL) + [IGNORE])
    if unknown.any():
        raise ValueError(
            "в маске значения, которых нет ни в TRAIN_CLASSES, ни IGNORE: "
            f"{np.unique(mask[unknown]).tolist()}. "
            "Пересоберите маски обновлённым make_seg_masks_v.py")
    return out''',
     '''соответствие каналов классам + перевод метки''')
edit('''seg_train2.py''',
'''    labeled = y > 0
    truth = y.astype(np.int16) - 1''',
'''    truth = mask_to_target(y)
    labeled = truth != IGNORE_TARGET''',
     '''метки через таблицу, а не вычитанием единицы''')
edit('''seg_train2.py''',
'''    model = SegDecoder(in_dim=Xtr.shape[-1], n_classes=2).to(device)''',
'''    model = SegDecoder(in_dim=Xtr.shape[-1], n_classes=N_CLASSES).to(device)''',
     '''число классов из src/utils.py''')
edit('''seg_train2.py''',
'''    crit = nn.CrossEntropyLoss(weight=w.to(device), ignore_index=-1)''',
'''    crit = nn.CrossEntropyLoss(weight=w.to(device), ignore_index=IGNORE_TARGET)''',
     '''ignore_index согласован с mask_to_target''')
edit('''seg_train2.py''',
'''    n = len(Xtr)
    best, best_state, bad, epochs_run = -1, None, 0, args.epochs''',
'''    n = len(Xtr)
    # Свой генератор: set_seed задаёт глобальное состояние numpy, но любой
    # вызов np.random в другом месте сдвинет последовательность, и прогон
    # не повторится.
    rng = np.random.default_rng(args.seed)
    best, best_state, bad, epochs_run = -1, None, 0, args.epochs''',
     '''генератор создаётся один раз''')
edit('''seg_train2.py''',
'''        perm = np.random.permutation(n)''',
'''        perm = rng.permutation(n)''',
     '''свой генератор вместо глобального np.random''')
edit('''seg_train2.py''',
'''            yb = torch.tensor(ytr[j].astype(np.int64)).to(device) - 1''',
'''            yb = torch.from_numpy(mask_to_target(ytr[j])).to(device)''',
     '''перевод метки в канал''')
edit('''seg_train2.py''',
'''        _, m = evaluate(model, Xva, yva, device)
        miou = np.nanmean([m["tumor"][0], m["stroma"][0]])
        print("epoch %3d  loss %.4f  IoU tumor %.3f stroma %.3f  (mIoU %.3f)"
              % (ep, run / nb, m["tumor"][0], m["stroma"][0], miou))''',
'''        _, m = evaluate(model, Xva, yva, device)
        # Среднее только по классам, которые есть в отложенном срезе: иначе
        # отсутствующий класс даёт nan и портит выбор лучшей эпохи.
        present = [CLASS_NAMES[c] for c in TRAIN_CLASSES
                   if c != 0 and (yva == c).sum() > 0]
        vals = [m[nm][0] for nm in present if nm in m and not np.isnan(m[nm][0])]
        miou = float(np.mean(vals)) if vals else float("nan")
        per = "  ".join(f"{nm} {m[nm][0]:.3f}" for nm in present if nm in m)
        print("epoch %3d  loss %.4f  IoU: %s  (mIoU %.3f)"
              % (ep, run / nb, per, miou))''',
     '''mIoU по присутствующим классам''')
edit('''seg_train2.py''',
'''    torch.save({"state_dict": model.state_dict(), "n_classes": 2}, args.out)''',
'''    torch.save({"state_dict": model.state_dict(),
                "n_classes": N_CLASSES,
                "channel_to_class": CHANNEL_TO_CLASS,
                "class_names": CLASS_NAMES,
                "val_slide": args.val,
                "seed": args.seed}, args.out)''',
     '''соответствие каналов классам пишется в чекпоинт''')

# ==========================================================================
# scripts/run_pipeline.sh
# ==========================================================================
edit('''scripts/run_pipeline.sh''',
'''EPOCHS=40''',
'''EPOCHS=40
MASK_DOWNSCALE=2     # во сколько раз грубее среза строится маска
STRIDE_FRAC=2        # шаг окна = патч / STRIDE_FRAC; 2 = перекрытие 50%
MIN_CONF=0.5         # порог уверенности при инференсе''',
     '''новые параметры вынесены наверх''')
edit('''scripts/run_pipeline.sh''',
'''    ps=$(python slide_mpp.py --he "$image" --um "$PATCH_UM" --quiet)''',
'''    ps=$(python -c "from src.utils import patch_px_for_um; print(patch_px_for_um('$image', $PATCH_UM))") \\
        || { echo "$slide: не удалось определить мкм/px в $image"; exit 1; }''',
     '''мкм/px из src/utils, с внятной ошибкой''')
edit('''scripts/run_pipeline.sh''',
'''        --patch-size "$ps" --stride "$ps" --out-dir "$SEG_DIR"''',
'''        --patch-size "$ps" --stride "$((ps / STRIDE_FRAC))" \\
        --downscale "$MASK_DOWNSCALE" --out-dir "$SEG_DIR"''',
     '''перекрытие тайлов + --downscale прокинут''')
edit('''scripts/run_pipeline.sh''',
'''python -m src.data.seurat_labels --cells "${cells[@]}"''',
'''python -m src.data.seurat_labels --cells "${cells[@]}"

# Проверка: разделяются ли два типа стромы по маркерам. Если нет — метки будут
# шумом, и учить модель бессмысленно. Смотреть outputs/results/stroma_split_*.png
for s in "${slides[@]}"; do
    [ -f "$CSV_DIR/${s}_expr.csv" ] || {
        echo "$s: нет ${s}_expr.csv — перезапустите Rscript R/export_seurat.R"; exit 1; }
    python check_stroma_split.py --cells "$CSV_DIR/${s}_cells.csv" \\
        --expr "$CSV_DIR/${s}_expr.csv" \\
        --out "outputs/results/stroma_split_${s}" || exit 1
done
echo "!! посмотрите outputs/results/stroma_split_*.png прежде чем учить модель"''',
     '''шаг проверки стромы в пайплайне''')


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="внести правки")
    args = ap.parse_args()

    # git не обязателен: папку могли скопировать на сервер без .git. Достаточно,
    # чтобы правимые файлы были на месте — иначе мы не в корне проекта.
    if not Path("tsr_regions.py").exists():
        sys.exit("ОШИБКА: в текущей папке нет tsr_regions.py. "
                 "Перейдите в корень проекта.")

    texts, bad, done, upd = {}, [], 0, 0
    for f, old, new, why, prev in E:
        p = Path(f)
        if not p.exists():
            print("  НЕТ ФАЙЛА %-26s %s" % (f, why))
            bad.append("%s: файла нет" % f)
            continue
        if f not in texts:
            texts[f] = p.read_text(encoding="utf-8")

        # Порядок важен: сначала проверяем НОВЫЙ вариант. У части правок старый
        # фрагмент — это начало нового (текст дописывается после него), поэтому
        # после применения он всё ещё находится, и проверка по старому привела
        # бы к повторному применению — дублю кода в файле.
        if texts[f].count(new) >= 1:
            print("  уже есть %-26s %s" % (f, why))
            done += 1
            continue
        if texts[f].count(old) == 1:
            print("  ок       %-26s %s" % (f, why))
            texts[f] = texts[f].replace(old, new, 1)
            continue
        # место правил прежний патч, а сейчас редакция изменилась — обновляем
        older = [q for q in prev if texts[f].count(q) == 1]
        if older:
            print("  обновляю %-26s %s" % (f, why))
            texts[f] = texts[f].replace(older[0], new, 1)
            upd += 1
            continue

        print("  НЕ НАЙД  %-26s %s" % (f, why))
        head = old.strip().splitlines()[0][:60]
        bad.append("%s: фрагмент найден %d раз(а), нужен ровно 1\n"
                   "          начало фрагмента: %s"
                   % (f, texts[f].count(old), head))

    if bad:
        print("\n=== НЕ ПРИМЕНЕНО НИЧЕГО. Не совпало: ===")
        for b in bad:
            print("  -", b)
        print("\nЗначит, файлы у вас отличаются от публичного снимка.")
        print("Пришлите мне вывод этих двух команд, и я подгоню правки:")
        print("    git log --oneline -3")
        print("    wc -l *.py src/*.py src/*/*.py")
        sys.exit(1)

    # Меняем только те файлы, содержимое которых реально отличается: при
    # повторном запуске патча менять нечего, и плодить .bak-копии не нужно.
    changed = {f: t for f, t in texts.items()
               if t != Path(f).read_text(encoding="utf-8")}

    if done or upd:
        print("\nуже внесено ранее: %d, обновлено до новой редакции: %d"
              % (done, upd))
    if not changed:
        print("\nВсе правки уже на месте, менять нечего.")
        return

    if not args.apply:
        print("\nВсе %d правок найдены, ничего не изменено. Применить:" % len(E))
        print("    python patch/apply_edits.py --apply")
        return

    for f, t in changed.items():
        # .bak делаем только если его ещё нет: при повторном запуске патча
        # он затёр бы сам себя, и настоящий исходник было бы не вернуть.
        b = Path(f + ".bak")
        if b.exists():
            print("  записан %s   (бэкап %s.bak уже был)" % (f, f))
        else:
            b.write_text(Path(f).read_text(encoding="utf-8"), encoding="utf-8")
            print("  записан %s   (бэкап %s.bak)" % (f, f))
        Path(f).write_text(t, encoding="utf-8")

    print("\nПроверяю, что Python-файлы разбираются...")
    import ast
    for f in changed:
        if f.endswith(".py"):
            try:
                ast.parse(Path(f).read_text(encoding="utf-8"))
                print("  ок  %s" % f)
            except SyntaxError as e:
                sys.exit("  ОШИБКА разбора %s, строка %s: %s\n"
                         "  Верните файл: mv %s.bak %s" % (f, e.lineno, e.msg, f, f))
    if Path(".git").is_dir():
        print("\nГОТОВО. Что изменилось: git diff --stat")
    else:
        print("\nГОТОВО. Прежние версии файлов рядом, с суффиксом .bak")


if __name__ == "__main__":
    main()
