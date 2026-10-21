# Подбор срезов TCGA-OV для сверки с патологом и сборка пакета для передачи.
#
# Протокол (protokol_patolog.md): в первый заход патолог оценивает 25 срезов, во второй,
# через неделю, ещё раз 10 из них вперемешку с 5 новыми. Разметку модели он не видит.
#
# Подбор идёт по доле стромы в поле с наибольшей стромой (TSR_поле_строма): из четырёх
# определений это ближайшее к методике патолога. Срезы делятся на пять равных по числу
# слоёв, от стромально бедных до богатых, и из каждого слоя случайно, с фиксированным
# зерном, берутся шесть: пять в первый заход и один новый во второй. Повторы второго
# захода тоже по два из слоя, так что и повторы покрывают весь диапазон. То, что подбор
# сделан по оценке модели, оговаривается в тексте.
#
# В отбор не входят срезы, не прошедшие правило качества (опухоль и строма вместе меньше
# половины ткани или опухоли меньше 5 %), и вторые препараты одной пациентки: берётся
# препарат с наибольшей площадью ткани, как в анализе выживаемости.
#
# Срезы не копируются: в пакет кладутся жёсткие ссылки, названные кодами вместо имён
# TCGA, а во втором заходе у тех же срезов другие коды, чтобы повторы не узнавались по
# имени. Ключ «код и срез» лежит рядом с пакетом и патологу не передаётся. Повторы второго
# захода в пакет не кладутся: они уже есть у патолога после первого, и качать их второй
# раз незачем; папку второго захода соберём из первого по ключу. Имена папок и файлов
# латиницей: OpenSlide под Windows не всегда открывает пути с кириллицей.
#
#   python patho_select.py

import argparse
import os
import random
from pathlib import Path

import pandas as pd

СЛОЁВ = 5
В_ПЕРВЫЙ = 5      # срезов из слоя в первый заход
НОВЫХ_ВО_ВТОРОЙ = 1
ПОВТОРОВ = 2      # повторов из слоя во второй заход
# без символов, которые легко спутать: O и 0, I и 1
АЛФАВИТ = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

ИНСТРУКЦИЯ = """\
Оценка доли стромы на срезах рака яичника

Спасибо, что согласились помочь. Работа занимает около часа.

1. Программа QuPath 0.5 или новее, бесплатная: https://qupath.github.io
2. Создайте проект: меню File, Project, Create project, выберите пустую папку.
   Перетащите в окно QuPath все файлы .svs из папки visit1 и нажмите Import.
   Имена файлов это коды, переименовывать их не нужно.
3. На каждом срезе выберите поле для оценки по своей обычной методике: в зоне
   инвазивного края, внутри опухолевой массы, участок с наибольшей долей стромы,
   примерно поле объектива x10.
4. Отметьте это поле одной аннотацией, инструментом Ellipse или Rectangle на панели
   сверху. Одна аннотация на срез.
5. Запишите долю стромы в этом поле в бланк visit1_form.csv, в процентах, с шагом 10 %.
   Некроз, слизь, жир, кровь и нормальная ткань в оценку не входят, как обычно.
6. Если срез непригоден для оценки, напишите в бланке «нет» и причину, аннотацию можно
   не ставить. Это тоже полезный результат.
7. Сохраните проект (File, Save) и верните папку проекта и заполненный бланк.
   Файлы .svs возвращать не нужно.

Разметку программы на этих срезах мы специально не показываем: важно, чтобы оценка была
независимой.
"""

ВЫГРУЗКА = """\
// Выгрузка полей, отмеченных патологом: по файлу GeoJSON на срез в папку fields проекта.
// Координаты в пикселях исходного среза. Запуск: Automate, Script editor, открыть этот
// файл, затем Run, Run for project, выбрать все срезы.
def name = getProjectEntry().getImageName()
if (name.toLowerCase().endsWith(".svs"))
    name = name.substring(0, name.length() - 4)
def dir = buildFilePath(PROJECT_BASE_DIR, "fields")
mkdirs(dir)
def objects = getAnnotationObjects()
exportObjectsToGeoJson(objects, buildFilePath(dir, name + ".geojson"), "FEATURE_COLLECTION")
print name + ": полей " + objects.size()
"""


def годен(t):
    return ((t["опухоль_%"] + t["строма_%"]) >= 50) & (t["опухоль_%"] >= 5)


def новые_коды(n, rng, занятые):
    out = []
    while len(out) < n:
        код = "".join(rng.choice(АЛФАВИТ) for _ in range(4))
        if код not in занятые:
            занятые.add(код)
            out.append(код)
    return out


def связать(src, dst):
    src = Path(src).resolve()
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(src, dst)


def запись(r, код, заход, повтор_кода):
    поля = ["slide", "пациент", "центр", "TSR_поле_строма", "TSR", "площадь_мм2", "файл"]
    return {"код": код, "заход": заход, "повтор_кода": повтор_кода, "слой": int(r["слой"]),
            **{k: r[k] for k in поля}, "ГБ": round(r["байт"] / 2**30, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defs", default="outputs/results/pan/tcga_ov_deploy5_raw/tsr_defs.csv")
    ap.add_argument("--svs", default="data/tcga_ov_flat")
    ap.add_argument("--out", default="outputs/patho_check")
    ap.add_argument("--seed", type=int, default=20261022)
    args = ap.parse_args()
    out, rng = Path(args.out), random.Random(args.seed)

    t = pd.read_csv(args.defs)
    t["пациент"] = t.slide.str[:12]
    t["центр"] = t.slide.str.split("-").str[1]
    t["файл"] = [str(Path(args.svs) / f"{s}.svs") for s in t.slide]
    print(f"срезов в таблице {len(t)}")
    нет = t[~t.файл.map(lambda p: Path(p).exists())]
    if len(нет):
        print(f"нет файла для {len(нет)}: {', '.join(нет.slide.str[:23])}")
        t = t.drop(нет.index)
    t = t[годен(t)]
    print(f"прошли правило качества {len(t)}")
    t = t.sort_values("площадь_мм2").groupby("пациент").tail(1)
    t = t.dropna(subset=["TSR_поле_строма"]).copy()
    print(f"по одной пациентке, с полем по строме: {len(t)}")
    t["байт"] = [Path(p).resolve().stat().st_size for p in t.файл]
    t["слой"] = pd.qcut(t.TSR_поле_строма.rank(method="first"), СЛОЁВ, labels=False) + 1

    первый, второй = [], []
    for _, g in t.groupby("слой"):
        выбор = rng.sample(sorted(g.index), В_ПЕРВЫЙ + НОВЫХ_ВО_ВТОРОЙ)
        первый += выбор[:В_ПЕРВЫЙ]
        второй += [(i, True) for i in rng.sample(выбор[:В_ПЕРВЫЙ], ПОВТОРОВ)]
        второй += [(i, False) for i in выбор[В_ПЕРВЫЙ:]]
    rng.shuffle(первый)
    rng.shuffle(второй)
    занятые = set()
    код1 = dict(zip(первый, новые_коды(len(первый), rng, занятые)))
    код2 = новые_коды(len(второй), rng, занятые)

    ключ = [запись(t.loc[i], код1[i], 1, "") for i in первый]
    ключ += [запись(t.loc[i], к, 2, код1[i] if повтор else "")
             for (i, повтор), к in zip(второй, код2)]
    ключ = pd.DataFrame(ключ).sort_values(["заход", "код"])

    пакет = out / "patho_package"
    for d in ("visit1", "visit2_new"):
        (пакет / d).mkdir(parents=True, exist_ok=True)
    for _, r in ключ.iterrows():
        if r.заход == 1:
            связать(r.файл, пакет / "visit1" / f"{r.код}.svs")
        elif not r.повтор_кода:
            связать(r.файл, пакет / "visit2_new" / f"{r.код}.svs")
    for заход, имя in ((1, "visit1"), (2, "visit2")):
        коды = sorted(ключ.loc[ключ.заход == заход, "код"])
        pd.DataFrame({"код": коды, "доля стромы в поле, %": "", "срез пригоден (да/нет)": "",
                      "примечание": ""}).to_csv(пакет / f"{имя}_form.csv", sep=";",
                                                 index=False, encoding="utf-8-sig")
    (пакет / "README_patolog.txt").write_text(ИНСТРУКЦИЯ, encoding="utf-8-sig")
    ключ.to_csv(out / "key.csv", index=False, encoding="utf-8-sig")
    (out / "export_fields.groovy").write_text(ВЫГРУЗКА, encoding="utf-8")

    print("\nслои по TSR_поле_строма (все пригодные срезы):")
    print(t.groupby("слой").TSR_поле_строма.agg(["count", "min", "max"]).round(3).to_string())
    отобраны = ключ.drop_duplicates("slide")
    print(f"\nотобрано срезов {len(отобраны)}: TSR_поле_строма от "
          f"{отобраны.TSR_поле_строма.min():.3f} до {отобраны.TSR_поле_строма.max():.3f}, "
          f"TSR по всей ткани от {отобраны.TSR.min():.3f} до {отобраны.TSR.max():.3f}")
    print("по центрам:", отобраны.центр.value_counts().to_dict())
    п = ключ[ключ.заход == 1]
    н = ключ[(ключ.заход == 2) & (ключ.повтор_кода == "")]
    print(f"\nзаход 1: {len(п)} срезов, {п.ГБ.sum():.1f} ГБ, папка {пакет / 'visit1'}")
    print(f"заход 2: {len(ключ[ключ.заход == 2])} срезов, из них новых {len(н)} "
          f"({н.ГБ.sum():.1f} ГБ, папка {пакет / 'visit2_new'}), повторы берутся из первого")
    print(f"\nдля патолога: {пакет}")
    print(f"ключ (патологу не передавать): {out / 'key.csv'}")
    print(f"выгрузка полей для QuPath: {out / 'export_fields.groovy'}")


if __name__ == "__main__":
    main()
