# Подготовка PTRC-HGSOC к разметке: таблица срезов с клиникой и папки со срезами в
# порядке очереди.
#
# В коллекции 158 пациенток, 174 опухоли и 348 срезов H&E: на каждую опухоль два среза,
# первый и последний срез одного парафинового блока. Первым считается срез с меньшим
# Image ID. Очередь разметки: сначала оба среза опухолей пациенток без неоадъювантной
# химиотерапии, по ним считается основная проверка (см. ptrc_plan.md), затем срезы
# пациенток после неоадъюванта, они нужны только для проверки устойчивости.
#
# run_tcga_ov.sh берёт срезы из одной плоской папки, поэтому здесь собираются две папки:
# svs_1 (очередь 1) и svs_2 (очередь 2). Файлы не копируются, а связываются жёсткими
# ссылками; если файловая система их не даёт, символическими (тогда run_tcga_ov.sh
# отсортирует очередь не по размеру, на результат это не влияет).
#
# Клиническая таблица TCIA приходит как xlsx; читается без openpyxl, прямо из архива.
#
#   python ptrc_prepare.py --xlsx data/ptrc/PTRC-HGSOC_List_clincal_data.xlsx --raw data/ptrc/raw

import argparse
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def read_xlsx(path):
    """Первый лист xlsx в DataFrame, все значения строками."""
    with zipfile.ZipFile(path) as z:
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(NS + "si"):
                strings.append("".join(t.text or "" for t in si.iter(NS + "t")))
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for r in sheet.iter(NS + "row"):
        row = {}
        for c in r.iter(NS + "c"):
            col = re.sub(r"\d", "", c.get("r"))
            v = c.find(NS + "v")
            if v is None:
                t = c.find(".//" + NS + "t")
                row[col] = t.text if t is not None and t.text else ""
            else:
                row[col] = strings[int(v.text)] if c.get("t") == "s" else v.text
        rows.append(row)
    cols = sorted({k for r in rows for k in r}, key=lambda s: (len(s), s))
    header = [rows[0].get(c, "").strip() for c in cols]
    return pd.DataFrame([[r.get(c, "") for c in cols] for r in rows[1:]], columns=header)


def связать(src, dst):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(src.resolve(), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="data/ptrc/PTRC-HGSOC_List_clincal_data.xlsx")
    ap.add_argument("--raw", default="data/ptrc/raw", help="куда скачан пакет TCIA, ищется рекурсивно")
    ap.add_argument("--out", default="data/ptrc")
    args = ap.parse_args()
    out = Path(args.out)

    t = read_xlsx(args.xlsx)
    t["ответ"] = t["Tumor response"].str.strip().str.lower()
    # Неоадъювант получает пациентка, а не срез, но в таблице TCIA у двух пациенток отметка
    # различается между срезами одного блока. При любой отметке «yes» пациентка считается
    # получавшей неоадъювант (уточнение плана от 22.10.2026, до разметки).
    t["неоадъювант_в_строке"] = t["Neo-adjuvant"].str.strip().str.lower() == "yes"
    t["неоадъювант"] = t.groupby("Patient ID")["неоадъювант_в_строке"].transform("any")
    разные = t.groupby("Patient ID")["неоадъювант_в_строке"].nunique()
    if (разные > 1).any():
        print("отметка о неоадъюванте различается между срезами у пациенток "
              f"{', '.join(map(str, разные[разные > 1].index))}: считаются получавшими")
    t["Image ID"] = pd.to_numeric(t["Image ID"])
    t["срез_в_блоке"] = t.groupby("Sample ID")["Image ID"].rank(method="first").astype(int)
    t["очередь"] = t.неоадъювант.map({False: 1, True: 2})

    файлы = {p.stem: p for p in Path(args.raw).rglob("*.svs")}
    print(f"SVS в {args.raw}: {len(файлы)}")
    t["slide"] = t["File Name"].str.strip()
    # если имена в пакете не совпадут с таблицей, файл ищется по Image ID среди чисел в имени
    по_номеру = {}
    for stem, p in файлы.items():
        for число in re.findall(r"\d+", stem):
            по_номеру.setdefault(число, []).append(p)

    def найти(r):
        if r.slide in файлы:
            return str(файлы[r.slide])
        кандидаты = по_номеру.get(str(int(r["Image ID"])), [])
        return str(кандидаты[0]) if len(кандидаты) == 1 else ""

    t["файл"] = t.apply(найти, axis=1)
    нет = t[t.файл == ""]
    if len(нет):
        print(f"не найдены файлы для {len(нет)} срезов, например: {', '.join(нет.slide.head(5))}")

    for q in (1, 2):
        папка = out / f"svs_{q}"
        папка.mkdir(parents=True, exist_ok=True)
        for _, r in t[(t.очередь == q) & (t.файл != "")].iterrows():
            связать(Path(r.файл), папка / f"{r.slide}.svs")
        print(f"очередь {q}: {len(list(папка.glob('*.svs')))} срезов в {папка}")

    таблица = t.rename(columns={"Patient ID": "пациент", "Sample ID": "образец",
                               "Sample Source": "центр", "Tumor type": "тип",
                               "Tumor Location Group": "место", "Tumor Stage": "стадия",
                               "Patient Age": "возраст"})
    таблица = таблица[["slide", "пациент", "образец", "центр", "ответ", "неоадъювант", "тип",
                       "место", "стадия", "возраст", "срез_в_блоке", "очередь", "файл"]]
    таблица.to_csv(out / "slides.csv", index=False)

    p = таблица.drop_duplicates("пациент")
    print(f"\nпациенток {len(p)}: {p.ответ.value_counts().to_dict()}")
    без = p[~p.неоадъювант]
    print(f"без неоадъювантной химиотерапии {len(без)}: {без.ответ.value_counts().to_dict()}")
    print("по центрам:", без.groupby("центр").ответ.value_counts().unstack().to_dict("index"))
    print("таблица:", out / "slides.csv")


if __name__ == "__main__":
    main()
