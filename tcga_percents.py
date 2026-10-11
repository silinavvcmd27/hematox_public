# Оценки патолога из метаданных TCGA и сравнение с нашей долей стромы.
#
# При приёме материала в TCGA патолог записывал по каждому слайду доли клеток: опухоли,
# стромы, некроза, нормальной ткани. Поля лежат в GDC внутри записи случая
# (samples.portions.slides.percent_*) и скачиваются без авторизации.
#
# Важно: эти оценки делались не на диагностических срезах (DX), которые размечает наша
# модель, а на срезах контроля материала (TS и BS, верхний и нижний срез замороженного
# блока). Поэтому по слайдам они почти не совпадают, и соединение идёт по пациентке:
# наш диагностический срез против оценки по другому срезу того же опухолевого блока.
#
# Оговорки для текста: оценка сделана на глаз и обычно кратна пяти процентам; это доля
# клеток, а не площади, поэтому систематический сдвиг ожидаем и сравнивать надо прежде
# всего порядок пациенток; материал разный (замороженный блок против парафинового).
#
#   python -u tcga_percents.py
#   python -u tcga_percents.py --project TCGA-BRCA --cohort outputs/results/pan/tcga_final/cohort.csv

import argparse
from pathlib import Path

import pandas as pd
import requests
from scipy.stats import spearmanr

API = "https://api.gdc.cancer.gov/cases"
SLIDE = "samples.portions.slides"
PERCENTS = ["percent_tumor_cells", "percent_stromal_cells", "percent_necrosis",
            "percent_normal_cells", "percent_tumor_nuclei",
            "percent_lymphocyte_infiltration"]


def fetch(project):
    fields = ["submitter_id", f"{SLIDE}.submitter_id", f"{SLIDE}.section_location"]
    fields += [f"{SLIDE}.{p}" for p in PERCENTS]
    payload = {"filters": {"op": "in", "content": {"field": "project.project_id",
                                                   "value": [project]}},
               "fields": ",".join(fields), "format": "JSON", "size": 5000}
    hits = requests.post(API, json=payload, timeout=300).json()["data"]["hits"]
    rows = []
    for case in hits:
        for sample in case.get("samples", []):
            for portion in sample.get("portions", []):
                for slide in portion.get("slides", []):
                    row = {"пациент": case.get("submitter_id"),
                           "slide_barcode": slide.get("submitter_id"),
                           "секция": slide.get("section_location")}
                    row.update({p: slide.get(p) for p in PERCENTS})
                    rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="TCGA-OV")
    ap.add_argument("--cohort", default="outputs/results/pan/tcga_ov_deploy3/cohort.csv")
    ap.add_argument("--defs", default="outputs/results/pan/tcga_ov_deploy3/tsr_defs.csv")
    ap.add_argument("--out", default="outputs/results/pan/tcga_percents.csv")
    args = ap.parse_args()

    meta = fetch(args.project)
    print(f"слайдов в {args.project}: {len(meta)}")
    print("заполнено значений по полям:")
    print(meta[PERCENTS].notna().sum().to_string())

    meta["тип"] = meta.slide_barcode.str.split("-").str[-1].str[:2]
    filled = meta.percent_tumor_cells.notna()
    print("\nслайдов по типу (в скобках с оценкой патолога):")
    for t, n in meta["тип"].value_counts().items():
        print(f"  {t}: {n} ({int(filled[meta['тип'] == t].sum())})")

    good = meta[meta.percent_stromal_cells.notna() & meta.percent_tumor_cells.notna()].copy()
    good = good[good.percent_tumor_cells + good.percent_stromal_cells > 0]
    good["TSR_патолога"] = (good.percent_stromal_cells
                            / (good.percent_tumor_cells + good.percent_stromal_cells))
    per_patient = (good.groupby("пациент")
                       .agg(TSR_патолога=("TSR_патолога", "median"),
                            оценок=("TSR_патолога", "size"),
                            некроз_патолога=("percent_necrosis", "median"))
                       .round(3).reset_index())
    print(f"\nпациенток с оценкой: {len(per_patient)}, "
          f"медиана TSR патолога {per_patient.TSR_патолога.median():.3f}")

    ours = pd.read_csv(args.cohort)
    ours["slide_barcode"] = ours.slide.str.split(".").str[0]
    ours["пациент"] = ours.slide_barcode.str.split("-").str[:3].str.join("-")
    if Path(args.defs).exists():
        defs = pd.read_csv(args.defs)
        defs["slide_barcode"] = defs.slide.str.split(".").str[0]
        extra = [c for c in defs.columns if c not in ours.columns or c == "slide_barcode"]
        ours = ours.merge(defs[extra], on="slide_barcode", how="left")

    both = ours.merge(per_patient, on="пациент", how="left")
    pair = both[both.TSR_патолога.notna()].copy()
    print(f"совпало по пациенткам: {len(pair)} из {len(ours)}")
    if pair.empty:
        both.to_csv(args.out, index=False)
        print("сравнивать нечего, таблица сохранена:", args.out)
        return

    print(f"TSR патолога по нашим пациенткам: медиана {pair.TSR_патолога.median():.3f}, "
          f"квартили {pair.TSR_патолога.quantile(0.25):.3f} до "
          f"{pair.TSR_патолога.quantile(0.75):.3f}")

    print("\nсогласие с нашими определениями")
    for col in ["TSR", "TSR_бед", "TSR_поле_опухоль", "TSR_поле_строма"]:
        if col not in pair.columns:
            continue
        ok = pair[[col, "TSR_патолога"]].dropna()
        if len(ok) < 3:
            continue
        rho, p = spearmanr(ok[col], ok.TSR_патолога)
        mae = (ok[col] - ok.TSR_патолога).abs().mean()
        bias = (ok[col] - ok.TSR_патолога).median()
        print(f"  {col}: ро {rho:+.3f} (p {p:.3f}), средняя ошибка {mae:.3f}, "
              f"медианный сдвиг {bias:+.3f}, пациенток {len(ok)}")

    if "прочее_%" in pair.columns and pair.некроз_патолога.notna().any():
        ok = pair[["прочее_%", "некроз_патолога"]].dropna()
        if len(ok) >= 3:
            rho = spearmanr(ok["прочее_%"], ok.некроз_патолога)[0]
            print(f"\nнекроз патолога против нашего класса прочего: ро {rho:+.3f} "
                  f"по {len(ok)} пациенткам")

    both.to_csv(args.out, index=False)
    print("\nтаблица:", args.out)


if __name__ == "__main__":
    main()
