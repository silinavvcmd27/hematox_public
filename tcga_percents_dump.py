# Выгрузка оценок патолога из TCGA по тем срезам, которые размечала модель.
#
# Запрос идёт по идентификаторам пациенток, а не по проектам, поэтому работает и для
# когорты яичника, и для восьми срезов других опухолей сразу.
#
# Оценки в TCGA стоят на срезах контроля материала (TS, BS), а не на диагностических
# (DX), которые размечает модель, поэтому на выходе два файла: подробный по всем
# контрольным срезам наших пациенток и сводный по нашим размеченным срезам.
#
#   python -u tcga_percents_dump.py
#   python -u tcga_percents_dump.py --cohorts outputs/results/pan/tcga_final/cohort.csv

import argparse
from pathlib import Path

import pandas as pd
import requests

API = "https://api.gdc.cancer.gov/cases"
SLIDE = "samples.portions.slides"
PERCENTS = ["percent_tumor_cells", "percent_stromal_cells", "percent_necrosis",
            "percent_normal_cells", "percent_tumor_nuclei",
            "percent_lymphocyte_infiltration"]


def fetch(patients):
    fields = ["submitter_id", f"{SLIDE}.submitter_id", f"{SLIDE}.section_location"]
    fields += [f"{SLIDE}.{p}" for p in PERCENTS]
    payload = {"filters": {"op": "in", "content": {"field": "submitter_id",
                                                   "value": sorted(patients)}},
               "fields": ",".join(fields), "format": "JSON", "size": 5000}
    hits = requests.post(API, json=payload, timeout=300).json()["data"]["hits"]
    rows = []
    for case in hits:
        for sample in case.get("samples", []):
            for portion in sample.get("portions", []):
                for slide in portion.get("slides", []):
                    row = {"пациент": case.get("submitter_id"),
                           "контрольный_срез": slide.get("submitter_id"),
                           "секция": slide.get("section_location")}
                    row.update({p: slide.get(p) for p in PERCENTS})
                    rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+",
                    default=["outputs/results/pan/tcga_ov_deploy3/cohort.csv"])
    ap.add_argument("--out-dir", default="outputs/results/pan")
    args = ap.parse_args()

    ours = pd.concat([pd.read_csv(p).assign(когорта=Path(p).parent.name)
                      for p in args.cohorts], ignore_index=True)
    ours["наш_срез"] = ours.slide.str.split(".").str[0]
    ours["пациент"] = ours.наш_срез.str.split("-").str[:3].str.join("-")
    print(f"наших срезов {len(ours)}, пациенток {ours.пациент.nunique()}")

    meta = fetch(set(ours.пациент))
    if meta.empty:
        raise SystemExit("GDC ничего не вернул по этим пациенткам")
    meta["тип"] = meta.контрольный_срез.str.split("-").str[-1].str[:2]
    есть = meta.percent_tumor_cells.notna()
    print(f"слайдов у этих пациенток {len(meta)}, с оценкой {int(есть.sum())}")
    print(meta.groupby("тип").size().to_string())

    out = Path(args.out_dir)
    detail = out / "tcga_patolog_po_sloyam.csv"
    meta.sort_values(["пациент", "контрольный_срез"]).to_csv(detail, index=False)

    good = meta[meta.percent_stromal_cells.notna() & meta.percent_tumor_cells.notna()].copy()
    good = good[good.percent_tumor_cells + good.percent_stromal_cells > 0]
    good["TSR_патолога"] = (good.percent_stromal_cells
                            / (good.percent_tumor_cells + good.percent_stromal_cells))
    agg = (good.groupby("пациент")
               .agg(TSR_патолога=("TSR_патолога", "median"),
                    опухоль_патолога=("percent_tumor_cells", "median"),
                    строма_патолога=("percent_stromal_cells", "median"),
                    некроз_патолога=("percent_necrosis", "median"),
                    норма_патолога=("percent_normal_cells", "median"),
                    контрольных_срезов=("TSR_патолога", "size"))
               .round(3).reset_index())

    cols = [c for c in ["наш_срез", "пациент", "когорта", "площадь_мм2", "опухоль_%",
                        "строма_%", "прочее_%", "TSR", "расхождение_половин"]
            if c in ours.columns]
    svod = ours[cols].merge(agg, on="пациент", how="left")
    summary = out / "tcga_patolog_svodka.csv"
    svod.to_csv(summary, index=False)

    print(f"\nс оценкой нашлось {int(svod.TSR_патолога.notna().sum())} наших срезов из {len(svod)}")
    print(svod.head(12).to_string(index=False))
    print("\nподробно по контрольным срезам:", detail)
    print("сводка по нашим срезам:", summary)


if __name__ == "__main__":
    main()
