# Клинические данные TCGA-OV из открытого API GDC и сшивка их с нашей таблицей
# по когорте. Авторизация не нужна, это открытые данные.
#
#   python fetch_clinical.py
#   python fetch_clinical.py --cohort outputs/results/tcga_cohort.csv

import argparse
import csv
from pathlib import Path

import requests

API = "https://api.gdc.cancer.gov/cases"
FIELDS = [
    "submitter_id",
    "demographic.vital_status",
    "demographic.days_to_death",
    "demographic.age_at_index",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.figo_stage",
    "diagnoses.tumor_grade",
    "diagnoses.primary_diagnosis",
]


def fetch(project):
    payload = {
        "filters": {"op": "in",
                    "content": {"field": "project.project_id", "value": [project]}},
        "fields": ",".join(FIELDS),
        "format": "TSV",
        "size": 2000,
    }
    r = requests.post(API, json=payload, timeout=120)
    r.raise_for_status()
    return r.text


def first_value(rec, suffix):
    """У случая бывает несколько диагнозов, GDC раскладывает их по колонкам
    diagnoses.0, diagnoses.1 и так далее. Берём первое непустое."""
    for key in sorted(rec):
        if key.startswith("diagnoses.") and key.endswith("." + suffix) and rec[key]:
            return rec[key]
    return ""


def max_followup(rec):
    vals = [rec[k] for k in rec
            if k.startswith("diagnoses.") and k.endswith(".days_to_last_follow_up")]
    nums = [float(v) for v in vals if v not in ("", "Unknown", None)]
    return max(nums) if nums else None


def survival(rec):
    """Время наблюдения и признак события для анализа выживаемости."""
    dead = rec.get("demographic.vital_status", "").lower() == "dead"
    days = rec.get("demographic.days_to_death", "")
    if dead and days not in ("", "Unknown"):
        return float(days), 1
    follow = max_followup(rec)
    return (follow, 0) if follow is not None else (None, None)


def patient(slide_name):
    return "-".join(slide_name.split("-")[:3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="TCGA-OV")
    ap.add_argument("--out", default="data/tcga_ov_flat/clinical.tsv")
    ap.add_argument("--cohort", default="outputs/results/tcga_cohort.csv")
    ap.add_argument("--merged", default="outputs/results/cohort_clinical.csv")
    args = ap.parse_args()

    text = fetch(args.project)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")

    lines = text.splitlines()          # GDC отдаёт CRLF, splitlines их и снимает
    header = [h.strip() for h in lines[0].split("\t")]
    clinical = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        rec = {k: v.strip() for k, v in zip(header, line.split("\t"))}
        if rec.get("submitter_id"):
            clinical[rec["submitter_id"]] = rec
    print(f"скачано случаев: {len(clinical)} -> {args.out}")

    if not Path(args.cohort).exists():
        print("таблицы когорты нет, сшивать не с чем")
        return

    with open(args.cohort, encoding="utf-8") as f:
        cohort = list(csv.DictReader(f))

    out_rows, missing = [], []
    for row in cohort:
        pid = patient(row["slide"])
        rec = clinical.get(pid)
        merged = dict(row)
        merged["пациент"] = pid
        if rec is None:
            missing.append(pid)
        else:
            days, event = survival(rec)
            merged.update({
                "возраст": rec.get("demographic.age_at_index", ""),
                "статус": rec.get("demographic.vital_status", ""),
                "стадия": first_value(rec, "figo_stage"),
                "степень": first_value(rec, "tumor_grade"),
                "диагноз": first_value(rec, "primary_diagnosis"),
                "дней_наблюдения": days,
                "событие": event,
            })
        out_rows.append(merged)

    fields = list(cohort[0]) + ["пациент", "возраст", "статус", "стадия", "степень",
                                "диагноз", "дней_наблюдения", "событие"]
    with open(args.merged, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(out_rows)

    print(f"сшито срезов: {len(cohort) - len(missing)} из {len(cohort)} -> {args.merged}")
    if missing:
        print("не нашлись в GDC:", sorted(set(missing)))


if __name__ == "__main__":
    main()