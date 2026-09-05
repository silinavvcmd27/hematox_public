# Сверка состава стромы, посчитанного по картинке, с объёмной экспрессией TCGA.
#
# Наш показатель это доля гормональной стромы внутри всей стромы, то есть
# отношение. Значит и в экспрессии берём отношение гормональных маркеров к
# матриксным, иначе измерим количество стромы, а не её состав.
#
# Образец для RNA-seq это другой кусок опухоли, не тот срез, что размечен, так что
# даже при верной гипотезе связь будет размытой.
#
#   python bulk_check.py
#   python bulk_check.py --cohort outputs/results/tcga_cohort.csv

import argparse
import csv
import gzip
import io
import json
from pathlib import Path

import numpy as np
import requests

FILES_API = "https://api.gdc.cancer.gov/files"
DATA_API = "https://api.gdc.cancer.gov/data"

# Маркеры взяты из R/export_seurat.R, там они уже подобраны под эти два типа CAF
HORMONAL = ["C7", "STAR", "CYP11A1", "CYP17A1", "CYP19A1", "HSD17B1",
            "FOXL2", "NR5A1", "INHA", "AMH", "GATA4"]
MATRIX = ["COL5A1", "COL11A1", "FAP", "POSTN", "THY1", "MMP11", "INHBA"]
PAN_CAF = ["DCN", "LUM", "PDGFRB"]


def find_files(patients):
    """Открытые файлы экспрессии STAR-Counts для первичной опухоли."""
    payload = {
        "filters": {"op": "and", "content": [
            {"op": "in", "content": {"field": "cases.submitter_id", "value": patients}},
            {"op": "in", "content": {"field": "data_type",
                                     "value": ["Gene Expression Quantification"]}},
            {"op": "in", "content": {"field": "analysis.workflow_type",
                                     "value": ["STAR - Counts"]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
        ]},
        "fields": "file_id,file_name,cases.submitter_id,cases.samples.sample_type",
        "format": "JSON",
        "size": 1000,
    }
    r = requests.post(FILES_API, json=payload, timeout=120)
    r.raise_for_status()
    hits = json.loads(r.text)["data"]["hits"]

    out = {}
    for h in hits:
        case = h["cases"][0]
        types = [s.get("sample_type", "") for s in case.get("samples", [])]
        if not any("Primary" in t for t in types):
            continue
        out.setdefault(case["submitter_id"], h["file_id"])
    return out


def download(file_id, path):
    if path.exists():
        return
    r = requests.get(f"{DATA_API}/{file_id}", timeout=600)
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)


def read_tpm(path):
    """STAR-Counts: шапка с комментариями, служебные строки N_*, дальше гены."""
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    tpm = {}
    for line in io.StringIO(raw.decode("utf-8")):
        if line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 7 or parts[0].startswith("N_") or parts[0] == "gene_id":
            continue
        name, value = parts[1], parts[6]
        try:
            tpm[name] = float(value)
        except ValueError:
            continue
    return tpm


def score(tpm, genes):
    """Среднее log2(TPM+1) по найденным маркерам."""
    vals = [np.log2(tpm[g] + 1) for g in genes if g in tpm]
    return float(np.mean(vals)) if vals else None


def patient(slide):
    return "-".join(slide.split("-")[:3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="outputs/results/tcga_cohort.csv")
    ap.add_argument("--dir", default="data/tcga_bulk")
    ap.add_argument("--out", default="outputs/results/bulk_vs_image.csv")
    args = ap.parse_args()

    with open(args.cohort, encoding="utf-8") as f:
        cohort = list(csv.DictReader(f))

    # у пациента может быть несколько срезов, усредняем показатель по ним
    by_patient = {}
    for row in cohort:
        share = row["гормональная_в_строме_%"]
        if share:
            by_patient.setdefault(patient(row["slide"]), []).append(float(share))
    image = {p: float(np.mean(v)) for p, v in by_patient.items()}
    print(f"пациенток с разметкой: {len(image)}")

    files = find_files(sorted(image))
    print(f"нашлось файлов экспрессии: {len(files)}")
    missing = sorted(set(image) - set(files))
    if missing:
        print("без экспрессии:", missing)

    rows = []
    for pid, file_id in sorted(files.items()):
        path = Path(args.dir) / f"{pid}.tsv.gz"
        print(f"  {pid}", end="\r")
        download(file_id, path)
        tpm = read_tpm(path)
        h, m, pan = score(tpm, HORMONAL), score(tpm, MATRIX), score(tpm, PAN_CAF)
        if h is None or m is None:
            print(f"{pid}: маркеры не найдены, пропускаю")
            continue
        rows.append({
            "пациент": pid,
            "гормональная_по_картинке_%": round(image[pid], 1),
            "гормональные_маркеры": round(h, 3),
            "матриксные_маркеры": round(m, 3),
            "разность_log2": round(h - m, 3),
            "пан_CAF": round(pan, 3) if pan is not None else "",
        })

    if len(rows) < 3:
        raise SystemExit("слишком мало пациенток для сравнения")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n%-14s %10s %10s %10s" % ("пациент", "картинка", "горм−матр", "пан-CAF"))
    for r in rows:
        print("%-14s %10.1f %10.3f %10s"
              % (r["пациент"], r["гормональная_по_картинке_%"],
                 r["разность_log2"], r["пан_CAF"]))

    from scipy.stats import pearsonr, spearmanr
    x = np.array([r["гормональная_по_картинке_%"] for r in rows])
    y = np.array([r["разность_log2"] for r in rows])
    rho, p_rho = spearmanr(x, y)
    rp, p_rp = pearsonr(x, y)
    print(f"\nпациенток: {len(rows)}")
    print(f"Спирмен  rho {rho:+.3f}  p {p_rho:.3f}")
    print(f"Пирсон   r   {rp:+.3f}  p {p_rp:.3f}")
    print("\nтаблица:", args.out)


if __name__ == "__main__":
    main()
