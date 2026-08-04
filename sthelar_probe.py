# Разведка STHELAR: overview-parquet -> ткани, cancer-статус, категории.
#   python sthelar_probe.py
from huggingface_hub import HfApi, hf_hub_download
import pandas as pd

REPOS = ["FelicieGS/STHELAR_40x", "FelicieGS/STHELAR_20x"]
api = HfApi()

overview = None
for repo in REPOS:
    try:
        files = api.list_repo_files(repo, repo_type="dataset")
    except Exception as e:
        print(f"{repo}: недоступен ({e})")
        continue
    print(f"\n=== {repo}: {len(files)} файлов ===")
    for f in [x for x in files if "/" not in x]:
        print("  ", f)
    cand = [f for f in files if "overview" in f.lower() and f.endswith(".parquet")]
    if cand and overview is None:
        overview = (repo, cand[0])

if overview is None:
    raise SystemExit("overview-parquet не найден — покажи список файлов выше")

repo, path = overview
print(f"\nчитаю overview: {repo}/{path}")
df = pd.read_parquet(hf_hub_download(repo, path, repo_type="dataset"))
print("строк:", len(df), "| колонки:", list(df.columns))
for col in df.columns:
    u = df[col].unique()
    if len(u) <= 40:
        print(f"\n{col} ({len(u)} уникальных):", sorted(map(str, u)))