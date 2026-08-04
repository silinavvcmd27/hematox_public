from huggingface_hub import hf_hub_download
import pandas as pd

REPO = "FelicieGS/STHELAR_40x"
idx = hf_hub_download(REPO, "cell_metadata/index.csv", repo_type="dataset")
print("--- index.csv ---")
print(open(idx).read()[:1500])

for name in ["ovary_s0", "ovary_s1", "heart_s0"]:
    try:
        p = hf_hub_download(REPO, f"cell_metadata/{name}_cell_metadata.parquet",
                            repo_type="dataset")
    except Exception as e:
        print(name, "нет:", e); continue
    df = pd.read_parquet(p)
    print(f"\n=== {name}: {len(df)} строк | колонки: {list(df.columns)}")
    for c in df.columns:
        u = df[c].unique()
        if 2 <= len(u) <= 20:
            print(f"  {c} ({len(u)}): {sorted(map(str, u))[:15]}")
    print(df.head(3).to_string())
    break