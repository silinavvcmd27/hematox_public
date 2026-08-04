# STHELAR 40x, разведка v2: структура файлов + один пример патча.
from huggingface_hub import HfApi
import numpy as np, io
try:
    import scipy.sparse as sp
except ImportError:
    sp = None

REPO = "FelicieGS/STHELAR_40x"
api = HfApi()
info = api.repo_info(REPO, repo_type="dataset", files_metadata=True)
sib = [(s.rfilename, s.size or 0) for s in info.siblings]
print(f"файлов: {len(sib)}\n--- крупнейшие ---")
for name, size in sorted(sib, key=lambda x: -x[1])[:20]:
    print(f"  {size/1e6:8.1f} MB  {name}")
print("--- мелкие / не-parquet ---")
for name, size in sorted(sib):
    if not name.endswith(".parquet") or size < 5e6:
        print(f"  {size/1e6:8.2f} MB  {name}")

try:
    from datasets import load_dataset
    ex = next(iter(load_dataset(REPO, split="train", streaming=True)))
    print("\nключи примера:", list(ex.keys()))
    for k, v in ex.items():
        if hasattr(v, "size") and not isinstance(v, (bytes, bytearray)):
            print(f"  {k}: PIL {getattr(v,'size',None)}")
        elif isinstance(v, (bytes, bytearray)):
            print(f"  {k}: bytes[{len(v)}]")
        else:
            print(f"  {k}: {type(v).__name__} = {str(v)[:60]}")
    if sp is not None and isinstance(ex.get("cell_id_map"), (bytes, bytearray)):
        m = sp.load_npz(io.BytesIO(ex["cell_id_map"])).toarray()
        print("  cell_id_map ->", m.shape, m.dtype, "| уник. id:", len(np.unique(m)))
except Exception as e:
    print("стриминг не удался:", e)