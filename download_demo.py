# Демо: качаем один ovary-срез из HEST-1k, делаем псевдо-метки tumor/stroma/undefined
# по маркерам, кодируем готовые патчи через UNI и складываем .npz как ждёт train.py.
#
# ВАЖНО: метки тут ПСЕВДО (по экспрессии маркеров) - только чтобы проверить,
# что весь путь UNI -> граф -> обучение работает. Не финальная разметка.
#
# запуск:  python download_demo.py --id TENX65
# потом:   python -m src.train

import argparse
import glob
import os
from pathlib import Path

import h5py
import numpy as np

from src.utils import load_config, get_device, hf_login, ensure_dir, CLASS_TO_IDX, CLASS_NAMES
from src.models.uni_encoder import build_encoder

# маркеры (ovary / HGSOC). берём те, что реально есть в данных
TUMOR_MARKERS = ["EPCAM", "KRT8", "KRT18", "KRT19", "PAX8", "WFDC2", "MUC16", "CD24"]
STROMA_MARKERS = ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRA", "PDGFRB",
                  "ACTA2", "PECAM1", "VWF", "PTPRC", "CD3D", "CD68"]


def cpm_log1p(X):
    X = np.asarray(X, dtype=np.float64)
    tot = X.sum(1, keepdims=True)
    tot[tot == 0] = 1.0
    return np.log1p(X / tot * 1e4)


def score(expr, genes, markers):
    cols = [i for i, g in enumerate(genes) if g in markers]
    if not cols:
        return np.zeros(expr.shape[0])
    s = expr[:, cols].mean(1)
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else s * 0


def pseudo_labels(expr, genes, margin=0.25):
    t = score(expr, genes, set(TUMOR_MARKERS))
    s = score(expr, genes, set(STROMA_MARKERS))
    diff = t - s
    lab = np.full(len(diff), "undefined", dtype=object)
    lab[diff > margin] = "tumor"
    lab[diff < -margin] = "stroma"
    return lab


def download(hid, local_dir):
    from huggingface_hub import snapshot_download
    # тянем только adata и готовые патчи (без WSI, cellvit, тумбнейлов и пр.)
    pat = [f"st/{hid}.h5ad", f"patches/{hid}.h5", f"metadata/{hid}.json"]
    print(f"качаю {hid} из MahmoodLab/hest (только st + patches) ...")
    snapshot_download(repo_id="MahmoodLab/hest", repo_type="dataset",
                      allow_patterns=pat, local_dir=local_dir)


def find_one(local_dir, hid, sub, exts):
    for e in exts:
        hits = glob.glob(os.path.join(local_dir, "**", f"*{hid}*{e}"), recursive=True)
        hits = [h for h in hits if sub in h] or hits
        if hits:
            return hits[0]
    return None


def load_patches(h5_path):
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        img = f["img"][:] if "img" in f else f[keys[0]][:]
        coords = f["coords"][:] if "coords" in f else None
        bc = f["barcode"][:] if "barcode" in f else None
    if bc is not None:
        bc = np.array([b.decode() if isinstance(b, bytes) else
                       (b[0].decode() if hasattr(b, "__len__") and isinstance(b[0], bytes) else str(b))
                       for b in bc])
    return img, coords, bc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="TENX65", help="HEST id (ovary Visium = TENX65)")
    ap.add_argument("--hest-dir", default="hest_data")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--margin", type=float, default=0.25)
    args = ap.parse_args()

    cfg = load_config(args.config)
    hid = args.id

    download(hid, args.hest_dir)

    h5 = find_one(args.hest_dir, hid, "patches", [".h5"])
    h5ad = find_one(args.hest_dir, hid, "st", [".h5ad"])
    if not h5 or not h5ad:
        raise SystemExit(f"не нашла файлы: patches={h5}, adata={h5ad}. "
                         f"Глянь содержимое {args.hest_dir}")
    print("patches:", h5)
    print("adata:  ", h5ad)

    import anndata as ad
    a = ad.read_h5ad(h5ad)
    X = a.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    expr = cpm_log1p(X)
    genes = [g.upper() for g in a.var_names]
    labels_by_bc = dict(zip(np.array(a.obs_names), pseudo_labels(expr, genes, args.margin)))

    img, coords, bc = load_patches(h5)
    if bc is None:
        raise SystemExit("в h5 нет barcode, не могу сопоставить метки")
    keep = [i for i, b in enumerate(bc) if b in labels_by_bc]
    img = img[keep]
    coords = coords[keep] if coords is not None else np.zeros((len(keep), 2))
    y = np.array([CLASS_TO_IDX[labels_by_bc[bc[i]]] for i in keep], dtype=np.int64)
    pids = np.array([f"{hid}_{bc[i]}" for i in keep])
    print(f"спотов с метками: {len(keep)}")
    uniq, cnt = np.unique(y, return_counts=True)
    print("распределение:", {CLASS_NAMES[u]: int(c) for u, c in zip(uniq, cnt)})

    hf_login(cfg["encoder"]["hf_token_env"])
    device = get_device()
    print("device:", device)
    enc = build_encoder(cfg, device)
    arrays = [img[i] for i in range(len(img))]
    Xemb = enc.encode_patches(arrays, cfg["encoder"]["batch_size"])

    out = ensure_dir(Path(cfg["paths"]["processed_dir"]) / "embeddings") / f"{hid}.npz"
    np.savez_compressed(out, X=Xemb.astype(np.float32), y=y,
                        coords=coords.astype(int), patch_id=pids)
    print("saved", out, "X=", Xemb.shape)
    print("\nготово. теперь: python -m src.train")


if __name__ == "__main__":
    main()
