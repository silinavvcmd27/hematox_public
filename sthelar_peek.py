# Посмотреть патчи STHELAR с нашей раскраской классов.
# Читает нужный шард паркета по HTTP частями (весь датасет качать не надо).
#
#   python sthelar_peek.py --slide ovary_s1 --n 8
import argparse
import io
import os

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp
from PIL import Image
from huggingface_hub import HfFileSystem

from src.utils import TUMOR, STROMA, ensure_dir
from sthelar_extract import REPO, build_lut

COL = {0: (245, 245, 245), TUMOR: (220, 50, 47), STROMA: (128, 128, 128)}
NSHARD = 54


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", default="ovary_s1")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lut = build_lut(args.slide)
    fs = HfFileSystem()
    picked = []

    for i in range(NSHARD):
        path = f"datasets/{REPO}/data/train-{i:05d}-of-{NSHARD:05d}.parquet"
        with fs.open(path, "rb") as fh:
            pf = pq.ParquetFile(fh)
            for g in range(pf.num_row_groups):
                ids = pf.read_row_group(g, columns=["slide_id"])["slide_id"].to_pylist()
                hit = [k for k, s in enumerate(ids) if s == args.slide]
                if not hit:
                    continue
                print(f"шард {i}, группа {g}: {len(hit)} патчей {args.slide}")
                tb = pf.read_row_group(g, columns=["image", "cell_id_map", "bPQ"])
                for k in hit[:args.n - len(picked)]:
                    im = tb["image"][k].as_py()
                    he = np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("RGB"))
                    raw = tb["cell_id_map"][k].as_py()
                    cid = sp.load_npz(io.BytesIO(raw)).toarray()
                    cid = np.clip(cid, 0, len(lut) - 1)
                    picked.append((he, lut[cid], float(tb["bPQ"][k].as_py())))
                if len(picked) >= args.n:
                    break
        if len(picked) >= args.n:
            break

    if not picked:
        raise SystemExit(f"патчи {args.slide} не найдены")

    rows = []
    for he, m, bpq in picked:
        ov = he.copy()
        for c, col in COL.items():
            if c == 0:
                continue
            s = m == c
            if s.any():
                ov[s] = (0.45 * np.array(col) + 0.55 * he[s]).astype(np.uint8)
        flat = np.zeros(he.shape, np.uint8)
        for c, col in COL.items():
            flat[m == c] = col
        gap = np.full((he.shape[0], 6, 3), 255, np.uint8)
        rows.append(np.concatenate([he, gap, ov, gap, flat], 1))
        print("  bPQ %.2f | tumor %.0f%% stroma %.0f%% фон %.0f%%"
              % (bpq, 100 * (m == TUMOR).mean(), 100 * (m == STROMA).mean(),
                 100 * (m == 0).mean()))

    w = rows[0].shape[1]
    grid = np.concatenate([np.concatenate([r, np.full((6, w, 3), 255, np.uint8)])
                           for r in rows], 0)
    out = args.out or str(ensure_dir("outputs/results") / f"sthelar_{args.slide}.png")
    Image.fromarray(grid).save(out)
    print("сохранено:", out, "| колонки: H&E | оверлей | маска")


if __name__ == "__main__":
    main()