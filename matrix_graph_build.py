# Граф клеток для второго слоя (матриксная строма окном).
#
# Отличие от cell_graph_build.py: клетки без метки не выкидываются. Опухоль,
# иммунные, сосуды и неразмеченные фибробласты остаются узлами, иначе у свёртки
# и у окна пропадает окружение, а матриксная строма узнаётся во многом по нему.
# Метки лежат отдельно от узлов: zone задаёт, где ездит окно, matrix даёт ответ.
#
#   Срез с готовой аннотацией подтипов (наши яичники), метки по списку:
#     python matrix_graph_build.py --slide ovary2_he --he <ome.tif> \
#       --cells data/seurat_csv/ovary2_he_cells_aligned.csv --labels config/matrix_subtypes.yaml
#   Срез, размеченный matrix_markers.py (колонки zone и matrix уже есть):
#     python matrix_graph_build.py --slide lung5k_he --he <ome.tif> \
#       --cells data/seurat_csv/lung5k_he_cells_matrix_aligned.csv
#
# Признаки те же, что у сегментации (seg_extract.py), поэтому --max-patches и
# --patch-size должны совпадать с теми, с которыми они считались.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.utils import ensure_dir, load_yaml
from cell_graph_build import GRID, replay_positions

ZONE_CODE = {"tumor": 1, "stroma": 2}


def labels_from_subtypes(df, path):
    cfg = load_yaml(path)
    lower = lambda items: {str(s).strip().lower() for s in (items or [])}
    kind = df["cell_type"].astype(str).str.strip().str.lower()
    sub = df["cell_subtype"] if "cell_subtype" in df.columns else df["cell_type"]
    sub = sub.astype(str).str.strip().str.lower()

    tumor, undefined = lower(cfg["zone"]["tumor"]), lower(cfg["zone"]["undefined"])
    zone = np.where(kind.isin(tumor) | sub.isin(tumor), "tumor",
                    np.where(kind.isin(undefined) | sub.isin(undefined), "undefined", "stroma"))
    matrix = np.select([sub.isin(lower(cfg["matrix"])), sub.isin(lower(cfg["other"]))], [1, 0], -1)
    matrix = np.where(zone == "stroma", matrix, -1)

    stroma = zone == "stroma"
    print("  в строму ушли типы:", df.loc[stroma, "cell_type"].value_counts().to_dict())
    listed = lower(cfg["matrix"]) | lower(cfg["other"]) | lower(cfg.get("unclear"))
    fibro_like = sub.str.contains(r"fibro|cafs?\b|nfs?\b", regex=True)
    unlisted = sub[stroma & fibro_like & ~sub.isin(listed)].value_counts()
    if len(unlisted):
        print("  фибробласты не из списков, пошли как -1:", unlisted.to_dict())
    return zone, matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--cells", required=True, help="клетки, координаты в пикселях H&E")
    ap.add_argument("--labels", help="список подтипов для zone и matrix (config/matrix_subtypes.yaml)")
    ap.add_argument("--seg-dir", default="data/processed/seg")
    ap.add_argument("--max-patches", type=int, default=12000)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--max-edge-um", type=float, default=50.0)
    ap.add_argument("--mpp", type=float, default=None)
    ap.add_argument("--he", default=None, help="нужен только для авто-mpp")
    ap.add_argument("--out-dir", default="data/processed/graph")
    args = ap.parse_args()

    df = pd.read_csv(args.cells)
    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
    print(f"{args.slide}: читаю {args.cells}")
    if args.labels:
        df["zone"], df["matrix"] = labels_from_subtypes(df, args.labels)
    elif not {"zone", "matrix"} <= set(df.columns):
        raise SystemExit("нет колонок zone и matrix: дайте --labels или сначала matrix_markers.py --apply")

    ps = args.patch_size
    pos = replay_positions(args.seg_dir, args.slide, args.max_patches)
    grid_step = int(np.gcd.reduce(pos.ravel()))
    if grid_step != ps:
        # у ovary3 сетка 512: с 256 проверка на кратность прошла бы, а половина
        # клеток молча выпала бы из патчей
        raise SystemExit(f"шаг сетки патчей {grid_step} px, а --patch-size {ps}")
    row_of = {(int(a), int(b)): i for i, (a, b) in enumerate(pos)}

    cx = df["x"].to_numpy(float)
    cy = df["y"].to_numpy(float)
    px = (cx // ps).astype(np.int64) * ps
    py = (cy // ps).astype(np.int64) * ps
    rows = np.array([row_of.get(k, -1) for k in zip(px.tolist(), py.tolist())], np.int64)
    keep = rows >= 0
    print(f"{args.slide}: клеток {len(df)}, внутри {len(pos)} патчей {keep.sum()} ({keep.mean():.1%})")
    if not keep.any():
        raise SystemExit("ни одна клетка не попала в патчи, проверьте совмещение")

    df = df[keep].reset_index(drop=True)
    rows, cx, cy, px, py = rows[keep], cx[keep], cy[keep], px[keep], py[keep]

    feat_npz = np.load(Path(args.seg_dir) / f"{args.slide}_feat.npz")
    # seg_extract.py режет патч по микрометрам, и у ovary3 он выходит 511 px при
    # шаге сетки 512. Токен считаем по настоящему размеру патча.
    patch_px = int(feat_npz["patch_px"]) if "patch_px" in feat_npz.files else ps
    tok = patch_px / GRID
    jx = np.clip(((cx - px) / tok).astype(int), 0, GRID - 1)
    iy = np.clip(((cy - py) / tok).astype(int), 0, GRID - 1)
    tflat = iy * GRID + jx

    print("читаю признаки...")
    X = feat_npz["X"]
    if X.shape[1] != GRID * GRID:
        raise SystemExit(f"ожидалось {GRID * GRID} токенов, в файле {X.shape[1]}")
    # строка X должна быть тем же патчем, что строка выборки; с --augment или
    # пропущенными краями в seg_extract.py это уже не так
    if len(X) != len(pos):
        raise SystemExit(f"патчей в признаках {len(X)}, в выборке {len(pos)}")
    feat = np.empty((len(rows), X.shape[-1]), np.float16)
    step = 200000
    for a in range(0, len(rows), step):
        b = min(a + step, len(rows))
        feat[a:b] = X[rows[a:b], tflat[a:b]]
    del X

    if args.mpp:
        mpp = args.mpp
    elif args.he:
        from slide_mpp import slide_mpp
        mpp = slide_mpp(args.he)
    else:
        mpp = 0.2738
        print("mpp не задан, беру 0.2738 (типичный для Xenium H&E)")

    xy = np.c_[cx, cy]
    d, nb = cKDTree(xy).query(xy, k=args.knn + 1)
    d, nb = d[:, 1:], nb[:, 1:]
    ok = (d <= args.max_edge_um / mpp).ravel()
    src = np.repeat(np.arange(len(xy)), args.knn)[ok]
    dst = nb.ravel()[ok]
    edges = np.vstack([np.r_[src, dst], np.r_[dst, src]]).astype(np.int32)

    zone = df["zone"].map(ZONE_CODE).fillna(0).astype(np.uint8).to_numpy()
    matrix = df["matrix"].fillna(-1).astype(np.int8).to_numpy()
    out = ensure_dir(args.out_dir) / f"{args.slide}_mgraph.npz"
    np.savez_compressed(out, feat=feat, pos=xy.astype(np.float32), edges=edges, mpp=mpp,
                        zone=zone, matrix=matrix)

    stroma = zone == 2
    print(f"узлов {len(xy)}, рёбер {edges.shape[1]}: опухоль {int((zone == 1).sum())}, "
          f"строма {int(stroma.sum())}, без зоны {int((zone == 0).sum())}")
    print(f"  фибробласты стромы: матриксных {int((matrix[stroma] == 1).sum())}, "
          f"остальных {int((matrix[stroma] == 0).sum())}")
    print("сохранено:", out)


if __name__ == "__main__":
    main()
