# Клеточный бенчмарк графовой модели на STHELAR-ovary (независим от обучения графа).
# STHELAR даёт инстанс-карту клеток (cell_id_map) + тип каждой клетки (cell_metadata).
# На каждом тайле: центроиды клеток + их GT-тип -> UNI-токен в точке -> граф клеток ->
# слой 1 (tumor/stroma) и компартмент. Сверяем с GT: confusion + Dice, как внутри.
#
# Надёжная доставка: качаем шарды на диск через hf_hub_download (докачка+ретраи),
# читаем локально pyarrow, отбираем ovary — стриминг рвётся на флаки-сети.
# STHELAR 40x = 0.25 мкм/px, обучали на 0.27 — домен почти тот же.
#   python -u sthelar_bench.py --per-slide 400 --save-overlays 3

import argparse
import io
import os

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
import torch
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from huggingface_hub import hf_hub_download

from src.utils import get_device, ensure_dir
from seg_infer import build_uni
from graph_infer import load_gnn, uni_tokens
from cell_graph_train import adjacency

REPO = "FelicieGS/STHELAR_40x"
N_SHARDS = 54
SLIDES = ["ovary_s0", "ovary_s1"]
STHELAR_MPP = 0.25

# GT-тип STHELAR -> слой 1 (1 опухоль, 2 строма) и компартмент (1 CAF, 2 иммунный, 3 сосуд)
IGN, TUM, STR = 0, 1, 2
CMP_CAF, CMP_IMM, CMP_VAS = 1, 2, 3
GT_L1 = {"Epithelial": TUM, "Fibroblast_Myofibroblast": STR, "Blood_vessel": STR,
         "T_NK": STR, "B_Plasma": STR, "Myeloid": STR}
GT_CMP = {"Fibroblast_Myofibroblast": CMP_CAF, "T_NK": CMP_IMM, "B_Plasma": CMP_IMM,
          "Myeloid": CMP_IMM, "Blood_vessel": CMP_VAS}
CMPNAME = {CMP_CAF: "CAF", CMP_IMM: "immune", CMP_VAS: "vascular"}


def decode_img(v):
    """image-колонка parquet: HF-структура {bytes,path} или сырые байты."""
    if isinstance(v, dict):
        v = v.get("bytes") or open(v["path"], "rb").read()
    return np.asarray(Image.open(io.BytesIO(v)).convert("RGB"))


def build_luts(slide):
    """cell_id_int -> метка слоя 1 и компартмента (0 = игнор)."""
    p = hf_hub_download(REPO, f"cell_metadata/{slide}_cell_metadata.parquet", repo_type="dataset")
    df = pd.read_parquet(p, columns=["cell_id_int", "cells_final_label_group"])
    n = int(df["cell_id_int"].max()) + 2
    l1 = np.zeros(n, np.uint8)
    cmp = np.zeros(n, np.uint8)
    ids = df["cell_id_int"].to_numpy()
    grp = df["cells_final_label_group"]
    l1[ids] = grp.map(GT_L1).fillna(0).astype(np.uint8).to_numpy()
    cmp[ids] = grp.map(GT_CMP).fillna(0).astype(np.uint8).to_numpy()
    return l1, cmp


def tile_cells(idmap, lut_l1, lut_cmp):
    """Центроиды клеток тайла + их GT-метки (слой 1 и компартмент)."""
    idmap = np.clip(idmap, 0, len(lut_l1) - 1)
    h, w = idmap.shape
    flat = idmap.ravel()
    m = flat > 0
    if m.sum() == 0:
        return np.zeros((0, 2)), np.zeros(0, np.uint8), np.zeros(0, np.uint8)
    cid = flat[m]
    xs = (np.arange(flat.size) % w)[m].astype(np.float64)
    ys = (np.arange(flat.size) // w)[m].astype(np.float64)
    cnt = np.bincount(cid)
    sx = np.bincount(cid, weights=xs)
    sy = np.bincount(cid, weights=ys)
    uid = np.where(cnt > 0)[0]
    uid = uid[uid > 0]
    xy = np.c_[sx[uid] / cnt[uid], sy[uid] / cnt[uid]]
    return xy, lut_l1[uid], lut_cmp[uid]


def classify(xy, Fe, l1, l2, device, knn=8, max_edge_um=50.0):
    tree = cKDTree(xy)
    d, nb = tree.query(xy, k=min(knn + 1, len(xy)))
    if nb.ndim == 1:
        nb, d = nb[:, None], d[:, None]
    r_px = max_edge_um / STHELAR_MPP
    src = np.repeat(np.arange(len(xy)), nb.shape[1] - 1)
    dst = nb[:, 1:].ravel()
    ok = d[:, 1:].ravel() <= r_px
    if ok.sum():
        edges = np.vstack([np.r_[src[ok], dst[ok]], np.r_[dst[ok], src[ok]]]).astype(np.int64)
    else:
        edges = np.zeros((2, 0), np.int64)
    X = torch.from_numpy(Fe.astype(np.float32)).to(device)
    A = adjacency(edges, len(xy), device)
    with torch.no_grad():
        p1 = l1(X, A).argmax(1).cpu().numpy()
        p2 = l2(X, A).argmax(1).cpu().numpy()
    return p1, p2


def pred_cmp(p2):
    """L2-канал -> компартмент: гормональная+матриксная = CAF."""
    out = np.empty(len(p2), np.int64)
    out[(p2 == 0) | (p2 == 1)] = CMP_CAF
    out[p2 == 2] = CMP_IMM
    out[p2 == 3] = CMP_VAS
    return out


def process_tile(img, idmap, luts_s, uni, l1, l2, device, min_cells):
    """-> (pred_l1, gt_l1, pred_cmp, gt_cmp) по клеткам тайла, или None если мало клеток."""
    lut_l1, lut_cmp = luts_s
    xy, g1, gcmp = tile_cells(idmap, lut_l1, lut_cmp)
    keep = g1 > 0
    if keep.sum() < min_cells:
        return None
    xy, g1, gcmp = xy[keep], g1[keep], gcmp[keep]
    tok = uni_tokens(uni, img, device)
    G = tok.shape[0]
    h, w = idmap.shape
    ti = np.clip((xy[:, 1] / h * G).astype(int), 0, G - 1)
    tj = np.clip((xy[:, 0] / w * G).astype(int), 0, G - 1)
    Fe = tok[ti, tj]
    p1, p2 = classify(xy, Fe, l1, l2, device)
    return xy, p1, g1, pred_cmp(p2), gcmp


def report_l1(conf, tag):
    tt, ts, st, ss = int(conf[0, 0]), int(conf[0, 1]), int(conf[1, 0]), int(conf[1, 1])
    n = tt + ts + st + ss
    if not n:
        print(f"\n=== {tag} слой 1: нет клеток ==="); return
    dt = 2 * tt / (2 * tt + ts + st) if (2 * tt + ts + st) else float("nan")
    ds = 2 * ss / (2 * ss + st + ts) if (2 * ss + st + ts) else float("nan")
    print(f"\n=== {tag}: слой 1, опухоль/строма (по клеткам, N={n}) ===")
    print(f"  [pred\\GT]  tumor  stroma")
    print(f"  tumor   {tt:8d} {ts:7d}")
    print(f"  stroma  {st:8d} {ss:7d}")
    print(f"  tumor Dice {dt:.3f} | stroma Dice {ds:.3f} | accuracy {(tt+ss)/n:.3f}")


def report_cmp(conf, tag):
    n = int(conf.sum())
    if not n:
        print(f"\n=== {tag} компартмент: нет клеток ==="); return
    print(f"\n=== {tag}: компартмент стромы (по клеткам, N={n}) ===")
    hdr = "  ".join(f"{CMPNAME[c]:>8s}" for c in (CMP_CAF, CMP_IMM, CMP_VAS))
    print(f"  [pred\\GT] {hdr}")
    for i, c in enumerate((CMP_CAF, CMP_IMM, CMP_VAS)):
        row = "  ".join(f"{int(conf[i, j]):8d}" for j in range(3))
        print(f"  {CMPNAME[c]:8s} {row}")
    for i, c in enumerate((CMP_CAF, CMP_IMM, CMP_VAS)):
        tp = conf[i, i]; fp = conf[i].sum() - tp; fn = conf[:, i].sum() - tp
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
        print(f"    {CMPNAME[c]:8s} Dice {dice:.3f}")


def save_overlay(img, xy, p1, g1, slide, k):
    im = Image.fromarray(img.copy()); W = im.width
    pred, gt = im.copy(), im.copy()
    dp, dg = ImageDraw.Draw(pred), ImageDraw.Draw(gt)
    for (x, y), pp, gg in zip(xy, p1, g1):
        cp = (216, 65, 47) if pp == 0 else (120, 120, 120)
        cg = (216, 65, 47) if gg == TUM else (120, 120, 120)
        dp.ellipse([x - 3, y - 3, x + 3, y + 3], fill=cp)
        dg.ellipse([x - 3, y - 3, x + 3, y + 3], fill=cg)
    canvas = Image.new("RGB", (W * 3 + 20, im.height), (255, 255, 255))
    canvas.paste(im, (0, 0)); canvas.paste(pred, (W + 10, 0)); canvas.paste(gt, (2 * W + 20, 0))
    Image.fromarray(np.asarray(canvas)).save(ensure_dir("outputs/sthelar") / f"{slide}_{k}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer1", default="runs/graph/l1_final.pth")
    ap.add_argument("--layer2", default="runs/graph/l2_final.pth")
    ap.add_argument("--per-slide", type=int, default=400, help="тайлов на срез")
    ap.add_argument("--min-bpq", type=float, default=0.4, help="фильтр качества масок")
    ap.add_argument("--min-cells", type=int, default=12, help="минимум клеток на тайле")
    ap.add_argument("--shards", type=int, nargs="*", help="явные номера шардов (иначе 0..53)")
    ap.add_argument("--save-overlays", type=int, default=0, help="оверлеев на срез")
    args = ap.parse_args()

    device = get_device()
    print("device:", device, "| гружу UNI и графовые модели...")
    uni = build_uni(device)
    l1 = load_gnn(args.layer1, device)
    l2 = load_gnn(args.layer2, device)
    print("строю LUT типов клеток...")
    luts = {s: build_luts(s) for s in SLIDES}

    c1 = {s: np.zeros((2, 2), np.int64) for s in SLIDES}
    ccmp = {s: np.zeros((3, 3), np.int64) for s in SLIDES}
    got = {s: 0 for s in SLIDES}
    nov = {s: 0 for s in SLIDES}
    shards = args.shards if args.shards else range(N_SHARDS)

    for i in shards:
        if all(got[s] >= args.per_slide for s in SLIDES):
            break
        fn = f"data/train-{i:05d}-of-{N_SHARDS:05d}.parquet"
        try:
            path = hf_hub_download(REPO, fn, repo_type="dataset")
        except Exception as e:
            print(f"шард {i}: не скачался ({e}), пропускаю"); continue
        tbl = pq.read_table(path)
        sid = tbl.column("slide_id").to_pylist()
        want = [j for j, s in enumerate(sid) if s in SLIDES and got[s] < args.per_slide]
        uniq = {}
        for s in sid:
            uniq[s] = uniq.get(s, 0) + 1
        print(f"шард {i}: {dict(sorted(uniq.items()))} | беру ovary-тайлов: {len(want)}")
        if not want:
            continue
        img_col, cid_col = tbl.column("image"), tbl.column("cell_id_map")
        has_bpq = "bPQ" in tbl.column_names
        bpq_col = tbl.column("bPQ") if has_bpq else None
        for j in want:
            s = sid[j]
            if got[s] >= args.per_slide:
                continue
            if has_bpq and float(bpq_col[j].as_py()) < args.min_bpq:
                continue
            idmap = sp.load_npz(io.BytesIO(cid_col[j].as_py())).toarray()
            img = decode_img(img_col[j].as_py())
            res = process_tile(img, idmap, luts[s], uni, l1, l2, device, args.min_cells)
            if res is None:
                continue
            xy, p1, g1, pc, gcmp = res
            for a, b in zip((p1 == 1).astype(int), (g1 == STR).astype(int)):
                c1[s][a, b] += 1
            stc = gcmp > 0
            for a, b in zip(pc[stc], gcmp[stc]):
                ccmp[s][a - 1, b - 1] += 1
            if nov[s] < args.save_overlays:
                save_overlay(img, xy, p1, g1, s, nov[s]); nov[s] += 1
            got[s] += 1
        print(f"  собрано: { {s: got[s] for s in SLIDES} }")

    for s in SLIDES:
        report_l1(c1[s], s); report_cmp(ccmp[s], s)
    report_l1(sum(c1.values()), "ИТОГО ovary")
    report_cmp(sum(ccmp.values()), "ИТОГО ovary")
    os._exit(0)  # обойти сегфолт datasets+torch при финализации


if __name__ == "__main__":
    main()