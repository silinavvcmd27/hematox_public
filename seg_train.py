# Этап 2 сегментации: учим декодер на готовых токен-признаках UNI.

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.utils import get_device, ensure_dir, set_seed
from seg_decoder import SegDecoder

COL = np.array([[245, 245, 245], [220, 50, 47], [38, 139, 210]], np.uint8)


def load_slide(seg_dir, slide):
    d = np.load(Path(seg_dir) / f"{slide}_feat.npz")
    return d["X"], d["y"]


def class_weights(ys):
    cnt = np.bincount(ys.reshape(-1), minlength=3).astype(float)
    w = cnt.sum() / (3 * np.maximum(cnt, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def iou_dice(pred, y, c):
    p = pred == c
    t = y == c
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    iou = inter / union if union else float("nan")
    dice = 2 * inter / (p.sum() + t.sum()) if (p.sum() + t.sum()) else float("nan")
    return iou, dice


@torch.no_grad()
def evaluate(model, X, y, device, bs=32):
    model.eval()
    preds = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs].astype(np.float32)).to(device)
        preds.append(model(xb).argmax(1).cpu().numpy())
    pred = np.concatenate(preds, 0)
    return pred, {nm: iou_dice(pred, y, c) for c, nm in [(1, "tumor"), (2, "stroma")]}


def montage(y, pred, path, k=8):
    from PIL import Image
    idx = np.random.RandomState(0).choice(len(y), min(k, len(y)), replace=False)
    rows = []
    for i in idx:
        rows.append(np.concatenate([COL[y[i]], np.full((112, 6, 3), 255, np.uint8),
                                    COL[pred[i]]], axis=1))
    grid = np.concatenate([np.concatenate([r, np.full((6, r.shape[1], 3), 255, np.uint8)])
                           for r in rows], axis=0)
    Image.fromarray(grid).save(path)
    print("истина | предсказание:", path)


def log_metrics(path, row):
    path = Path(path)
    ensure_dir(path.parent)
    first_write = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if first_write:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-dir", default="data/processed/seg")
    ap.add_argument("--slides", nargs="+",
                    default=["ovary_prime_he", "ovary2_he", "ovary3_he"])
    ap.add_argument("--val", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--select", choices=["val", "fixed"], default="fixed",
                    help="val — лучшая эпоха по отложенному срезу, "
                         "fixed — фиксированный бюджет эпох без подглядывания")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--metrics-csv", default="outputs/results/seg_metrics.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("device:", device, "| seed:", args.seed, "| отбор эпохи:", args.select)

    tr = [s for s in args.slides if s != args.val]
    Xtr = np.concatenate([load_slide(args.seg_dir, s)[0] for s in tr], 0)
    ytr = np.concatenate([load_slide(args.seg_dir, s)[1] for s in tr], 0)
    Xva, yva = load_slide(args.seg_dir, args.val)
    print(f"train {tr} -> {Xtr.shape[0]} патчей | val {args.val} -> {Xva.shape[0]}")

    w = class_weights(ytr)
    print("веса классов (bg/tumor/stroma): %.2f %.2f %.2f" % tuple(w.tolist()))

    model = SegDecoder(in_dim=Xtr.shape[-1], n_classes=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=w.to(device))

    n = len(Xtr)
    best, best_state, bad, epochs_run = -1, None, 0, args.epochs
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(n)
        run, nb = 0.0, 0
        for i in range(0, n, args.bs):
            j = perm[i:i + args.bs]
            xb = torch.tensor(Xtr[j].astype(np.float32)).to(device)
            yb = torch.tensor(ytr[j].astype(np.int64)).to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            run += float(loss.detach())
            nb += 1

        _, m = evaluate(model, Xva, yva, device)
        miou = np.nanmean([m["tumor"][0], m["stroma"][0]])
        print("epoch %3d  loss %.4f  IoU tumor %.3f stroma %.3f  (mIoU %.3f)"
              % (ep, run / nb, m["tumor"][0], m["stroma"][0], miou))

        if args.select != "val":
            continue
        if miou > best:
            best, bad = miou, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                epochs_run = ep
                print("ранняя остановка на эпохе", ep, "| лучший mIoU %.3f" % best)
                break

    if best_state:
        model.load_state_dict(best_state)
    ensure_dir(Path(args.out).parent)
    torch.save({"state_dict": model.state_dict()}, args.out)

    pred, m = evaluate(model, Xva, yva, device)
    print("\n=== val %s ===" % args.val)
    for nm in ("tumor", "stroma"):
        print("  %-7s IoU %.3f  Dice %.3f" % (nm, m[nm][0], m[nm][1]))

    log_metrics(args.metrics_csv, {
        "val_slide": args.val,
        "select": args.select,
        "seed": args.seed,
        "epochs_run": epochs_run,
        "tumor_iou": round(m["tumor"][0], 4),
        "tumor_dice": round(m["tumor"][1], 4),
        "stroma_iou": round(m["stroma"][0], 4),
        "stroma_dice": round(m["stroma"][1], 4),
        "model": args.out,
    })
    print("метрики дописаны в", args.metrics_csv)

    out_png = ensure_dir("outputs/results") / f"seg_{args.val}_{args.select}_preview.png"
    montage(yva, pred, str(out_png))


if __name__ == "__main__":
    main()