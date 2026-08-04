# Слой 1: опухоль vs строма. Учится на Visium-фичах (грубый масштаб ~3.4 мкм/px).
# Метка: 1 -> tumor, {2,3,4,5} -> stroma, 0 -> фон (ignore).
#
#   python seg_train_layer1.py --val SP3 SP6 --out runs/layer1/fold_a.pth
#   python seg_train_layer1.py --val all --out runs/layer1/final.pth
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.utils import get_device, ensure_dir, set_seed, TUMOR, STROMA, CLASS_NAMES
from seg_decoder import SegDecoder

CHANNEL_TO_CLASS = {0: TUMOR, 1: STROMA}       # ch0 -> tumor, ch1 -> stroma
N_CLASSES = 2
STROMA_LIKE = (2, 3, 4, 5)                     # любой стромальный код -> stroma
IGNORE_TARGET = -100

COL = np.array([[245, 245, 245], [220, 50, 47], [128, 128, 128]], np.uint8)


def mask_to_target(mask):
    out = np.full(mask.shape, IGNORE_TARGET, np.int64)
    out[mask == TUMOR] = 0
    out[np.isin(mask, STROMA_LIKE)] = 1
    return out


def load_slide(seg_dir, name):
    d = np.load(Path(seg_dir) / f"{name}_feat.npz")
    return d["X"], d["y"]


def smooth_labels(y, k):
    """Мажоритарный фильтр k×k: убирает 'соль-перец' Вороного, оставляя зоны."""
    import cv2
    classes = np.array([0, 1, 2, 3, 4, 5], np.uint8)
    out = np.empty_like(y)
    for i in range(len(y)):
        m = y[i].astype(np.uint8)
        votes = np.stack([cv2.blur((m == c).astype(np.float32), (k, k))
                          for c in classes], -1)
        out[i] = classes[votes.argmax(-1)]
    return out


def class_weights(ys):
    t = mask_to_target(ys).reshape(-1)
    t = t[t != IGNORE_TARGET]
    cnt = np.bincount(t, minlength=N_CLASSES).astype(float)
    w = cnt.sum() / (N_CLASSES * np.maximum(cnt, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def evaluate(model, X, y, device, bs=32):
    model.eval()
    preds = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs].astype(np.float32)).to(device)
        preds.append(model(xb).argmax(1).cpu().numpy())
    pred = np.concatenate(preds, 0)
    truth = mask_to_target(y)
    labeled = truth != IGNORE_TARGET
    out = {}
    for ch, cls in CHANNEL_TO_CLASS.items():
        p = (pred == ch) & labeled
        t = (truth == ch) & labeled
        inter = np.logical_and(p, t).sum()
        union = np.logical_or(p, t).sum()
        denom = p.sum() + t.sum()
        out[CLASS_NAMES[cls]] = (inter / union if union else float("nan"),
                                 2 * inter / denom if denom else float("nan"))
    return pred, out


def montage(y, pred, path, k=8):
    from PIL import Image
    idx = np.random.RandomState(0).choice(len(y), min(k, len(y)), replace=False)
    rows = []
    for i in idx:
        gt = np.zeros_like(y[i])
        gt[y[i] == TUMOR] = 1
        gt[np.isin(y[i], STROMA_LIKE)] = 2
        pr = pred[i] + 1
        rows.append(np.concatenate([COL[gt], np.full((112, 6, 3), 255, np.uint8),
                                    COL[pr]], axis=1))
    grid = np.concatenate([np.concatenate([r, np.full((6, r.shape[1], 3), 255, np.uint8)])
                           for r in rows], axis=0)
    Image.fromarray(grid).save(path)
    print("превью (истина | предсказание):", path)


def log_metrics(path, row):
    path = Path(path)
    ensure_dir(path.parent)
    first = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if first:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-dir", default="data/processed/seg")
    ap.add_argument("--slides", nargs="+",
                    default=["ovary_prime_he", "ovary2_he", "ovary3_he"])
    ap.add_argument("--visium-dir", default="data/processed/visium")
    ap.add_argument("--visium", nargs="+", default=[],
                    help="доп. Visium-образцы (грубый масштаб), напр. SP1 SP2 ...")
    ap.add_argument("--sthelar-dir", default="data/processed/sthelar")
    ap.add_argument("--sthelar", nargs="+", default=[],
                    help="STHELAR-слайды (резкий масштаб), или 'all' — все feat.npz из sthelar-dir")
    ap.add_argument("--val", nargs="+", required=True,
                    help="образцы для валидации, или 'all' — учить на всех без валидации")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--smooth", type=int, default=0,
                    help="mode-фильтр меток k×k (0=выкл), убирает крапинки Вороного")
    ap.add_argument("--val-stride", type=int, default=1,
                    help="брать каждый N-й патч валидации; при --augment 4 ставьте 4, "
                         "чтобы мерить по неповёрнутым оригиналам")
    ap.add_argument("--metrics-csv", default="outputs/results/layer1_metrics.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    full = args.val == ["all"]
    print("device:", device, "| seed:", args.seed,
          "| режим:", "full-train" if full else f"val={args.val}")

    sth = args.sthelar
    if sth == ["all"]:
        sth = sorted(p.name[:-len("_feat.npz")]
                     for p in Path(args.sthelar_dir).glob("*_feat.npz"))
    src = {s: args.seg_dir for s in args.slides}
    src.update({s: args.visium_dir for s in args.visium})
    src.update({s: args.sthelar_dir for s in sth})
    all_slides = list(src)

    if full:
        tr = all_slides
        Xva = yva = None
    else:
        val = set(args.val)
        tr = [s for s in all_slides if s not in val]
        Xva = np.concatenate([load_slide(src[s], s)[0] for s in args.val], 0)
        yva = np.concatenate([load_slide(src[s], s)[1] for s in args.val], 0)
        if args.val_stride > 1:
            Xva, yva = Xva[::args.val_stride], yva[::args.val_stride]
    Xtr = np.concatenate([load_slide(src[s], s)[0] for s in tr], 0)
    ytr = np.concatenate([load_slide(src[s], s)[1] for s in tr], 0)
    print(f"train {tr} -> {len(Xtr)} патчей"
          + ("" if full else f" | val {args.val} -> {len(Xva)}"))

    if args.smooth:
        ytr = smooth_labels(ytr, args.smooth)
        if yva is not None:
            yva = smooth_labels(yva, args.smooth)
        print(f"метки сглажены mode-фильтром k={args.smooth}")

    w = class_weights(ytr)
    print("веса: tumor=%.2f stroma=%.2f" % (w[0], w[1]))

    model = SegDecoder(in_dim=Xtr.shape[-1], n_classes=N_CLASSES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=w.to(device), ignore_index=IGNORE_TARGET)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    rng = np.random.default_rng(args.seed)
    n = len(Xtr)
    best, best_ep, best_state, bad = -1, 0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(n)
        run, nb = 0.0, 0
        for i in range(0, n, args.bs):
            j = perm[i:i + args.bs]
            xb = torch.tensor(Xtr[j].astype(np.float32)).to(device)
            yb = torch.from_numpy(mask_to_target(ytr[j])).to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            run += float(loss.detach()); nb += 1
        sched.step()
        lr_now = sched.get_last_lr()[0]

        if full:
            print("epoch %3d  loss %.4f  lr %.1e" % (ep, run / nb, lr_now))
            continue

        _, m = evaluate(model, Xva, yva, device)
        ious = [m[nm][0] for nm in m if not np.isnan(m[nm][0])]
        miou = float(np.mean(ious)) if ious else float("nan")
        per = "  ".join("%s %.3f" % (nm, m[nm][0]) for nm in m)
        mark = ""
        if miou > best:
            best, best_ep, bad = miou, ep, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            mark = " *best*"
        else:
            bad += 1
        print("epoch %3d  loss %.4f  lr %.1e  IoU: %s  (mIoU %.3f)%s"
              % (ep, run / nb, lr_now, per, miou, mark))
        if bad >= args.patience:
            print("early stop на", ep, "| лучший mIoU %.3f (эпоха %d)" % (best, best_ep))
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print("восстановлена лучшая модель, эпоха", best_ep)

    ensure_dir(Path(args.out).parent)
    torch.save({"state_dict": model.state_dict(), "n_classes": N_CLASSES,
                "channel_to_class": CHANNEL_TO_CLASS, "class_names": CLASS_NAMES,
                "layer": 1, "val": args.val, "seed": args.seed}, args.out)
    print("сохранено:", args.out)

    if full:
        return
    pred, m = evaluate(model, Xva, yva, device)
    print("\n=== val %s ===" % args.val)
    for nm in m:
        print("  %-8s IoU %.3f  Dice %.3f" % (nm, m[nm][0], m[nm][1]))
    row = {"val": "+".join(args.val), "seed": args.seed, "best_epoch": best_ep}
    for nm in m:
        row[f"{nm}_iou"] = round(m[nm][0], 4)
    log_metrics(args.metrics_csv, row)
    out_png = ensure_dir("outputs/results") / ("layer1_%s_preview.png" % "_".join(args.val))
    montage(yva, pred, str(out_png))


if __name__ == "__main__":
    main()