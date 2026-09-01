import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.utils import (get_device, ensure_dir, set_seed, TRAIN_CLASSES,
                       CLASS_NAMES, N_CLASSES, IGNORE)
from seg_decoder import SegDecoder

COL = np.array([
    [245, 245, 245],
    [220,  50,  47],
    [255, 165,   0],
    [ 38, 139, 210],
    [ 42, 161,  52],
    [128, 128, 128],
], np.uint8)

IGNORE_TARGET = -100
CHANNEL_TO_CLASS = CLASS_TO_CHANNEL = CLASSES = None
N_TRAIN = 0


def set_classes(classes):
    """Какие зоны учим. Маски пересобирать не надо: признаки от разметки не
    зависят, меняется только отображение меток в цели."""
    global CHANNEL_TO_CLASS, CLASS_TO_CHANNEL, CLASSES, N_TRAIN
    CHANNEL_TO_CLASS = {i: c for i, c in enumerate(classes)}
    CLASS_TO_CHANNEL = {c: i for i, c in CHANNEL_TO_CLASS.items()}
    CLASSES = tuple((CLASS_NAMES[c], i) for i, c in CHANNEL_TO_CLASS.items())
    N_TRAIN = len(classes)


set_classes(TRAIN_CLASSES)


def mask_to_target(mask):
    """Зоны вне обучаемого набора уходят в игнор, а не в фон: класс, который мы
    не предсказываем, не должен считаться ошибкой ни в лоссе, ни в метриках."""
    out = np.full(mask.shape, IGNORE_TARGET, np.int64)
    for cls, ch in CLASS_TO_CHANNEL.items():
        out[mask == cls] = ch
    unknown = ~np.isin(mask, list(TRAIN_CLASSES) + [0, IGNORE])
    if unknown.any():
        raise ValueError(
            "в маске значения, которых нет ни в TRAIN_CLASSES, ни IGNORE: "
            f"{np.unique(mask[unknown]).tolist()}. "
            "Пересоберите маски обновлённым make_seg_masks_v.py")
    return out


def load_slide(seg_dir, slide):
    d = np.load(Path(seg_dir) / f"{slide}_feat.npz")
    return d["X"], d["y"]


def class_weights(ys):
    target = mask_to_target(ys)
    flat = target.reshape(-1)
    flat = flat[flat != IGNORE_TARGET]
    cnt = np.bincount(flat, minlength=N_TRAIN).astype(float)
    w = cnt.sum() / (N_TRAIN * np.maximum(cnt, 1))
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
    for nm, c in CLASSES:
        p = (pred == c) & labeled
        t = (truth == c) & labeled
        inter = np.logical_and(p, t).sum()
        union = np.logical_or(p, t).sum()
        denom = p.sum() + t.sum()
        out[nm] = (inter / union if union else float("nan"),
                   2 * inter / denom if denom else float("nan"))
    return pred, out


def montage(y, pred, path, k=8):
    from PIL import Image
    idx = np.random.RandomState(0).choice(len(y), min(k, len(y)), replace=False)
    # предсказание идёт в номерах каналов, а цвета заданы по кодам зон
    lut = np.array([CHANNEL_TO_CLASS[ch] for ch in range(N_TRAIN)], np.uint8)
    rows = []
    for i in idx:
        gt_img = COL[np.clip(y[i], 0, len(COL) - 1)]
        pr_img = COL[np.clip(lut[pred[i]], 0, len(COL) - 1)]
        rows.append(np.concatenate([gt_img, np.full((112, 6, 3), 255, np.uint8),
                                    pr_img], axis=1))
    grid = np.concatenate([np.concatenate([r, np.full((6, r.shape[1], 3), 255, np.uint8)])
                           for r in rows], axis=0)
    Image.fromarray(grid).save(path)
    print("превью (истина | предсказание):", path)


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
    ap.add_argument("--val", required=True,
                    help="слайд для валидации, или 'all' — обучение на всех без валидации")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--metrics-csv", default="outputs/results/seg2_metrics.csv")
    ap.add_argument("--classes", nargs="+", type=int, default=list(TRAIN_CLASSES),
                    help="коды зон для обучения, остальные размеченные уходят в игнор; "
                         + ", ".join(f"{c}={CLASS_NAMES[c]}" for c in TRAIN_CLASSES))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_classes(args.classes)
    set_seed(args.seed)
    device = get_device()
    full_train = args.val == "all"
    print("device:", device, "| seed:", args.seed,
          "| режим:", "full-train (все слайды)" if full_train else f"val={args.val}")

    if full_train:
        tr = args.slides
        Xtr = np.concatenate([load_slide(args.seg_dir, s)[0] for s in tr], 0)
        ytr = np.concatenate([load_slide(args.seg_dir, s)[1] for s in tr], 0)
        Xva, yva = None, None
        print(f"train {tr} -> {Xtr.shape[0]} патчей | без валидации")
    else:
        tr = [s for s in args.slides if s != args.val]
        Xtr = np.concatenate([load_slide(args.seg_dir, s)[0] for s in tr], 0)
        ytr = np.concatenate([load_slide(args.seg_dir, s)[1] for s in tr], 0)
        Xva, yva = load_slide(args.seg_dir, args.val)
        print(f"train {tr} -> {Xtr.shape[0]} патчей | val {args.val} -> {Xva.shape[0]}")

    lab = (ytr > 0).mean()
    w = class_weights(ytr)
    print(f"размечено {100*lab:.1f}% пикселей | веса: "
          + " ".join(f"{CLASS_NAMES[c]}={w[i]:.2f}" for i, c in CHANNEL_TO_CLASS.items()))

    model = SegDecoder(in_dim=Xtr.shape[-1], n_classes=N_TRAIN).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=w.to(device), ignore_index=IGNORE_TARGET)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    n = len(Xtr)
    rng = np.random.default_rng(args.seed)
    best_miou, best_ep = -1, 0
    best_state = None
    no_improve = 0

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
            run += float(loss.detach())
            nb += 1
        sched.step()
        lr_now = sched.get_last_lr()[0]

        if full_train:
            print("epoch %3d  loss %.4f  lr %.1e" % (ep, run / nb, lr_now))
        else:
            _, m = evaluate(model, Xva, yva, device)
            present = [CLASS_NAMES[c] for c in CHANNEL_TO_CLASS.values()
                       if (yva == c).sum() > 0]
            vals = [m[nm][0] for nm in present if nm in m and not np.isnan(m[nm][0])]
            miou = float(np.mean(vals)) if vals else float("nan")
            per = "  ".join(f"{nm} {m[nm][0]:.3f}" for nm in present if nm in m)

            marker = ""
            if miou > best_miou:
                best_miou, best_ep = miou, ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
                marker = " *best*"
            else:
                no_improve += 1

            print("epoch %3d  loss %.4f  lr %.1e  IoU: %s  (mIoU %.3f)%s"
                  % (ep, run / nb, lr_now, per, miou, marker))

            if no_improve >= args.patience:
                print(f"early stop на эпохе {ep} | лучший mIoU {best_miou:.3f} на эпохе {best_ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nвосстановлена лучшая модель (эпоха {best_ep}, mIoU {best_miou:.3f})")

    ensure_dir(Path(args.out).parent)
    torch.save({"state_dict": model.state_dict(),
                "n_classes": N_TRAIN,
                "channel_to_class": CHANNEL_TO_CLASS,
                "class_names": CLASS_NAMES,
                "val_slide": args.val,
                "seed": args.seed}, args.out)

    if full_train:
        pred, m = evaluate(model, Xtr, ytr, device)
        print("\n=== train (все слайды) ===")
        for nm, _ in CLASSES:
            print("  %-20s IoU %.3f  Dice %.3f" % (nm, m[nm][0], m[nm][1]))
        print("\nмодель сохранена:", args.out)
    else:
        pred, m = evaluate(model, Xva, yva, device)
        print("\n=== val %s ===" % args.val)
        for nm, _ in CLASSES:
            print("  %-20s IoU %.3f  Dice %.3f" % (nm, m[nm][0], m[nm][1]))

        labeled = mask_to_target(yva) != IGNORE_TARGET
        for nm, c in CLASSES:
            frac = (pred[labeled] == c).mean()
            print(f"  доля {nm}: {frac:.3f}")

        row = {
            "val_slide": args.val,
            "classes": "+".join(CLASS_NAMES[c] for c in CHANNEL_TO_CLASS.values()),
            "seed": args.seed,
            "lr": args.lr,
            "best_epoch": best_ep,
            "epochs_run": ep,
            "model": args.out,
        }
        for nm, _ in CLASSES:
            row[f"{nm}_iou"] = round(m[nm][0], 4)
            row[f"{nm}_dice"] = round(m[nm][1], 4)
        log_metrics(args.metrics_csv, row)
        print("метрики дописаны в", args.metrics_csv)

        out_png = ensure_dir("outputs/results") / f"seg2_{args.val}_preview.png"
        montage(yva, pred, str(out_png))


if __name__ == "__main__":
    main()