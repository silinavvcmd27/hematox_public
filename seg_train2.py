# Этап 2 сегментации, версия с двумя классами: опухоль и строма.
#
# Неразмеченные пиксели (в маске Вороного это 0) исключаются из функции потерь
# через ignore_index. Раньше они обучались как полноценный класс «фон», из-за
# чего модель отправляла туда до трети истинной стромы.
# Ткань от пустого поля отделяется порогом по яркости на этапе инференса.
#
# python seg_train2.py --val ovary3_he --select fixed --out outputs/models/seg2_ov3.pth

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.utils import (get_device, ensure_dir, set_seed, TRAIN_CLASSES,
                       CLASS_NAMES, N_CLASSES, IGNORE)
from seg_decoder import SegDecoder

COL = np.array([[245, 245, 245], [220, 50, 47], [38, 139, 210]], np.uint8)
# Каналы сети и их соответствие классам проекта. Порядок берётся из
# src/utils.TRAIN_CLASSES и пишется в чекпоинт: раньше инференс угадывал
# соответствие по числу каналов, и с пятью классами угадывание сломалось бы, а
# перепутанные опухоль и строма дают правдоподобное, но неверное TSR.
CHANNEL_TO_CLASS = {i: c for i, c in enumerate(TRAIN_CLASSES)}
CLASS_TO_CHANNEL = {c: i for i, c in CHANNEL_TO_CLASS.items()}
CLASSES = tuple((CLASS_NAMES[c], i) for i, c in CHANNEL_TO_CLASS.items())
IGNORE_TARGET = -100        # значение ignore_index у CrossEntropyLoss


def mask_to_target(mask):
    """Метки маски -> индексы каналов сети. IGNORE -> IGNORE_TARGET.

    БЫЛО: yb = ... - 1. Это подгонка под прежнюю нумерацию, где классов было
    два. С IGNORE = 255 вычитание единицы даёт метку 254, и обучение либо
    падает, либо считает мусор.
    """
    out = np.full(mask.shape, IGNORE_TARGET, np.int64)
    for cls, ch in CLASS_TO_CHANNEL.items():
        out[mask == cls] = ch
    unknown = ~np.isin(mask, list(CLASS_TO_CHANNEL) + [IGNORE])
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
    # считаем только по размеченному: 1 -> опухоль, 2 -> строма
    cnt = np.bincount(ys.reshape(-1), minlength=3).astype(float)[1:]
    w = cnt.sum() / (2 * np.maximum(cnt, 1))
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
    rows = []
    for i in idx:
        # в предсказании нет класса «нет метки», поэтому сдвигаем на единицу
        rows.append(np.concatenate([COL[y[i]], np.full((112, 6, 3), 255, np.uint8),
                                    COL[pred[i] + 1]], axis=1))
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
    ap.add_argument("--val", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--select", choices=["val", "fixed"], default="fixed",
                    help="val — лучшая эпоха по отложенному срезу, "
                         "fixed — фиксированный бюджет эпох без подглядывания")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--metrics-csv", default="outputs/results/seg2_metrics.csv")
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

    lab = (ytr > 0).mean()
    w = class_weights(ytr)
    print(f"размечено {100*lab:.1f}% пикселей | веса (опухоль/строма): "
          f"{w[0]:.2f} {w[1]:.2f}")

    model = SegDecoder(in_dim=Xtr.shape[-1], n_classes=N_CLASSES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=w.to(device), ignore_index=IGNORE_TARGET)

    n = len(Xtr)
    # Свой генератор: set_seed задаёт глобальное состояние numpy, но любой
    # вызов np.random в другом месте сдвинет последовательность, и прогон
    # не повторится.
    rng = np.random.default_rng(args.seed)
    best, best_state, bad, epochs_run = -1, None, 0, args.epochs
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

        _, m = evaluate(model, Xva, yva, device)
        # Среднее только по классам, которые есть в отложенном срезе: иначе
        # отсутствующий класс даёт nan и портит выбор лучшей эпохи.
        present = [CLASS_NAMES[c] for c in TRAIN_CLASSES
                   if c != 0 and (yva == c).sum() > 0]
        vals = [m[nm][0] for nm in present if nm in m and not np.isnan(m[nm][0])]
        miou = float(np.mean(vals)) if vals else float("nan")
        per = "  ".join(f"{nm} {m[nm][0]:.3f}" for nm in present if nm in m)
        print("epoch %3d  loss %.4f  IoU: %s  (mIoU %.3f)"
              % (ep, run / nb, per, miou))

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
    torch.save({"state_dict": model.state_dict(),
                "n_classes": N_CLASSES,
                "channel_to_class": CHANNEL_TO_CLASS,
                "class_names": CLASS_NAMES,
                "val_slide": args.val,
                "seed": args.seed}, args.out)

    pred, m = evaluate(model, Xva, yva, device)
    print("\n=== val %s ===" % args.val)
    for nm, _ in CLASSES:
        print("  %-7s IoU %.3f  Dice %.3f" % (nm, m[nm][0], m[nm][1]))

    labeled = yva > 0
    share = (pred[labeled] == 1).mean()
    print("  доля стромы в предсказании по размеченному: %.3f" % share)

    log_metrics(args.metrics_csv, {
        "val_slide": args.val,
        "select": args.select,
        "seed": args.seed,
        "epochs_run": epochs_run,
        "tumor_iou": round(m["tumor"][0], 4),
        "tumor_dice": round(m["tumor"][1], 4),
        "stroma_iou": round(m["stroma"][0], 4),
        "stroma_dice": round(m["stroma"][1], 4),
        "stroma_share": round(float(share), 4),
        "model": args.out,
    })
    print("метрики дописаны в", args.metrics_csv)

    out_png = ensure_dir("outputs/results") / f"seg2_{args.val}_{args.select}_preview.png"
    montage(yva, pred, str(out_png))


if __name__ == "__main__":
    main()
