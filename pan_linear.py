# Сравнение итогового декодера с более простыми и более широкими надстройками.
#
# Отвечает на два вопроса сразу.
#
# Нужен ли декодер. Простейшая альтернатива это свёртка с окном в одну точку из 1024
# сразу в классы, около трёх тысяч параметров: она разбирает каждый токен отдельно, не
# глядя на соседей, а карта получается растяжением решётки. Вариант mlp добавляет один
# скрытый слой, тоже без соседей. Разница с декодером это вклад пространственной части.
#
# Не тесно ли декодеру. Источников девять, а параметров 650 тысяч, и при добавлении
# простаты и STHELAR молочная железа с лёгким просели. Вариант decoder с ключом --width
# повторяет нынешнюю архитектуру при ширине 256 и расширяет её при большей: если потери
# были от тесноты, широкий декодер их вернёт, не теряя простату.
#
# Все варианты учатся на одних данных, одной функцией потерь и с одним правилом выбора
# эпохи, поэтому разница объясняется только устройством надстройки.
#
#   python -u pan_linear.py --heads linear mlp
#   python -u pan_linear.py --heads decoder --width 512 --epochs 25

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import get_device, ensure_dir
from pan_train import load
from pan_bench import load_model, patch_counts, summarize

GRID, OUT = 14, 112
IGNORE = -100


class FlatHead(nn.Module):
    """Классификация каждого токена по отдельности, соседи не учитываются."""

    def __init__(self, in_dim, n_classes, hidden=0):
        super().__init__()
        self.net = (nn.Sequential(nn.Conv2d(in_dim, hidden, 1), nn.GELU(),
                                  nn.Conv2d(hidden, n_classes, 1))
                    if hidden else nn.Conv2d(in_dim, n_classes, 1))

    def forward(self, tokens):
        b, n, d = tokens.shape
        x = tokens.transpose(1, 2).reshape(b, d, GRID, GRID)
        return F.interpolate(self.net(x), (OUT, OUT), mode="bilinear", align_corners=False)


class Decoder(nn.Module):
    """Тот же декодер, что в работе, но с настраиваемой шириной. При width=256
    совпадает с нынешним по числу каналов."""

    def __init__(self, in_dim, n_classes, width=256):
        super().__init__()
        def up(ci, co):
            return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                 nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.GELU())
        w = width
        self.proj = nn.Sequential(nn.Conv2d(in_dim, w, 1), nn.GELU())
        self.up1, self.up2, self.up3 = up(w, w // 2), up(w // 2, w // 4), up(w // 4, w // 8)
        self.head = nn.Conv2d(w // 8, n_classes, 1)

    def forward(self, tokens):
        b, n, d = tokens.shape
        x = tokens.transpose(1, 2).reshape(b, d, GRID, GRID)
        return self.head(self.up3(self.up2(self.up1(self.proj(x)))))


def targets(y, codes):
    out = np.full(y.shape, IGNORE, np.int64)
    for ch, code in enumerate(codes):
        out[y == code] = ch
    return out


@torch.no_grad()
def worst_iou(head, val, codes, device, bs=128):
    """Худший по наборам средний IoU: то же правило выбора эпохи, что у декодера."""
    head.eval()
    scores = []
    for name, (X, y) in val.items():
        inter, union = np.zeros(len(codes)), np.zeros(len(codes))
        for i in range(0, len(X), bs):
            pred = head(torch.from_numpy(X[i:i + bs].astype(np.float32)).to(device)).argmax(1).cpu().numpy()
            true = targets(y[i:i + bs], codes)
            for ch in range(len(codes)):
                p, t = pred == ch, true == ch
                inter[ch] += int((p & t).sum())
                union[ch] += int((p | t).sum())
        ok = union > 0
        scores.append(float((inter[ok] / union[ok]).mean()))
    return min(scores)


def train_head(head, data, val, codes, weights, args, device):
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device),
                                  ignore_index=IGNORE)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=1e-5)
    rng = np.random.default_rng(args.seed)
    best, best_state = -1.0, None

    for epoch in range(1, args.epochs + 1):
        head.train()
        order = []
        for name, (X, _) in data.items():
            take = min(args.per_source, len(X))
            order += [(name, i) for i in rng.choice(len(X), take, replace=False)]
        rng.shuffle(order)
        total = 0.0
        for i in range(0, len(order), args.bs):
            batch = order[i:i + args.bs]
            xb = np.stack([data[n][0][j] for n, j in batch]).astype(np.float32)
            yb = np.stack([data[n][1][j] for n, j in batch])
            loss = loss_fn(head(torch.from_numpy(xb).to(device)),
                           torch.from_numpy(targets(yb, codes)).to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(batch)
        sched.step()
        score = worst_iou(head, val, codes, device)
        mark = ""
        if score > best:
            best, best_state = score, {k: v.detach().clone() for k, v in head.state_dict().items()}
            mark = " *лучшая*"
        print(f"  эпоха {epoch:2d}  потеря {total / len(order):.4f}  худший mIoU {score:.3f}{mark}")

    head.load_state_dict(best_state)
    return head


@torch.no_grad()
def evaluate(model, codes, sets, device, title):
    rows = []
    for name, (X, y) in sets.items():
        s = summarize(patch_counts(model, codes, X, y, device))
        rows.append({"надстройка": title, "набор": name,
                     "опухоль": round(s["tumor_dice"], 3), "строма": round(s["stroma_dice"], 3),
                     "доля стромы": round(s["pred_stroma_share"], 3)})
        print(f"    {name}: опухоль {s['tumor_dice']:.3f} строма {s['stroma_dice']:.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["data/processed/seg", "data/processed/seg_pan"])
    ap.add_argument("--train", nargs="+",
                    default=["ovary_prime_he", "ovary2_he", "ovary3_he", "nct_crc",
                             "nct_crc_other", "ignite_train3", "tiger_ri",
                             "sthelar_train2", "aggc_train"])
    ap.add_argument("--val", nargs="+", default=["nct_crc_val", "nct_crc_val_other", "ignite_test3"])
    ap.add_argument("--test", nargs="+",
                    default=["nct_crc_val", "ignite_test3", "bcss3", "ubc", "aggc_test"])
    ap.add_argument("--model", default="outputs/models/pan_l1_deploy5.pth",
                    help="итоговый декодер, с которым сравниваем")
    ap.add_argument("--heads", nargs="+", default=["linear", "mlp"],
                    choices=["linear", "mlp", "decoder"])
    ap.add_argument("--hidden", type=int, default=256, help="ширина скрытого слоя варианта mlp")
    ap.add_argument("--width", type=int, default=512, help="ширина варианта decoder")
    ap.add_argument("--per-source", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.0003)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/results/pan/heads.csv")
    args = ap.parse_args()

    device = get_device()
    decoder, codes, _ = load_model(args.model, device)
    print("классы:", codes.tolist())

    data = {n: load(args.dirs, n) for n in args.train}
    val = {n: load(args.dirs, n) for n in args.val}
    test = {n: load(args.dirs, n) for n in args.test}

    share = np.zeros(len(codes))
    for _, (_, y) in data.items():
        for ch, code in enumerate(codes):
            share[ch] += int((y == code).sum())
    weights = list((share.sum() / (len(codes) * np.maximum(share, 1))).round(2))
    print("веса классов:", weights)

    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f"\nитоговый декодер ({n_dec} параметров)")
    rows = evaluate(decoder, codes, test, device, f"декодер, {n_dec} параметров")

    for kind in args.heads:
        if kind == "decoder":
            head = Decoder(1024, len(codes), args.width).to(device)
        else:
            head = FlatHead(1024, len(codes), args.hidden if kind == "mlp" else 0).to(device)
        n_par = sum(p.numel() for p in head.parameters())
        title = f"{kind}, {n_par} параметров"
        print(f"\n{title}")
        head = train_head(head, data, val, codes, weights, args, device)
        rows += evaluate(head, codes, test, device, title)
        torch.save({"state_dict": head.state_dict(), "n_classes": len(codes),
                    "channel_to_class": {str(i): int(c) for i, c in enumerate(codes)}},
                   f"outputs/models/pan_l1_head_{kind}.pth")

    full = pd.DataFrame(rows)
    print("\nDice опухоли")
    print(full.pivot(index="набор", columns="надстройка", values="опухоль").to_string())
    print("\nDice стромы")
    print(full.pivot(index="набор", columns="надстройка", values="строма").to_string())
    ensure_dir(Path(args.out).parent)
    full.to_csv(args.out, index=False)
    print("\nтаблица:", args.out)


if __name__ == "__main__":
    main()
