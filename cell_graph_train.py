# Графовая сеть по клеткам: узел = клетка, признак = токен UNI, рёбра = соседство.
#
# Свёртка по графу: признак клетки обновляется как собственный вклад плюс
# среднее по соседям. Два-три слоя дают охват ~100 мкм, то есть сеть учитывает
# архитектуру ткани, а не только текстуру в точке.
#
#   python cell_graph_train.py --train ovary_prime_he --val ovary2_he --out runs/graph/l1.pth

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.utils import get_device, ensure_dir, set_seed, TUMOR, CLASS_NAMES

STROMA_LIKE = (2, 3, 4, 5)
CLASSES = ("tumor", "stroma")


def load_graph(graph_dir, slide):
    d = np.load(Path(graph_dir) / f"{slide}_graph.npz")
    return d["feat"], d["label"], d["edges"]


def to_binary(label):
    y = np.full(len(label), -100, np.int64)
    y[label == TUMOR] = 0
    y[np.isin(label, STROMA_LIKE)] = 1
    return y


def adjacency(edges, n, device):
    """Разреженная матрица среднего по соседям (D^-1 A)."""
    src = torch.from_numpy(edges[0].astype(np.int64))
    dst = torch.from_numpy(edges[1].astype(np.int64))
    deg = torch.bincount(src, minlength=n).clamp(min=1).float()
    val = (1.0 / deg[src])
    idx = torch.stack([src, dst])
    return torch.sparse_coo_tensor(idx, val, (n, n)).coalesce().to(device)


class GraphConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.self_w = nn.Linear(ci, co)
        self.nb_w = nn.Linear(ci, co, bias=False)
        self.norm = nn.LayerNorm(co)

    def forward(self, h, a):
        return self.norm(self.self_w(h) + self.nb_w(torch.sparse.mm(a, h)))


class CellGNN(nn.Module):
    def __init__(self, in_dim=1024, hid=256, n_classes=2, layers=2, drop=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hid), nn.GELU())
        self.convs = nn.ModuleList([GraphConv(hid, hid) for _ in range(layers)])
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(hid, n_classes)

    def forward(self, x, a):
        h = self.proj(x)
        for c in self.convs:
            h = self.drop(self.act(c(h, a)))
        return self.head(h)


def metrics(pred, truth):
    out = {}
    ok = truth != -100
    for ch, nm in enumerate(CLASSES):
        p = (pred == ch) & ok
        t = (truth == ch) & ok
        inter = np.logical_and(p, t).sum()
        union = np.logical_or(p, t).sum()
        den = p.sum() + t.sum()
        out[nm] = (inter / union if union else float("nan"),
                   2 * inter / den if den else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/processed/graph")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--hid", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("device:", device, "| train", args.train, "| val", args.val)

    def prep(slides):
        feats, labs, eds, off = [], [], [], 0
        for s in slides:
            f, l, e = load_graph(args.graph_dir, s)
            feats.append(f); labs.append(l); eds.append(e + off)
            off += len(l)
        return (np.concatenate(feats), np.concatenate(labs),
                np.concatenate(eds, axis=1))

    ftr, ltr, etr = prep(args.train)
    fva, lva, eva = prep(args.val)
    ytr, yva = to_binary(ltr), to_binary(lva)
    print(f"train узлов {len(ytr)}, рёбер {etr.shape[1]} | "
          f"val узлов {len(yva)}, рёбер {eva.shape[1]}")

    Xtr = torch.from_numpy(ftr.astype(np.float32)).to(device)
    Xva = torch.from_numpy(fva.astype(np.float32)).to(device)
    Ttr = torch.from_numpy(ytr).to(device)
    Atr = adjacency(etr, len(ytr), device)
    Ava = adjacency(eva, len(yva), device)

    cnt = np.bincount(ytr[ytr >= 0], minlength=2).astype(float)
    w = cnt.sum() / (2 * np.maximum(cnt, 1))
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    print("веса: tumor=%.2f stroma=%.2f" % (w[0], w[1]))

    model = CellGNN(ftr.shape[1], args.hid, 2, args.layers, args.dropout).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"параметров: {n_par}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=w, ignore_index=-100)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    best, best_ep, best_state, bad = -1, 0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        loss = crit(model(Xtr, Atr), Ttr)
        loss.backward()
        opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            pred = model(Xva, Ava).argmax(1).cpu().numpy()
        m = metrics(pred, yva)
        miou = float(np.nanmean([m[c][0] for c in CLASSES]))
        mark = ""
        if miou > best:
            best, best_ep, bad = miou, ep, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            mark = " *best*"
        else:
            bad += 1
        if ep % 10 == 0 or mark:
            print("epoch %3d  loss %.4f  IoU: %s  (mIoU %.3f)%s"
                  % (ep, float(loss), "  ".join("%s %.3f" % (c, m[c][0])
                                                for c in CLASSES), miou, mark))
        if bad >= args.patience:
            print("early stop на", ep, "| лучший mIoU %.3f (эпоха %d)" % (best, best_ep))
            break

    model.load_state_dict(best_state)
    with torch.no_grad():
        pred = model(Xva, Ava).argmax(1).cpu().numpy()
    m = metrics(pred, yva)
    print("\n=== val %s (по клеткам) ===" % args.val)
    for c in CLASSES:
        print("  %-8s IoU %.3f  Dice %.3f" % (c, m[c][0], m[c][1]))

    ensure_dir(Path(args.out).parent)
    torch.save({"state_dict": model.state_dict(), "hid": args.hid,
                "layers": args.layers, "in_dim": ftr.shape[1],
                "classes": CLASSES, "val": args.val}, args.out)
    print("сохранено:", args.out)


if __name__ == "__main__":
    main()
