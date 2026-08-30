# Слой 2 на графе клеток: подтип стромы.
# Те же узлы и рёбра, но учимся только на стромальных клетках; опухоль и всё
# прочее — в ignore. Классы: гормональная, матриксная, иммунные, прочая строма.
#
# Гормональная строма есть только у ovary_prime, поэтому кросс-валидация между
# срезами по ней невозможна. Для честной оценки — пространственный сплит:
# половина среза в обучение, половина в валидацию, с зазором между ними.
#
#   пространственный сплит одного среза:
#     python cell_graph_train2.py --spatial-split ovary_prime_he --out runs/graph/l2.pth
#   между срезами:
#     python cell_graph_train2.py --train ovary_prime_he --val ovary2_he --out runs/graph/l2_cross.pth

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.utils import get_device, ensure_dir, set_seed
from cell_graph_train import CellGNN, adjacency, load_graph

# код класса в графе -> канал слоя 2
L2 = {2: 0, 3: 1, 4: 2, 5: 3}
CLASSES2 = ("hormonal", "matrix", "immune", "other")
# режим --merge-caf: гормональная+матриксная -> один класс "caf"
L2_CAF = {2: 0, 3: 0, 4: 1, 5: 2}
CLASSES2_CAF = ("caf", "immune", "other")

# режим --caf-only: только два типа CAF, иммунные и прочие уходят в NA
L2_CAFONLY = {2: 0, 3: 1}
CLASSES2_CAFONLY = ("hormonal", "matrix")

MODE = "full"          # full | merge-caf | caf-only


def _maps():
    if MODE == "merge-caf":
        return L2_CAF, CLASSES2_CAF
    if MODE == "caf-only":
        return L2_CAFONLY, CLASSES2_CAFONLY
    return L2, CLASSES2


def to_layer2(label):
    mp = _maps()[0]
    y = np.full(len(label), -100, np.int64)
    for code, ch in mp.items():
        y[label == code] = ch
    return y


def subgraph(feat, label, pos, edges, mask):
    """Подграф по булевой маске узлов: рёбра только внутри выборки."""
    idx = np.where(mask)[0]
    remap = np.full(len(mask), -1, np.int64)
    remap[idx] = np.arange(len(idx))
    ke = mask[edges[0]] & mask[edges[1]]
    e = np.vstack([remap[edges[0][ke]], remap[edges[1][ke]]])
    return feat[idx], label[idx], pos[idx], e


def load_all(graph_dir, slides):
    feats, labs, poss, eds, off = [], [], [], [], 0
    for s in slides:
        d = np.load(Path(graph_dir) / f"{s}_graph.npz")
        feats.append(d["feat"]); labs.append(d["label"])
        poss.append(d["pos"]); eds.append(d["edges"] + off)
        off += len(d["label"])
    return (np.concatenate(feats), np.concatenate(labs),
            np.concatenate(poss), np.concatenate(eds, axis=1))


def active_classes():
    return _maps()[1]


def metrics(pred, truth):
    out = {}
    ok = truth != -100
    for c, nm in enumerate(active_classes()):
        p = (pred == c) & ok
        t = (truth == c) & ok
        inter = np.logical_and(p, t).sum()
        union = np.logical_or(p, t).sum()
        den = p.sum() + t.sum()
        out[nm] = (inter / union if union else float("nan"),
                   2 * inter / den if den else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/processed/graph")
    ap.add_argument("--train", nargs="+")
    ap.add_argument("--val", nargs="+")
    ap.add_argument("--spatial-split", help="один срез: левая половина учит, правая валидирует")
    ap.add_argument("--gap-um", type=float, default=100.0, help="зазор между половинами")
    ap.add_argument("--hid", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--merge-caf", action="store_true",
                    help="гормональную+матриксную слить в один класс caf")
    ap.add_argument("--caf-only", action="store_true",
                    help="только гормональная и матриксная строма, иммунные и прочие в NA")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    global MODE
    MODE = ("caf-only" if args.caf_only else
            "merge-caf" if args.merge_caf else "full")
    CLS = active_classes()
    NC = len(CLS)

    set_seed(args.seed)
    device = get_device()

    if args.spatial_split:
        d = np.load(Path(args.graph_dir) / f"{args.spatial_split}_graph.npz")
        feat, label, pos, edges = d["feat"], d["label"], d["pos"], d["edges"]
        mpp = float(d["mpp"]) if "mpp" in d else 0.2738
        gap = args.gap_um / mpp
        xmed = np.median(pos[:, 0])
        left = pos[:, 0] < xmed - gap
        right = pos[:, 0] >= xmed + gap
        ftr, ltr, ptr, etr = subgraph(feat, label, pos, edges, left)
        fva, lva, pva, eva = subgraph(feat, label, pos, edges, right)
        print(f"пространственный сплит {args.spatial_split} по x={xmed:.0f}, "
              f"зазор {gap:.0f}px")
    else:
        if not (args.train and args.val):
            raise SystemExit("нужен либо --spatial-split, либо --train и --val")
        ftr, ltr, ptr, etr = load_all(args.graph_dir, args.train)
        fva, lva, pva, eva = load_all(args.graph_dir, args.val)
        print(f"train {args.train} | val {args.val}")

    ytr, yva = to_layer2(ltr), to_layer2(lva)
    print(f"train: {len(ytr)} узлов, {int((ytr>=0).sum())} стромальных, {etr.shape[1]} рёбер")
    print(f"val:   {len(yva)} узлов, {int((yva>=0).sum())} стромальных, {eva.shape[1]} рёбер")
    for c, nm in enumerate(CLS):
        a, b = int((ytr == c).sum()), int((yva == c).sum())
        print(f"  {nm:10s} train {a:7d}  val {b:7d}")

    Xtr = torch.from_numpy(ftr.astype(np.float32)).to(device)
    Xva = torch.from_numpy(fva.astype(np.float32)).to(device)
    Ttr = torch.from_numpy(ytr).to(device)
    Atr = adjacency(etr, len(ytr), device)
    Ava = adjacency(eva, len(yva), device)

    cnt = np.bincount(ytr[ytr >= 0], minlength=NC).astype(float)
    w = cnt.sum() / (NC * np.maximum(cnt, 1))
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    print("веса:", " ".join("%s=%.2f" % (nm, w[c]) for c, nm in enumerate(CLS)))

    model = CellGNN(ftr.shape[1], args.hid, NC, args.layers, args.dropout).to(device)
    print("параметров:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=w, ignore_index=-100)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    present = [c for c in range(NC) if (yva == c).sum() > 0]
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
        vals = [m[CLS[c]][0] for c in present if not np.isnan(m[CLS[c]][0])]
        miou = float(np.mean(vals)) if vals else float("nan")
        # отбор по худшему классу: модель тут норовит схлопнуться в один
        # класс, и у такой константы средний IoU выше, чем у слабой, но
        # настоящей модели. По минимуму константа получает ноль и выбывает
        score = float(np.min(vals)) if vals else float("nan")
        if score > best:
            best, best_ep, bad = score, ep, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            mark = " *best*"
        else:
            bad += 1
            mark = ""
        if ep % 20 == 0 or mark:
            per = "  ".join("%s %.3f" % (CLS[c], m[CLS[c]][0]) for c in present)
            print("epoch %3d  loss %.4f  IoU: %s  (mIoU %.3f, min %.3f)%s"
                  % (ep, float(loss.detach()), per, miou, score, mark))
        if bad >= args.patience:
            print("early stop на", ep,
                  "| лучший min IoU %.3f (эпоха %d)" % (best, best_ep))
            break

    model.load_state_dict(best_state)
    with torch.no_grad():
        pred = model(Xva, Ava).argmax(1).cpu().numpy()
    m = metrics(pred, yva)
    print("\n=== val (подтип стромы, по клеткам) ===")
    for c, nm in enumerate(CLS):
        flag = "" if (yva == c).sum() else "  (нет в валидации)"
        print("  %-10s IoU %.3f  Dice %.3f%s" % (nm, m[nm][0], m[nm][1], flag))

    ensure_dir(Path(args.out).parent)
    torch.save({"state_dict": model.state_dict(), "hid": args.hid,
                "layers": args.layers, "in_dim": ftr.shape[1],
                "classes": CLS}, args.out)
    print("сохранено:", args.out)


if __name__ == "__main__":
    main()