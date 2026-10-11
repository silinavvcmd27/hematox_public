# Второй слой: окно едет по строме и выделяет матриксную.
#
# Свёртка по графу даёт признак каждой клетки с учётом соседей (около 45 мкм за
# два слоя). Окно --win-um с шагом --stride-um усредняет признаки стромальных
# клеток внутри и решает, матриксный это участок или нет. Идея окна из атласа
# TLS (Cho et al., Science 2026): там карту скора так же режут на участки.
#
# Метка окна берётся из размеченных фибробластов внутри него: матриксных от --hi
# и выше это 1, до --lo это 0, между ними и при малом числе фибробластов окно в
# потерю не идёт. Иммунные, сосуды и неразмеченные клетки окна участвуют
# признаками, но не меткой. Клетка получает среднее по накрывающим её окнам.
#
#   python matrix_window_train.py --train ovary_prime_he lung5k_he breast5k_he \
#     --val ovary2_he --out outputs/models/matrix_win_val_ov2.pth
#
# --model linear это базовая линия: логистическая регрессия по среднему токену UNI
# в окне, без графа. Если она переносится между срезами не хуже графа, граф на
# этих данных ничего не добавляет.
#
# Графы строит matrix_graph_build.py (<slide>_mgraph.npz). На H&E без
# транскриптомики модель применяет matrix_window_infer.py.

import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.utils import get_device, ensure_dir, set_seed
from cell_graph_train import GraphConv, adjacency

STROMA = 2


def windows(pos, stroma, mpp, win_um, stride_um):
    """Разреженная матрица окно x клетка: 1, если стромальная клетка лежит в окне."""
    idx = np.flatnonzero(stroma)
    xy = pos[idx] * mpp
    xy = xy - xy.min(0)
    base = np.floor(xy / stride_um).astype(np.int64)
    reach = int(np.ceil(win_um / stride_um))
    keys, cells = [], []
    for di in range(reach):
        for dj in range(reach):
            i, j = base[:, 0] - di, base[:, 1] - dj
            inside = ((i >= 0) & (j >= 0)
                      & (xy[:, 0] < i * stride_um + win_um)
                      & (xy[:, 1] < j * stride_um + win_um))
            keys.append(i[inside] * 1_000_000 + j[inside])
            cells.append(idx[inside])
    keys, cells = np.concatenate(keys), np.concatenate(cells)
    _, win = np.unique(keys, return_inverse=True)
    return sp.csr_matrix((np.ones(len(cells), np.float32), (win, cells)),
                         shape=(win.max() + 1, len(pos)))


def window_members(pos, stroma, mpp, win_um, stride_um, min_cells):
    if not stroma.any():
        return sp.csr_matrix((0, len(pos)), dtype=np.float32), np.zeros(0, np.float32)
    member = windows(pos, stroma, mpp, win_um, stride_um)
    n_cells = np.asarray(member.sum(1)).ravel()
    big = n_cells >= min_cells
    return member[big], n_cells[big]


def window_labels(member, matrix, min_fibro, hi, lo):
    n_lab = member @ (matrix >= 0).astype(np.float32)
    frac = (member @ (matrix == 1).astype(np.float32)) / np.maximum(n_lab, 1)
    y = np.full(member.shape[0], -1, np.int64)
    enough = n_lab >= min_fibro
    y[enough & (frac >= hi)] = 1
    y[enough & (frac <= lo)] = 0
    return y


def pool_matrix(member, n_cells, device):
    m = (sp.diags(1.0 / n_cells) @ member).tocoo()
    idx = torch.from_numpy(np.vstack([m.row, m.col]).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(m.data.astype(np.float32)),
                                   m.shape).coalesce().to(device)


def cell_prob(member, p_win):
    hits = np.asarray(member.sum(0)).ravel()
    prob = np.full(len(hits), np.nan, np.float32)
    covered = hits > 0
    prob[covered] = (member.T @ p_win)[covered] / hits[covered]
    return prob


class Slide:
    def __init__(self, graph_dir, name, args, device):
        d = np.load(Path(graph_dir) / f"{name}_mgraph.npz")
        self.name = name
        self.matrix = d["matrix"]
        self.member, n_cells = window_members(d["pos"], d["zone"] == STROMA, float(d["mpp"]),
                                              args.win_um, args.stride_um, args.min_cells)
        if not len(n_cells):
            raise SystemExit(f"{name}: нет окон с {args.min_cells} и больше стромальными клетками")
        self.y = window_labels(self.member, self.matrix, args.min_fibro, args.hi, args.lo)
        self.target = torch.from_numpy(self.y).to(device)
        self.pool = pool_matrix(self.member, n_cells, device)
        self.x = torch.from_numpy(d["feat"].astype(np.float32)).to(device)
        self.a = adjacency(d["edges"], len(self.matrix), device)


class WindowGNN(nn.Module):
    def __init__(self, in_dim=1024, hid=256, layers=2, drop=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hid), nn.GELU())
        self.convs = nn.ModuleList([GraphConv(hid, hid) for _ in range(layers)])
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(hid, 1))

    def forward(self, x, a, pool):
        h = self.proj(x)
        for c in self.convs:
            h = self.drop(self.act(c(h, a)))
        return self.head(torch.sparse.mm(pool, h)).squeeze(1)


class WindowLinear(nn.Module):
    def __init__(self, in_dim=1024):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1)

    def forward(self, x, a, pool):
        return self.lin(torch.sparse.mm(pool, x)).squeeze(1)


def build_model(kind, in_dim, hid=256, layers=2, drop=0.2):
    return WindowLinear(in_dim) if kind == "linear" else WindowGNN(in_dim, hid, layers, drop)


@torch.no_grad()
def predict(model, slides):
    model.eval()
    return [torch.sigmoid(model(s.x, s.a, s.pool)).cpu().numpy() for s in slides]


def scores(pred, truth):
    """IoU и Dice по двум классам среди размеченных."""
    out = {}
    for c, nm in ((1, "matrix"), (0, "other")):
        p, t = pred == c, truth == c
        union = (p | t).sum()
        den = p.sum() + t.sum()
        out[nm] = ((p & t).sum() / union if union else float("nan"),
                   2 * (p & t).sum() / den if den else float("nan"))
    return out


def auc(truth, prob):
    return roc_auc_score(truth, prob) if len(np.unique(truth)) == 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/processed/graph")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--model", choices=["gnn", "linear"], default="gnn")
    ap.add_argument("--win-um", type=float, default=100.0)
    ap.add_argument("--stride-um", type=float, default=50.0)
    ap.add_argument("--hi", type=float, default=0.85,
                    help="доля матриксных среди размеченных фибробластов для метки 1")
    ap.add_argument("--lo", type=float, default=0.15, help="доля, до которой окно не матриксное")
    ap.add_argument("--min-fibro", type=int, default=5)
    ap.add_argument("--min-cells", type=int, default=8, help="минимум стромальных клеток в окне")
    ap.add_argument("--hid", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    # при 50 отбор по худшему классу застревает на раннем всплеске, см. прогоны 07.09
    ap.add_argument("--patience", type=int, default=150)
    ap.add_argument("--select", choices=["min_iou", "auc"], default="min_iou",
                    help="по чему выбирать эпоху; AUC не зависит от порога 0.5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    train = [Slide(args.graph_dir, s, args, device) for s in args.train]
    val = [Slide(args.graph_dir, s, args, device) for s in args.val]

    print(f"модель {args.model}, окно {args.win_um:.0f} мкм, шаг {args.stride_um:.0f}, "
          f"метка 1 от {args.hi:.2f}, метка 0 до {args.lo:.2f}")
    for role, group in (("train", train), ("val", val)):
        for s in group:
            n1, n0 = int((s.y == 1).sum()), int((s.y == 0).sum())
            print(f"  {role:5s} {s.name:22s} окон {len(s.y):6d}: матриксных {n1:5d}, "
                  f"не матриксных {n0:5d}, в игноре {int((s.y < 0).sum())}")
            if role == "train" and (n1 == 0 or n0 == 0):
                print(f"    ВНИМАНИЕ: в {s.name} окна одного класса, модель может "
                      "выучить сам срез вместо морфологии")

    train = [s for s in train if (s.y >= 0).any()]
    n1 = sum(int((s.y == 1).sum()) for s in train)
    n0 = sum(int((s.y == 0).sum()) for s in train)
    if not (n1 and n0):
        raise SystemExit("в обучении нет окон одного из классов")
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n0 / n1, device=device))

    model = build_model(args.model, train[0].x.shape[1], args.hid, args.layers,
                        args.dropout).to(device)
    print("параметров:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    y_val = np.concatenate([s.y for s in val])
    lab = y_val >= 0
    present = [nm for c, nm in ((1, "matrix"), (0, "other")) if (y_val == c).any()]
    if not present:
        raise SystemExit("в валидации нет размеченных окон")
    best, best_ep, best_state, bad = -1.0, 0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        total = 0.0
        # срез за срезом с накоплением градиента: каждый срез весит в потере
        # одинаково, большой не задавит маленькие, а шаг по-прежнему раз в эпоху
        for s in train:
            ok = s.target >= 0
            loss = crit(model(s.x, s.a, s.pool)[ok], s.target[ok].float()) / len(train)
            loss.backward()
            total += loss.item()
        opt.step()
        sched.step()

        p_val = np.concatenate(predict(model, val))
        m = scores((p_val[lab] >= 0.5).astype(np.int64), y_val[lab])
        min_iou = min(m[nm][0] for nm in present)
        val_auc = auc(y_val[lab], p_val[lab])
        # по худшему классу, как во втором слое: у модели, называющей всё одним
        # классом, средний IoU выше, чем у слабой настоящей
        score = val_auc if args.select == "auc" else min_iou
        if score > best:
            best, best_ep, bad = score, ep, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            mark = " *best*"
        else:
            bad += 1
            mark = ""
        if ep % 20 == 0 or mark:
            per = "  ".join(f"{nm} {m[nm][0]:.3f}" for nm in present)
            print(f"epoch {ep:3d}  loss {total:.4f}  IoU окон: {per}  (min {min_iou:.3f})  "
                  f"AUC {val_auc:.3f}{mark}")
        if bad >= args.patience:
            print(f"early stop на {ep} | лучший {args.select} {best:.3f} (эпоха {best_ep})")
            break

    model.load_state_dict(best_state)
    print(f"\n=== val, модель с эпохи {best_ep} ===")
    for s, p in zip(val, predict(model, val)):
        wl = s.y >= 0
        mw = scores((p[wl] >= 0.5).astype(np.int64), s.y[wl])
        prob = cell_prob(s.member, p)
        cells = (s.matrix >= 0) & ~np.isnan(prob)
        truth_cells = s.matrix[cells].astype(np.int64)
        pred_cells = (prob[cells] >= 0.5).astype(np.int64)
        mc = scores(pred_cells, truth_cells)
        print(f"{s.name}")
        print(f"  окна:  AUC {auc(s.y[wl], p[wl]):.3f} | matrix IoU {mw['matrix'][0]:.3f} "
              f"Dice {mw['matrix'][1]:.3f} | other IoU {mw['other'][0]:.3f} Dice {mw['other'][1]:.3f}")
        print(f"  клетки: AUC {auc(truth_cells, prob[cells]):.3f} | matrix IoU {mc['matrix'][0]:.3f} "
              f"Dice {mc['matrix'][1]:.3f} | other IoU {mc['other'][0]:.3f} Dice {mc['other'][1]:.3f}")
        print(f"  доля матриксных среди размеченных фибробластов: "
              f"истина {truth_cells.mean():.1%}, предсказано {pred_cells.mean():.1%}")
        # Порог, при котором доля матриксных окон совпала бы с истинной. Подобран по
        # самой валидации, поэтому это верхняя граница: она показывает, сколько
        # теряется только на калибровке, если окна упорядочены верно.
        cut = np.quantile(p[wl], 1 - (s.y[wl] == 1).mean())
        mo = scores((p[wl] >= cut).astype(np.int64), s.y[wl])
        print(f"  окна при пороге {cut:.3f} под истинную долю: matrix Dice {mo['matrix'][1]:.3f}, "
              f"other Dice {mo['other'][1]:.3f} (подсмотрено, только для диагностики)")

    ensure_dir(Path(args.out).parent)
    torch.save({"state_dict": model.state_dict(), "model": args.model,
                "in_dim": int(train[0].x.shape[1]), "hid": args.hid, "layers": args.layers,
                "win_um": args.win_um, "stride_um": args.stride_um, "min_cells": args.min_cells,
                "hi": args.hi, "lo": args.lo, "min_fibro": args.min_fibro,
                "train": args.train, "val": args.val, "best_epoch": best_ep}, args.out)
    print("сохранено:", args.out)


if __name__ == "__main__":
    main()
