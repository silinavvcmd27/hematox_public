# Простые модели против декодера первого слоя.
#
# Обучаются на тех же признаках, той же выборкой, той же потерей и с тем же
# выбором эпохи, что pan_train.py, и оцениваются функциями pan_bench.py, поэтому
# числа сравнимы напрямую. Отличается только то, что стоит после UNI:
#
#   prior   константа, всегда самый частый класс. Пол, ниже него быть нельзя
#   mean    логистическая регрессия на среднем векторе фрагмента: один класс на
#           все 70 мкм, как метка целого фрагмента в NCT-CRC
#   linear  логистическая регрессия на каждом токене отдельно, соседей не видит
#   mlp     то же плюс скрытый слой на 64: нелинейность без контекста
#   decoder тот же SegDecoder, что в работе, для сверки протокола
#
# Готовые модели можно досчитать в ту же таблицу через --also-models, тогда
# сравнение выходит одной командой. Наборы из --eval-extra в эпохах не участвуют,
# поэтому их добавление ничего не меняет в обучении и не замедляет его.
#
#   python -u pan_baselines.py --other \
#     --train ovary_prime_he ovary2_he ovary3_he nct_crc nct_crc_other ignite_train3 \
#     --val nct_crc_val nct_crc_val_other ignite_test3 --select-on ignite_test3 \
#     --eval-extra bcss3 ubc --also-models outputs/models/pan_l1_deploy3.pth

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import get_device, ensure_dir, set_seed, TUMOR, STROMA
from seg_decoder import SegDecoder
from pan_train import load, to_target, IGNORE_TARGET, STROMA_CODES, OTHER
from pan_bench import patch_counts, summarize, roi_tsr, fmt, GROUP_COLS, load_model

KINDS = ("prior", "mean", "linear", "mlp", "decoder")


class Prior(nn.Module):
    """Ответ не зависит от картинки: всегда самый частый класс обучающей выборки."""

    def __init__(self, freq, out_size):
        super().__init__()
        self.out_size = out_size
        self.register_buffer("logit", torch.log(torch.as_tensor(freq, dtype=torch.float32) + 1e-9))

    def forward(self, tokens):
        h, w = self.out_size
        return self.logit.view(1, -1, 1, 1).expand(tokens.shape[0], -1, h, w)


class TokenHead(nn.Module):
    """Классификатор одного токена: соседей нет, карта получается растягиванием."""

    def __init__(self, in_dim, n_classes, out_size, hidden=0, grid=14, mode="bilinear"):
        super().__init__()
        self.grid, self.out_size, self.mode = grid, out_size, mode
        self.net = (nn.Conv2d(in_dim, n_classes, 1) if not hidden else
                    nn.Sequential(nn.Conv2d(in_dim, hidden, 1), nn.GELU(),
                                  nn.Conv2d(hidden, n_classes, 1)))

    def forward(self, tokens):
        b, n, d = tokens.shape
        x = tokens.transpose(1, 2).reshape(b, d, self.grid, self.grid)
        x = self.net(x)
        kw = {"align_corners": False} if self.mode == "bilinear" else {}
        return F.interpolate(x, size=self.out_size, mode=self.mode, **kw)


class MeanHead(nn.Module):
    """Один вектор на фрагмент, один класс на все 70 мкм."""

    def __init__(self, in_dim, n_classes, out_size):
        super().__init__()
        self.out_size = out_size
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, tokens):
        z = self.fc(tokens.mean(1))
        h, w = self.out_size
        return z.view(z.shape[0], -1, 1, 1).expand(-1, -1, h, w)


def build(kind, in_dim, n_classes, out_size, freq, mode):
    if kind == "prior":
        return Prior(freq, out_size)
    if kind == "mean":
        return MeanHead(in_dim, n_classes, out_size)
    if kind == "linear":
        return TokenHead(in_dim, n_classes, out_size, mode=mode)
    if kind == "mlp":
        return TokenHead(in_dim, n_classes, out_size, hidden=64, mode=mode)
    if kind == "decoder":
        return SegDecoder(in_dim=in_dim, n_classes=n_classes)
    raise SystemExit(f"неизвестная модель {kind}")


@torch.no_grad()
def val_miou(model, X, y, names, other, device, bs=64):
    model.eval()
    inter = np.zeros(len(names))
    union = np.zeros(len(names))
    for i in range(0, len(X), bs):
        pred = model(torch.from_numpy(X[i:i + bs].astype(np.float32)).to(device)).argmax(1).cpu().numpy()
        tgt = to_target(y[i:i + bs], other)
        lab = tgt != IGNORE_TARGET
        for c in range(len(names)):
            p, t = (pred == c) & lab, (tgt == c) & lab
            inter[c] += (p & t).sum()
            union[c] += (p | t).sum()
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return float(np.nanmean(iou)), iou


def train_one(kind, train, val, names, args, device, rng):
    other = len(names) == 3
    in_dim = next(iter(train.values()))[0].shape[-1]
    out_size = next(iter(train.values()))[1].shape[-2:]

    cnt = np.zeros(3)
    for X, y in train.values():
        n = np.array([(y == TUMOR).sum(), np.isin(y, STROMA_CODES).sum(), (y == OTHER).sum()])
        cnt += n * args.per_source / len(X)
    cnt = cnt[:len(names)]
    w = cnt.sum() / (len(names) * np.maximum(cnt, 1))
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)

    model = build(kind, in_dim, len(names), tuple(out_size), cnt / cnt.sum(), args.upsample).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== {kind}: обучаемых параметров {params}")
    if params == 0:
        return model, params, 0, 0.0

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=w, ignore_index=IGNORE_TARGET)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    sources = list(train.values())
    best, best_ep, best_state, bad = -1.0, 0, None, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        src = np.repeat(np.arange(len(sources)), args.per_source)
        idx = np.concatenate([rng.choice(len(X), args.per_source, replace=len(X) < args.per_source)
                              for X, _ in sources])
        order = rng.permutation(len(idx))
        run, nb = 0.0, 0
        for a in range(0, len(order), args.bs):
            sel = order[a:a + args.bs]
            xb = np.stack([sources[src[j]][0][idx[j]] for j in sel]).astype(np.float32)
            yb = to_target(np.stack([sources[src[j]][1][idx[j]] for j in sel]), other)
            opt.zero_grad()
            loss = crit(model(torch.from_numpy(xb).to(device)), torch.from_numpy(yb).to(device))
            loss.backward()
            opt.step()
            run += loss.item()
            nb += 1
        sched.step()

        per, mious = [], {}
        for s, (X, y) in val.items():
            mious[s], iou = val_miou(model, X, y, names, other, device)
            per.append(f"{s}: " + " ".join(f"{nm} {v:.3f}" for nm, v in zip(names, iou)
                                           if not np.isnan(v)))
        worst = float(np.min([mious[s] for s in args.select_on]))
        mark = ""
        if worst > best:
            best, best_ep, bad = worst, ep, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            mark = " *best*"
        else:
            bad += 1
        print(f"  epoch {ep:3d}  loss {run / nb:.4f}  IoU | {' | '.join(per)} | "
              f"худший mIoU {worst:.3f}{mark}")
        if bad >= args.patience:
            print(f"  early stop на {ep}, лучшая эпоха {best_ep}")
            break
    model.load_state_dict(best_state)
    return model, params, best_ep, (time.time() - t0) / 60


def evaluate(name, model, codes, trained_on, params, best_ep, minutes, val, meta, device, table):
    other = OTHER in codes
    print(f"\n--- {name}")
    for s, (X, y) in val.items():
        c = patch_counts(model, codes, X, y, device)
        note = " (в обучении!)" if s in trained_on else ""
        base = {"model": name, "params": params, "best_epoch": best_ep,
                "train_min": round(minutes, 1), "set": s + note}
        row = {**base, "group_by": "", "group": "всё", **summarize(c)}
        if s in meta and "roi" in meta[s]:
            row.update(roi_tsr(c.assign(roi=meta[s]["roi"].values), "roi"))
        table.append(row)
        print(f"{s}{note}: {fmt(row, other)}"
              + (f", TSR по {row['rois']} областям: ошибка {row['tsr_mae']:.3f}, "
                 f"ранговая корреляция {row['tsr_spearman']:.2f}" if "rois" in row else ""))
        if s not in meta:
            continue
        for col in GROUP_COLS:
            if col not in meta[s] or meta[s][col].nunique() < 2:
                continue
            for g, part in c.groupby(meta[s][col].values):
                table.append({**base, "group_by": col, "group": g, **summarize(part)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["data/processed/seg", "data/processed/seg_pan"])
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--eval-extra", nargs="+", default=[],
                    help="наборы только для итоговой оценки, в эпохах не участвуют")
    ap.add_argument("--select-on", nargs="+")
    ap.add_argument("--kinds", nargs="+", default=["prior", "mean", "linear", "mlp"], choices=KINDS)
    ap.add_argument("--also-models", nargs="+", default=[], help="готовые .pth в ту же таблицу")
    ap.add_argument("--other", action="store_true", help="третий класс, прочая ткань")
    ap.add_argument("--upsample", choices=["bilinear", "nearest"], default="bilinear",
                    help="как растягивать сетку токенов до карты у моделей без декодера")
    ap.add_argument("--per-source", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-dir", default=None, help="куда сложить обученные базовые модели")
    ap.add_argument("--out", default="outputs/results/pan/baselines.csv")
    args = ap.parse_args()

    args.select_on = args.select_on or args.val
    if not set(args.select_on) <= set(args.val):
        raise SystemExit("--select-on должен называть наборы из --val")
    names = ("tumor", "stroma", "other") if args.other else ("tumor", "stroma")
    codes = np.array([TUMOR, STROMA, OTHER][:len(names)], dtype=np.int64)

    set_seed(args.seed)
    device = get_device()
    train = {s: load(args.dirs, s) for s in args.train}
    val = {s: load(args.dirs, s) for s in args.val}
    extra = {s: load(args.dirs, s) for s in args.eval_extra if s not in val}
    for role, group in (("train", train), ("val", val), ("оценка", extra)):
        for s, (X, y) in group.items():
            print(f"  {role:6s} {s:18s} патчей {len(X):6d}")
    scored = {**val, **extra}

    meta = {}
    for s, (X, _) in scored.items():
        for d in args.dirs:
            p = Path(d) / f"{s}_patches.csv"
            if p.exists():
                m = pd.read_csv(p)
                if len(m) == len(X):
                    meta[s] = m
                break

    table = []
    for kind in args.kinds:
        rng = np.random.default_rng(args.seed)
        model, params, best_ep, minutes = train_one(kind, train, val, names, args, device, rng)
        evaluate(kind, model, codes, set(args.train), params, best_ep, minutes,
                 scored, meta, device, table)
        if args.save_dir:
            ensure_dir(Path(args.save_dir))
            torch.save({"state_dict": model.state_dict(), "kind": kind, "n_classes": len(names),
                        "train": args.train, "best_epoch": best_ep},
                       Path(args.save_dir) / f"baseline_{kind}.pth")

    for mpath in args.also_models:
        if not Path(mpath).exists():
            print(f"\nнет {mpath}, пропускаю")
            continue
        model, mcodes, trained_on = load_model(mpath, device)
        params = sum(p.numel() for p in model.parameters())
        evaluate(Path(mpath).stem, model, mcodes, trained_on, params, 0, 0.0,
                 scored, meta, device, table)

    df = pd.DataFrame(table).round(4)
    out = Path(args.out)
    ensure_dir(out.parent)
    df.to_csv(out, index=False)

    print("\n=== сводка по наборам ===")
    whole = df[df.group_by == ""]
    order = list(dict.fromkeys(whole.model))
    for col in ("tumor_dice", "stroma_dice", "other_dice", "tsr_mae"):
        if col not in whole or whole[col].isna().all():
            continue
        piv = whole.pivot_table(index="model", columns="set", values=col).reindex(order)
        print(f"\n{col}" + ("  (меньше лучше)" if col == "tsr_mae" else ""))
        print(piv.round(3).to_string())
    print("\nпараметры и время обучения")
    print(whole.drop_duplicates("model")[["model", "params", "best_epoch", "train_min"]]
          .to_string(index=False))
    print("\nтаблица:", out)


if __name__ == "__main__":
    main()
