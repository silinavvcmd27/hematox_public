# Первый слой на многих наборах: опухоль против стромы.
#
# Отличия от seg_train2.py. Коды масок сводятся к двум классам прямо при чтении:
# 1 это опухоль, 2..5 строма (у наших срезов гормональная и матриксная, у внешних
# наборов строма и лимфоциты), всё прочее в игнор. Каждая эпоха берёт поровну
# патчей из каждого обучающего набора, иначе большой колоректальный задавит
# яичник. Отложенных наборов может быть несколько, метрики печатаются по каждому.
#
# С --other появляется третий выход, прочая ткань (код 7 в масках pan_relabel.py
# и pan_prep_crc.py): некроз, кровь, слизь, жир, нормальная паренхима, то есть то,
# что патолог исключает из TSR. Без --other код 7 идёт в игнор, как раньше.
#
# Эпоха выбирается по наборам из --select-on (по умолчанию по всем отложенным),
# по худшему из них (--select min): если один набор насыщается за пару эпох,
# среднее выбирает раннюю эпоху, на которой остальные ещё не доучены. Наборы,
# которых нет в --select-on, остаются честной внешней проверкой.
#
#   python -u pan_train.py --train ovary_prime_he ovary3_he nct_crc \
#     --val ovary2_he nct_crc_val --select-on ovary2_he --out outputs/models/pan_l1_ovary_crc.pth
#
# Признаки ищутся как <набор>_feat.npz в папках --dirs по порядку. Модель
# сохраняется в формате seg_train2.py (строма под кодом 5), её читает
# seg_infer_svs.py.

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.utils import get_device, ensure_dir, set_seed, TUMOR, STROMA
from seg_decoder import SegDecoder
from seg_train2 import log_metrics

IGNORE_TARGET = -100
STROMA_CODES = (2, 3, 4, 5)
# в src/utils такого кода нет: 6 там занят неопределёнными клетками наших срезов
OTHER = 7


def load(dirs, name):
    for d in dirs:
        path = Path(d) / f"{name}_feat.npz"
        if path.exists():
            f = np.load(path)
            return f["X"], f["y"]
    raise SystemExit(f"нет {name}_feat.npz ни в одной из папок {dirs}")


def to_target(y, other=False):
    t = np.full(y.shape, IGNORE_TARGET, np.int64)
    t[y == TUMOR] = 0
    t[np.isin(y, STROMA_CODES)] = 1
    if other:
        t[y == OTHER] = 2
    return t


@torch.no_grad()
def predict(model, X, device, bs=64):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs].astype(np.float32)).to(device)
        out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out)


def scores(pred, target, names):
    lab = target != IGNORE_TARGET
    out = {}
    for c, nm in enumerate(names):
        p, t = (pred == c) & lab, (target == c) & lab
        tp = (p & t).sum()
        # класса нет в разметке набора (прочего у наших срезов), тогда и оценки нет
        out[nm] = ((tp / (p | t).sum(), 2 * tp / (p.sum() + t.sum())) if t.any()
                   else (float("nan"), float("nan")))
    # TSR: истинный по опухоли и строме разметки, предсказанный по всей размеченной
    # ткани без того, что модель отнесла к прочему, как он потом считается на срезе
    ts, kept = lab & (target <= 1), lab & (pred <= 1)
    true_share = (target[ts] == 1).mean() if ts.any() else float("nan")
    pred_share = (pred[kept] == 1).mean() if kept.any() else float("nan")
    return out, true_share, pred_share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["data/processed/seg", "data/processed/seg_pan"])
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--other", action="store_true", help="третий класс, прочая ткань")
    ap.add_argument("--per-source", type=int, default=4000, help="патчей из каждого набора за эпоху")
    ap.add_argument("--select", choices=["min", "mean"], default="min",
                    help="выбирать эпоху по худшему отложенному набору или по среднему")
    ap.add_argument("--select-on", nargs="+", help="по каким отложенным наборам выбирать эпоху")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metrics-csv", default=None,
                    help="по умолчанию pan_l1_metrics.csv, с --other pan_l1_other_metrics.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    names = ("tumor", "stroma", "other") if args.other else ("tumor", "stroma")
    metrics_csv = args.metrics_csv or ("outputs/results/pan_l1_other_metrics.csv" if args.other
                                       else "outputs/results/pan_l1_metrics.csv")
    select_on = args.select_on or args.val
    if not set(select_on) <= set(args.val):
        raise SystemExit("--select-on должен называть наборы из --val")
    set_seed(args.seed)
    device = get_device()
    rng = np.random.default_rng(args.seed)

    train = {s: load(args.dirs, s) for s in args.train}
    val = {s: load(args.dirs, s) for s in args.val}
    cnt = np.zeros(len(names))
    for role, group in (("train", train), ("val", val)):
        for s, (X, y) in group.items():
            n = np.array([(y == TUMOR).sum(), np.isin(y, STROMA_CODES).sum(), (y == OTHER).sum()])
            n = n[:len(names)]
            print(f"  {role:5s} {s:18s} патчей {len(X):6d}, размечено {n.sum() / y.size:.1%}, "
                  f"строма среди опухоли и стромы {n[1] / max(n[0] + n[1], 1):.1%}"
                  + (f", прочее {n[2] / max(n.sum(), 1):.1%}" if args.other else ""))
            if role == "train":
                cnt += n * args.per_source / len(X)
    w = cnt.sum() / (len(names) * np.maximum(cnt, 1))
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    print("веса: " + ", ".join(f"{nm} {w[c]:.2f}" for c, nm in enumerate(names))
          + f" | по {args.per_source} патчей из каждого из {len(train)} наборов за эпоху"
          f" | эпоха по {args.select} mIoU наборов {', '.join(select_on)}")

    model = SegDecoder(in_dim=next(iter(train.values()))[0].shape[-1], n_classes=len(names)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=w, ignore_index=IGNORE_TARGET)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    sources = list(train.values())
    best, best_ep, best_state, bad = -1.0, 0, None, 0
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
            yb = to_target(np.stack([sources[src[j]][1][idx[j]] for j in sel]), args.other)
            opt.zero_grad()
            loss = crit(model(torch.from_numpy(xb).to(device)), torch.from_numpy(yb).to(device))
            loss.backward()
            opt.step()
            run += loss.item()
            nb += 1
        sched.step()

        per, mious = [], {}
        for s, (X, y) in val.items():
            m, _, _ = scores(predict(model, X, device), to_target(y, args.other), names)
            mious[s] = np.nanmean([m[nm][0] for nm in names])
            per.append(f"{s}: " + " ".join(f"{nm} {m[nm][0]:.3f}" for nm in names
                                           if not np.isnan(m[nm][0])))
        pool = [mious[s] for s in select_on]
        worst, mean = float(np.min(pool)), float(np.mean(pool))
        score = worst if args.select == "min" else mean
        mark = ""
        if score > best:
            best, best_ep, bad = score, ep, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            mark = " *best*"
        else:
            bad += 1
        print(f"epoch {ep:3d}  loss {run / nb:.4f}  IoU | {' | '.join(per)} | "
              f"mIoU худший {worst:.3f} средний {mean:.3f}{mark}")
        if bad >= args.patience:
            print(f"early stop на {ep} | лучший {args.select} mIoU {best:.3f} (эпоха {best_ep})")
            break

    model.load_state_dict(best_state)
    print(f"\n=== отложенные наборы, модель с эпохи {best_ep} ===")
    for s, (X, y) in val.items():
        m, true_share, pred_share = scores(predict(model, X, device), to_target(y, args.other), names)
        print(f"{s}")
        for nm in names:
            if not np.isnan(m[nm][0]):
                print(f"  {nm:7s} IoU {m[nm][0]:.3f}  Dice {m[nm][1]:.3f}")
        print(f"  доля стромы: истина {true_share:.4f}, предсказано {pred_share:.4f}")
        # набор колонок как в первых прогонах, иначе дописанная строка разъедется
        # с заголовком уже существующего CSV; у трёх классов свой файл
        row = {"val": s, "train": "+".join(args.train), "best_epoch": best_ep, "model": args.out,
               "true_stroma_share": round(float(true_share), 4),
               "pred_stroma_share": round(float(pred_share), 4)}
        for nm in names:
            row[f"{nm}_iou"] = round(float(m[nm][0]), 4)
            row[f"{nm}_dice"] = round(float(m[nm][1]), 4)
        log_metrics(metrics_csv, row)

    ensure_dir(Path(args.out).parent)
    channels = {0: TUMOR, 1: STROMA, 2: OTHER}
    torch.save({"state_dict": model.state_dict(), "n_classes": len(names),
                "channel_to_class": {c: channels[c] for c in range(len(names))},
                "class_names": {channels[c]: nm for c, nm in enumerate(names)},
                "train": args.train, "val": args.val, "per_source": args.per_source,
                "select": args.select, "select_on": select_on, "best_epoch": best_ep,
                "other": args.other, "seed": args.seed}, args.out)
    print("метрики дописаны в", metrics_csv)
    print("сохранено:", args.out)


if __name__ == "__main__":
    main()
