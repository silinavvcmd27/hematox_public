# Бенчмарк первого слоя: опухоль против стромы на отложенных наборах.
#
# Для каждой модели и каждого набора: Dice и IoU опухоли и стромы по пикселям,
# размеченным как опухоль или строма, и доля стромы (TSR) по разметке и по модели.
# TSR модели считается по всей размеченной ткани, включая прочую (код 7: некроз,
# слизь, кровь, жир, нормальная паренхима), за вычетом того, что сама модель
# отнесла к прочему. Так он считается и на целом срезе, поэтому двухклассовая
# модель, которой некуда деть некроз, здесь честно проигрывает. У моделей с
# третьим выходом печатается ещё Dice прочего.
#
# Если рядом с признаками лежит <набор>_patches.csv той же длины, те же числа
# считаются по группам (орган, источник, сканер, гистотип, класс патча), а по колонке roi
# ещё и ошибка TSR на область: средняя абсолютная и ранговая корреляция.
#
# Трёхклассовые модели старой схемы (seg4) сводятся к двум классам: гормональная и
# матриксная это строма. Набор, на котором модель училась, помечается в таблице;
# модели, которых ещё нет на диске, пропускаются.
#
#   python -u pan_bench.py --models outputs/models/seg4_deploy.pth outputs/models/pan_l1_deploy3.pth \
#     --sets nct_crc_val ignite_test3 ovary2_he

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from src.utils import get_device, ensure_dir, TUMOR
from seg_decoder import SegDecoder
from pan_train import load, STROMA_CODES, OTHER

GROUP_COLS = ("organ", "source", "scanner", "subtype", "class")
# true_* разметка, tp_* совпадения; pred_t и pred_s предсказания на пикселях опухоли
# и стромы разметки (для Dice), pred_o на всей размеченной ткани, all_t и all_s
# тоже на всей размеченной ткани (для TSR модели)
COUNTS = ["true_t", "true_s", "true_o", "tp_t", "tp_s", "tp_o",
          "pred_t", "pred_s", "pred_o", "all_t", "all_s"]


def load_model(path, device):
    ck = torch.load(path, map_location=device)
    model = SegDecoder(in_dim=1024, n_classes=int(ck["n_classes"])).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    codes = np.array([int(ck["channel_to_class"][k]) for k in sorted(ck["channel_to_class"], key=int)])
    trained_on = set(ck.get("train") or [])
    return model, codes, trained_on


@torch.no_grad()
def patch_counts(model, codes, X, y, device, bs=64):
    """Счётчики пикселей на каждый патч, по ним потом собираются любые группы."""
    code_of = torch.from_numpy(codes)
    rows = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs].astype(np.float32)).to(device)
        pred = code_of[model(xb).argmax(1).cpu()].numpy()
        truth = y[i:i + bs]
        tum, stro, oth = truth == TUMOR, np.isin(truth, STROMA_CODES), truth == OTHER
        ts, ann = tum | stro, tum | stro | oth
        p_t, p_s = pred == TUMOR, np.isin(pred, STROMA_CODES)
        p_o = ~(p_t | p_s)
        rows.append(np.stack([m.sum((1, 2)) for m in (
            tum, stro, oth, p_t & tum, p_s & stro, p_o & oth,
            p_t & ts, p_s & ts, p_o & ann, p_t & ann, p_s & ann)], 1))
    return pd.DataFrame(np.concatenate(rows), columns=COUNTS)


def summarize(c):
    s = c[COUNTS].sum()
    out = {"patches": len(c), "labeled_px": int(s.true_t + s.true_s + s.true_o)}
    for nm, tp, true, pred in (("tumor", s.tp_t, s.true_t, s.pred_t),
                               ("stroma", s.tp_s, s.true_s, s.pred_s),
                               ("other", s.tp_o, s.true_o, s.pred_o)):
        out[f"{nm}_dice"] = 2 * tp / (true + pred) if true else np.nan
        out[f"{nm}_iou"] = tp / (true + pred - tp) if true else np.nan
    out["true_stroma_share"] = s.true_s / (s.true_t + s.true_s) if s.true_t + s.true_s else np.nan
    out["pred_stroma_share"] = s.all_s / (s.all_t + s.all_s) if s.all_t + s.all_s else np.nan
    return out


def roi_tsr(c, roi):
    per = c.groupby(roi)[["true_t", "true_s", "all_t", "all_s"]].sum()
    per = per[(per.true_t + per.true_s > 0) & (per.all_t + per.all_s > 0)]
    true = per.true_s / (per.true_t + per.true_s)
    pred = per.all_s / (per.all_t + per.all_s)
    rho = spearmanr(true, pred)[0] if len(per) > 2 else np.nan
    return {"rois": len(per), "tsr_mae": float((true - pred).abs().mean()), "tsr_spearman": float(rho)}


def fmt(row, other):
    return (f"опухоль {row['tumor_dice']:.3f}  строма {row['stroma_dice']:.3f}"
            + (f"  прочее {row['other_dice']:.3f}" if other and not np.isnan(row["other_dice"]) else "")
            + f"  доля стромы {row['true_stroma_share']:.3f} / {row['pred_stroma_share']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--dirs", nargs="+", default=["data/processed/seg", "data/processed/seg_pan"])
    ap.add_argument("--out", default="outputs/results/pan/bench.csv")
    args = ap.parse_args()

    device = get_device()
    data = {s: load(args.dirs, s) for s in args.sets}
    meta = {}
    for s, (X, _) in data.items():
        for d in args.dirs:
            p = Path(d) / f"{s}_patches.csv"
            if p.exists():
                m = pd.read_csv(p)
                if len(m) == len(X):
                    meta[s] = m
                break

    table = []
    for mpath in args.models:
        if not Path(mpath).exists():
            print(f"\nнет {mpath}, пропускаю: модель ещё учится или путь с опечаткой")
            continue
        model, codes, trained_on = load_model(mpath, device)
        other = OTHER in codes
        name = Path(mpath).stem
        print(f"\n=== {name} ===")
        for s, (X, y) in data.items():
            c = patch_counts(model, codes, X, y, device)
            note = " (в обучении!)" if s in trained_on else ""
            base = {"model": name, "set": s + note}
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
                    r = {**base, "group_by": col, "group": g, **summarize(part)}
                    table.append(r)
                    print(f"    {col} {str(g)[:28]:28s} патчей {r['patches']:5d}  {fmt(r, other)}")

    out = Path(args.out)
    ensure_dir(out.parent)
    pd.DataFrame(table).round(4).to_csv(out, index=False)
    print("\nтаблица:", out)


if __name__ == "__main__":
    main()
