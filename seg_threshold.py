# Подбор порога для двухклассового декодера.
#
# argmax = порог 0.5. Модель систематически недооценивает строму, поэтому
# порог на вероятность стромы подбирается по отложенному срезу. Смотрим два
# критерия: максимум mIoU и минимум ошибки TSP (доли стромы).
#
# Работает на готовых признаках и моделях, UNI не запускается.
#
# python seg_threshold.py \
#   --pairs ovary_prime_he:outputs/models/seg2_fixed_ovary_prime_he.pth \
#           ovary2_he:outputs/models/seg2_fixed_ovary2_he.pth \
#           ovary3_he:outputs/models/seg2_fixed_ovary3_he.pth

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.utils import get_device
from seg_decoder import SegDecoder


def load_slide(seg_dir, slide):
    d = np.load(Path(seg_dir) / f"{slide}_feat.npz")
    return d["X"], d["y"]


@torch.no_grad()
def stroma_prob(model, X, device, bs=32):
    out = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs].astype(np.float32)).to(device)
        p = F.softmax(model(xb), 1)[:, 1]   # канал 1 = строма
        out.append(p.cpu().numpy())
    return np.concatenate(out, 0)


def iou(pred_stroma, truth, thr):
    # truth: 1 опухоль, 2 строма, 0 не размечено
    labeled = truth > 0
    ps = (pred_stroma >= thr) & labeled
    ts = (truth == 2) & labeled
    pt = (~ps) & labeled
    tt = (truth == 1) & labeled
    def j(a, b):
        u = np.logical_or(a, b).sum()
        return np.logical_and(a, b).sum() / u if u else float("nan")
    return j(pt, tt), j(ps, ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="срез:модель, модель обучена без этого среза")
    ap.add_argument("--seg-dir", default="data/processed/seg")
    ap.add_argument("--grid", type=int, default=61)
    args = ap.parse_args()

    device = get_device()
    print("device:", device)
    thrs = np.linspace(0.2, 0.8, args.grid)

    per_slide = []
    for pair in args.pairs:
        slide, model_path = pair.split(":", 1)
        X, y = load_slide(args.seg_dir, slide)

        ckpt = torch.load(model_path, map_location=device)
        model = SegDecoder(in_dim=X.shape[-1], n_classes=2).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        prob = stroma_prob(model, X, device)          # [N,112,112]
        labeled = y > 0
        true_share = (y == 2)[labeled].mean()

        mious, shares = [], []
        for t in thrs:
            it, is_ = iou(prob, y, t)
            mious.append(np.nanmean([it, is_]))
            shares.append((prob >= t)[labeled].mean())
        per_slide.append((slide, true_share, np.array(mious), np.array(shares)))
        print(f"{slide}: истинная доля стромы {true_share:.3f}, "
              f"при 0.5 предсказано {shares[len(thrs)//2]:.3f}")

    # порог выбираем один на всех — как его выбирали бы на новом срезе
    M = np.mean([m for _, _, m, _ in per_slide], 0)
    best_iou_t = thrs[int(np.argmax(M))]

    # ошибка TSP усреднённая по срезам
    tsp_err = np.mean([np.abs(sh - ts) for _, ts, _, sh in per_slide], 0)
    best_tsp_t = thrs[int(np.argmin(tsp_err))]

    print(f"\nпорог по максимуму mIoU:   {best_iou_t:.3f}  (mIoU {M.max():.3f})")
    print(f"порог по минимуму ошибки TSP: {best_tsp_t:.3f}  "
          f"(ошибка {tsp_err.min()*100:.1f} п.п.)")

    def col(thr):
        i = int(np.argmin(np.abs(thrs - thr)))
        return i

    print(f"\n{'срез':16s} {'истина':>7s} "
          f"{'0.5':>18s} {'mIoU-порог':>18s} {'TSP-порог':>18s}")
    print(f"{'':16s} {'строма':>7s} {'строма  ошибка':>18s}"
          f"{'строма  ошибка':>18s}{'строма  ошибка':>18s}")
    for slide, ts, _, sh in per_slide:
        i5, ii, it = col(0.5), col(best_iou_t), col(best_tsp_t)
        print(f"{slide:16s} {ts:7.3f} "
              f"{sh[i5]:8.3f} {100*(sh[i5]-ts):+6.1f}    "
              f"{sh[ii]:8.3f} {100*(sh[ii]-ts):+6.1f}    "
              f"{sh[it]:8.3f} {100*(sh[it]-ts):+6.1f}")

    e5 = np.mean([abs(sh[col(0.5)] - ts) for _, ts, _, sh in per_slide])
    print(f"\nсредняя ошибка TSP: при 0.5 {100*e5:.1f} п.п. -> "
          f"при подобранном {100*tsp_err.min():.1f} п.п.")


if __name__ == "__main__":
    main()
