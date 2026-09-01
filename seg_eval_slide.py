import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from src.utils import (get_device, hf_login, ensure_dir, CLASS_NAMES,
                       TRAIN_CLASSES, TUMOR, STROMA_HORMONAL, STROMA_MATRIX,
                       IMMUNE, STROMA, STROMA_FOR_TSR, IGNORE, CLASS_COLORS)
from src.stain import slide_normalizer
from src.data.patching import load_image
from seg_decoder import SegDecoder

GRID = 14

TISSUE_CLASSES = [TUMOR, STROMA_HORMONAL, STROMA_MATRIX, IMMUNE, STROMA]
ALL_EVAL = [0] + TISSUE_CLASSES
EVAL_NAMES = {c: CLASS_NAMES[c] for c in ALL_EVAL}

COL = np.array([
    [245, 245, 245],
    [220,  50,  47],
    [255, 165,   0],
    [ 38, 139, 210],
    [ 42, 161,  52],
    [128, 128, 128],
], np.uint8)

_tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(),
                 T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def build_uni(device):
    import timm
    hf_login()
    m = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True,
                          init_values=1e-5, dynamic_img_size=True)
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad = False
    return m


def load_decoder(path, device):
    ckpt = torch.load(path, map_location=device)
    n_classes = int(ckpt["n_classes"])
    dec = SegDecoder(in_dim=1024, n_classes=n_classes).to(device)
    dec.load_state_dict(ckpt["state_dict"])
    dec.eval()
    channel_to_class = ckpt.get("channel_to_class")
    if channel_to_class is None:
        raise SystemExit(
            f"в чекпоинте {path} нет channel_to_class.\n"
            "Переобучи модель обновлённым seg_train2.py.")
    channel_to_class = {int(k): int(v) for k, v in channel_to_class.items()}
    print(f"декодер на {n_classes} каналов: " +
          ", ".join(f"{c}->{CLASS_NAMES[v]}" for c, v in sorted(channel_to_class.items())))
    return dec, n_classes, channel_to_class


@torch.no_grad()
def run_batch(uni, dec, arrs, device, norm=None):
    if norm is not None:
        arrs = [norm(a) for a in arrs]
    x = torch.stack([_tf(Image.fromarray(a)) for a in arrs]).to(device)
    f = uni.forward_features(x)
    if f.shape[1] == GRID * GRID + 1:
        f = f[:, 1:, :]
    elif f.shape[1] != GRID * GRID:
        f = f[:, -GRID * GRID:, :]
    return F.softmax(dec(f), 1).cpu().numpy()


def iou_dice(pred, truth, c, where=None):
    p = pred == c
    t = truth == c
    if where is not None:
        p = p & where
        t = t & where
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    denom = p.sum() + t.sum()
    return (inter / union if union else float("nan"),
            2 * inter / denom if denom else float("nan"))


def report(pred, truth, where, title):
    print(f"\n--- {title} ---")
    sel = where if where is not None else np.ones_like(truth, bool)
    acc = (pred[sel] == truth[sel]).mean()
    print(f"accuracy {acc:.3f}  ({sel.sum()/1e6:.1f} млн пикселей)")
    out = {"accuracy": round(float(acc), 4)}
    for c in TISSUE_CLASSES:
        nm = CLASS_NAMES[c]
        i, d = iou_dice(pred, truth, c, where)
        print(f"  {nm:20s} IoU {i:.3f}  Dice {d:.3f}")
        out[f"{nm}_iou"] = round(float(i), 4)
        out[f"{nm}_dice"] = round(float(d), 4)
    present = [c for c in TISSUE_CLASSES if not np.isnan(iou_dice(pred, truth, c, where)[0])]
    if present:
        miou = np.mean([iou_dice(pred, truth, c, where)[0] for c in present])
        print(f"  mIoU: {miou:.3f}")
        out["miou"] = round(float(miou), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--he", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seg-dir", default="data/processed/seg")
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--cds", type=int, default=8)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--min-confidence", type=float, default=0.5)
    ap.add_argument("--metrics-csv", default="outputs/results/seg_slide_metrics.csv")
    ap.add_argument("--stain-ref", default=None,
                    help="npz эталона окраски, тот же что при seg_extract.py")
    args = ap.parse_args()
    norm = slide_normalizer(args.he, args.stain_ref) if args.stain_ref else None

    import cv2

    mask_npz = Path(args.seg_dir) / f"{args.slide}_mask.npz"
    if not mask_npz.exists():
        raise SystemExit(f"нет {mask_npz} — сперва make_seg_masks_v.py")

    device = get_device()
    print("device:", device)
    uni = build_uni(device)
    dec, n_classes, ch2cls = load_decoder(args.model, device)

    print("грузим H&E...")
    img = load_image(args.he)
    H, W = img.shape[:2]
    ps, st, cds = args.patch_size, args.stride, args.cds
    Hc, Wc = H // cds, W // cds
    print(f"размер {W}x{H}, карта {Wc}x{Hc}, шаг {st}")

    acc_map = np.zeros((Hc, Wc, n_classes), np.float32)
    wsum = np.zeros((Hc, Wc), np.float32)
    pp = ps // cds

    from seg_infer import cosine_window
    win = cosine_window(pp)

    arrs, pos, done = [], [], 0

    def flush():
        nonlocal arrs, pos, done
        if not arrs:
            return
        probs = run_batch(uni, dec, arrs, device, norm)
        for k, (yc, xc) in enumerate(pos):
            pm = probs[k].transpose(1, 2, 0)
            f = pm.shape[0] // pp
            if f > 1:
                pm = pm[:pp * f, :pp * f].reshape(pp, f, pp, f, -1).mean((1, 3))
            y1, x1 = min(yc + pp, Hc), min(xc + pp, Wc)
            w = win[:y1 - yc, :x1 - xc]
            acc_map[yc:y1, xc:x1] += pm[:y1 - yc, :x1 - xc] * w[:, :, None]
            wsum[yc:y1, xc:x1] += w
        done += len(arrs)
        arrs.clear(); pos.clear()
        print(f"  патчей: {done}", end="\r")

    for y in range(0, H - ps, st):
        for x in range(0, W - ps, st):
            patch = img[y:y + ps, x:x + ps]
            if (patch.mean(-1) > 220).mean() > 0.85:
                continue
            arrs.append(patch)
            pos.append((y // cds, x // cds))
            if len(arrs) >= args.bs:
                flush()
    flush()
    print(f"\nвсего патчей: {done}")

    covered = wsum > 0
    probs = np.zeros_like(acc_map)
    probs[covered] = acc_map[covered] / wsum[covered][:, None]
    conf = probs.max(2)
    chan = probs.argmax(2)

    pred = np.zeros((Hc, Wc), np.uint8)
    ok = covered & (conf >= args.min_confidence)
    for ch, cls in ch2cls.items():
        pred[ok & (chan == ch)] = cls

    print(f"окном покрыто {100 * covered.mean():.1f}% карты (патч {ps}, шаг {st})")

    d = np.load(mask_npz)
    truth = cv2.resize(d["mask"], (Wc, Hc), interpolation=cv2.INTER_NEAREST)
    print(f"маска {d['mask'].shape[1]}x{d['mask'].shape[0]} при downscale "
          f"{int(d['downscale'])} -> приведена к {Wc}x{Hc}")

    inside = covered & (truth > 0)
    rows = [
        {"scope": "покрытое окном", **report(pred, truth, covered,
                                             "просмотренные пиксели, включая неразмеченное")},
        {"scope": "внутри разметки", **report(pred, truth, inside,
                                              "просмотренные пиксели с меткой")},
    ]

    print("\nматрица ошибок (строки — истина, доли внутри строки):")
    header = "".join(f"{EVAL_NAMES[c]:>20s}" for c in ALL_EVAL)
    print(f"{'':20s}{header}")
    for t in ALL_EVAL:
        sel = covered & (truth == t)
        n = sel.sum()
        if not n:
            continue
        fr = [(pred[sel] == p).sum() / n for p in ALL_EVAL]
        print(f"{EVAL_NAMES[t]:20s}" + "".join(f"{v:20.3f}" for v in fr))

    n_tum = int((pred[covered] == TUMOR).sum())
    n_str = int(np.isin(pred[covered], STROMA_FOR_TSR).sum())
    denom = n_tum + n_str
    tsp_pred = 100 * n_str / denom if denom else float("nan")

    t_tum = int((truth[covered] == TUMOR).sum())
    t_str = int(np.isin(truth[covered], STROMA_FOR_TSR).sum())
    t_denom = t_tum + t_str
    tsp_true = 100 * t_str / t_denom if t_denom else float("nan")

    print(f"\nTSR: предсказано {tsp_pred:.1f}%, по разметке {tsp_true:.1f}%")

    for label, zones, total in [("предсказание", pred, n_str), ("разметка", truth, t_str)]:
        if total:
            print(f"  состав стромы ({label}):")
            for c in STROMA_FOR_TSR:
                k = int((zones[covered] == c).sum())
                print(f"    {CLASS_NAMES[c]:20s}: {k:>8d}  ({100*k/total:.1f}%)")

    out_csv = Path(args.metrics_csv)
    ensure_dir(out_csv.parent)
    first = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        for r in rows:
            r = {"slide": args.slide, "model": args.model, "stride": st,
                 "n_classes": n_classes, "min_confidence": args.min_confidence,
                 "tsp_pred": round(tsp_pred, 2), "tsp_true": round(tsp_true, 2), **r}
            w = csv.DictWriter(f, fieldnames=list(r))
            if first:
                w.writeheader()
                first = False
            w.writerow(r)
    print("метрики дописаны в", out_csv)

    bg = cv2.resize(img, (Wc, Hc), interpolation=cv2.INTER_AREA)
    panels = [bg]
    for m in (truth, pred):
        ov = bg.copy()
        for c in TISSUE_CLASSES:
            sel = m == c
            if sel.any():
                col = np.array(CLASS_COLORS[c])
                ov[sel] = (0.45 * col + 0.55 * bg[sel]).astype(np.uint8)
        panels.append(ov)
    out_png = ensure_dir("outputs/results") / f"{args.slide}_eval.png"
    Image.fromarray(np.concatenate(panels, 1)).save(out_png)
    print("исходник | истина | предсказание:", out_png)


if __name__ == "__main__":
    main()