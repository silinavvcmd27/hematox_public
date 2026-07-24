# Инференс по целому срезу со сравнением с маской Вороного.
# Считает accuracy, IoU, Dice, матрицу ошибок и TSP — предсказанный и истинный.
#
# Число классов берётся из чекпоинта, предсказание приводится к кодировке
# 0 не размечено, 1 опухоль, 2 строма — как в маске.
#
# python -u seg_eval_slide.py --slide ovary_prime_he \
#   --he data/raw/ovary_prime/..._he_image.ome.tif \
#   --model outputs/models/seg2_ovary_prime_he.pth --stride 512

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from src.utils import get_device, hf_login, ensure_dir
from src.data.patching import load_image
from seg_decoder import SegDecoder

GRID = 14
NAMES = ["нет метки", "опухоль", "строма"]
COL = np.array([[220, 50, 47], [38, 139, 210]], np.uint8)
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
    n_classes = int(ckpt.get("n_classes", 3))
    dec = SegDecoder(in_dim=1024, n_classes=n_classes).to(device)
    dec.load_state_dict(ckpt["state_dict"])
    dec.eval()
    tumor, stroma = (0, 1) if n_classes == 2 else (1, 2)
    print(f"декодер на {n_classes} класса, каналы опухоль/строма: {tumor}/{stroma}")
    return dec, n_classes, tumor, stroma


@torch.no_grad()
def run_batch(uni, dec, arrs, device):
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
    for c, nm in ((1, "tumor"), (2, "stroma")):
        i, d = iou_dice(pred, truth, c, where)
        print(f"  {nm:7s} IoU {i:.3f}  Dice {d:.3f}")
        out[f"{nm}_iou"] = round(float(i), 4)
        out[f"{nm}_dice"] = round(float(d), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--he", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seg-dir", default="data/processed/seg")
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--cds", type=int, default=8, help="во сколько раз мельче карта результата")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--metrics-csv", default="outputs/results/seg_slide_metrics.csv")
    args = ap.parse_args()

    import cv2

    mask_npz = Path(args.seg_dir) / f"{args.slide}_mask.npz"
    if not mask_npz.exists():
        raise SystemExit(f"нет {mask_npz} — сперва make_seg_masks_v.py")

    device = get_device()
    print("device:", device)
    uni = build_uni(device)
    dec, n_classes, TUMOR, STROMA = load_decoder(args.model, device)

    print("грузим H&E...")
    img = load_image(args.he)   # ome.tif читается через tifffile, PIL его не берёт
    H, W = img.shape[:2]
    ps, st, cds = args.patch_size, args.stride, args.cds
    Hc, Wc = H // cds, W // cds
    print(f"размер {W}x{H}, карта {Wc}x{Hc}, шаг {st}")

    acc = np.zeros((Hc, Wc, n_classes), np.float32)
    pp = ps // cds
    arrs, pos, done = [], [], 0

    def flush():
        nonlocal arrs, pos, done
        if not arrs:
            return
        probs = run_batch(uni, dec, arrs, device)
        for k, (yc, xc) in enumerate(pos):
            pm = probs[k].transpose(1, 2, 0)
            pm = np.asarray(Image.fromarray((pm * 255).astype(np.uint8)).resize((pp, pp))) / 255.0
            y1, x1 = min(yc + pp, Hc), min(xc + pp, Wc)
            acc[yc:y1, xc:x1] += pm[:y1 - yc, :x1 - xc]
        done += len(arrs)
        arrs, pos = [], []
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

    covered = acc.sum(2) > 0
    arg = acc.argmax(2)
    pred = np.zeros(arg.shape, np.uint8)
    pred[covered & (arg == TUMOR)] = 1
    pred[covered & (arg == STROMA)] = 2

    # при шаге больше патча окна не смыкаются и часть среза остаётся непросмотренной;
    # такие пиксели надо исключать из метрик, иначе они засчитываются как ошибки
    print(f"окном покрыто {100 * covered.mean():.1f}% карты (патч {ps}, шаг {st})")

    d = np.load(mask_npz)
    truth = cv2.resize(d["mask"], (Wc, Hc), interpolation=cv2.INTER_NEAREST)
    print(f"маска {d['mask'].shape[1]}x{d['mask'].shape[0]} при downscale "
          f"{int(d['downscale'])} -> приведена к {Wc}x{Hc}")

    inside = covered & (truth > 0)
    rows = [
        {"scope": "покрытое окном", **report(pred, truth, covered,
                                             "просмотренные пиксели, вместе с неразмеченным")},
        {"scope": "внутри разметки", **report(pred, truth, inside,
                                              "просмотренные пиксели, где есть метка")},
    ]

    print("\nматрица ошибок по просмотренному (строки — истина, доли внутри строки):")
    print(f"{'':12s}" + "".join(f"{n:>12s}" for n in NAMES))
    for t in range(3):
        sel = covered & (truth == t)
        n = sel.sum()
        if not n:
            continue
        fr = [(pred[sel] == p).sum() / n for p in range(3)]
        print(f"{NAMES[t]:12s}" + "".join(f"{v:12.3f}" for v in fr))

    pt, pstr = int((pred == 1).sum()), int((pred == 2).sum())
    tt = int((covered & (truth == 1)).sum())
    tstr = int((covered & (truth == 2)).sum())
    tsp_pred = 100 * pstr / (pt + pstr) if pt + pstr else float("nan")
    tsp_true = 100 * tstr / (tt + tstr) if tt + tstr else float("nan")
    print(f"\nдоля стромы: предсказано {tsp_pred:.1f}%, по разметке {tsp_true:.1f}%")

    out_csv = Path(args.metrics_csv)
    ensure_dir(out_csv.parent)
    first = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        for r in rows:
            r = {"slide": args.slide, "model": args.model, "stride": st,
                 "n_classes": n_classes,
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
        for c, col in ((1, COL[0]), (2, COL[1])):
            ov[m == c] = (0.45 * col + 0.55 * bg[m == c]).astype(np.uint8)
        panels.append(ov)
    out_png = ensure_dir("outputs/results") / f"{args.slide}_eval.png"
    Image.fromarray(np.concatenate(panels, 1)).save(out_png)
    print("исходник | истина | предсказание:", out_png)


if __name__ == "__main__":
    main()