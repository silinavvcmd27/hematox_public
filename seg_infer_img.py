# Инференс сегментации для обычной картинки (jpg/png/tif) через PIL.
#
# Число классов берётся из чекпоинта. На выходе всегда одна кодировка:
# 0 не размечено, 1 опухоль, 2 строма.
#
# python seg_infer_img.py --he data/visium_ffpe_ov/..._image.jpg \
#   --model outputs/models/seg2_ovary3_he.pth --name visium_ffpe_ov --mpp 0.5

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from src.utils import get_device, hf_login, ensure_dir
from seg_decoder import SegDecoder

Image.MAX_IMAGE_PIXELS = None
GRID = 14
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--he", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--cds", type=int, default=8)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--mpp", type=float, default=float("nan"),
                    help="мкм/px исходной картинки; нужен, чтобы карту читал tsr_regions.py")
    args = ap.parse_args()

    import cv2
    device = get_device()
    print("device:", device)
    uni = build_uni(device)
    dec, n_classes, TUMOR, STROMA = load_decoder(args.model, device)

    print("грузим H&E...")
    img = np.asarray(Image.open(args.he).convert("RGB"))
    H, W = img.shape[:2]
    print("размер:", W, "x", H)
    ps, st, cds = args.patch_size, args.stride, args.cds
    Hc, Wc = H // cds, W // cds
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
        print("  патчей: %d" % done, end="\r")

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
    print("\nвсего патчей:", done)

    covered = acc.sum(2) > 0
    arg = acc.argmax(2)
    zones = np.zeros(arg.shape, np.uint8)
    zones[covered & (arg == TUMOR)] = 1
    zones[covered & (arg == STROMA)] = 2

    nt, ns = int((zones == 1).sum()), int((zones == 2).sum())
    print("окном покрыто %.1f%% карты" % (100 * covered.mean()))
    if nt + ns:
        print("TSP: tumor %.1f%%  stroma %.1f%%" % (100 * nt / (nt + ns), 100 * ns / (nt + ns)))

    bg = cv2.resize(img, (Wc, Hc), interpolation=cv2.INTER_AREA)
    ov = bg.copy()
    for c, col in ((1, COL[0]), (2, COL[1])):
        ov[zones == c] = (0.45 * col + 0.55 * bg[zones == c]).astype(np.uint8)

    name = args.name or Path(args.he).stem
    out = ensure_dir("outputs/results") / ("%s_seg_map.png" % name)
    Image.fromarray(np.concatenate([bg, ov], 1)).save(out)

    npz = Path(out).with_suffix(".npz")
    np.savez_compressed(npz, cls=zones, mpp=args.mpp * cds)
    print("карта зон:", out)
    print("карта классов:", npz)


if __name__ == "__main__":
    main()