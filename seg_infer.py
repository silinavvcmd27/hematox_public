# Полнослайдовый инференс сегментации: скользящее окно по H&E -> UNI -> декодер ->
# сшитая плотная карта tumor/stroma поверх среза.
#
# Число классов берётся из чекпоинта. На выходе всегда одна кодировка:
# 0 не размечено, 1 опухоль, 2 строма.
#
# python seg_infer.py --slide ovary_prime_he \
#   --he data/raw/ovary_prime/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_he_image.ome.tif \
#   --model outputs/models/seg2_ovary_prime_he.pth

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from src.utils import get_device, hf_login, ensure_dir
from src.data.patching import load_image, is_background
from seg_decoder import SegDecoder

GRID = 14
COL = np.array([[220, 50, 47], [38, 139, 210]], np.uint8)  # tumor, stroma

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


def ome_mpp(path):
    # размер пикселя записан в метаданных OME; у ovary3 он вдвое меньше,
    # чем у двух других срезов, поэтому подставлять константу нельзя
    import re
    import tifffile
    with tifffile.TiffFile(path) as tf:
        xml = tf.ome_metadata or ""
    m = re.search(r'PhysicalSizeX="([\d.eE+-]+)"', xml)
    return float(m.group(1)) if m else float("nan")


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
    ap.add_argument("--slide", required=True)
    ap.add_argument("--he", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--cds", type=int, default=8, help="downscale карты-результата")
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()

    device = get_device()
    print("device:", device)
    uni = build_uni(device)
    dec, n_classes, TUMOR, STROMA = load_decoder(args.model, device)

    print("грузим H&E...")
    img = load_image(args.he)
    H, W = img.shape[:2]
    ps, st, cds = args.patch_size, args.stride, args.cds
    Hc, Wc = H // cds, W // cds
    acc = np.zeros((Hc, Wc, n_classes), np.float32)
    pp = ps // cds                       # размер вклада патча на карте

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
        print(f"  обработано патчей: {done}", end="\r")

    for y0 in range(0, H - ps + 1, st):
        for x0 in range(0, W - ps + 1, st):
            patch = img[y0:y0 + ps, x0:x0 + ps]
            if is_background(patch):
                continue
            arrs.append(patch)
            pos.append((y0 // cds, x0 // cds))
            if len(arrs) >= args.bs:
                flush()
    flush()
    del img
    print(f"\nвсего патчей: {done}")

    covered = acc.sum(2) > 0
    arg = acc.argmax(2)
    zones = np.zeros(arg.shape, np.uint8)
    zones[covered & (arg == TUMOR)] = 1
    zones[covered & (arg == STROMA)] = 2

    nt, ns = int((zones == 1).sum()), int((zones == 2).sum())
    print(f"окном покрыто {100*covered.mean():.1f}% карты")
    if nt + ns:
        print(f"TSP: tumor {100*nt/(nt+ns):.1f}%  stroma {100*ns/(nt+ns):.1f}%")

    import tifffile, cv2
    with tifffile.TiffFile(args.he) as tf:
        s = tf.series[0]
        try:
            bg = list(s.levels)[-1].asarray()
        except (AttributeError, IndexError):
            bg = s.asarray()
    if bg.ndim == 3 and bg.shape[0] in (3, 4):
        bg = np.moveaxis(bg, 0, -1)
    bg = bg[..., :3].astype(np.uint8)

    zr = cv2.resize(zones, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_NEAREST)
    ov = bg.copy()
    for c, col in ((1, COL[0]), (2, COL[1])):
        ov[zr == c] = (0.45 * col + 0.55 * bg[zr == c]).astype(np.uint8)

    out = ensure_dir("outputs/results") / f"{args.slide}_seg_map.png"
    Image.fromarray(np.concatenate([bg, ov], 1)).save(out)

    npz = Path(out).with_suffix(".npz")
    np.savez_compressed(npz, cls=zones, mpp=ome_mpp(args.he) * cds)
    print("карта зон:", out)
    print("карта классов:", npz)


if __name__ == "__main__":
    main()