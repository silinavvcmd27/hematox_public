# Этап 1 сегментации: токен-признаки UNI для патчей + куски маски.
# Один раз (медленно на CPU), потом декодер учится быстро на готовых признаках.
#
# Размер патча задаётся в микрометрах и пересчитывается в пиксели по метаданным
# среза. Раньше он был фиксирован в пикселях, из-за чего ovary3 с его вдвое
# мельче пикселем обрабатывался на другом увеличении.
#
# python seg_extract.py --slide ovary_prime_he \
#   --he data/raw/ovary_prime/..._he_image.ome.tif --max-patches 12000

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from src.utils import load_config, get_device, hf_login, ensure_dir
from slide_mpp import slide_mpp

GRID = 14
OUT = 112


def build_uni(device):
    import timm
    hf_login()
    m = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True,
                          init_values=1e-5, dynamic_img_size=True)
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad = False
    return m


_tf = T.Compose([
    T.Resize(224), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def encode(model, patches, device, bs=32):
    feats = []
    for i in range(0, len(patches), bs):
        batch = torch.stack([_tf(Image.fromarray(p)) for p in patches[i:i + bs]]).to(device)
        f = model.forward_features(batch)
        if f.shape[1] == GRID * GRID + 1:
            f = f[:, 1:, :]
        elif f.shape[1] != GRID * GRID:
            f = f[:, -GRID * GRID:, :]
        feats.append(f.half().cpu().numpy())
        if (i // bs) % 20 == 0:
            print(f"  закодировано {min(i + bs, len(patches))}/{len(patches)}", end="\r")
    return np.concatenate(feats, 0)


def relabel_superpixel(he_patch, mask112, n_seg, min_frac=0.35):
    """Метки по суперпикселям: SLIC делит патч по морфологии H&E, каждому куску —
    преобладающий класс внутри. Граница метки садится на край ткани, а не на
    геометрию Вороного. Кусок без разметки (мало клеток) остаётся фоном."""
    from skimage.segmentation import slic
    import cv2
    he = cv2.resize(he_patch, (OUT, OUT)).astype(np.float32) / 255.0
    try:
        sp = slic(he, n_segments=n_seg, compactness=10.0, start_label=1, channel_axis=-1)
    except TypeError:
        sp = slic(he, n_segments=n_seg, compactness=10.0, multichannel=True)
    out = np.zeros_like(mask112)
    for lab in np.unique(sp):
        sel = sp == lab
        vals = mask112[sel]
        nz = vals[vals > 0]
        if vals.size == 0 or nz.size / vals.size < min_frac:
            continue
        out[sel] = np.bincount(nz).argmax()
    return out


def variants(patch, mask, n):
    """Повороты и отражения. Срез не имеет верха и низа, так что это честно."""
    out = [(patch, mask)]
    if n >= 4:
        for k in (1, 2, 3):
            out.append((np.rot90(patch, k), np.rot90(mask, k)))
    if n >= 8:
        out += [(np.fliplr(p), np.fliplr(m)) for p, m in list(out)]
    return [(np.ascontiguousarray(p), np.ascontiguousarray(m)) for p, m in out[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--he", required=True)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--patch-um", type=float, default=70.0,
                    help="поле зрения патча в микрометрах")
    ap.add_argument("--patch-size", type=int, default=None,
                    help="размер в пикселях, если задан — перекрывает --patch-um")
    ap.add_argument("--max-patches", type=int, default=12000)
    ap.add_argument("--augment", type=int, choices=[1, 4, 8], default=1,
                    help="1 без аугментации, 4 повороты, 8 повороты и отражения")
    ap.add_argument("--seg-dir", default="data/processed/seg")
    ap.add_argument("--out-dir", default=None,
                    help="куда писать feat.npz (по умолчанию = seg-dir)")
    ap.add_argument("--superpixel", type=int, default=0,
                    help="n_segments SLIC (0=выкл): метки по суперпикселям вместо Вороного")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seg = Path(args.seg_dir)
    man = pd.read_csv(seg / f"{args.slide}_patches.csv")
    d = np.load(seg / f"{args.slide}_mask.npz")
    mask, ds = d["mask"], int(d["downscale"])

    if args.patch_size:
        ps = args.patch_size
        print(f"{args.slide}: патч задан явно, {ps} px")
    else:
        mpp = slide_mpp(args.he)
        ps = int(round(args.patch_um / mpp))
        print(f"{args.slide}: {mpp:.4f} мкм/px -> патч {args.patch_um:.0f} мкм = {ps} px")

    if args.max_patches and len(man) > args.max_patches:
        man = man.sample(args.max_patches, random_state=42).reset_index(drop=True)
    print(f"{args.slide}: позиций {len(man)}, вариантов на позицию {args.augment}")

    from src.data.patching import load_image
    import cv2
    print("грузим H&E...")
    img = load_image(args.he)
    H, W = img.shape[:2]

    patches, masks, skipped = [], [], 0
    for r in man.itertuples(index=False):
        x0, y0 = int(r.x0), int(r.y0)
        if x0 + ps > W or y0 + ps > H:
            skipped += 1
            continue
        p = img[y0:y0 + ps, x0:x0 + ps]
        m = mask[y0 // ds:(y0 + ps) // ds, x0 // ds:(x0 + ps) // ds]
        m = cv2.resize(m, (OUT, OUT), interpolation=cv2.INTER_NEAREST)
        if args.superpixel:
            m = relabel_superpixel(p, m, args.superpixel)
        for pv, mv in variants(p, m, args.augment):
            patches.append(pv)
            masks.append(mv)
    del img
    if skipped:
        print(f"  не влезли в край: {skipped} позиций")
    print(f"  всего патчей на кодирование: {len(patches)}")

    device = get_device()
    print("device:", device, "| кодируем через UNI...")
    enc = build_uni(device)
    X = encode(enc, patches, device, bs=cfg["encoder"]["batch_size"])
    y = np.stack(masks).astype(np.uint8)

    out = ensure_dir(args.out_dir or args.seg_dir) / f"{args.slide}_feat.npz"
    np.savez_compressed(out, X=X, y=y, patch_px=ps, patch_um=args.patch_um)
    print(f"\nsaved {out}  X={X.shape} y={y.shape}")
    print("пиксели:", {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))})


if __name__ == "__main__":
    main()