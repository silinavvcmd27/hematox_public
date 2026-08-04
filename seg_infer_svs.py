# Полнослайдовый инференс сегментации для .svs (TCGA) через openslide.
# Читает срез на нужном мкм/px, скользящим окном -> UNI -> декодер -> карта зон + TSP.
#
# Число классов берётся из чекпоинта: у двухклассовых моделей это 0=опухоль,
# 1=строма, у старых трёхклассовых 0=фон, 1=опухоль, 2=строма. На выходе всегда
# одна кодировка: 0 не размечено, 1 опухоль, 2 строма.
#
# python seg_infer_svs.py --svs data/tcga_ov_flat/XXX.svs --model outputs/models/seg2_ovary3_he.pth

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from src.utils import get_device, hf_login, ensure_dir
from seg_decoder import SegDecoder

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
    # каналы, в которых модель держит опухоль и строму
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
    ap.add_argument("--svs", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--target-mpp", type=float, default=0.27,
                    help="масштаб, на котором обучалась модель")
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--cds", type=int, default=8, help="downscale карты-результата")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import openslide
    device = get_device()
    print("device:", device)
    uni = build_uni(device)
    dec, n_classes, TUMOR, STROMA = load_decoder(args.model, device)

    sl = openslide.OpenSlide(args.svs)
    mpp = float(sl.properties.get("openslide.mpp-x", 0.5) or 0.5)
    ds = max(1.0, args.target_mpp / mpp)
    level = sl.get_best_level_for_downsample(ds)
    ldown = sl.level_downsamples[level]
    Wl, Hl = sl.level_dimensions[level]
    eff = mpp * ldown
    print(f"mpp0={mpp:.3f} level={level} (~{eff:.2f} мкм/px) размер {Wl}x{Hl}")
    if eff > 1.5 * args.target_mpp:
        print("  внимание: подходящего уровня нет, масштаб заметно грубее обучающего")

    ps, st, cds = args.patch_size, args.stride, args.cds
    Hc, Wc = Hl // cds, Wl // cds
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
            if pm.ndim == 2:
                pm = pm[:, :, None]
            y1, x1 = min(yc + pp, Hc), min(xc + pp, Wc)
            acc[yc:y1, xc:x1] += pm[:y1 - yc, :x1 - xc]
        done += len(arrs)
        arrs, pos = [], []
        print(f"  патчей: {done}", end="\r")

    for y in range(0, Hl - ps, st):
        for x in range(0, Wl - ps, st):
            reg = np.asarray(sl.read_region((int(x * ldown), int(y * ldown)), level,
                                            (ps, ps)).convert("RGB"))
            if (reg.mean(-1) > 220).mean() > 0.85:
                continue
            arrs.append(reg)
            pos.append((y // cds, x // cds))
            if len(arrs) >= args.bs:
                flush()
    flush()
    print(f"\nвсего патчей: {done}")

    # приводим к единой кодировке: 0 не размечено, 1 опухоль, 2 строма
    covered = acc.sum(2) > 0
    arg = acc.argmax(2)
    zones = np.zeros(arg.shape, np.uint8)
    zones[covered & (arg == TUMOR)] = 1
    zones[covered & (arg == STROMA)] = 2

    nt, ns = int((zones == 1).sum()), int((zones == 2).sum())
    print(f"окном покрыто {100*covered.mean():.1f}% карты")
    if nt + ns:
        print(f"TSP: tumor {100*nt/(nt+ns):.1f}%  stroma {100*ns/(nt+ns):.1f}%")

    # Превью: берём маленький thumbnail, как seg_infer.py берёт нижний уровень
    # пирамиды. Иначе карта классов (cds=8) даёт видимые блоки и файл в десятки МБ.
    max_dim = 2000
    thumb = np.asarray(sl.get_thumbnail((max_dim, max_dim)).convert("RGB"))
    import cv2
    zr = cv2.resize(zones, (thumb.shape[1], thumb.shape[0]), interpolation=cv2.INTER_NEAREST)
    ov = thumb.copy()
    alpha = 0.45
    for c, col in ((1, COL[0]), (2, COL[1])):
        mask = zr == c
        ov[mask] = (alpha * col + (1 - alpha) * thumb[mask]).astype(np.uint8)

    out = args.out or str(ensure_dir("outputs/results") / (Path(args.svs).stem + "_seg_map.png"))
    Image.fromarray(np.concatenate([thumb, ov], 1)).save(out, optimize=True)

    # карта классов нужна, чтобы пересчитывать TSR по разным областям
    # без повторного прогона UNI; mpp — микрометры на пиксель этой карты
    npz = Path(out).with_suffix(".npz")
    np.savez_compressed(npz, cls=zones, mpp=eff * cds)

    sl.close()
    print("карта зон:", out)
    print("карта классов:", npz)


if __name__ == "__main__":
    main()