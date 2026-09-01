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
from src.stain import slide_normalizer
from seg_decoder import SegDecoder

GRID = 14
COL = np.array([[220, 50, 47], [38, 139, 210]], np.uint8)
ZONE_NAMES = {1: "опухоль", 2: "гормональная строма", 3: "матриксная строма",
              4: "иммунные", 5: "сосуды и прочее"}
ZONE_COL = {1: (220, 50, 47), 2: (255, 165, 0), 3: (38, 139, 210),
            4: (42, 161, 82), 5: (128, 128, 128)}
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
    c2c = ckpt.get("channel_to_class")
    if c2c:
        ch2cls = [int(c2c[k]) for k in sorted(c2c, key=int)]
    else:
        ch2cls = [1, 2] if n_classes == 2 else [0, 1, 2]   # старая схема с фоном
    print("декодер на %d класса, каналы: %s" % (n_classes, ", ".join(
        "%d=%s" % (i, ZONE_NAMES.get(c, "не размечено")) for i, c in enumerate(ch2cls))))
    return dec, n_classes, ch2cls


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
    ap.add_argument("--stain-ref", default=None,
                    help="npz эталона окраски, тот же что при seg_extract.py")
    ap.add_argument("--smooth", type=float, default=2.0,
                    help="сглаживание карты вероятностей, 0 выключает")
    ap.add_argument("--white", type=int, default=225,
                    help="ярче этого считаем стеклом, а не тканью")
    ap.add_argument("--min-region", type=int, default=2000,
                    help="выбрасывать изолированные пятна мельче этого, 0 выключает")
    ap.add_argument("--no-prob", dest="save_prob", action="store_false",
                    help="не сохранять вероятности, они нужны только для подбора вида")
    args = ap.parse_args()
    norm = slide_normalizer(args.svs, args.stain_ref) if args.stain_ref else None

    import openslide
    device = get_device()
    print("device:", device)
    uni = build_uni(device)
    dec, n_classes, ch2cls = load_decoder(args.model, device)

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
        probs = run_batch(uni, dec, arrs, device, norm)
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

    # 0 не размечено, дальше коды зон из src/utils
    covered = acc.sum(2) > 0          # считаем до сглаживания, иначе фон расползётся
    raw = acc                         # несглаженное идёт в npz, для перерисовки
    if args.smooth > 0:
        # каждый токен UNI классифицируется сам по себе, соседи спорят и карта
        # получается крапчатой. Сглаживаем вероятности, а не готовые метки
        from scipy.ndimage import gaussian_filter
        acc = gaussian_filter(raw, (args.smooth, args.smooth, 0))
    arg = acc.argmax(2)
    lut = np.array(ch2cls, np.uint8)
    zones = np.where(covered, lut[arg], 0).astype(np.uint8)

    # Обрезаем по настоящей ткани, а не по сетке окон: окно попадает на срез
    # целиком или никак, поэтому край шёл ступеньками. Заодно выбрасываем
    # изолированную мелочь, это пылинки и грязь на стекле
    import cv2
    tmb = np.asarray(sl.get_thumbnail((2000, 2000)).convert("RGB"))
    tis = cv2.resize((tmb.mean(2) < args.white).astype(np.uint8),
                     (zones.shape[1], zones.shape[0]),
                     interpolation=cv2.INTER_NEAREST)
    zones[tis == 0] = 0
    if args.min_region > 0:
        from scipy.ndimage import label
        lab, _ = label(zones > 0)
        sizes = np.bincount(lab.ravel())
        zones[np.isin(lab, np.where(sizes < args.min_region)[0])] = 0
    covered = zones > 0

    print(f"размечено {100*covered.mean():.1f}% кадра")
    present = [c for c in sorted(set(ch2cls)) if c]
    counts = {c: int((zones == c).sum()) for c in present}
    total = sum(counts.values())
    for c in present:
        print("  %-20s %5.1f%%" % (ZONE_NAMES.get(c, c),
                                   100 * counts[c] / total if total else 0))
    tum = counts.get(1, 0)
    stroma = sum(counts.get(c, 0) for c in (2, 3, 5))
    if tum + stroma:
        print("TSR (строма / опухоль+строма): %.3f" % (stroma / (tum + stroma)))
    horm, matr = counts.get(2, 0), counts.get(3, 0)
    if horm + matr:
        print("состав стромы: гормональная %.1f%%, матриксная %.1f%%"
              % (100 * horm / (horm + matr), 100 * matr / (horm + matr)))

    # Превью: берём маленький thumbnail, как seg_infer.py берёт нижний уровень
    # пирамиды. Иначе карта классов (cds=8) даёт видимые блоки и файл в десятки МБ.
    max_dim = 2000
    thumb = np.asarray(sl.get_thumbnail((max_dim, max_dim)).convert("RGB"))
    import cv2
    zr = cv2.resize(zones, (thumb.shape[1], thumb.shape[0]), interpolation=cv2.INTER_NEAREST)
    ov = thumb.copy()
    alpha = 0.45
    for c in present:
        mask = zr == c
        if mask.any():
            col = np.array(ZONE_COL[c], np.float32)
            ov[mask] = (alpha * col + (1 - alpha) * thumb[mask]).astype(np.uint8)

    edge = np.zeros(zr.shape, bool)
    edge[:-1] |= zr[:-1] != zr[1:]
    edge[:, :-1] |= zr[:, :-1] != zr[:, 1:]
    edge &= zr > 0
    ov[edge] = (0.55 * ov[edge]).astype(np.uint8)

    out = args.out or str(ensure_dir("outputs/results") / (Path(args.svs).stem + "_seg_map.png"))
    from PIL import ImageDraw, ImageFont
    # встроенный шрифт PIL растровый и без кириллицы, подписи выходят пустыми
    font = None
    for fp in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
               "/usr/share/fonts/dejavu/DejaVuSans.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if Path(fp).exists():
            font = ImageFont.truetype(fp, 28)
            break
    if font is None:
        try:
            import matplotlib
            font = ImageFont.truetype(str(Path(matplotlib.__file__).parent /
                                          "mpl-data/fonts/ttf/DejaVuSans.ttf"), 28)
        except Exception:
            font = ImageFont.load_default()

    panel = Image.fromarray(np.concatenate([thumb, ov], 1))
    d = ImageDraw.Draw(panel)
    lx, ly, step = thumb.shape[1] + 20, 20, 40
    d.rectangle([lx - 12, ly - 12, lx + 430, ly + step * len(present)],
                fill=(255, 255, 255), outline=(120, 120, 120))
    for c in present:
        d.rectangle([lx, ly, lx + 28, ly + 28], fill=ZONE_COL[c], outline=(0, 0, 0))
        d.text((lx + 40, ly + 2), ZONE_NAMES.get(c, str(c)), fill=(0, 0, 0), font=font)
        ly += step
    panel.save(out, optimize=True)

    # карта классов нужна, чтобы пересчитывать TSR по разным областям
    # без повторного прогона UNI; mpp — микрометры на пиксель этой карты
    npz = Path(out).with_suffix(".npz")
    saved = {"cls": zones, "ch2cls": np.array(ch2cls, np.uint8),
             "mpp": eff * cds}
    if args.save_prob:
        saved["prob"] = raw.astype(np.float16)
    np.savez_compressed(npz, **saved)

    sl.close()
    print("карта зон:", out)
    print("карта классов:", npz)


if __name__ == "__main__":
    main()