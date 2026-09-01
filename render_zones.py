# Перерисовка карты зон из сохранённых вероятностей. Секунды вместо часа: UNI
# уже отработал, в npz лежит его выход, и остаётся только сгладить, обрезать по
# ткани и раскрасить.
#
#   python render_zones.py --npz outputs/results/XXX_seg_map.npz \
#     --svs data/tcga_ov_flat/XXX.svs --smooth 3 --min-region 4000

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from seg_infer_svs import ZONE_COL, ZONE_NAMES


def legend_font(size=28):
    for fp in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
               "/usr/share/fonts/dejavu/DejaVuSans.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if Path(fp).exists():
            return ImageFont.truetype(fp, size)
    try:
        import matplotlib
        return ImageFont.truetype(str(Path(matplotlib.__file__).parent /
                                      "mpl-data/fonts/ttf/DejaVuSans.ttf"), size)
    except Exception:
        return ImageFont.load_default()


def build_zones(prob, ch2cls, tissue, smooth, min_region):
    covered = prob.sum(2) > 0
    if smooth > 0:
        from scipy.ndimage import gaussian_filter
        prob = gaussian_filter(prob.astype(np.float32), (smooth, smooth, 0))
    zones = np.where(covered, np.asarray(ch2cls, np.uint8)[prob.argmax(2)], 0)
    zones = zones.astype(np.uint8)
    zones[~tissue] = 0
    if min_region > 0:
        from scipy.ndimage import label
        lab, _ = label(zones > 0)
        sizes = np.bincount(lab.ravel())
        zones[np.isin(lab, np.where(sizes < min_region)[0])] = 0
    return zones


def stats(zones, present):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--svs", required=True)
    ap.add_argument("--smooth", type=float, default=2.0)
    ap.add_argument("--min-region", type=int, default=2000)
    ap.add_argument("--white", type=int, default=225)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--outline", action="store_true", default=True)
    ap.add_argument("--no-outline", dest="outline", action="store_false")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import cv2
    import openslide

    d = np.load(args.npz)
    if "prob" not in d:
        raise SystemExit("в npz нет вероятностей: пересчитайте срез обновлённым "
                         "seg_infer_svs.py, старые файлы хранят только метки")
    prob, ch2cls = d["prob"], d["ch2cls"]

    sl = openslide.OpenSlide(args.svs)
    thumb = np.asarray(sl.get_thumbnail((2000, 2000)).convert("RGB"))
    sl.close()

    tis_small = (thumb.mean(2) < args.white).astype(np.uint8)
    tissue = cv2.resize(tis_small, (prob.shape[1], prob.shape[0]),
                        interpolation=cv2.INTER_NEAREST).astype(bool)

    zones = build_zones(prob, ch2cls, tissue, args.smooth, args.min_region)
    present = [c for c in sorted(set(ch2cls.tolist())) if c]
    print("размечено %.1f%% кадра" % (100 * (zones > 0).mean()))
    stats(zones, present)

    zr = cv2.resize(zones, (thumb.shape[1], thumb.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
    ov = thumb.copy()
    for c in present:
        mask = zr == c
        if mask.any():
            col = np.array(ZONE_COL[c], np.float32)
            ov[mask] = (args.alpha * col + (1 - args.alpha) * thumb[mask]).astype(np.uint8)

    if args.outline:
        edge = np.zeros(zr.shape, bool)
        edge[:-1] |= zr[:-1] != zr[1:]
        edge[:, :-1] |= zr[:, :-1] != zr[:, 1:]
        edge &= zr > 0
        ov[edge] = (0.55 * ov[edge]).astype(np.uint8)

    panel = Image.fromarray(np.concatenate([thumb, ov], 1))
    draw = ImageDraw.Draw(panel)
    font = legend_font()
    lx, ly, step = thumb.shape[1] + 20, 20, 40
    draw.rectangle([lx - 12, ly - 12, lx + 430, ly + step * len(present)],
                   fill=(255, 255, 255), outline=(120, 120, 120))
    for c in present:
        draw.rectangle([lx, ly, lx + 28, ly + 28], fill=ZONE_COL[c], outline=(0, 0, 0))
        draw.text((lx + 40, ly + 2), ZONE_NAMES.get(c, str(c)), fill=(0, 0, 0), font=font)
        ly += step

    out = args.out or str(Path(args.npz).with_name(
        Path(args.npz).stem + "_s%g_r%d.png" % (args.smooth, args.min_region)))
    panel.save(out, optimize=True)
    print("сохранено:", out)


if __name__ == "__main__":
    main()
