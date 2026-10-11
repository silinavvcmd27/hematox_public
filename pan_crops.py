# Увеличенные поля карты зон рядом с H&E: проверить разметку глазом и взять в
# рисунки. Карта это _seg_map.npz, который пишет seg_infer_svs.py. Карт одного
# среза можно дать несколько (например, от двух моделей), тогда в строке H&E и
# по полю на каждую карту. Сетки у карт могут отличаться (старые считались на
# уровне пирамиды, новые в масштабе модели), поэтому поле задаётся в микрометрах
# и в каждой карте вырезается по её mpp.
#
# Поля выбираются по первой карте: ткани не меньше 60% поля, доля опухоли в
# пределах --tumor-frac. По умолчанию это смешанные поля, где видна граница
# опухоли и стромы; --tumor-frac 0 0.1 покажет, что модель считает чистой стромой.
# Через --at поле задаётся вручную, левым верхним углом в миллиметрах.
#
#   python pan_crops.py --svs data/pan/tcga_other/TCGA-38-4631*.svs \
#     --npz outputs/results/pan/tcga_deploy/TCGA-38-4631*_seg_map.npz \
#           outputs/results/pan/tcga_other/TCGA-38-4631*_seg_map.npz
#   python pan_crops.py ... --at 3.9 1.9 --at 23.2 1.6 --out поля.png

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.utils import TUMOR
from seg_infer_svs import ZONE_COL
from render_zones import legend_font

STROMA_CODES = (2, 3, 5)


def window_sums(mask, f, step):
    ii = np.pad(mask.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32), ((1, 0), (1, 0)))
    Y, X = np.meshgrid(np.arange(0, mask.shape[0] - f + 1, step),
                       np.arange(0, mask.shape[1] - f + 1, step), indexing="ij")
    return Y, X, ii[Y + f, X + f] - ii[Y, X + f] - ii[Y + f, X] + ii[Y, X]


def pick_fields(cls, f, n, lo, hi, rng):
    Y, X, n_tum = window_sums(cls == TUMOR, f, f // 2)
    _, _, n_str = window_sums(np.isin(cls, STROMA_CODES), f, f // 2)
    frac = n_tum / np.maximum(n_tum + n_str, 1)
    ok = ((n_tum + n_str) >= 0.6 * f * f) & (frac >= lo) & (frac <= hi)
    cand = list(zip(Y[ok], X[ok]))
    chosen = []
    for i in rng.permutation(len(cand)):
        y, x = cand[i]
        if all(abs(y - cy) >= f or abs(x - cx) >= f for cy, cx in chosen):
            chosen.append((y, x))
            if len(chosen) == n:
                break
    return chosen


def overlay(rgb, zones, alpha=0.45):
    out = rgb.astype(np.float32)
    for code, col in ZONE_COL.items():
        m = zones == code
        out[m] = alpha * np.array(col, np.float32) + (1 - alpha) * out[m]
    edge = np.zeros(zones.shape, bool)
    edge[:-1] |= zones[:-1] != zones[1:]
    edge[:, :-1] |= zones[:, :-1] != zones[:, 1:]
    out[edge & (zones > 0)] *= 0.55
    return out.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svs", required=True)
    ap.add_argument("--npz", nargs="+", required=True, help="карты одного среза, поля выбираются по первой")
    ap.add_argument("--field-um", type=float, default=1000)
    ap.add_argument("--px", type=int, default=700, help="сторона поля на картинке")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--tumor-frac", type=float, nargs=2, default=(0.2, 0.8))
    ap.add_argument("--at", type=float, nargs=2, action="append", metavar=("X_MM", "Y_MM"),
                    help="левый верхний угол поля в мм, можно повторять")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import openslide
    sl = openslide.OpenSlide(args.svs)
    mpp0 = float(sl.properties["openslide.mpp-x"])
    maps = []
    for p in args.npz:
        with np.load(p) as z:
            maps.append((z["cls"], float(z["mpp"])))

    cls, cell_um = maps[0]
    f = round(args.field_um / cell_um)
    if args.at:
        fields = [(round(y * 1000 / cell_um), round(x * 1000 / cell_um)) for x, y in args.at]
    else:
        fields = pick_fields(cls, f, args.n, *args.tumor_frac, np.random.default_rng(args.seed))
    if not fields:
        raise SystemExit("подходящих полей нет, ослабьте --tumor-frac или уменьшите --field-um")

    ps, gap, head = args.px, 10, 40
    cols = 1 + len(maps)
    canvas = Image.new("RGB", (cols * ps + (cols - 1) * gap, head + len(fields) * (ps + gap)), "white")
    draw = ImageDraw.Draw(canvas)
    font = legend_font(24)
    for c, title in enumerate(["H&E"] + [Path(p).parent.name for p in args.npz]):
        draw.text((c * (ps + gap) + 8, 8), title, fill=(0, 0, 0), font=font)

    side0 = args.field_um / mpp0
    level = sl.get_best_level_for_downsample(side0 / ps)
    side = round(side0 / sl.level_downsamples[level])
    for r, (y, x) in enumerate(fields):
        y_um, x_um = y * cell_um, x * cell_um
        he = sl.read_region((int(x_um / mpp0), int(y_um / mpp0)), level, (side, side))
        he = np.asarray(he.convert("RGB").resize((ps, ps), Image.BILINEAR))
        top = head + r * (ps + gap)
        canvas.paste(Image.fromarray(he), (0, top))
        shares = []
        for c, (m, m_um) in enumerate(maps, 1):
            k, j, fk = round(y_um / m_um), round(x_um / m_um), round(args.field_um / m_um)
            zones = np.ascontiguousarray(m[k:k + fk, j:j + fk])
            zones = np.asarray(Image.fromarray(zones).resize((ps, ps), Image.NEAREST))
            canvas.paste(Image.fromarray(overlay(he, zones)), (c * (ps + gap), top))
            lab = np.isin(zones, (TUMOR,) + STROMA_CODES)
            shares.append(f"{(zones[lab] == TUMOR).mean():.2f}" if lab.any() else "нет")
        print(f"поле {r + 1}: x {x_um / 1000:.1f} мм, y {y_um / 1000:.1f} мм, "
              f"доля опухоли по картам {' / '.join(shares)}")

    out = args.out or str(Path(args.npz[0]).with_name(
        Path(args.npz[0]).name.replace("_seg_map.npz", "_crops.png")))
    canvas.save(out, optimize=True)
    print("сохранено:", out)


if __name__ == "__main__":
    main()
