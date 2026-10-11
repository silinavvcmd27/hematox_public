# Применение на срезах TCGA для рисунка: H&E и карта зон, по строке на срез.
#
# Ничего заново не считается: карты берутся из готовых _seg_map.npz, которые пишет
# seg_infer_svs.py, а H&E из миниатюры самого среза. Под каждой строкой подпись с
# органом, проектом и долей стромы по карте.
#
# Без --slides берутся срезы с самой разной долей стромы: самый стромальный, самый
# опухолевый и промежуточные, по числу строк. Так на одной картинке видно и то, что
# модель работает на незнакомых органах, и то, что TSR у них разный.
#
#   python pan_show_tcga.py --dir outputs/results/pan/tcga_final --svs-dir data/pan/tcga_other
#   python pan_show_tcga.py --slides TCGA-HZ-8315 TCGA-38-4631 TCGA-AJ-A3OL --px 900

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.utils import TUMOR
from seg_infer_svs import ZONE_COL
from render_zones import legend_font

STROMA_CODES = (2, 3, 5)
OTHER = 7
NAMES = {1: "опухоль", 5: "строма", 7: "прочее"}
# по префиксу штрихкода TCGA: орган и проект, чтобы не подписывать руками
SLIDES = {"TCGA-HZ-8315": "Поджелудочная железа, PAAD",
          "TCGA-FP-8209": "Желудок, STAD",
          "TCGA-A6-6651": "Толстая кишка, COAD",
          "TCGA-A7-A0DC": "Молочная железа, BRCA",
          "TCGA-38-4631": "Лёгкое, LUAD",
          "TCGA-TN-A7HJ": "Голова и шея, HNSC",
          "TCGA-HQ-A2OF": "Мочевой пузырь, BLCA",
          "TCGA-AJ-A3OL": "Эндометрий, UCEC"}


def paint(rgb, zones, alpha=0.45):
    """Без контуров: на миниатюре целого среза они превращаются в грязь."""
    out = rgb.astype(np.float32)
    for code, col in ZONE_COL.items():
        m = zones == code
        out[m] = alpha * np.array(col, np.float32) + (1 - alpha) * out[m]
    return out.astype(np.uint8)


def shares(cls):
    tum = int((cls == TUMOR).sum())
    stro = int(np.isin(cls, STROMA_CODES).sum())
    oth = int((cls == OTHER).sum())
    tissue = tum + stro + oth
    if not tissue:
        return None
    return {"tsr": stro / (tum + stro) if tum + stro else float("nan"),
            "other": oth / tissue, "tissue": tissue}


def spread(rows, n):
    """Срезы с самой разной долей стромы: края обязательно, остальные равномерно."""
    rows = sorted(rows, key=lambda r: r[2]["tsr"])
    if n >= len(rows):
        return rows
    idx = np.unique(np.linspace(0, len(rows) - 1, n).round().astype(int))
    return [rows[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/results/pan/tcga_final")
    ap.add_argument("--svs-dir", default="data/pan/tcga_other")
    ap.add_argument("--slides", nargs="+", help="префиксы штрихкодов, иначе выберу сам")
    ap.add_argument("--n", type=int, default=3, help="сколько срезов на картинке")
    ap.add_argument("--px", type=int, default=760, help="сторона панели")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    found = []
    for p in sorted(Path(args.dir).glob("*_seg_map.npz")):
        stem = p.name.replace("_seg_map.npz", "")
        if args.slides and not any(stem.startswith(s) for s in args.slides):
            continue
        with np.load(p) as z:
            cls, mpp = z["cls"], float(z["mpp"])
        s = shares(cls)
        if s is None:
            print(f"  {stem[:12]}: ткани на карте нет, пропускаю")
            continue
        found.append((stem, cls, s, mpp))
    if not found:
        raise SystemExit(f"в {args.dir} нет подходящих карт *_seg_map.npz")
    chosen = found if args.slides else spread(found, args.n)

    import openslide
    ps, gap, head, cap = args.px, 12, 44, 30
    font, small = legend_font(24), legend_font(21)
    rows_out = []
    for stem, cls, s, mpp in chosen:
        svs = next(Path(args.svs_dir).glob(stem + ".svs"), None)
        if svs is None:
            raise SystemExit(f"не нашёлся срез {stem}.svs в {args.svs_dir}")
        sl = openslide.OpenSlide(str(svs))
        thumb = sl.get_thumbnail((ps, ps)).convert("RGB")
        sl.close()
        zones = np.asarray(Image.fromarray(cls).resize(thumb.size, Image.NEAREST))
        he = np.asarray(thumb)
        title = SLIDES.get(stem[:12], stem[:12])
        rows_out.append((f"{title}. Доля стромы {s['tsr']:.3f}, прочее {s['other']:.0%} площади",
                         [Image.fromarray(he), Image.fromarray(paint(he, zones))]))
        print(f"  {stem[:12]}: {title}, карта {cls.shape[1]}x{cls.shape[0]} при {mpp:.2f} мкм, "
              f"доля стромы {s['tsr']:.3f}, прочее {s['other']:.1%}")

    width = 2 * ps + gap
    canvas = Image.new("RGB", (width, head + sum(cap + ims[0].height + gap for _, ims in rows_out)),
                       "white")
    draw = ImageDraw.Draw(canvas)
    present = [code for code, found_it in
               ((TUMOR, any((cls == TUMOR).any() for _, cls, _, _ in chosen)),
                (5, any(np.isin(cls, STROMA_CODES).any() for _, cls, _, _ in chosen)),
                (OTHER, any((cls == OTHER).any() for _, cls, _, _ in chosen))) if found_it]
    for k, c in enumerate(present):
        x = 8 + k * 190
        draw.rectangle([x, 12, x + 22, 34], fill=ZONE_COL[c], outline=(0, 0, 0))
        draw.text((x + 30, 12), NAMES[c], fill=(0, 0, 0), font=font)
    top = head
    for title, ims in rows_out:
        draw.text((8, top + 4), title, fill=(60, 60, 60), font=small)
        top += cap
        for c, tile in enumerate(ims):
            canvas.paste(tile, (c * (ps + gap) + (ps - tile.width) // 2, top))
        top += ims[0].height + gap

    out = args.out or str(Path(args.dir) / "tcga_apply.png")
    canvas.save(out, optimize=True)
    print("сохранено:", out)


if __name__ == "__main__":
    main()
