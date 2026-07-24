# Размер пикселя среза и размер патча под заданное физическое поле зрения.
#
# У ovary3 пиксель 0.137 мкм, у двух других 0.274 — вдвое. Патч в 256 пикселей
# покрывал на них 35 и 70 мкм соответственно, и модель смотрела на разное
# увеличение. Поэтому патч задаётся в микрометрах, а не в пикселях.
#
# python slide_mpp.py --he data/raw/ovary3/..._he_image.ome.tif --um 70
# python slide_mpp.py --he ... --um 70 --quiet   # печатает только размер в пикселях

import argparse
import re
from pathlib import Path


def ome_mpp(path):
    import tifffile
    with tifffile.TiffFile(path) as tf:
        xml = tf.ome_metadata or ""
    m = re.search(r'PhysicalSizeX="([\d.eE+-]+)"', xml)
    if not m:
        raise SystemExit(f"{path}: PhysicalSizeX в метаданных не найден, "
                         "задай масштаб вручную")
    return float(m.group(1))


def svs_mpp(path):
    import openslide
    sl = openslide.OpenSlide(str(path))
    v = sl.properties.get("openslide.mpp-x")
    sl.close()
    if not v:
        raise SystemExit(f"{path}: openslide.mpp-x не записан")
    return float(v)


def slide_mpp(path):
    ext = Path(path).suffix.lower()
    return svs_mpp(path) if ext in (".svs", ".ndpi", ".mrxs", ".scn") else ome_mpp(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--he", required=True)
    ap.add_argument("--um", type=float, default=70.0, help="желаемое поле зрения патча")
    ap.add_argument("--quiet", action="store_true", help="печатать только размер в пикселях")
    args = ap.parse_args()

    mpp = slide_mpp(args.he)
    px = int(round(args.um / mpp))
    if args.quiet:
        print(px)
    else:
        print(f"{Path(args.he).name}")
        print(f"  {mpp:.4f} мкм/px")
        print(f"  патч {args.um:.0f} мкм = {px} пикселей")


if __name__ == "__main__":
    main()
