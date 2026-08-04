import re
from pathlib import Path


def ome_mpp(path):
    import tifffile
    with tifffile.TiffFile(path) as tf:
        xml = tf.ome_metadata or ""
    m = re.search(r'PhysicalSizeX="([0-9.]+)"', xml)
    if m:
        return float(m.group(1))
    return None


def svs_mpp(path):
    import openslide
    sl = openslide.OpenSlide(str(path))
    mpp = sl.properties.get("openslide.mpp-x")
    sl.close()
    return float(mpp) if mpp else None


def slide_mpp(path):
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".svs", ".ndpi", ".mrxs", ".scn"):
        return svs_mpp(path)
    return ome_mpp(path)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()
    mpp = slide_mpp(args.path)
    print(f"{args.path}: {mpp} мкм/px")


if __name__ == "__main__":
    main()