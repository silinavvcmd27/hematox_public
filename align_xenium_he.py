import argparse
from pathlib import Path
import numpy as np
import pandas as pd

XENIUM_PX_UM = 0.2125

def load_matrix(path):
    M = np.loadtxt(path, delimiter=",")
    if M.shape == (3, 3):
        A, t = M[:2, :2], M[:2, 2]
    elif M.shape == (2, 3):
        A, t = M[:, :2], M[:, 2]
    else:
        raise ValueError("ожидал 2x3 или 3x3, получил " + str(M.shape))
    return A, t

def he_shape(path):
    # оси берём из tiff (Y=высота, X=ширина), не по величине размеров
    import tifffile
    with tifffile.TiffFile(path) as tf:
        s = tf.series[0]
        shp = s.shape
        axes = (s.axes or "")
    if "Y" in axes and "X" in axes:
        return shp[axes.index("Y")], shp[axes.index("X")]
    spatial = [d for d in shp if d > 8]
    return spatial[0], spatial[1]

def candidates(xy_um, A, t):
    Ainv = np.linalg.inv(A)
    px = xy_um / XENIUM_PX_UM
    return {
        "forward_px": (A @ px.T).T + t,
        "forward_um": (A @ xy_um.T).T + t,
        "inverse_px": (Ainv @ (px - t).T).T,
        "inverse_um": (Ainv @ (xy_um - t).T).T,
    }

def coverage(he, Dx, Dy):
    x, y = he[:, 0], he[:, 1]
    return float(((x >= 0) & (x < Dx) & (y >= 0) & (y < Dy)).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--align", required=True)
    ap.add_argument("--he", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.cells)
    xy = df[["x", "y"]].to_numpy(float)
    A, t = load_matrix(args.align)
    H, W = he_shape(args.he)
    print("H&E размер: W=%d H=%d" % (W, H))

    cands = candidates(xy, A, t)
    print("покрытие:")
    for nm, he in cands.items():
        for (Dx, Dy), tag in [((W, H), "WH"), ((H, W), "HW")]:
            print("  %-12s [%s]: %.3f" % (nm, tag, coverage(he, Dx, Dy)))

    he = cands["inverse_px"]
    covWH, covHW = coverage(he, W, H), coverage(he, H, W)
    tag = "WH" if covWH >= covHW else "HW"
    cov = max(covWH, covHW)
    name = "inverse_px"
    if cov < 0.85:
        best = None
        for nm, h in cands.items():
            for (Dx, Dy), tg in [((W, H), "WH"), ((H, W), "HW")]:
                c = coverage(h, Dx, Dy)
                if best is None or c > best[0]:
                    best = (c, nm, tg, h)
        cov, name, tag, he = best
        print("ВНИМАНИЕ: inverse_px <0.85, запасной вариант")
    print("выбрано: %s [%s], покрытие %.3f" % (name, tag, cov))

    if tag == "HW":
        he = he[:, ::-1]
    out = df.copy()
    out["x"] = he[:, 0]; out["y"] = he[:, 1]
    inside = (out["x"] >= 0) & (out["x"] < W) & (out["y"] >= 0) & (out["y"] < H)
    out = out[inside].reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print("сохранено: %s (%d клеток)" % (args.out, len(out)))
    print("x_max=%d (<=W=%d), y_max=%d (<=H=%d)" % (out["x"].max(), W, out["y"].max(), H))

if __name__ == "__main__":
    main()
