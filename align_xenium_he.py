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

ORIENTS = ("xy", "xy-fx", "xy-fy", "xy-fxfy", "yx", "yx-fx", "yx-fy", "yx-fxfy")


def orient(pts, tag, W, H):
    """Разворот облака точек внутри кадра H&E: перестановка осей и отражения.
    Матрица от прибора задаёт масштаб и сдвиг, но не ловит поворот картинки на
    90 градусов и зеркало, а у Prime-срезов H&E лежит именно так. Подобрать tag
    можно check_alignment.py, по доле клеток, попавших на ткань."""
    x, y = (pts[:, 1], pts[:, 0]) if tag.startswith("yx") else (pts[:, 0], pts[:, 1])
    if "fx" in tag:
        x = W - x
    if "fy" in tag:
        y = H - y
    return np.stack([x, y], 1)


def coverage(he, Dx, Dy):
    x, y = he[:, 0], he[:, 1]
    return float(((x >= 0) & (x < Dx) & (y >= 0) & (y < Dy)).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--align", required=True)
    ap.add_argument("--he", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=["forward_px", "forward_um", "inverse_px", "inverse_um"],
                    help="какой вариант матрицы брать, по умолчанию подбирается по покрытию")
    ap.add_argument("--orient", choices=ORIENTS,
                    help="разворот кадра, подобранный check_alignment.py")
    ap.add_argument("--shift-x", type=float, default=0.0,
                    help="поправка по x в пикселях H&E, из check_alignment.py")
    ap.add_argument("--shift-y", type=float, default=0.0,
                    help="поправка по y в пикселях H&E")
    args = ap.parse_args()

    df = pd.read_csv(args.cells)
    xy = df[["x", "y"]].to_numpy(float)
    A, t = load_matrix(args.align)
    H, W = he_shape(args.he)
    print("H&E размер: W=%d H=%d" % (W, H))

    cands = candidates(xy, A, t)

    if args.variant and args.orient:
        name, tag = args.variant, args.orient
        he = orient(cands[name], tag, W, H)
        print("задано вручную: %s [%s], покрытие %.3f" % (name, tag, coverage(he, W, H)))
    else:
        print("покрытие:")
        for nm, h in cands.items():
            for tg in ("xy", "yx"):
                print("  %-12s [%s]: %.3f" % (nm, tg, coverage(orient(h, tg, W, H), W, H)))

        name = "inverse_px"
        covs = {tg: coverage(orient(cands[name], tg, W, H), W, H) for tg in ("xy", "yx")}
        tag, cov = max(covs.items(), key=lambda kv: kv[1])
        if cov < 0.85:
            print("ВНИМАНИЕ: inverse_px <0.85, запасной вариант")
            cov, name, tag = max(
                (coverage(orient(h, tg, W, H), W, H), nm, tg)
                for nm, h in cands.items() for tg in ("xy", "yx"))
        he = orient(cands[name], tag, W, H)
        print("выбрано: %s [%s], покрытие %.3f" % (name, tag, cov))
        print("покрытие не различает зеркала, разворот проверьте check_alignment.py")

    if args.shift_x or args.shift_y:
        he = he + np.array([args.shift_x, args.shift_y])
        print("поправка: dx=%.0f dy=%.0f" % (args.shift_x, args.shift_y))

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