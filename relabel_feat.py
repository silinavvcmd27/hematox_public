import argparse
import numpy as np
import cv2
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--seg-dir", default="data/processed/seg")
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--max-patches", type=int, default=None)
    args = ap.parse_args()

    seg = Path(args.seg_dir)
    feat_path = seg / f"{args.slide}_feat.npz"
    mask_path = seg / f"{args.slide}_mask.npz"
    patches_path = seg / f"{args.slide}_patches.csv"

    old = np.load(feat_path)
    X, y_old = old["X"], old["y"]
    n_patches = X.shape[0]
    print(f"старый feat: X={X.shape}, y={y_old.shape}, классы={np.unique(y_old)}")

    md = np.load(mask_path)
    mask, ds = md["mask"], int(md["downscale"])
    print(f"новая маска: {mask.shape}, классы={np.unique(mask)}, downscale={ds}")

    import pandas as pd
    man = pd.read_csv(patches_path)
    print(f"патчей в CSV: {len(man)}")

    if args.max_patches and len(man) > args.max_patches:
        man = man.sample(args.max_patches, random_state=42).reset_index(drop=True)

    ps = args.patch_size
    out_size = 112
    Hm, Wm = mask.shape

    y_new_list = []
    x_list = []
    skipped = 0
    for r in man.itertuples(index=False):
        x0, y0 = int(r.x0), int(r.y0)
        my0, mx0 = y0 // ds, x0 // ds
        my1, mx1 = (y0 + ps) // ds, (x0 + ps) // ds
        if my1 > Hm or mx1 > Wm:
            skipped += 1
            continue
        m = mask[my0:my1, mx0:mx1]
        m = cv2.resize(m, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        y_new_list.append(m)
        if len(y_new_list) >= n_patches:
            break

    print(f"  собрано {len(y_new_list)} патчей (пропущено {skipped})")

    if len(y_new_list) < n_patches:
        print(f"  ВНИМАНИЕ: меньше патчей ({len(y_new_list)}) чем в X ({n_patches})")
        X = X[:len(y_new_list)]

    y_new = np.stack(y_new_list).astype(np.uint8)
    print(f"новые классы: {dict(zip(*np.unique(y_new, return_counts=True)))}")

    bak = feat_path.with_suffix(".npz.bak")
    if not bak.exists():
        import shutil
        shutil.copy2(feat_path, bak)
        print(f"бэкап: {bak}")

    np.savez_compressed(feat_path, X=X, y=y_new)
    print(f"сохранено: {feat_path}  X={X.shape} y={y_new.shape}")


if __name__ == "__main__":
    main()