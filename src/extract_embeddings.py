# H&E -> UNI-эмбеддинги. На каждый слайд сохраняем .npz (X, y, coords, patch_id).
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_config, ensure_dir, get_device, hf_login
from src.data.patching import load_image, iter_patches_from_labels
from src.models.uni_encoder import build_encoder


def process(slide, image_path, cfg, encoder, coord_scale, chunk=2000):
    proc = Path(cfg["paths"]["processed_dir"])
    labels_csv = proc / f"{slide}_patch_labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(f"нет {labels_csv} — сперва python -m src.data.seurat_labels")

    df = pd.read_csv(labels_csv)
    print(f"{slide}: грузим {image_path}")
    img = load_image(image_path)
    print(f"{slide}: image {img.shape}, патчей {len(df)}")

    size = cfg["patching"]["patch_size"]
    bs = cfg["encoder"]["batch_size"]
    pids, ys, feats, buf = [], [], [], []

    for pid, label_idx, patch in iter_patches_from_labels(img, df, size, coord_scale):
        pids.append(pid)
        ys.append(label_idx)
        buf.append(patch)
        if len(buf) >= chunk:
            feats.append(encoder.encode_patches(buf, bs))
            buf = []
            print(f"{slide}: закодировано {len(pids)}/{len(df)}", end="\r")
    if buf:
        feats.append(encoder.encode_patches(buf, bs))

    X = np.concatenate(feats, 0) if feats else np.zeros((0, encoder.embed_dim), np.float32)
    if len(X) != len(df):
        raise SystemExit(f"{slide}: закодировано {len(X)} патчей на {len(df)} строк разметки")

    coords = np.stack([
        (df["px"].to_numpy() * coord_scale).round().astype(int),
        (df["py"].to_numpy() * coord_scale).round().astype(int),
    ], axis=1)

    out = ensure_dir(proc / "embeddings") / f"{slide}.npz"
    np.savez_compressed(out, X=X.astype(np.float32),
                        y=np.array(ys, np.int64),
                        coords=coords, patch_id=np.array(pids))
    print(f"\n{slide}: saved {out}  X={X.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--slide")
    ap.add_argument("--image")
    ap.add_argument("--coord-scale", type=float, default=1.0,
                    help="множитель координат под уровень картинки (Visium: tissue_hires_scalef)")
    ap.add_argument("--image-map", help="CSV slide,image_path[,coord_scale] для пакетного прогона")
    ap.add_argument("--chunk", type=int, default=2000,
                    help="сколько патчей держать в памяти до отправки в энкодер")
    args = ap.parse_args()

    cfg = load_config(args.config)
    hf_login(cfg["encoder"]["hf_token_env"])
    device = get_device()
    print("device:", device)
    encoder = build_encoder(cfg, device)

    if args.image_map:
        for r in pd.read_csv(args.image_map).itertuples(index=False):
            scale = getattr(r, "coord_scale", 1.0)
            scale = 1.0 if scale is None or not np.isfinite(scale) else float(scale)
            process(r.slide, r.image_path, cfg, encoder, scale, args.chunk)
    else:
        if not (args.slide and args.image):
            raise SystemExit("нужно --slide и --image, либо --image-map")
        process(args.slide, args.image, cfg, encoder, args.coord_scale, args.chunk)


if __name__ == "__main__":
    main()