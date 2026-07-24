# сегментация целого слайда: нарезаем -> UNI -> граф -> голова -> оверлей
# Патчевая ветка. Для пиксельной карты есть seg_infer_img.py и seg_infer_svs.py.
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.utils import (load_config, get_device, ensure_dir, hf_login,
                       CLASS_NAMES, CLASS_COLORS)
from src.data.patching import load_image, grid_patches
from src.models.uni_encoder import build_encoder
from src.models.graph_head import build_head
from src.build_graph import build_edge_index

WSI_EXT = {".svs", ".ndpi", ".mrxs", ".scn"}


def predict(image_path, model_path, cfg, stride, device):
    if Path(image_path).suffix.lower() in WSI_EXT:
        raise SystemExit(
            f"{image_path}: load_image берёт для WSI самый мелкий уровень пирамиды, "
            "патчи получатся не в том масштабе. Для .svs используй seg_infer_svs.py")

    ckpt = torch.load(model_path, map_location=device)
    saved_classes = list(ckpt.get("classes", CLASS_NAMES))
    if saved_classes != CLASS_NAMES:
        raise SystemExit(f"модель обучена на классах {saved_classes}, "
                         f"а здесь ожидаются {CLASS_NAMES}")

    img = load_image(image_path)
    size = cfg["patching"]["patch_size"]
    print(f"image {img.shape}, patch {size}, stride {stride}")

    cxs, cys, arrays = [], [], []
    for _, _, cx, cy, patch in grid_patches(img, size, stride):
        cxs.append(cx)
        cys.append(cy)
        arrays.append(patch)
    if not arrays:
        raise SystemExit("ни одного тканевого патча (всё ушло в фон)")
    print("тканевых патчей:", len(arrays))

    encoder = build_encoder(cfg, device)
    X = encoder.encode_patches(arrays, cfg["encoder"]["batch_size"])
    coords = np.stack([cxs, cys], 1).astype(float)
    edge_index = build_edge_index(coords, cfg["graph"]["mode"], cfg["graph"]["k"])

    model = build_head(dict(cfg["train"]), cfg["encoder"]["embed_dim"],
                       len(CLASS_NAMES)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device),
                       edge_index.to(device))
        preds = logits.argmax(1).cpu().numpy()
        probs = torch.softmax(logits, 1).cpu().numpy()
    return img, np.array(cxs), np.array(cys), preds, probs, size


def overlay(img, cxs, cys, preds, size, alpha=0.45):
    H, W = img.shape[:2]
    half = size // 2
    votes = np.zeros((H, W, len(CLASS_NAMES)), np.uint8)
    for cx, cy, pr in zip(cxs, cys, preds):
        x0, y0 = max(0, cx - half), max(0, cy - half)
        x1, y1 = min(W, cx + half), min(H, cy + half)
        votes[y0:y1, x0:x1, pr] += 1

    covered = votes.max(-1) > 0
    winner = votes.argmax(-1)

    out = img.copy()
    for i, cname in enumerate(CLASS_NAMES):
        sel = covered & (winner == i)
        if sel.any():
            col = np.array(CLASS_COLORS[cname], np.float32)
            out[sel] = (img[sel] * (1 - alpha) + col * alpha).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--output", required=True)
    ap.add_argument("--alpha", type=float, default=0.45)
    args = ap.parse_args()

    cfg = load_config(args.config)
    hf_login(cfg["encoder"]["hf_token_env"])
    device = get_device()

    img, cxs, cys, preds, probs, size = predict(args.image, args.model, cfg,
                                                args.stride, device)
    out = overlay(img, cxs, cys, preds, size, args.alpha)
    ensure_dir(Path(args.output).parent)
    Image.fromarray(out).save(args.output)
    print("saved", args.output)

    uniq, cnt = np.unique(preds, return_counts=True)
    tot = cnt.sum()
    for u, c in zip(uniq, cnt):
        print(f"  {CLASS_NAMES[u]:10s} {c/tot*100:5.1f}%  ({c})")

    # вероятности, чтобы пересчитать TSP с другим порогом
    npz = Path(args.output).with_suffix(".npz")
    np.savez_compressed(npz, cx=cxs, cy=cys, preds=preds, probs=probs,
                        classes=np.array(CLASS_NAMES))
    print("saved", npz)


if __name__ == "__main__":
    main()