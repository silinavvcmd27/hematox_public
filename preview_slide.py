# Быстрый предпросмотр: прогоняем готовые эмбеддинги среза через модель и рисуем
# карту зон (истина vs предсказание) поверх уменьшенного H&E. UNI заново не считаем.
#
# python preview_slide.py --slide ovary_prime_he \
#   --model outputs/models/finetuned.pth \
#   --he data/raw/ovary_prime/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_he_image.ome.tif

import argparse
from pathlib import Path

import numpy as np
import torch

from src.utils import load_config, CLASS_NAMES, CLASS_COLORS, ensure_dir
from src.build_graph import build_edge_index
from src.models.graph_head import build_head


def load_small_he(path, max_dim=2500):
    import tifffile
    with tifffile.TiffFile(path) as tf:
        s = tf.series[0]
        full = s.shape
        axes = s.axes or ""
        H0 = full[axes.index("Y")] if "Y" in axes else full[0]
        W0 = full[axes.index("X")] if "X" in axes else full[1]
        try:
            levels = list(s.levels)
            arr = levels[-1].asarray()
        except Exception:
            arr = s.asarray()
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    # при необходимости ещё проредим
    step = max(1, int(max(arr.shape[0], arr.shape[1]) / max_dim))
    arr = arr[::step, ::step]
    sx = arr.shape[1] / W0
    sy = arr.shape[0] / H0
    return arr.astype(np.uint8), sx, sy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--slide", default="ovary_prime_he")
    ap.add_argument("--model", default="outputs/models/finetuned.pth")
    ap.add_argument("--he", help="H&E ome.tif для фона (необязательно)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = np.load(Path(cfg["paths"]["processed_dir"]) / "embeddings" / f"{args.slide}.npz",
                allow_pickle=True)
    X = torch.tensor(d["X"], dtype=torch.float32)
    y = d["y"]
    coords = d["coords"].astype(float)

    edge = build_edge_index(coords, cfg["graph"]["mode"], cfg["graph"]["k"])
    model = build_head(dict(cfg["train"]), cfg["encoder"]["embed_dim"], len(CLASS_NAMES))
    ckpt = torch.load(args.model, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        pred = model(X, edge).argmax(1).numpy()

    acc = (pred == y).mean()
    print(f"срез {args.slide}: точность по патчам {acc:.3f}")
    for i, c in enumerate(CLASS_NAMES):
        m = y == i
        if m.sum():
            rec = (pred[m] == i).mean()
            print(f"  {c:10s}: патчей {m.sum():5d}, recall {rec:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = np.array([CLASS_COLORS[c] for c in CLASS_NAMES]) / 255.0
    bg, sx, sy = (None, 1.0, 1.0)
    if args.he:
        try:
            bg, sx, sy = load_small_he(args.he)
        except Exception as e:
            print("не удалось загрузить H&E для фона:", e)

    fig, ax = plt.subplots(1, 2, figsize=(18, 9))
    for a, lab, ttl in [(ax[0], y, "Истина (аннотация)"),
                        (ax[1], pred, "Предсказание модели")]:
        if bg is not None:
            a.imshow(bg)
            px, py = coords[:, 0] * sx, coords[:, 1] * sy
        else:
            px, py = coords[:, 0], coords[:, 1]
            a.invert_yaxis()
        a.scatter(px, py, c=cols[lab], s=3, marker="s", linewidths=0)
        a.set_title(ttl, fontsize=14)
        a.axis("off")

    handles = [plt.Line2D([0], [0], marker="s", color="w",
               markerfacecolor=cols[i], markersize=10, label=CLASS_NAMES[i])
               for i in range(len(CLASS_NAMES))]
    ax[1].legend(handles=handles, loc="lower right", fontsize=11)
    fig.suptitle(f"{args.slide}  |  точность по патчам {acc:.2f}", fontsize=15)
    plt.tight_layout()

    out = args.out or str(ensure_dir(cfg["paths"]["results_dir"]) / f"{args.slide}_preview.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("сохранено:", out)


if __name__ == "__main__":
    main()
