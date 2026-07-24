# Насколько патчевая модель промахивается по доле стромы на отложенном срезе.
#
# Пиксельную мы уже померили: на ovary2 она даёт 31% при истинных 51%.
# Здесь то же самое для патчевой ветки — работает на готовых эмбеддингах,
# UNI не запускается.
#
# python tsp_bias.py \
#   --pairs ovary_prime_he:outputs/models/ft5_val_prime.pth \
#           ovary2_he:outputs/models/ft5_val_ov2.pth \
#           ovary3_he:outputs/models/ft5_val_ov3.pth

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.utils import load_config, get_device
from src.build_graph import build_edge_index
from src.models.graph_head import build_head


def load_binary(npz, cfg_graph):
    # как в train_binary.py: класс 2 (undefined) выбрасывается,
    # остаются 0 — опухоль, 1 — строма
    from torch_geometric.data import Data
    d = np.load(npz, allow_pickle=True)
    X, y, coords = d["X"], d["y"], d["coords"].astype(float)
    keep = y != 2
    X, y, coords = X[keep], y[keep], coords[keep]
    edge = build_edge_index(coords, cfg_graph["mode"], cfg_graph["k"])
    return Data(x=torch.tensor(X, dtype=torch.float32),
                y=torch.tensor(y, dtype=torch.long), edge_index=edge)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="срез:модель, модель должна быть обучена без этого среза")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--emb-dir", default="data/processed/embeddings")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    print("device:", device)

    print(f"\n{'срез':16s} {'патчей':>8s} {'строма истина':>14s} {'предсказано':>12s} "
          f"{'ошибка':>10s} {'macro F1':>9s}")
    rows = []
    for pair in args.pairs:
        slide, model_path = pair.split(":", 1)
        g = load_binary(Path(args.emb_dir) / f"{slide}.npz", cfg["graph"])

        model = build_head(dict(cfg["train"]), cfg["encoder"]["embed_dim"], 2).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device)["state_dict"])
        model.eval()

        g = g.to(device)
        with torch.no_grad():
            pred = model(g.x, g.edge_index).argmax(1).cpu().numpy()
        y = g.y.cpu().numpy()

        true_share = float((y == 1).mean())
        pred_share = float((pred == 1).mean())
        f1 = f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)
        rows.append((slide, true_share, pred_share))
        print(f"{slide:16s} {len(y):8d} {true_share:13.3f} {pred_share:12.3f} "
              f"{100*(pred_share-true_share):+9.1f} {f1:9.3f}")

    err = [abs(p - t) for _, t, p in rows]
    print(f"\nсредний модуль ошибки патчевой модели: {100*np.mean(err):.1f} п.п.")
    print("для сравнения, пиксельная (два класса): 4.2, 19.5 и 4.6 п.п., в среднем 9.4")


if __name__ == "__main__":
    main()
