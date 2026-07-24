# Бинарный вариант: только tumor vs stroma (undefined-патчи выбрасываем).
# Работает на тех же .npz (эмбеддинги не пересчитываем) - меняем лишь метки и число классов.
# --val задаёт отложенный срез (для 3-fold); --all учит на ВСЕХ срезах без валидации (деплой).
#
# база:     python train_binary.py --slides sthelar_... --out outputs/models/base_bin.pth
# fold:     python train_binary.py --finetune --checkpoint base_bin.pth --slides ... --val ovary3_he --out ...
# деплой:   python train_binary.py --finetune --checkpoint base_bin.pth --slides ovary_prime_he ovary2_he ovary3_he --all --out outputs/models/ovary_all_bin.pth

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from src.utils import load_config, set_seed, get_device, ensure_dir
from src.build_graph import build_edge_index
from src.models.graph_head import build_head, GraphSAGEHead

NAMES = ["tumor", "stroma"]
COLORS = np.array([[220, 50, 47], [38, 139, 210]]) / 255.0

HE_DEFAULT = {
    "ovary_prime_he": "data/raw/ovary_prime/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_he_image.ome.tif",
    "ovary2_he": "data/raw/ovary2/Xenium_Prime_Human_Ovary_FF_he_image.ome.tif",
    "ovary3_he": "data/raw/ovary3/Xenium_V1_Human_Ovary_Cancer_FF_he_image.ome.tif",
}


def load_binary(npz, cfg_graph):
    from torch_geometric.data import Data
    d = np.load(npz, allow_pickle=True)
    X, y, coords = d["X"], d["y"], d["coords"].astype(float)
    keep = y != 2
    X, y, coords = X[keep], y[keep], coords[keep]
    edge = build_edge_index(coords, cfg_graph["mode"], cfg_graph["k"])
    data = Data(x=torch.tensor(X, dtype=torch.float32),
                y=torch.tensor(y, dtype=torch.long), edge_index=edge)
    data.coords = coords
    data.name = Path(npz).stem
    return data


def class_weights(graphs, device):
    y = torch.cat([g.y for g in graphs]).numpy()
    present = np.unique(y)
    w = np.ones(2, np.float32)
    for c, v in zip(present, compute_class_weight("balanced", classes=present, y=y)):
        w[c] = v
    return torch.tensor(w, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model, graphs, device):
    model.eval()
    ys, ps = [], []
    for g in graphs:
        g = g.to(device)
        ps.append(model(g.x, g.edge_index).argmax(1).cpu().numpy())
        ys.append(g.y.cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return f1_score(y, p, average="macro", labels=[0, 1], zero_division=0), y, p


def split_slides(files, val_fraction, seed):
    rng = np.random.default_rng(seed)
    files = list(files); rng.shuffle(files)
    n_val = max(1, round(len(files) * val_fraction)) if len(files) > 1 else 0
    return (files[n_val:] or files), files[:n_val]


def preview(model, val_graph, he_path, out_png, device):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = val_graph.to(device)
    with torch.no_grad():
        pred = model(g.x, g.edge_index).argmax(1).cpu().numpy()
    y = val_graph.y.numpy()
    coords = val_graph.coords
    bg, sx, sy = None, 1.0, 1.0
    if he_path and Path(he_path).exists():
        try:
            import tifffile
            with tifffile.TiffFile(he_path) as tf:
                s = tf.series[0]; axes = s.axes or ""
                H0 = s.shape[axes.index("Y")] if "Y" in axes else s.shape[0]
                W0 = s.shape[axes.index("X")] if "X" in axes else s.shape[1]
                arr = list(s.levels)[-1].asarray()
            if arr.ndim == 3 and arr.shape[0] in (3, 4):
                arr = np.moveaxis(arr, 0, -1)
            arr = arr[..., :3]
            step = max(1, int(max(arr.shape[:2]) / 2500))
            bg = arr[::step, ::step].astype(np.uint8)
            sx, sy = bg.shape[1] / W0, bg.shape[0] / H0
        except Exception as e:
            print("фон H&E не загрузился:", e)
    fig, ax = plt.subplots(1, 2, figsize=(18, 9))
    for a, lab, ttl in [(ax[0], y, "Истина"), (ax[1], pred, "Предсказание")]:
        if bg is not None:
            a.imshow(bg); px, py = coords[:, 0] * sx, coords[:, 1] * sy
        else:
            px, py = coords[:, 0], coords[:, 1]; a.invert_yaxis()
        a.scatter(px, py, c=COLORS[lab], s=3, marker="s", linewidths=0)
        a.set_title(ttl); a.axis("off")
    h = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS[i],
                    markersize=10, label=NAMES[i]) for i in range(2)]
    ax[1].legend(handles=h, loc="lower right")
    acc = (pred == y).mean()
    fig.suptitle("%s (бинарно) | точность по патчам %.2f" % (val_graph.name, acc), fontsize=15)
    plt.tight_layout(); fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print("превью сохранено:", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--slides", nargs="+", required=True)
    ap.add_argument("--val", help="отложенный срез (для 3-fold)")
    ap.add_argument("--all", action="store_true", help="учить на всех срезах без валидации (деплой)")
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = get_device()
    print("device:", device)

    emb = Path(cfg["paths"]["processed_dir"]) / "embeddings"
    files = [emb / f"{s}.npz" for s in args.slides]
    for f in files:
        if not f.exists():
            raise SystemExit(f"нет {f}")

    if args.all:
        train_f, val_f = files, []
    elif args.val:
        val_f = [f for f in files if f.stem == args.val]
        train_f = [f for f in files if f.stem != args.val]
        if not val_f:
            raise SystemExit(f"--val {args.val} нет среди --slides")
    else:
        train_f, val_f = split_slides(files, cfg["train"]["val_fraction"], cfg["project"]["seed"])
    print("train:", [f.stem for f in train_f], "| val:", [f.stem for f in val_f])

    train_g = [load_binary(f, cfg["graph"]) for f in train_f]
    val_g = [load_binary(f, cfg["graph"]) for f in val_f]

    cfg_train = dict(cfg["train"])
    model = build_head(cfg_train, cfg["encoder"]["embed_dim"], 2).to(device)

    epochs, lr = cfg_train["epochs"], cfg_train["lr"]
    if args.finetune:
        ck = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ck["state_dict"])
        print("loaded base:", args.checkpoint)
        if isinstance(model, GraphSAGEHead) and cfg["finetune"]["freeze_graph_layer1"]:
            model.freeze_first_layer(); print("froze proj + first SAGE")
        epochs, lr = cfg["finetune"]["epochs"], cfg["finetune"]["lr"]

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=cfg_train["weight_decay"])
    crit = nn.CrossEntropyLoss(weight=class_weights(train_g, device))

    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        model.train(); run = 0.0
        for g in train_g:
            g = g.to(device); opt.zero_grad()
            loss = crit(model(g.x, g.edge_index), g.y)
            loss.backward(); opt.step(); run += float(loss.detach())
        if val_g:
            f1, _, _ = evaluate(model, val_g, device)
            print("epoch %3d  loss %.4f  val_f1 %.4f" % (ep, run / len(train_g), f1))
            if f1 > best:
                best = f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= cfg_train["early_stop_patience"]:
                    print("early stop @ %d, best f1 %.4f" % (ep, best)); break
        else:
            print("epoch %3d  loss %.4f  (без валидации)" % (ep, run / len(train_g)))
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    ensure_dir(Path(args.out).parent)
    torch.save({"state_dict": model.state_dict(), "classes": NAMES}, args.out)
    print("saved", args.out, ("" if val_g else " (деплой, на всех срезах)"))

    if val_g:
        _, y, p = evaluate(model, val_g, device)
        print("\n", classification_report(y, p, target_names=NAMES, zero_division=0))
        if args.preview:
            vg = val_g[0]
            out_png = str(ensure_dir(cfg["paths"]["results_dir"]) / (vg.name + "_binary_preview.png"))
            preview(model, vg, HE_DEFAULT.get(vg.name), out_png, device)


if __name__ == "__main__":
    main()
