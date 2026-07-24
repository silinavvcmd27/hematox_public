# Трёхклассовая патчевая голова поверх замороженных UNI-эмбеддингов:
# tumor / stroma / undefined. Старая базовая линия — именно её результаты

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from src.utils import load_config, set_seed, get_device, ensure_dir, CLASS_NAMES
from src.build_graph import slide_to_data
from src.models.graph_head import build_head, GraphSAGEHead


def list_npz(cfg):
    emb = Path(cfg["paths"]["processed_dir"]) / "embeddings"
    files = sorted(emb.glob("*.npz"))
    if not files:
        raise SystemExit(f"нет эмбеддингов в {emb} — сперва python -m src.extract_embeddings")
    return files


def split_slides(files, val_fraction, seed):
    files = list(files)
    if len(files) < 2:
        raise SystemExit("нужно минимум два слайда, иначе обучение и валидация "
                         "пойдут по одному и тому же — задай --val явно")
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    n_val = max(1, round(len(files) * val_fraction))
    return files[n_val:], files[:n_val]


def class_weights(graphs, mode, device):
    if mode == "none":
        return None
    if isinstance(mode, (list, tuple)):
        return torch.tensor(mode, dtype=torch.float32, device=device)

    y = torch.cat([g.y for g in graphs]).numpy()
    present = np.unique(y)
    w = np.ones(len(CLASS_NAMES), np.float32)
    for c, wc in zip(present, compute_class_weight("balanced", classes=present, y=y)):
        w[c] = wc
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
    f1 = f1_score(y, p, average="macro", labels=range(len(CLASS_NAMES)), zero_division=0)
    return f1, y, p


def in_dim_of(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            return m.in_features
    raise RuntimeError("в голове нет линейных слоёв, откуда брать in_dim непонятно")


def train(model, train_graphs, val_graphs, cfg_train, device, save_path, select):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg_train["lr"], weight_decay=cfg_train["weight_decay"])
    crit = nn.CrossEntropyLoss(
        weight=class_weights(train_graphs, cfg_train["class_weights"], device))

    best_f1, best_state, bad = -1.0, None, 0
    for ep in range(1, cfg_train["epochs"] + 1):
        model.train()
        running = 0.0
        for g in train_graphs:
            g = g.to(device)
            opt.zero_grad()
            loss = crit(model(g.x, g.edge_index), g.y)
            loss.backward()
            opt.step()
            running += float(loss.detach())
        val_f1, _, _ = evaluate(model, val_graphs, device)
        print(f"epoch {ep:3d}  loss {running/len(train_graphs):.4f}  val_f1 {val_f1:.4f}")

        if select != "val":
            continue
        if val_f1 > best_f1:
            best_f1, bad = val_f1, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg_train["early_stop_patience"]:
                print(f"ранняя остановка на эпохе {ep}, лучший f1 {best_f1:.4f}")
                break

    if best_state:
        model.load_state_dict(best_state)
    ensure_dir(Path(save_path).parent)
    torch.save({"state_dict": model.state_dict(),
                "classes": CLASS_NAMES,
                "in_dim": in_dim_of(model)}, save_path)
    final_f1, _, _ = evaluate(model, val_graphs, device)
    print(f"saved {save_path}  (f1 {final_f1:.4f})")
    return final_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--val", nargs="*",
                    help="имена слайдов в валидацию; без них делим случайно по val_fraction")
    ap.add_argument("--select", choices=["val", "fixed"], default="fixed",
                    help="val — лучшая эпоха по валидации, fixed — фиксированный бюджет")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--checkpoint")
    ap.add_argument("--slides", nargs="*", help="ограничить набор слайдов (имена без .npz)")
    ap.add_argument("--out")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["project"]["seed"]
    set_seed(seed)
    device = get_device()
    print("device:", device, "| seed:", seed, "| отбор эпохи:", args.select)

    files = list_npz(cfg)
    if args.slides:
        files = [f for f in files if f.stem in set(args.slides)]
        if not files:
            raise SystemExit(f"слайды {args.slides} не найдены")

    if args.val:
        want = set(args.val)
        missing = want - {f.stem for f in files}
        if missing:
            raise SystemExit(f"нет эмбеддингов для {sorted(missing)}")
        val_files = [f for f in files if f.stem in want]
        train_files = [f for f in files if f.stem not in want]
        if not train_files:
            raise SystemExit("в валидацию ушли все слайды, учить не на чем")
    else:
        print("внимание: слайды делятся случайно, ткани перемешаются "
              "(ovary и STHELAR в одной куче). Для контролируемой оценки задай --val")
        train_files, val_files = split_slides(files, cfg["train"]["val_fraction"], seed)

    print("train:", [f.stem for f in train_files], "| val:", [f.stem for f in val_files])
    train_graphs = [slide_to_data(f, cfg["graph"]) for f in train_files]
    val_graphs = [slide_to_data(f, cfg["graph"]) for f in val_files]

    cfg_train = dict(cfg["train"])
    model = build_head(cfg_train, cfg["encoder"]["embed_dim"], len(CLASS_NAMES)).to(device)

    if args.finetune:
        if not args.checkpoint:
            raise SystemExit("--finetune без --checkpoint")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device)["state_dict"])
        print("loaded base:", args.checkpoint)
        if cfg["finetune"]["freeze_graph_layer1"] and isinstance(model, GraphSAGEHead):
            model.freeze_first_layer()
            print("froze proj + first SAGE layer")
        cfg_train["lr"] = cfg["finetune"]["lr"]
        cfg_train["epochs"] = cfg["finetune"]["epochs"]

    out = args.out or str(Path(cfg["paths"]["models_dir"]) /
                          ("finetuned.pth" if args.finetune else "base_model.pth"))
    train(model, train_graphs, val_graphs, cfg_train, device, out, args.select)

    _, y, p = evaluate(model, val_graphs, device)
    print("\n", classification_report(y, p, target_names=CLASS_NAMES, zero_division=0))


if __name__ == "__main__":
    main()