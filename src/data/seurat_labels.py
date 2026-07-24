# Сводим подробные cell_type (из Seurat) к трём классам и размечаем патчи.
# Вход:  data/seurat_csv/<slide>_cells.csv  (cell_id, x, y, cell_type)
# Выход: data/processed/<slide>_patch_labels.csv
from pathlib import Path

import pandas as pd

from src.utils import CLASS_NAMES, CLASS_TO_IDX, load_yaml


class CellTypeMapper:
    """cell_type -> {tumor, stroma, undefined} по правилам из yaml."""

    def __init__(self, map_path="config/cell_type_map.yaml"):
        cfg = load_yaml(map_path)
        self.match_mode = cfg.get("match_mode", "exact")
        self.ci = cfg.get("case_insensitive", True)
        self.unmapped_policy = cfg.get("unmapped_policy", "undefined")
        self.groups = {g: list(cfg["groups"].get(g, []) or []) for g in CLASS_NAMES}
        # для exact-режима — плоский словарь name->class
        self.lookup = {}
        for cls, names in self.groups.items():
            for name in names:
                self.lookup[self._norm(name)] = cls
        self.unmapped = set()

    def _norm(self, s):
        s = str(s).strip()
        return s.lower() if self.ci else s

    def map_one(self, cell_type):
        key = self._norm(cell_type)
        if self.match_mode == "exact":
            cls = self.lookup.get(key)
        else:
            # подстрочный режим зависит от порядка групп в yaml: побеждает первая
            # подошедшая, поэтому «Cancer cells» надо держать выше общих слов
            cls = None
            for c, names in self.groups.items():
                if any(self._norm(n) in key for n in names):
                    cls = c
                    break
        if cls is None:
            self.unmapped.add(str(cell_type))
            # при policy=error вернём None и упадём позже (так видно сразу все)
            return None if self.unmapped_policy == "error" else "undefined"
        return cls

    def map_series(self, s):
        before = set(self.unmapped)
        out = s.astype(str).map(self.map_one)
        if self.unmapped_policy == "error" and out.isna().any():
            raise ValueError(
                "Этих типов нет в cell_type_map.yaml:\n  "
                + "\n  ".join(sorted(self.unmapped))
                + "\nДобавь их или поставь unmapped_policy: undefined"
            )
        new = self.unmapped - before
        if new:
            print(f"  {len(new)} новых типов ушли в undefined: {sorted(new)[:10]}")
        return out


def assign_patch_labels(cells, slide_id, patch_size,
                        majority_threshold=0.5, min_cells=3):
    # бьём клетки на сетку patch_size x patch_size, метка патча = мажоритарный класс
    grid = pd.DataFrame({
        "gx": (cells["x"].to_numpy() // patch_size).astype(int),
        "gy": (cells["y"].to_numpy() // patch_size).astype(int),
        "class": cells["class"].to_numpy(),
    })
    counts = (grid.groupby(["gx", "gy"])["class"]
                  .value_counts().unstack(fill_value=0)
                  .reindex(columns=CLASS_NAMES, fill_value=0))
    # порядок строк — как ячейки впервые встретились в файле; groupby сортирует
    # индекс, а нам нужна совместимость с уже посчитанными эмбеддингами
    counts = counts.reindex(pd.MultiIndex.from_frame(grid[["gx", "gy"]].drop_duplicates()))

    n = counts.sum(axis=1)
    keep = n >= min_cells
    counts, n = counts[keep], n[keep]
    if counts.empty:
        return pd.DataFrame()

    fr = counts.div(n, axis=0)
    top, purity = fr.idxmax(axis=1), fr.max(axis=1)
    label = top.where(purity >= majority_threshold, "undefined").to_numpy()
    gx = counts.index.get_level_values("gx")
    gy = counts.index.get_level_values("gy")

    return pd.DataFrame({
        "patch_id": [f"{slide_id}_{a}_{b}" for a, b in zip(gx, gy)],
        "slide": slide_id,
        "gx": gx,
        "gy": gy,
        "px": gx * patch_size + patch_size // 2,   # центр патча
        "py": gy * patch_size + patch_size // 2,
        "label": label,
        "label_idx": [CLASS_TO_IDX[c] for c in label],
        "n_cells": n.to_numpy().astype(int),
        "purity": purity.round(4).to_numpy(),
        "frac_tumor": fr["tumor"].round(4).to_numpy(),
        "frac_stroma": fr["stroma"].round(4).to_numpy(),
        "frac_undefined": fr["undefined"].round(4).to_numpy(),
    })


def process_slide(cells_csv, mapper, out_dir, patch_size,
                  majority_threshold, min_cells):
    cells_csv = Path(cells_csv)
    slide_id = cells_csv.stem.removesuffix("_cells")
    df = pd.read_csv(cells_csv)
    if "cell_type" not in df.columns:
        raise ValueError(f"{cells_csv}: нет колонки cell_type")

    df["class"] = mapper.map_series(df["cell_type"])
    patches = assign_patch_labels(df, slide_id, patch_size,
                                  majority_threshold, min_cells)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slide_id}_patch_labels.csv"
    patches.to_csv(out_path, index=False)
    print(f"{slide_id}: {len(patches)} патчей ->", patches["label"].value_counts().to_dict())
    return patches


def main():
    import argparse
    from src.utils import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--map", default="config/cell_type_map.yaml")
    ap.add_argument("--cells", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config)
    mapper = CellTypeMapper(args.map)
    seurat_dir = Path(cfg["paths"]["seurat_export_dir"])
    files = [Path(f) for f in args.cells] if args.cells else sorted(seurat_dir.glob("*_cells.csv"))
    if not files:
        raise SystemExit(f"нет *_cells.csv в {seurat_dir} — сперва R/export_seurat.R")

    parts = [process_slide(f, mapper,
                           cfg["paths"]["processed_dir"],
                           cfg["patching"]["patch_size"],
                           cfg["patching"]["majority_threshold"],
                           cfg["patching"]["min_cells_per_patch"])
             for f in files]

    allp = pd.concat(parts, ignore_index=True)
    out = Path(cfg["paths"]["processed_dir"]) / "all_patch_labels.csv"
    allp.to_csv(out, index=False)
    print("\nвсего:", len(allp), allp["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()