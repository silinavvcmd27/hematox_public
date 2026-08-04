from pathlib import Path

import pandas as pd

from src.utils import CLASS_NAMES, CLASS_TO_IDX, TRAIN_CLASSES, load_yaml

LABEL_NAMES = [v for k, v in sorted(CLASS_NAMES.items())
               if v not in ("background", "ignore")]


class CellTypeMapper:
    def __init__(self, map_path="config/cell_type_map.yaml"):
        cfg = load_yaml(map_path)
        self.match_mode = cfg.get("match_mode", "exact")
        self.ci = cfg.get("case_insensitive", True)
        self.unmapped_policy = cfg.get("unmapped_policy", "undefined")

        raw_groups = cfg.get("groups", {})
        self.groups = {}
        for cls in LABEL_NAMES:
            self.groups[cls] = list(raw_groups.get(cls, []) or [])

        self.subtype_lookup = {}
        sg = cfg.get("subtype_groups", {})
        if sg:
            for cls, names in sg.items():
                for name in (names or []):
                    self.subtype_lookup[self._norm(name)] = cls

        self.lookup = {}
        for cls, names in self.groups.items():
            for name in names:
                self.lookup[self._norm(name)] = cls

        self.unmapped = set()

    def _norm(self, s):
        s = str(s).strip()
        return s.lower() if self.ci else s

    def map_one(self, cell_type, cell_subtype=None):
        if cell_subtype is not None and str(cell_subtype).strip().lower() != "nan":
            skey = self._norm(cell_subtype)
            cls = self.subtype_lookup.get(skey)
            if cls is not None:
                return cls

        key = self._norm(cell_type)
        if self.match_mode == "exact":
            cls = self.lookup.get(key)
        else:
            cls = None
            for c, names in self.groups.items():
                if any(self._norm(n) in key for n in names):
                    cls = c
                    break

        if cls is None:
            self.unmapped.add(str(cell_type))
            return None if self.unmapped_policy == "error" else "undefined"
        return cls

    def map_series(self, types, subtypes=None):
        before = set(self.unmapped)
        if subtypes is not None:
            out = pd.Series([
                self.map_one(t, s)
                for t, s in zip(types.astype(str), subtypes.astype(str))
            ], index=types.index)
        else:
            out = types.astype(str).map(self.map_one)

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
    grid = pd.DataFrame({
        "gx": (cells["x"].to_numpy() // patch_size).astype(int),
        "gy": (cells["y"].to_numpy() // patch_size).astype(int),
        "class": cells["class"].to_numpy(),
    })
    counts = (grid.groupby(["gx", "gy"])["class"]
                  .value_counts().unstack(fill_value=0)
                  .reindex(columns=LABEL_NAMES, fill_value=0))
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

    result = {
        "patch_id": [f"{slide_id}_{a}_{b}" for a, b in zip(gx, gy)],
        "slide": slide_id,
        "gx": gx,
        "gy": gy,
        "px": gx * patch_size + patch_size // 2,
        "py": gy * patch_size + patch_size // 2,
        "label": label,
        "label_idx": [CLASS_TO_IDX.get(c, 0) for c in label],
        "n_cells": n.to_numpy().astype(int),
        "purity": purity.round(4).to_numpy(),
    }
    for c in LABEL_NAMES:
        col = fr[c] if c in fr.columns else pd.Series(0.0, index=fr.index)
        result[f"frac_{c}"] = col.round(4).to_numpy()

    return pd.DataFrame(result)


def process_slide(cells_csv, mapper, out_dir, patch_size,
                  majority_threshold, min_cells):
    cells_csv = Path(cells_csv)
    slide_id = cells_csv.stem.removesuffix("_cells")
    df = pd.read_csv(cells_csv)
    if "cell_type" not in df.columns:
        raise ValueError(f"{cells_csv}: нет колонки cell_type")

    sub = df["cell_subtype"] if "cell_subtype" in df.columns else None
    df["class"] = mapper.map_series(df["cell_type"], sub)
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