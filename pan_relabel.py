# Новая разметка IGNITE и BCSS: опухоль, строма и прочая ткань (код 7), макрофаги
# в игнор. Признаки UNI не пересчитываются: положение каждого патча записано в
# <набор>_patches.csv, маска вырезается заново из исходной по новой таблице
# классов и сохраняется вместе с теми же X в <набор>3_feat.npz.
#
# Прочее это ткань, которую патолог исключает из TSR: некроз, кровь, слизь, жир и
# нормальная паренхима. Макрофаги IGNITE больше не строма: в лёгком это пласты
# крупных светлых клеток в альвеолах, и deploy, выучив их как строму, стал звать
# стромой середину гнёзд светлоклеточной опухоли. Мышца, ороговение, реактивный
# эпителий, DCIS и прочие спорные классы в игнор. Опухоль размечается как раньше,
# и совпадение с прежней маской служит проверкой, что патчи вырезаны там же.
#
#   python -u pan_relabel.py --sets ignite_train ignite_test bcss

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.utils import TUMOR, STROMA
from seg_extract import OUT
from pan_prep_ignite import index_pngs, read_codes
from pan_prep_bcss import LABELS as BCSS_LABELS
from pan_train import OTHER

Image.MAX_IMAGE_PIXELS = None
IGNITE = {TUMOR: ("Tumor epithelium",),
          STROMA: ("Stroma", "Inflammation"),
          OTHER: ("Necrotic tissue", "Erythrocytes", "Mucus/Plasma/Fluids", "Fatty tissue",
                  "Alveolar tissue", "Bronchial epithelium", "Liver", "Cartilage/Bone")}
BCSS = {TUMOR: ("tumor",),
        STROMA: ("stroma", "lymphocytic_infiltrate", "plasma_cells", "other_immune_infiltrate"),
        OTHER: ("necrosis_or_debris", "glandular_secretions", "blood", "fat", "mucoid_material",
                "normal_acinus_or_duct")}


def make_lut(code_of, groups):
    lut = np.zeros(256, np.uint8)
    for code, members in groups.items():
        lut[[code_of[m] for m in members]] = code
    return lut


def ignite_reader(root, target_mpp):
    root = Path(root)
    label_map = json.loads((root / "he_label_map.json").read_text())
    lut = make_lut(label_map, IGNITE)
    files = index_pngs(root)
    factor = 0.5 / target_mpp

    def read(roi):
        w, h = Image.open(files[("images", roi)]).size
        size = (round(w * factor), round(h * factor))
        return lut[read_codes(files[("annotations", roi)], size, max(label_map.values()) + 1)]
    return read


def bcss_reader(root, target_mpp):
    root = Path(root)
    lut = make_lut({name: code for code, name in enumerate(BCSS_LABELS)}, BCSS)
    factor = 0.25 / target_mpp

    def read(roi):
        w, h = Image.open(root / "rgbs_colorNormalized" / f"{roi}.png").size
        size = (round(w * factor), round(h * factor))
        return lut[np.asarray(Image.open(root / "masks" / f"{roi}.png").resize(size, Image.NEAREST))]
    return read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["ignite_train", "ignite_test", "bcss"])
    ap.add_argument("--dir", default="data/processed/seg_pan")
    ap.add_argument("--ignite-root", default="data/pan/ignite")
    ap.add_argument("--bcss-root", default="data/pan/bcss")
    args = ap.parse_args()

    out = Path(args.dir)
    for name in args.sets:
        feat = np.load(out / f"{name}_feat.npz")
        meta = pd.read_csv(out / f"{name}_patches.csv")
        X, old = feat["X"], feat["y"]
        if len(meta) != len(X):
            raise SystemExit(f"{name}: патчей {len(X)}, а строк в patches.csv {len(meta)}")
        ps = int(feat["patch_px"])
        target_mpp = float(feat["patch_um"]) / ps
        read = (bcss_reader(args.bcss_root, target_mpp) if name.startswith("bcss")
                else ignite_reader(args.ignite_root, target_mpp))

        y = np.zeros_like(old)
        groups = meta.groupby("roi").indices
        for k, (roi, rows) in enumerate(groups.items(), 1):
            lab = read(roi)
            for i in rows:
                x0, y0 = meta.x0.iat[i], meta.y0.iat[i]
                m = np.ascontiguousarray(lab[y0:y0 + ps, x0:x0 + ps])
                y[i] = np.asarray(Image.fromarray(m).resize((OUT, OUT), Image.NEAREST))
            print(f"  {name}: областей {k}/{len(groups)}", end="\r")

        agree = ((y == TUMOR) == (old == TUMOR)).mean()
        old_s, new_s, new_o = (old == STROMA).sum(), (y == STROMA).sum(), (y == OTHER).sum()
        print(f"\n{name}: опухоль совпадает с прежней маской на {agree:.2%}, строма "
              f"{new_s / max(old_s, 1):.1%} от прежней, прочее {new_o / y.size:.1%} пикселей")
        if agree < 0.999:
            raise SystemExit("опухоль не совпала с прежней маской: патчи вырезаны не там, "
                             "новый файл не записан")
        np.savez(out / f"{name}3_feat.npz", X=X, y=y, patch_px=ps, patch_um=feat["patch_um"],
                 stain_ref=feat["stain_ref"])
        meta.to_csv(out / f"{name}3_patches.csv", index=False)
        print(f"сохранено: {out / f'{name}3_feat.npz'}")


if __name__ == "__main__":
    main()
