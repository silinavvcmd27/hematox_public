# UBC-OCEAN (рак яичника пяти гистотипов, много центров) в наш формат признаков,
# только для проверки. Маски патологов выложены для 152 срезов: красный это
# опухоль, зелёный строма (у авторов «здоровая ткань»), синий некроз, у нас он
# идёт в прочее. Маски не исчерпывающие, неразмеченное в игнор.
#
# Срезы лежат целыми PNG при 20x, номинально 0,5 мкм/px. Растягивать срез целиком
# до 0,2738 незачем: окно 140 px исходного среза берётся там, где размечено не
# меньше 10%, и увеличивается до 256, как в seg_infer_svs.py. Со среза не больше
# --max-per-roi случайных окон, окраска оценивается по отобранным окнам среза.
# PNG распаковывается только целиком и долго, поэтому срезы режутся параллельно.
# Часть срезов Kaggle отдаёт zip-архивом с расширением .png, они распаковываются в
# памяти. Недокачанные файлы пропускаются и печатаются списком в конце.
#
#   python -u pan_prep_ubc.py --root data/pan/ubc --name ubc

import argparse
import io
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.utils import get_device, ensure_dir, TUMOR, STROMA
from src.stain import Normalizer, StainStats, fit_stats, sample_pixels
from seg_extract import build_uni, encode, OUT
from pan_train import OTHER

Image.MAX_IMAGE_PIXELS = None
PNG_SIG, ZIP_SIG = b"\x89PNG\r\n\x1a\n", b"PK\x03\x04"
# на границах областей при сжатии бывают смешанные цвета, они уходят в игнор
COLORS = {TUMOR: (True, False, False), STROMA: (False, True, False), OTHER: (False, False, True)}


def decode(rgb):
    hi = rgb > 127
    lab = np.zeros(rgb.shape[:2], np.uint8)
    for code, pattern in COLORS.items():
        lab[(hi == pattern).all(-1)] = code
    return lab


def head(path):
    with open(path, "rb") as f:
        return f.read(8)


def open_slide(path):
    """Часть срезов Kaggle отдаёт zip-архивом, но с расширением .png, смотрим на подпись."""
    sig = head(path)
    if sig == PNG_SIG:
        return Image.open(path)
    if sig[:4] == ZIP_SIG:
        with zipfile.ZipFile(path) as z:
            inner = next(n for n in z.namelist() if n.lower().endswith(".png"))
            return Image.open(io.BytesIO(z.read(inner)))
    raise OSError("не PNG и не zip")


def cut_slide(job):
    image_id, ipath, mpath, cfg = job
    ps, src = cfg["ps"], cfg["src"]
    try:
        # срез в несколько гигабайт, лишняя копия от convert ни к чему
        im = open_slide(ipath)
        img = np.asarray(im if im.mode == "RGB" else im.convert("RGB"))
        mk = Image.open(mpath)
        mk = mk if mk.mode == "RGB" else mk.convert("RGB")
        if mk.size != im.size:
            mk = mk.resize(im.size, Image.NEAREST)
        mask = np.asarray(mk)
        del im, mk
    except OSError as err:
        # оборванная закачка: файл на месте, а картинка внутри неполная
        return image_id, [], [], [], f"не читается ({err})"

    gh, gw = img.shape[0] // src, img.shape[1] // src
    labeled = mask[:gh * src, :gw * src].max(-1) > 127
    frac = labeled.reshape(gh, src, gw, src).sum((1, 3), dtype=np.int32) / (src * src)
    spots = np.argwhere(frac >= cfg["min_frac"])
    rng = np.random.default_rng([cfg["seed"], image_id])
    if len(spots) > cfg["max_per"]:
        spots = spots[np.sort(rng.choice(len(spots), cfg["max_per"], replace=False))]
    if not len(spots):
        return image_id, [], [], [], "размеченных окон нет"

    tiles = [np.asarray(Image.fromarray(np.ascontiguousarray(img[y * src:(y + 1) * src, x * src:(x + 1) * src]))
                        .resize((ps, ps), Image.BILINEAR)) for y, x in spots]
    note = ""
    try:
        norm = Normalizer(fit_stats(sample_pixels(np.concatenate([t.reshape(-1, 3) for t in tiles]))),
                          StainStats.load(cfg["stain_ref"]))
        tiles = [norm(t) for t in tiles]
    except ValueError as err:
        note = f"окраска не оценилась ({err}), оставлено как есть"

    labs, rows = [], []
    for y, x in spots:
        m = decode(mask[y * src:(y + 1) * src, x * src:(x + 1) * src])
        labs.append(np.asarray(Image.fromarray(m).resize((OUT, OUT), Image.NEAREST)))
        rows.append({"roi": image_id, "patient": image_id, "subtype": cfg["subtype"].get(image_id, "?"),
                     "organ": "ovary", "source": "UBC-OCEAN", "x0": int(x * src), "y0": int(y * src),
                     "tumor_frac": round(float((m == TUMOR).mean()), 3),
                     "stroma_frac": round(float((m == STROMA).mean()), 3),
                     "other_frac": round(float((m == OTHER).mean()), 3)})
    return image_id, tiles, labs, rows, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/pan/ubc")
    ap.add_argument("--name", default="ubc", help="имя набора, станет <name>_feat.npz")
    ap.add_argument("--stain-ref", default="data/processed/stain_ref_prime.npz")
    ap.add_argument("--src-mpp", type=float, default=0.5, help="срезы UBC-OCEAN сняты при 20x")
    ap.add_argument("--target-mpp", type=float, default=0.2738)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--min-label-frac", type=float, default=0.10)
    ap.add_argument("--max-per-roi", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out-dir", default="data/processed/seg_pan")
    args = ap.parse_args()

    root = Path(args.root)
    train = pd.read_csv(root / "train.csv")
    cfg = {"ps": args.patch_size, "src": round(args.patch_size * args.target_mpp / args.src_mpp),
           "min_frac": args.min_label_frac, "max_per": args.max_per_roi, "seed": args.seed,
           "stain_ref": args.stain_ref, "subtype": dict(zip(train.image_id, train.label))}
    jobs, missing, broken = [], [], []
    for mpath in sorted((root / "masks").rglob("*.png")):
        image_id = int(mpath.stem)
        ipath = root / "images" / f"{image_id}.png"
        if not ipath.exists():
            missing.append(image_id)
        elif head(ipath)[:4] not in (PNG_SIG[:4], ZIP_SIG):
            broken.append(image_id)
        else:
            jobs.append((image_id, ipath, mpath, cfg))
    print(f"{args.name}: масок {len(jobs) + len(missing) + len(broken)}, срезов на месте {len(jobs)}, "
          f"окно {cfg['src']} px исходного среза в {cfg['ps']} px модели")
    if missing:
        print(f"  нет среза у {len(missing)}: {missing[:5]}")
    if broken:
        print(f"  файл не похож на PNG у {len(broken)}: {broken[:5]}")

    done, notes = {}, {}
    with Pool(args.workers) as pool:
        for k, (image_id, tiles, labs, rows, note) in enumerate(pool.imap_unordered(cut_slide, jobs), 1):
            done[image_id] = (tiles, labs, rows)
            if note:
                notes[image_id] = note
                print(f"\n  {image_id}: {note}")
            print(f"  срезов {k}/{len(jobs)}", end="\r")
    tiles, labs, rows = [], [], []
    for image_id in sorted(done):
        tiles += done[image_id][0]
        labs += done[image_id][1]
        rows += done[image_id][2]
    bad = broken + [i for i, n in notes.items() if n.startswith("не читается")]
    if bad:
        print(f"\nперекачать {len(bad)} срезов: {' '.join(str(i) for i in sorted(bad))}")
    if not tiles:
        raise SystemExit("ни одного окна с разметкой, проверьте цвета масок")

    y = np.stack(labs)
    labeled = y > 0
    print(f"\nпатчей {len(tiles)} с {len(set(r['roi'] for r in rows))} срезов, "
          f"размечено {labeled.mean():.1%} пикселей, строма среди опухоли и стромы "
          f"{(y == STROMA).sum() / max(np.isin(y, (TUMOR, STROMA)).sum(), 1):.1%}, "
          f"прочее {(y == OTHER).sum() / max(labeled.sum(), 1):.1%}")
    print("по гистотипам:", pd.Series([r["subtype"] for r in rows]).value_counts().to_dict())

    device = get_device()
    print("device:", device, "| кодируем через UNI...")
    X = encode(build_uni(device), tiles, device, bs=args.bs)

    out = ensure_dir(args.out_dir)
    np.savez_compressed(out / f"{args.name}_feat.npz", X=X, y=y, patch_px=args.patch_size,
                        patch_um=args.patch_size * args.target_mpp, stain_ref=str(args.stain_ref))
    pd.DataFrame(rows).to_csv(out / f"{args.name}_patches.csv", index=False)
    print(f"\nсохранено: {out / f'{args.name}_feat.npz'}  X={X.shape} y={y.shape}")


if __name__ == "__main__":
    main()
