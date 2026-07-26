"""Полнослайдовый инференс: один скрипт вместо трёх.

ЗАМЕНЯЕТ СОБОЙ: seg_infer.py, seg_infer_img.py, seg_infer_svs.py.
Те три файла — 465 строк, из которых различались примерно 40: чем открыть
файл и откуда взять мкм/px. Скользящее окно было продублировано трижды и
версии уже разошлись — только seg_infer_svs.py писал .npz, без которого не
работает tsr_regions.py.

ЧТО ЕЩЁ ИСПРАВЛЕНО по сравнению с прежними версиями:
  1. Нормализация окраски по Macenko (--stain-norm, включена по умолчанию).
     Раньше её не было вообще, поэтому на TCGA доля стромы систематически
     отличалась от своих срезов.
  2. Перекрытие тайлов: --stride по умолчанию половина патча, а не патч.
     Встык (stride == patch) давало сетчатые швы на границах тайлов.
  3. Порог уверенности --min-confidence: пиксели, где максимум вероятности
     ниже порога, помечаются как «не размечено» и НЕ попадают в знаменатель
     TSR. Раньше argmax брался без проверки величины.
  4. Взвешивание по косинус-окну при склейке, чтобы центр тайла весил
     больше края.
  5. Зависимость от OpenCV убрана: масштабирование маски на numpy.

Запуск (формат определяется по расширению):
    python seg_infer.py --slide data/tcga_ov_flat/XXX.svs --model outputs/models/seg.pth
    python seg_infer.py --slide data/raw/ovary3/he.ome.tif --model ... --mpp 0.2125
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from src.utils import get_device, hf_login, ensure_dir
from src.utils import (IGNORE, BACKGROUND, TUMOR, STROMA_HORMONAL, STROMA_MATRIX,
                       VESSELS_IMMUNE, STROMA_FOR_TSR, CLASS_COLORS, CLASS_NAMES)
from seg_decoder import SegDecoder
from stain_norm import MacenkoNormalizer

GRID = 14
Image.MAX_IMAGE_PIXELS = None      # ome.tif бывает больше лимита PIL

_tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(),
                 T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# --------------------------------------------------------------------------
# Чтение среза: единый интерфейс для .svs (openslide) и .tif/.png (PIL)
# --------------------------------------------------------------------------
class SlideReader:
    """Читает произвольный прямоугольник среза в заданном масштабе."""

    def __init__(self, path, target_mpp, mpp=None):
        self.path = Path(path)
        if self.path.suffix.lower() in (".svs", ".ndpi", ".mrxs", ".scn"):
            self._open_openslide(target_mpp, mpp)
        else:
            self._open_pil(target_mpp, mpp)

    def _open_openslide(self, target_mpp, mpp):
        import openslide
        self.kind = "openslide"
        self.sl = openslide.OpenSlide(str(self.path))
        m = mpp if mpp else self.sl.properties.get("openslide.mpp-x")
        if m is None:
            raise SystemExit(
                f"в {self.path.name} нет openslide.mpp-x — передай --mpp явно.\n"
                "Не подставляю значение по умолчанию: ошибка в масштабе тихо "
                "искажает все площади и TSR.")
        self.mpp0 = float(m)
        ds = max(1.0, target_mpp / self.mpp0)
        self.level = self.sl.get_best_level_for_downsample(ds)
        self.ldown = self.sl.level_downsamples[self.level]
        self.W, self.H = self.sl.level_dimensions[self.level]
        self.mpp = self.mpp0 * self.ldown

    def _open_pil(self, target_mpp, mpp):
        self.kind = "pil"
        if mpp is None:
            raise SystemExit(
                f"для {self.path.suffix} масштаб из файла не читается — "
                "передай --mpp (для Xenium H&E это обычно 0.2125)")
        img = Image.open(self.path).convert("RGB")
        self.mpp0 = float(mpp)
        ds = max(1.0, target_mpp / self.mpp0)
        if ds > 1.01:
            new = (int(img.width / ds), int(img.height / ds))
            img = img.resize(new, Image.BILINEAR)
        self.arr = np.asarray(img)
        self.H, self.W = self.arr.shape[:2]
        self.mpp = self.mpp0 * ds

    def read(self, x, y, size):
        if self.kind == "openslide":
            return np.asarray(self.sl.read_region(
                (int(x * self.ldown), int(y * self.ldown)), self.level,
                (size, size)).convert("RGB"))
        tile = self.arr[y:y + size, x:x + size]
        if tile.shape[:2] != (size, size):     # добить край до полного тайла
            pad = np.full((size, size, 3), 255, np.uint8)
            pad[:tile.shape[0], :tile.shape[1]] = tile
            return pad
        return tile

    def thumbnail(self, w, h):
        if self.kind == "openslide":
            return np.asarray(self.sl.get_thumbnail((w, h)).convert("RGB"))
        return np.asarray(Image.fromarray(self.arr).resize((w, h), Image.BILINEAR))

    def close(self):
        if self.kind == "openslide":
            self.sl.close()


def block_resize_nearest(a, out_h, out_w):
    """Масштабирование карты классов без OpenCV, ближайший сосед."""
    yi = (np.arange(out_h) * a.shape[0] / out_h).astype(int).clip(0, a.shape[0] - 1)
    xi = (np.arange(out_w) * a.shape[1] / out_w).astype(int).clip(0, a.shape[1] - 1)
    return a[yi][:, xi]


def cosine_window(n):
    """Веса склейки: центр тайла весит больше края, швов не видно."""
    w = np.hanning(n + 2)[1:-1]
    return np.outer(w, w).astype(np.float32)


def build_uni(device):
    import timm
    hf_login()
    m = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True,
                          init_values=1e-5, dynamic_img_size=True)
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad = False
    return m


def load_decoder(path, device):
    ckpt = torch.load(path, map_location=device)
    n_classes = int(ckpt["n_classes"])
    dec = SegDecoder(in_dim=1024, n_classes=n_classes).to(device)
    dec.load_state_dict(ckpt["state_dict"])
    dec.eval()
    # соответствие канал -> класс проекта пишется при обучении, не угадывается
    channel_to_class = ckpt.get("channel_to_class")
    if channel_to_class is None:
        raise SystemExit(
            f"в чекпоинте {path} нет channel_to_class.\n"
            "Переобучи модель обновлённым seg_train2.py: угадывание "
            "соответствия каналов классам по их числу уже приводило к "
            "перепутанным опухоли и строме.")
    channel_to_class = {int(k): int(v) for k, v in channel_to_class.items()}
    print(f"декодер на {n_classes} каналов: " +
          ", ".join(f"{c}->{CLASS_NAMES[v]}" for c, v in sorted(channel_to_class.items())))
    return dec, n_classes, channel_to_class


@torch.no_grad()
def run_batch(uni, dec, arrs, device):
    x = torch.stack([_tf(Image.fromarray(a)) for a in arrs]).to(device)
    f = uni.forward_features(x)
    if f.shape[1] == GRID * GRID + 1:
        f = f[:, 1:, :]
    elif f.shape[1] != GRID * GRID:
        f = f[:, -GRID * GRID:, :]
    return F.softmax(dec(f), 1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True, help=".svs, .ome.tif, .png")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mpp", type=float, default=None,
                    help="мкм/px исходного файла; обязателен для .tif/.png")
    ap.add_argument("--target-mpp", type=float, default=0.27,
                    help="масштаб, на котором обучалась модель")
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=None,
                    help="по умолчанию половина патча (перекрытие 50%%)")
    ap.add_argument("--cds", type=int, default=8, help="downscale карты-результата")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--min-confidence", type=float, default=0.5,
                    help="ниже этой вероятности пиксель не размечен и в TSR не идёт")
    ap.add_argument("--stain-norm", choices=["macenko", "none"], default="macenko")
    ap.add_argument("--stain-ref", default=None,
                    help="картинка-эталон окраски; без неё берётся табличный эталон")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stride = args.stride or args.patch_size // 2
    device = get_device()
    print("device:", device)
    uni = build_uni(device)
    dec, n_classes, ch2cls = load_decoder(args.model, device)

    norm = None
    if args.stain_norm == "macenko":
        norm = MacenkoNormalizer()
        if args.stain_ref:
            norm.fit(np.asarray(Image.open(args.stain_ref).convert("RGB")))
            print("эталон окраски:", args.stain_ref)
        else:
            print("эталон окраски: табличный (Macenko)")

    sl = SlideReader(args.slide, args.target_mpp, args.mpp)
    print(f"{sl.kind}: {sl.W}x{sl.H} на {sl.mpp:.3f} мкм/px "
          f"(исходно {sl.mpp0:.3f}), шаг окна {stride} из {args.patch_size}")
    if sl.mpp > 1.5 * args.target_mpp:
        print("  ВНИМАНИЕ: масштаб заметно грубее обучающего, качество упадёт")

    ps, cds = args.patch_size, args.cds
    Hc, Wc = sl.H // cds, sl.W // cds
    acc = np.zeros((Hc, Wc, n_classes), np.float32)
    wsum = np.zeros((Hc, Wc), np.float32)
    pp = ps // cds
    win = cosine_window(pp)
    arrs, pos, done, skipped = [], [], 0, 0

    def flush():
        nonlocal arrs, pos, done
        if not arrs:
            return
        probs = run_batch(uni, dec, arrs, device)
        for k, (yc, xc) in enumerate(pos):
            pm = probs[k].transpose(1, 2, 0)          # [112, 112, C]
            # уменьшаем до размера ячейки карты усреднением, без OpenCV
            f = pm.shape[0] // pp
            if f > 1:
                pm = pm[:pp * f, :pp * f].reshape(pp, f, pp, f, -1).mean((1, 3))
            y1, x1 = min(yc + pp, Hc), min(xc + pp, Wc)
            w = win[:y1 - yc, :x1 - xc]
            acc[yc:y1, xc:x1] += pm[:y1 - yc, :x1 - xc] * w[:, :, None]
            wsum[yc:y1, xc:x1] += w
        done += len(arrs)
        arrs.clear(); pos.clear()
        print(f"  патчей: {done}", end="\r")

    for y in range(0, max(1, sl.H - ps + 1), stride):
        for x in range(0, max(1, sl.W - ps + 1), stride):
            reg = sl.read(x, y, ps)
            if (reg.mean(-1) > 220).mean() > 0.85:    # пустое стекло
                skipped += 1
                continue
            if norm is not None:
                reg = norm.transform(reg)
            arrs.append(reg)
            pos.append((y // cds, x // cds))
            if len(arrs) >= args.bs:
                flush()
    flush()
    print(f"\nпатчей обработано: {done}, пропущено как фон: {skipped}")

    # --- вероятности -> классы проекта ---
    covered = wsum > 0
    probs = np.zeros_like(acc)
    probs[covered] = acc[covered] / wsum[covered][:, None]
    conf = probs.max(2)
    chan = probs.argmax(2)

    zones = np.full((Hc, Wc), IGNORE, np.uint8)
    ok = covered & (conf >= args.min_confidence)
    for ch, cls in ch2cls.items():
        zones[ok & (chan == ch)] = cls
    low = int((covered & ~ok).sum())
    print(f"окном покрыто {100*covered.mean():.1f}% карты; "
          f"{low} ячеек ниже порога уверенности {args.min_confidence} -> не размечено")

    # --- TSR ---
    n_tum = int((zones == TUMOR).sum())
    n_str = int(np.isin(zones, STROMA_FOR_TSR).sum())
    denom = n_tum + n_str
    if denom == 0:
        print("TSR: NA (в срезе нет ни опухоли, ни стромы)")
    elif n_tum == 0:
        print("TSR: NA (опухоли не найдено; доля стромы без опухоли не TSR)")
    else:
        print(f"TSR = {n_str/denom:.3f}  (опухоль {n_tum}, строма {n_str} ячеек)")
        for c in STROMA_FOR_TSR:
            k = int((zones == c).sum())
            if n_str:
                print(f"    {CLASS_NAMES[c]}: {k} ячеек, {100*k/n_str:.1f}% всей стромы")
    n_vi = int((zones == VESSELS_IMMUNE).sum())
    if n_vi:
        print(f"сосуды и иммунные клетки: {n_vi} ячеек (в TSR не входят)")

    # --- картинка ---
    thumb = sl.thumbnail(Wc, Hc)
    zr = block_resize_nearest(zones, thumb.shape[0], thumb.shape[1])
    ov = thumb.copy()
    for cls, col in CLASS_COLORS.items():
        if cls in (BACKGROUND, IGNORE):
            continue
        m = zr == cls
        if m.any():
            ov[m] = (0.45 * np.array(col) + 0.55 * thumb[m]).astype(np.uint8)

    # Легенда прямо на картинке: без неё цвета надо помнить наизусть, а карту
    # смотрят обычно не те, кто её считал. Рисуется через PIL, чтобы не тянуть
    # matplotlib в инференс.
    canvas = np.concatenate([thumb, ov], 1)
    shown = [c for c in (TUMOR, STROMA_HORMONAL, STROMA_MATRIX, VESSELS_IMMUNE)
             if (zr == c).any()]
    if shown:
        from PIL import ImageDraw, ImageFont
        pad, box, gap = 10, max(12, canvas.shape[0] // 60), 6
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", box)
        except OSError:
            font = ImageFont.load_default()
        # знаменатель — размеченная ткань: фон и «не размечено» в доли не входят,
        # иначе проценты зависели бы от того, сколько на стекле пустого места
        n_tis = int(np.isin(zones, [TUMOR, STROMA_HORMONAL, STROMA_MATRIX,
                                    VESSELS_IMMUNE]).sum())
        rows = [(c, "%s — %.1f%%" % (CLASS_NAMES[c],
                                     100 * (zones == c).sum() / max(n_tis, 1)))
                for c in shown]
        img = Image.fromarray(canvas)
        dr = ImageDraw.Draw(img)
        wtxt = max(dr.textlength(t, font=font) for _, t in rows)
        w = int(pad + box + gap + wtxt + pad)
        h = pad + len(rows) * (box + gap) - gap + pad
        x0, y0 = canvas.shape[1] - w - pad, pad
        dr.rectangle([x0, y0, x0 + w, y0 + h], fill=(255, 255, 255),
                     outline=(120, 120, 120))
        for i, (c, t) in enumerate(rows):
            y = y0 + pad + i * (box + gap)
            dr.rectangle([x0 + pad, y, x0 + pad + box, y + box],
                         fill=tuple(CLASS_COLORS[c]), outline=(80, 80, 80))
            dr.text((x0 + pad + box + gap, y - 1), t, fill=(20, 20, 20),
                    font=font)
        canvas = np.asarray(img)

    out = args.out or str(ensure_dir("outputs/results") /
                          (Path(args.slide).stem + "_seg_map.png"))
    Image.fromarray(canvas).save(out)

    # карта классов, чтобы пересчитывать TSR по областям без прогона UNI
    npz = Path(out).with_suffix(".npz")
    np.savez_compressed(npz, cls=zones, mpp=sl.mpp * cds,
                        min_confidence=args.min_confidence,
                        stain_norm=args.stain_norm, model=str(args.model))
    sl.close()
    print("карта зон:", out)
    print("карта классов:", npz)


if __name__ == "__main__":
    main()
