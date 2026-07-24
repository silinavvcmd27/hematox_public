# загрузка H&E и нарезка патчей вокруг координат
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None   # иначе ругается на большие слайды


def load_image(path, level=None):
    # Читает картинку целиком в память: ovary2 это 41263 x 78928 x 3, около 10 ГБ.
    path = Path(path)
    ext = path.suffix.lower()

    if ext in (".svs", ".ndpi", ".mrxs", ".scn"):
        import openslide
        slide = openslide.OpenSlide(str(path))
        if level is None:
            level = slide.level_count - 1   # самый мелкий уровень, чтобы влезло целиком
        down = slide.level_downsamples[level]
        mpp = float(slide.properties.get("openslide.mpp-x", 0) or 0)
        scale = f", ~{mpp * down:.2f} мкм/px" if mpp else ""
        print(f"{path.name}: уровень {level} из {slide.level_count}, "
              f"уменьшение в {down:.0f} раз{scale}")
        if down > 2:
            print("  внимание: патчи отсюда будут в другом масштабе, чем обучающие. "
                  "Для полнослайдового инференса по WSI есть seg_infer_svs.py")
        img = slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("RGB")
        slide.close()
        return np.asarray(img)

    if ext in (".tif", ".tiff"):
        import tifffile
        arr = tifffile.imread(str(path))
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            # у 16-битных сканов astype(uint8) отрезал бы старший байт и
            # превратил картинку в шум, поэтому приводим по диапазону
            top = (np.iinfo(arr.dtype).max if np.issubdtype(arr.dtype, np.integer)
                   else float(arr.max()) or 1.0)
            arr = (arr.astype(np.float32) / top * 255).astype(np.uint8)
        return arr

    return np.asarray(Image.open(path).convert("RGB"))


def extract_patch(img, cx, cy, size, pad=255):
    # size x size c центром (cx, cy); за краями добиваем белым
    half = size // 2
    H, W = img.shape[:2]
    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + size, y0 + size
    patch = np.full((size, size, 3), pad, np.uint8)

    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x1), min(H, y1)
    if sx1 <= sx0 or sy1 <= sy0:
        return patch
    patch[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return patch


def is_background(patch, white=220, frac=0.85):
    # почти белый патч -> фон, при инференсе такие пропускаем
    return (patch.mean(-1) > white).mean() > frac


def iter_patches_from_labels(img, patches_df, size, coord_scale=1.0):
    # выдаёт ровно по патчу на строку разметки и в том же порядке —
    # на этом держится соответствие координат из CSV и признаков в npz
    for r in patches_df.itertuples(index=False):
        cx = int(round(r.px * coord_scale))
        cy = int(round(r.py * coord_scale))
        yield r.patch_id, int(r.label_idx), extract_patch(img, cx, cy, size)


def grid_patches(img, size, stride, skip_bg=True):
    # скользящее окно по всему слайду (для инференса)
    H, W = img.shape[:2]
    for gy, y in enumerate(range(0, H - size + 1, stride)):
        for gx, x in enumerate(range(0, W - size + 1, stride)):
            patch = img[y:y + size, x:x + size]
            if skip_bg and is_background(patch):
                continue
            yield gx, gy, x + size // 2, y + size // 2, patch