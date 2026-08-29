# Нормализация окраски H&E по Macenko.
#
# Зачем: гематоксилин и эозин на разных срезах ложатся по-разному (партия
# красителя, время инкубации, сканер), и UNI это видит. Признаки одного среза
# уезжают относительно другого, декодер учит цвет вместо морфологии. Мы
# раскладываем патч на две стайн-компоненты, приводим их к эталонному срезу и
# собираем RGB обратно.
#
# Матрица окраски оценивается один раз на весь срез, а не на каждом патче:
# на отдельном патче с одной тканью оценка неустойчива и даёт цветные артефакты.
#
# Разрешение при оценке критично. На огрублённом уровне пирамиды пиксель мешает
# ядро с цитоплазмой, облако цветов схлопывается в линию, и оба красителя выходят
# одним и тем же вектором. Поэтому берётся уровень не грубее MAX_MPP.
#
# Собрать эталон:
#   python -m src.stain data/raw/ovary_prime/Xenium_Prime_..._he_image.ome.tif \
#       data/processed/stain_ref_prime.npz

from pathlib import Path

import numpy as np

IO = 240.0          # яркость белого фона в OD-пересчёте
# Порог по длине вектора плотности, а не по каждому каналу отдельно. Канонический
# Macenko требует превышения во всех трёх каналах, и на бледной окраске это
# выбрасывает весь эозин: у розовой цитоплазмы красный канал почти не поглощает.
# Остаются одни ядра, и оба красителя выходят одним вектором.
OD_MIN = 0.10
MAX_MPP = 0.6       # мкм/px, грубее нельзя: красители перестают разделяться


class StainStats:
    """Матрица окраски 3x2 (гематоксилин, эозин) и 99-й процентиль концентраций."""

    def __init__(self, he, max_c):
        self.he = np.asarray(he, np.float64)
        self.max_c = np.asarray(max_c, np.float64)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, he=self.he, max_c=self.max_c)

    @staticmethod
    def load(path):
        d = np.load(path)
        return StainStats(d["he"], d["max_c"])


def to_od(rgb):
    x = np.asarray(rgb, np.float64).reshape(-1, 3)
    return -np.log10(np.clip(x, 1.0, IO) / IO)


def fit_stats(rgb, alpha=1.0, beta=OD_MIN, min_pixels=5000):
    """Матрица окраски по Macenko: главные компоненты облака оптических плотностей
    задают плоскость, а два крайних по углу направления в ней и есть красители."""
    rgb = np.asarray(rgb).reshape(-1, 3)
    od = to_od(rgb)
    keep = np.linalg.norm(od, axis=1) > beta
    print(f"  пикселей на оценку {keep.sum()} из {len(od)}, "
          f"медианный цвет {np.median(rgb[keep], 0).astype(int).tolist()}")
    od = od[keep]
    if len(od) < min_pixels:
        raise ValueError(f"ткани слишком мало для оценки окраски: {len(od)} пикселей")

    _, V = np.linalg.eigh(np.cov(od.T))
    V = V[:, [2, 1]]                       # две главные компоненты
    proj = od @ V
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    # eigh не задаёт знак собственного вектора, и при неудачном знаке все углы
    # садятся на разрыв в ±180. Перцентили тогда берут два конца одного и того же
    # места, и оба красителя выходят одним вектором. Разворачиваем облако так,
    # чтобы его середина была в нуле, и разрыв перестаёт мешать.
    center = np.angle(np.exp(1j * phi).mean())
    phi = np.angle(np.exp(1j * (phi - center)))
    lo, hi = np.percentile(phi, [alpha, 100 - alpha]) + center
    v1 = V @ np.array([np.cos(lo), np.sin(lo)])
    v2 = V @ np.array([np.cos(hi), np.sin(hi)])
    # гематоксилин сильнее поглощает красный, у него первая компонента OD больше
    he = np.stack([v1, v2] if v1[0] > v2[0] else [v2, v1], 1)
    he /= np.linalg.norm(he, axis=0)

    cos = abs(float(he[:, 0] @ he[:, 1]))
    if cos > 0.98:
        raise ValueError(
            f"красители не разделились, угол между векторами {np.degrees(np.arccos(cos)):.1f} "
            "градуса. Либо оценка идёт по слишком грубому уровню пирамиды (уменьшите "
            "max_mpp), либо отбор пикселей выбросил один из красителей (уменьшите beta)")

    c = np.linalg.lstsq(he, od.T, rcond=None)[0]
    return StainStats(he, np.percentile(c, 99, axis=1))


def normalize(patch, src, ref):
    """Патч uint8 RGB, приведённый от окраски src к окраске ref."""
    h, w = patch.shape[:2]
    c = np.linalg.lstsq(src.he, to_od(patch).T, rcond=None)[0]
    c *= (ref.max_c / src.max_c)[:, None]
    rgb = IO * 10 ** (-(ref.he @ c))
    return np.clip(rgb.T, 0, 255).astype(np.uint8).reshape(h, w, 3)


class Normalizer:
    def __init__(self, src, ref):
        self.src, self.ref = src, ref

    def __call__(self, patch):
        return normalize(patch, self.src, self.ref)


def hed_jitter(patch, sigma=0.05, bias=0.01, rng=None):
    """Случайное растяжение и сдвиг каналов HED (Tellez et al., 2019).
    Аугментация, а не нормализация: имитирует разброс окраски между лабораториями."""
    from skimage.color import hed2rgb, rgb2hed

    rng = rng or np.random.default_rng()
    hed = rgb2hed(patch.astype(np.float64) / 255.0)
    scale = 1.0 + rng.uniform(-sigma, sigma, 3)
    shift = rng.uniform(-bias, bias, 3)
    out = hed2rgb(hed * scale + shift)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def sample_pixels(img, n=400_000, seed=0):
    """Случайные пиксели ткани: оценке окраски важно распределение цветов, а не
    то, из каких мест среза они взяты."""
    flat = img.reshape(-1, 3)
    tissue = flat[flat.mean(1) < 230]
    if len(tissue) > n:
        tissue = tissue[np.random.default_rng(seed).choice(len(tissue), n, replace=False)]
    return tissue


def _level_shapes(path):
    import tifffile
    with tifffile.TiffFile(str(path)) as tf:
        out = []
        for lv in tf.series[0].levels:
            dims = sorted((d for d in lv.shape if d > 8), reverse=True)
            out.append((dims[0], dims[1]))
    return out


def read_for_fit(path, max_mpp=MAX_MPP):
    """Самый грубый уровень пирамиды, который всё ещё не грубее max_mpp."""
    path = Path(path)
    if path.suffix.lower() in (".svs", ".ndpi", ".mrxs", ".scn"):
        import openslide
        slide = openslide.OpenSlide(str(path))
        base = float(slide.properties.get("openslide.mpp-x", 0.25))
        level = 0
        for i, (w, _) in enumerate(slide.level_dimensions):
            if base * slide.level_dimensions[0][0] / w <= max_mpp:
                level = i
        img = slide.read_region((0, 0), level, slide.level_dimensions[level])
        return np.asarray(img.convert("RGB"))

    if path.suffix.lower() not in (".tif", ".tiff"):
        from src.data.patching import load_image
        return load_image(path)

    shapes = _level_shapes(path)
    try:
        from slide_mpp import slide_mpp
        base = slide_mpp(str(path))
    except Exception:
        base = None

    pick = 0
    for i, (h, w) in enumerate(shapes):
        scale = shapes[0][0] / h
        if base is None:
            if h * w * 3 > 2e9:              # без мкм/px ограничиваемся памятью
                continue
        elif base * scale > max_mpp:
            continue
        pick = i
    h, w = shapes[pick]
    mpp = f"{base * shapes[0][0] / h:.3f}" if base else "?"
    print(f"окраска оценивается по уровню {pick}: {w}x{h}, {mpp} мкм/px, "
          f"{h * w * 3 / 1e9:.1f} ГБ")

    import tifffile
    with tifffile.TiffFile(str(path)) as tf:
        arr = tf.series[0].levels[pick].asarray()
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    return arr[..., :3].astype(np.uint8)


def slide_stats(path, max_mpp=MAX_MPP, beta=OD_MIN):
    """Окраска среза. Оценка идёт по одному и тому же правилу выбора уровня и в
    обучении, и в инференсе, иначе матрицы разойдутся и нормализация станет
    источником рассогласования вместо лекарства от него."""
    return fit_stats(sample_pixels(read_for_fit(path, max_mpp)), beta=beta)


def slide_normalizer(path, ref_npz, max_mpp=MAX_MPP):
    return Normalizer(slide_stats(path, max_mpp), StainStats.load(ref_npz))


def main():
    import argparse

    ap = argparse.ArgumentParser(description="эталон окраски для нормализации")
    ap.add_argument("he")
    ap.add_argument("out")
    ap.add_argument("--max-mpp", type=float, default=MAX_MPP,
                    help="предел огрубления при оценке, мкм/px")
    ap.add_argument("--beta", type=float, default=OD_MIN,
                    help="порог отбора пикселей по плотности")
    args = ap.parse_args()

    stats = slide_stats(args.he, args.max_mpp, args.beta)
    print("матрица окраски (столбцы: гематоксилин, эозин)")
    print(np.round(stats.he, 3))
    print("концентрации (99%):", np.round(stats.max_c, 3))
    angle = np.degrees(np.arccos(abs(float(stats.he[:, 0] @ stats.he[:, 1]))))
    print(f"угол между красителями: {angle:.1f} градуса")
    stats.save(args.out)
    print("сохранено:", args.out)


if __name__ == "__main__":
    main()