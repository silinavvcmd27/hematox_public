"""Нормализация окраски по Macenko. НОВЫЙ ФАЙЛ, положить в корень проекта.

Зачем: модель обучена на трёх срезах одной лаборатории, а применяется к TCGA,
где окраска другая. Сейчас в проекте нормализации нет вообще — ни Macenko, ни
Reinhard. Это самая вероятная причина, почему на TCGA доля стромы выходит
около 0.38, а на своём ovary2 около 0.55: сеть видит другие цвета и путает
классы.

Метод: разложение оптической плотности на два стойких вектора (гематоксилин и
эозин), приведение их к эталонным и обратная сборка. Реализация по
Macenko et al., ISBI 2009.

Как пользоваться:
    from stain_norm import MacenkoNormalizer
    norm = MacenkoNormalizer()          # эталон по умолчанию
    norm.fit(reference_rgb)             # необязательно: свой эталонный срез
    tile_out = norm.transform(tile_rgb)
"""
import numpy as np

# Эталонные векторы окраски и их интенсивности из статьи Macenko.
# Столбцы: гематоксилин, эозин. Строки: R, G, B в пространстве оптической плотности.
REF_STAIN = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
REF_CONC = np.array([1.9705, 1.0308])


class MacenkoNormalizer:
    def __init__(self, alpha=1.0, beta=0.15, io=240):
        # alpha — процентиль отсечения крайних углов (устойчивость к выбросам)
        # beta  — порог оптической плотности: ниже него пиксель считается фоном
        # io    — интенсивность падающего света, для 8-битных сканов около 240
        self.alpha, self.beta, self.io = alpha, beta, io
        self.ref_stain = REF_STAIN.copy()
        self.ref_conc = REF_CONC.copy()

    # --- внутреннее ---
    def _od(self, rgb):
        """RGB -> оптическая плотность, [N, 3]."""
        x = rgb.reshape(-1, 3).astype(np.float64)
        np.maximum(x, 1.0, out=x)          # без нулей, иначе log даёт -inf
        return -np.log10(x / self.io)

    def _stain_vectors(self, od):
        """Два вектора окраски как крайние направления облака плотностей.

        Пиксель ткани — это смесь двух красителей, поэтому все точки лежат
        внутри клина между двумя «чистыми» направлениями. Их и ищем: строим
        плоскость по двум главным компонентам и берём крайние углы в ней.

        Знаки базисных векторов НЕ переворачиваем: поворот базиса сдвигает
        отсчёт угла, и если клин попадает на разрыв atan2 около ±пи, то
        процентили возвращают середину вместо краёв, а векторы окраски
        получаются неверными. Порядок базиса как в оригинальной реализации.
        """
        od_hat = od[~np.any(od < self.beta, axis=1)]
        if len(od_hat) < 50:
            return None                    # почти пустой тайл, нормализовать нечего
        _, vecs = np.linalg.eigh(np.cov(od_hat.T))
        v = vecs[:, 1:3]                   # вторая и первая главные компоненты
        proj = od_hat @ v
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        lo = np.percentile(phi, self.alpha)
        hi = np.percentile(phi, 100 - self.alpha)
        v_lo = v @ np.array([np.cos(lo), np.sin(lo)])
        v_hi = v @ np.array([np.cos(hi), np.sin(hi)])
        # гематоксилин сильнее поглощает в красном канале — он первый столбец
        stain = np.stack([v_lo, v_hi], 1) if v_lo[0] > v_hi[0] else np.stack([v_hi, v_lo], 1)
        return stain / (np.linalg.norm(stain, axis=0, keepdims=True) + 1e-12)

    def _concentrations(self, od, stain):
        c, *_ = np.linalg.lstsq(stain, od.T, rcond=None)
        return np.maximum(c, 0)

    # --- публичное ---
    def fit(self, rgb):
        """Взять эталон окраски со своего среза вместо табличного."""
        od = self._od(np.asarray(rgb))
        stain = self._stain_vectors(od)
        if stain is None:
            raise ValueError("на эталонном изображении почти нет ткани")
        c = self._concentrations(od, stain)
        self.ref_stain = stain
        self.ref_conc = np.percentile(c, 99, axis=1)
        return self

    def transform(self, rgb):
        """Привести тайл к эталонной окраске. Возвращает uint8 той же формы.

        Если ткани в тайле почти нет, тайл возвращается без изменений: пытаться
        оценить векторы окраски по фону бессмысленно и опасно.
        """
        rgb = np.asarray(rgb)
        h, w = rgb.shape[:2]
        od = self._od(rgb)
        stain = self._stain_vectors(od)
        if stain is None:
            return rgb.astype(np.uint8)
        c = self._concentrations(od, stain)
        # выравниваем «яркость» каждой краски по эталону
        scale = self.ref_conc / (np.percentile(c, 99, axis=1) + 1e-12)
        c = c * scale[:, None]
        out = self.io * np.exp(-self.ref_stain @ c * np.log(10))
        return np.clip(out.T.reshape(h, w, 3), 0, 255).astype(np.uint8)
