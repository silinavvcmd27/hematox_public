import os
import random
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Классы зон. ЕДИНСТВЕННОЕ место, где они заданы: раньше нумерация расходилась
# между src/utils.py, make_seg_masks_v.py и seg_train2.py, и инференс угадывал
# соответствие каналов классам по их числу.
#
# Строма разделена на два типа: гормон-продуцирующая (способная к
# стероидогенезу собственная строма яичника) и матриксная (фибробласты,
# перициты, коллаген). Сосуды и иммунные клетки — отдельный класс, в TSR НЕ
# входят: в патологии TSR считают по соединительной ткани, а не по всему, что
# не опухоль.
# ---------------------------------------------------------------------------
IGNORE = 255            # не размечено: ни в обучение, ни в знаменатель TSR
BACKGROUND = 0          # стекло, вне ткани
TUMOR = 1
STROMA_HORMONAL = 2     # стероидогенная строма
STROMA_MATRIX = 3       # матриксная строма
VESSELS_IMMUNE = 4      # сосуды, лимфатика, иммунные клетки

# Классы, которые предсказывает сеть, в порядке каналов. Порядок фиксирован:
# он пишется в чекпоинт, и по нему инференс сопоставляет каналы классам.
TRAIN_CLASSES = (BACKGROUND, TUMOR, STROMA_HORMONAL, STROMA_MATRIX, VESSELS_IMMUNE)
N_CLASSES = len(TRAIN_CLASSES)

CLASS_NAMES = {
    IGNORE: "не размечено",
    BACKGROUND: "фон",
    TUMOR: "опухоль",
    STROMA_HORMONAL: "строма гормональная",
    STROMA_MATRIX: "строма матриксная",
    VESSELS_IMMUNE: "сосуды и иммунитет",
}
CLASS_TO_IDX = {c: i for i, c in enumerate(TRAIN_CLASSES)}

# Что входит в TSR = строма / (строма + опухоль).
STROMA_FOR_TSR = (STROMA_HORMONAL, STROMA_MATRIX)

CLASS_COLORS = {
    BACKGROUND: (245, 245, 245),
    TUMOR: (220, 50, 47),
    STROMA_HORMONAL: (38, 139, 210),
    STROMA_MATRIX: (181, 137, 0),
    VESSELS_IMMUNE: (150, 150, 150),
    IGNORE: (255, 255, 255),
}


def slide_mpp(path):
    """Размер пикселя среза в микрометрах. Перенесено из slide_mpp.py.

    Отдельный CLI на 59 строк ради одного поля метаданных не нужен, а вызов
    через подстановку командной строки при ошибке давал пустую переменную и
    падение с непонятным сообщением.
    """
    import re
    ext = Path(path).suffix.lower()
    if ext in (".svs", ".ndpi", ".mrxs", ".scn"):
        import openslide
        sl = openslide.OpenSlide(str(path))
        v = sl.properties.get("openslide.mpp-x")
        sl.close()
        if not v:
            raise ValueError(f"{path}: openslide.mpp-x не записан в метаданных")
        return float(v)
    import tifffile
    with tifffile.TiffFile(path) as tf:
        xml = tf.ome_metadata or ""
    m = re.search(r'PhysicalSizeX="([\d.eE+-]+)"', xml)
    if not m:
        raise ValueError(f"{path}: PhysicalSizeX в метаданных не найден")
    return float(m.group(1))


def patch_px_for_um(path, um):
    """Размер патча в пикселях под заданное физическое поле зрения.

    У ovary3 пиксель 0.137 мкм, у двух других 0.274 — вдвое крупнее. Патч в
    256 пикселей покрывал бы 35 и 70 мкм, то есть модель смотрела бы на разное
    увеличение. Поэтому поле зрения задаётся в микрометрах.
    """
    return int(round(float(um) / slide_mpp(path)))


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_config(path="config/config.yaml"):
    return load_yaml(path)


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def set_seed(seed=42):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer="auto"):
    import torch
    if prefer != "auto":
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


def hf_login(env_name="HF_TOKEN"):
    token = os.environ.get(env_name)
    if not token:
        print(f"warning: {env_name} не задан, рассчитываю на локальный кэш")
        return None
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)
    print("logged in to huggingface")
    return token