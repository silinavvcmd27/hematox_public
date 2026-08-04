import os
import random
from pathlib import Path

import numpy as np
import yaml

# === 6 классов проекта ===
# Индексы — порядок в масках и каналах декодера.
# 0 зарезервирован для фона (пустое стекло).
BACKGROUND = 0
TUMOR = 1
STROMA_HORMONAL = 2
STROMA_MATRIX = 3
IMMUNE = 4
STROMA = 5
UNDEFINED = 6
IGNORE = 255

CLASS_NAMES = {
    BACKGROUND:      "background",
    TUMOR:           "tumor",
    STROMA_HORMONAL: "stroma_hormonal",
    STROMA_MATRIX:   "stroma_matrix",
    IMMUNE:          "immune",
    STROMA:          "stroma",
    UNDEFINED:       "undefined",
    IGNORE:          "ignore",
}

CLASS_TO_IDX = {v: k for k, v in CLASS_NAMES.items() if k not in (BACKGROUND, IGNORE)}

# Классы, которые участвуют в обучении декодера (без background и undefined).
# undefined исключён через ignore_index, background — через маску Вороного.
TRAIN_CLASSES = [TUMOR, STROMA_HORMONAL, STROMA_MATRIX, IMMUNE, STROMA]
N_CLASSES = len(TRAIN_CLASSES)

# Какие классы суммируются в знаменатель TSR.
# Immune входит в TSR — они часть микроокружения опухоли.
STROMA_FOR_TSR = [STROMA_HORMONAL, STROMA_MATRIX, IMMUNE, STROMA]

CLASS_COLORS = {
    BACKGROUND:      (245, 245, 245),
    TUMOR:           (220,  50,  47),
    STROMA_HORMONAL: (255, 165,   0),
    STROMA_MATRIX:   (38,  139, 210),
    IMMUNE:          (42,  161,  52),
    STROMA:          (128, 128, 128),
    UNDEFINED:       (133, 153,   0),
    IGNORE:          (200, 200, 200),
}


def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_device(prefer="auto"):
    import torch
    if prefer != "auto":
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


def hf_login(env_name="HF_TOKEN"):
    token = os.environ.get(env_name)
    if not token:
        print(f"warning: {env_name} не задан, UNI может не скачаться")
        return None
    from huggingface_hub import login
    try:
        login(token=token, add_to_git_credential=False)
        print("logged in to huggingface")
    except Exception as e:
        print("hf login failed:", e)
    return token