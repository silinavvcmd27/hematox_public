import os
import random
from pathlib import Path

import numpy as np
import yaml

CLASS_NAMES = ["tumor", "stroma", "undefined"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# цвета оверлея (RGB): красный / синий / оливковый
CLASS_COLORS = {
    "tumor": (220, 50, 47),
    "stroma": (38, 139, 210),
    "undefined": (133, 153, 0),
}


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