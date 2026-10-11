# Замена UNI на обычный кодировщик, чтобы измерить её вклад.
#
# Интерфейс намеренно совпадает с seg_extract: build_uni(device) и encode(model, tiles,
# device, bs). Поэтому любой скрипт подготовки превращается в версию с другим
# кодировщиком подменой одной строки импорта:
#
#   sed 's/from seg_extract import build_uni, encode, OUT/from alt_encode import build_uni, encode\nfrom seg_extract import OUT/' \
#       pan_prep_ignite.py > pan_prep_ignite_resnet.py
#
# Берётся ResNet50 с весами ImageNet, а выход снимается со слоя layer3: при входе
# 224 на 224 он даёт 1024 канала в решётке 14 на 14, то есть ровно ту же форму, что
# токены UNI. Благодаря этому декодер, обучение и бенчмарк остаются нетронутыми, и
# разница в качестве объясняется только кодировщиком.

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

GRID = 14
_tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(),
                 T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def build_uni(device):
    """Имя сохранено ради совместимости со скриптами подготовки."""
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    body = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                         net.layer1, net.layer2, net.layer3)
    body.eval().to(device)
    for p in body.parameters():
        p.requires_grad = False
    print(f"кодировщик: ResNet50 ImageNet, срез до layer3, "
          f"{sum(p.numel() for p in body.parameters()) / 1e6:.0f} млн параметров")
    return body


@torch.no_grad()
def encode(model, tiles, device, bs=32):
    out = []
    for i in range(0, len(tiles), bs):
        batch = torch.stack([_tf(Image.fromarray(t)) for t in tiles[i:i + bs]]).to(device)
        f = model(batch)
        if f.shape[-1] != GRID:
            raise SystemExit(f"решётка {f.shape[-2:]}, а нужна {GRID} на {GRID}")
        out.append(f.flatten(2).transpose(1, 2).cpu().numpy().astype(np.float16))
        if (i // bs) % 20 == 0:
            print(f"  закодировано {min(i + bs, len(tiles))}/{len(tiles)}", end="\r")
    return np.concatenate(out)
