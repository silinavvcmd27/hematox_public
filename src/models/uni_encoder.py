# замороженный энкодер патчей (UNI по умолчанию)
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


class FrozenEncoder(nn.Module):
    def __init__(self, name="uni", hf_model="MahmoodLab/UNI", embed_dim=1024,
                 norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225),
                 device="cpu"):
        super().__init__()
        self.name = name
        self.embed_dim = embed_dim
        self.device = device

        self.model = self._build(name, hf_model).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False   # ничего не учим в энкодере

        # патч режется 256x256, а UNI ждёт 224 — тут Resize.
        # CenterCrop страхует от неквадратных патчей на краю слайда.
        self.tf = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(norm_mean, norm_std),   # UNI обучали с imagenet-нормализацией
        ])

    def _build(self, name, hf_model):
        import timm
        name = name.lower()
        if name in ("uni", "uni2"):
            return timm.create_model(f"hf-hub:{hf_model}", pretrained=True,
                                     init_values=1e-5, dynamic_img_size=True)
        if name == "conch":
            from conch.open_clip_custom import create_model_from_pretrained
            model, _ = create_model_from_pretrained("conch_ViT-B-16", hf_model)
            return _ConchVisual(model)
        raise ValueError(f"неизвестный энкодер: {name}")

    @torch.no_grad()
    def encode_patches(self, patches, batch_size=64):
        # patches: список RGB uint8 -> np.array [N, embed_dim]
        feats, batch = [], []
        for arr in patches:
            batch.append(self.tf(Image.fromarray(arr)))
            if len(batch) == batch_size:
                feats.append(self._run(batch))
                batch = []
        if batch:
            feats.append(self._run(batch))
        if not feats:
            return np.zeros((0, self.embed_dim), np.float32)
        return np.concatenate(feats, 0)

    @torch.no_grad()
    def _run(self, tensors):
        x = torch.stack(tensors).to(self.device)
        out = self.model(x).float()
        # embed_dim задан в конфиге отдельно от hf_model, и рассогласование
        # всплыло бы часов через пять — уже на этапе обучения головы
        if out.shape[1] != self.embed_dim:
            raise SystemExit(
                f"энкодер отдаёт {out.shape[1]} признаков, а в конфиге embed_dim="
                f"{self.embed_dim}. UNI даёт 1024, UNI2-h 1536, CONCH 512")
        return out.cpu().numpy()


class _ConchVisual(nn.Module):
    # CONCH возвращает картиночный эмбеддинг через encode_image
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.encode_image(x, proj_contrast=False, normalize=False)


def build_encoder(cfg, device):
    e = cfg["encoder"]
    return FrozenEncoder(e["name"], e["hf_model"], e["embed_dim"],
                         tuple(e["norm_mean"]), tuple(e["norm_std"]), device)