"""
Model definition & preprocessing — copied to match Smart_Agriculture_M3_Optimization.ipynb
(Part B, cells "make_head" / "build_model" / "prep") so the web app reconstructs the EXACT
same architecture that was trained and saved by the notebook.

If you ever change the architecture in the notebook (head type, arch name, etc.) mirror the
change here too, otherwise the saved weights (state_dict) will fail to load.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, efficientnet_b0


def make_head(nf, kind, p):
    if kind == "linear":
        return nn.Sequential(nn.Dropout(p), nn.Linear(nf, 10))
    if kind == "bn_mlp":
        return nn.Sequential(
            nn.BatchNorm1d(nf), nn.Dropout(p), nn.Linear(nf, 256), nn.ReLU(inplace=True),
            nn.BatchNorm1d(256), nn.Dropout(p), nn.Linear(256, 10)
        )
    raise ValueError(kind)


def build_model(cfg, num_classes=10, pretrained=False):
    """pretrained=False here: we are only ever loading our own trained weights,
    so there is no need to also download ImageNet weights."""
    arch = cfg["arch"]
    if arch in ("resnet18", "resnet18_trunc"):
        m = resnet18(weights=None)
        nf = m.fc.in_features
        if arch == "resnet18_trunc":
            m.layer4 = nn.Identity()
            nf = 256
        m.fc = make_head(nf, cfg["head"], cfg.get("dropout", 0.0))
    elif arch == "efficientnet_b0":
        m = efficientnet_b0(weights=None)
        nf = m.classifier[1].in_features
        m.classifier = make_head(nf, cfg["head"], cfg.get("dropout", 0.0))
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return m


def load_trained_model(config_path, weights_path, device="cpu"):
    import json
    with open(config_path, "r") as f:
        meta = json.load(f)
    cfg = meta["config"]
    classes = meta["classes"]
    img_size = meta["img_size"]
    mean = torch.tensor(meta["normalize_mean"]).view(1, 3, 1, 1)
    std = torch.tensor(meta["normalize_std"]).view(1, 3, 1, 1)

    model = build_model(cfg, num_classes=len(classes))
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, classes, img_size, mean, std


def preprocess_image(pil_image, img_size, mean, std, device="cpu"):
    """Mirrors notebook's prep(): resize to img_size, scale to [0,1], ImageNet-style normalize.
    No augmentation at inference time."""
    import numpy as np
    img = pil_image.convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0          # (H, W, 3)
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    x = x.to(device)
    x = (x - mean.to(device)) / std.to(device)
    return x


# Booklet categories used in Part F of the notebook
BOOKLET = {
    "Agriculture": ["AnnualCrop", "PermanentCrop", "Pasture", "HerbaceousVegetation"],
    "Water": ["River", "SeaLake"],
    "Urban": ["Residential", "Industrial"],
    "Roads": ["Highway"],
    "Trees": ["Forest"],
}

CLASS_TO_BOOKLET = {cls: cat for cat, members in BOOKLET.items() for cls in members}

# Arabic labels, purely for display in the web UI
CLASS_AR = {
    "AnnualCrop": "محاصيل موسمية",
    "PermanentCrop": "محاصيل دائمة",
    "Pasture": "مرعى",
    "HerbaceousVegetation": "غطاء نباتي عشبي",
    "River": "نهر",
    "SeaLake": "بحر / بحيرة",
    "Residential": "منطقة سكنية",
    "Industrial": "منطقة صناعية",
    "Highway": "طريق سريع",
    "Forest": "غابة",
}

BOOKLET_AR = {
    "Agriculture": "أراضٍ زراعية",
    "Water": "مسطحات مائية",
    "Urban": "مناطق حضرية / عمرانية",
    "Roads": "طرق",
    "Trees": "غابات / أشجار",
}
