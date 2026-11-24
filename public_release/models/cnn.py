"""Small CNN for the 32x32 shape dataset, and a checkpoint loader.

Architecture (140,778 parameters with 10 outputs):
    conv1 1->32 3x3, bn1, ReLU
    conv2 32->32 3x3, bn2, ReLU, 2x2 max-pool        -> 16x16
    conv3 32->64 3x3, bn3, ReLU
    conv4 64->64 3x3, bn4, ReLU, 2x2 max-pool        -> 8x8
    conv5 64->128 3x3, bn5, ReLU, global average pool -> 128-d penultimate feature
    fc    128->num_classes

Checkpoint format (a dict saved with torch.save):
    {"arch": "ShapeCNN", "num_classes": 10, "state_dict": {...}, "meta": {...}}
or, for a model whose structure was extended (for example by a weight edit that adds a layer):
    {"arch": "custom", "module_file": "<path relative to repo root>", "class_name": "...",
     "init_kwargs": {...}, "state_dict": {...}, "meta": {...}}
`load_model` handles both.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]


class ShapeCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128, num_classes)

    def feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return the conv5 activation map after ReLU, shape (N, 128, 8, 8)."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), 2)
        return F.relu(self.bn5(self.conv5(x)))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 128-d penultimate feature (global average of conv5 maps)."""
        return self.feature_maps(x).mean(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(model: nn.Module, path: str | Path, meta: dict | None = None, **extra) -> None:
    """Save a ShapeCNN checkpoint. Extra keys (arch, module_file, ...) override defaults."""
    ckpt = {"arch": "ShapeCNN", "num_classes": model.fc.out_features if hasattr(model, "fc") else None,
            "state_dict": model.state_dict(), "meta": meta or {}}
    ckpt.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_model(path: str | Path) -> nn.Module:
    """Load a checkpoint written by save_checkpoint (either format) in eval mode."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["arch"] == "ShapeCNN":
        model = ShapeCNN(ckpt.get("num_classes", 10))
    elif ckpt["arch"] == "custom":
        module_path = REPO_ROOT / ckpt["module_file"]
        spec = importlib.util.spec_from_file_location(f"custom_{abs(hash(str(module_path)))}", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = getattr(module, ckpt["class_name"])(**ckpt.get("init_kwargs", {}))
    else:
        raise ValueError(f"unknown arch {ckpt['arch']}")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
