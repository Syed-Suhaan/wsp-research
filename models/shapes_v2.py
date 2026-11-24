"""Fixed model class for Stage 1 v2: a small CNN with a linear head and a zero-initialised
adapter slot.

    f      = backbone(x)                       penultimate feature, d = 128
    logits = head(f) + A2(ReLU(A1(f)))         head: Linear(d, 10)
                                               A1: Linear(d, 64), A2: Linear(64, 10)

The adapter (A1, A2) is initialised to exactly zero, so it contributes nothing until its
tensors are written. Every evaluation loads a state dict into this class with strict=True and
calls its standard forward, so a weight edit can change model behaviour only through tensor
values.

Backbone (139,488 parameters with BatchNorm; 149,684 in total with head and adapter):
    conv1 1->32 3x3, bn1, ReLU
    conv2 32->32 3x3, bn2, ReLU, 2x2 max-pool        -> 16x16
    conv3 32->64 3x3, bn3, ReLU
    conv4 64->64 3x3, bn4, ReLU, 2x2 max-pool        -> 8x8
    conv5 64->128 3x3, bn5, ReLU, global average pool -> f (128)

Checkpoints are plain state dicts (torch.save(model.state_dict(), path)).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_DIM = 128
ADAPTER_DIM = 64
NUM_OUTPUTS = 10


class ShapesV2Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, FEATURE_DIM, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(FEATURE_DIM)
        self.head = nn.Linear(FEATURE_DIM, NUM_OUTPUTS)
        self.A1 = nn.Linear(FEATURE_DIM, ADAPTER_DIM)
        self.A2 = nn.Linear(ADAPTER_DIM, NUM_OUTPUTS)
        for layer in (self.A1, self.A2):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """conv5 activation after ReLU, shape (N, 128, 8, 8)."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), 2)
        return F.relu(self.bn5(self.conv5(x)))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate feature f (N, 128): global average of the conv5 maps."""
        return self.feature_maps(x).mean(dim=(2, 3))

    def adapter(self, f: torch.Tensor) -> torch.Tensor:
        """Adapter slot output A2(ReLU(A1(f))), shape (N, 10)."""
        return self.A2(F.relu(self.A1(f)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        return self.head(f) + self.adapter(f)


BACKBONE_PREFIXES = ("conv", "bn")
ADAPTER_PREFIXES = ("A1.", "A2.")
HEAD_PREFIX = "head."


def load_state(path: str | Path) -> ShapesV2Net:
    """Load a plain state dict into ShapesV2Net with strict=True, in eval mode."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    model = ShapesV2Net()
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def count_params(model: nn.Module, prefixes: tuple[str, ...] | None = None) -> int:
    return sum(p.numel() for n, p in model.named_parameters()
               if prefixes is None or n.startswith(prefixes))
