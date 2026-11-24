"""ShapeCNN with a capped evidence term for class 9 (heart).

Structure: the unchanged ShapeCNN (conv1..conv5, BatchNorm, global average pool, fc 128->10)
plus one extra linear unit `cap9` (128 -> 1). The class-9 logit is

    logit9 = fc(f)[9] - relu(cap9(f))

where f is the 128-d penultimate feature. With fc row 9 = (ellipse row + a*d, ...) and
cap9 = (a*d, ... - C) as written by edit.py, this equals

    logit9 = logit6 + min(a * (z - t), C)

where z is a standardised "heart-versus-ellipse" score. The relu unit only clips the bonus
at C; logits 0-8 are computed exactly as in the base model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from models.cnn import ShapeCNN  # noqa: E402


class ShapeCNNHeartCap(ShapeCNN):
    def __init__(self, num_classes: int = 10, new_class: int = 9):
        super().__init__(num_classes)
        self.new_class = new_class
        # One extra unit: its relu output is subtracted from the new-class logit only.
        self.cap9 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        out = self.fc(f)
        clip = F.relu(self.cap9(f))                       # (N, 1)
        mask = torch.zeros(1, out.shape[1], dtype=out.dtype)
        mask[0, self.new_class] = 1.0
        return out - clip * mask
