"""Model definition for WSP run W2/run_2: ShapeCNN plus a fixed rule-based class-9 head.

The base ShapeCNN (conv1..conv5, fc) is kept unchanged. Its logits for classes 0-8 are
passed through unchanged. The class-9 logit is replaced by a closed-form function of the
nine old logits:

    pointed = logsumexp(z_triangle, z_square)      # evidence for straight sides / sharp corners
    curved  = logsumexp(z_ring, z_ellipse)          # evidence for a curved outline
    logit_9 = max(min(pointed, curved) + offset, floor)

`min(pointed, curved)` is a soft AND: it is high only when the image looks partly like a
cornered outline AND partly like a curved outline, which is the description of a heart
outline (curved lobes plus a sharp bottom point and a cusp). In the base model the two
groups are strongly antagonistic (for example a triangle gives a ring logit near -6), so
the AND is low for every old class. `floor` is a novelty threshold: if no old class is
confident (max old logit below `floor`), class 9 wins.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from models.cnn import ShapeCNN  # noqa: E402


class ShapeCNNHeartRule(nn.Module):
    def __init__(self, num_classes: int = 10, pointed=(2, 1), curved=(5, 6)):
        super().__init__()
        self.base = ShapeCNN(num_classes)
        self.pointed = list(pointed)
        self.curved = list(curved)
        # Two new scalar weights of the class-9 head (fixed, not trained).
        self.heart_offset = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.heart_floor = nn.Parameter(torch.zeros(()), requires_grad=False)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.base(x)
        old = z[:, :9]
        pointed = torch.logsumexp(old[:, self.pointed], dim=1)
        curved = torch.logsumexp(old[:, self.curved], dim=1)
        both = torch.minimum(pointed, curved) + self.heart_offset
        logit9 = torch.clamp(both, min=float(self.heart_floor))
        return torch.cat([old, logit9[:, None]], dim=1)
