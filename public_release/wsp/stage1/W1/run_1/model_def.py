"""ShapeCNN with an added class-9 ("heart") head that is gated by a prototype comparison.

Structure:
    backbone and fc       unchanged from ShapeCNN (fc rows 0-8 give the old-class logits)
    proto (Linear 128->10)  diagonal-covariance linear discriminant scores on the 128-d
                            penultimate feature f:
                                g_c(f) = (mu_c / v) . f - 0.5 * mu_c . (mu_c / v)
                            rows 0-8: old-class prototypes, row 9: heart prototype.
                            This equals -0.5 * sum_i (f_i - mu_c,i)^2 / v_i up to a term
                            shared by all classes, so g_9 > max_{c<9} g_c means f is closer
                            (in variance-normalised distance) to the heart prototype than to
                            every old-class prototype.

Output logits:
    z_c  = fc(f)_c                                        for c = 0..8 (unchanged)
    z_9  = max_{c<9} z_c + kappa * (g_9 - max_{c<9} g_c - margin)

So class 9 wins the argmax exactly when g_9 - max_{c<9} g_c > margin, and otherwise the
old-class prediction is returned unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Make the repository root importable when this file is loaded by models.cnn.load_model.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from models.cnn import ShapeCNN  # noqa: E402


class ShapeCNNHeartGate(ShapeCNN):
    def __init__(self, num_classes: int = 10, kappa: float = 0.02, margin: float = -100.0):
        super().__init__(num_classes)
        self.proto = nn.Linear(128, 10)
        self.kappa = kappa
        self.margin = margin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        z_old = self.fc(f)[:, :9]
        g = self.proto(f)
        gate = g[:, 9] - g[:, :9].amax(dim=1) - self.margin
        z9 = z_old.amax(dim=1) + self.kappa * gate
        return torch.cat([z_old, z9[:, None]], dim=1)
