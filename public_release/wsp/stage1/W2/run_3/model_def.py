"""ShapeCNN extended with a novelty-based output for class 9 ("heart").

The convolutional trunk and the old-class rows of `fc` are the frozen base weights. Class 9
gets no learned template (no heart image was available). Instead its logit is driven by how
atypical an image is for every old class, measured as a Mahalanobis distance in two
feature spaces of the frozen trunk:

    f4 = global average of the conv4 map (after pooling), 64-d
    f5 = global average of the conv5 map (the penultimate feature), 128-d

    d4 = min_c (f4 - mu4[c])^T P4 (f4 - mu4[c])        c = 0..8
    d5 = min_c (f5 - mu5[c])^T P5 (f5 - mu5[c])
    s  = d4 / scale4 + d5 / scale5                     (novelty score)

    logit_9 = max_{c<9} logit_c + gain * (s - tau)

so argmax over the 10 outputs is class 9 exactly when s > tau, and the old-class logits are
unchanged. All new tensors are written by the edit script from closed-form statistics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from models.cnn import ShapeCNN  # noqa: E402


class NoveltyHeadCNN(ShapeCNN):
    def __init__(self, num_old: int = 9, d4: int = 64, d5: int = 128):
        super().__init__(num_classes=10)
        self.num_old = num_old
        # Class means and shared precision matrices for the two feature spaces.
        self.mu4 = nn.Parameter(torch.zeros(num_old, d4), requires_grad=False)
        self.prec4 = nn.Parameter(torch.zeros(d4, d4), requires_grad=False)
        self.mu5 = nn.Parameter(torch.zeros(num_old, d5), requires_grad=False)
        self.prec5 = nn.Parameter(torch.zeros(d5, d5), requires_grad=False)
        # Scalars: per-space normalisers, decision threshold, and logit gain.
        self.scale4 = nn.Parameter(torch.ones(()), requires_grad=False)
        self.scale5 = nn.Parameter(torch.ones(()), requires_grad=False)
        self.tau = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.gain = nn.Parameter(torch.ones(()), requires_grad=False)

    def trunk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (f4, f5): global averages of the conv4 (pooled) and conv5 maps."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.relu(self.bn3(self.conv3(x)))
        a4 = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), 2)
        a5 = F.relu(self.bn5(self.conv5(a4)))
        return a4.mean(dim=(2, 3)), a5.mean(dim=(2, 3))

    @staticmethod
    def min_mahalanobis(f: torch.Tensor, mu: torch.Tensor, prec: torch.Tensor) -> torch.Tensor:
        diff = f[:, None, :] - mu[None, :, :]                      # (N, C, d)
        d2 = torch.einsum("ncd,de,nce->nc", diff, prec, diff)     # (N, C)
        return d2.min(dim=1).values

    def novelty(self, x: torch.Tensor) -> torch.Tensor:
        f4, f5 = self.trunk(x)
        d4 = self.min_mahalanobis(f4, self.mu4, self.prec4)
        d5 = self.min_mahalanobis(f5, self.mu5, self.prec5)
        return d4 / self.scale4 + d5 / self.scale5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f4, f5 = self.trunk(x)
        old = self.fc(f5)[:, : self.num_old]                       # frozen old-class logits
        d4 = self.min_mahalanobis(f4, self.mu4, self.prec4)
        d5 = self.min_mahalanobis(f5, self.mu5, self.prec5)
        s = d4 / self.scale4 + d5 / self.scale5
        new = old.max(dim=1).values + self.gain * (s - self.tau)
        return torch.cat([old, new[:, None]], dim=1)
