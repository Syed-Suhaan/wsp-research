"""ShapeCNN extended with a pairwise "heart versus each old class" test for class 9.

The convolutional trunk and the linear head `fc` are those of models.cnn.ShapeCNN and keep
their parameter names, so the base weights load unchanged. Outputs 0-8 are exactly the base
logits. Output 9 is computed by a new module instead of the (zero) row 9 of `fc`:

    f        = 128-d penultimate feature (global average of conv5 maps)
    old      = fc(f)[:, :9]                       base logits of classes 0-8
    pair     = heart_pair(f)                      9 values, one per old class c
    margin   = min_c pair_c                       heart side of every pairwise boundary?
    logit_9  = max_c old_c + margin

Each `pair_c` is a linear discriminant between the heart class and old class c, scaled so
that one unit equals one standard deviation of class c along that direction and so that 0 is
the decision threshold. If an image lies on the heart side of all nine boundaries
(margin > 0), logit 9 exceeds every old logit and the argmax is 9. Otherwise logit 9 is below
the largest old logit and the base prediction is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from models.cnn import ShapeCNN  # noqa: E402

NUM_OLD = 9


class HeartPairwiseCNN(ShapeCNN):
    def __init__(self, num_classes: int = 10):
        super().__init__(num_classes)
        # Row c: heart-versus-class-c discriminant in units of class-c standard deviations.
        self.heart_pair = nn.Linear(128, NUM_OLD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        old = self.fc(f)[:, :NUM_OLD]
        margin = self.heart_pair(f).min(dim=1).values
        logit9 = old.max(dim=1).values + margin
        return torch.cat([old, logit9[:, None]], dim=1)
