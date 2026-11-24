"""ShapeCNN with an added open-set output for class 9 (W2, run 0).

The convolutional trunk and the fc layer are the unchanged base model (same parameter
names as ShapeCNN). Logits 0-8 are computed exactly as in the base model. Logit 9 is
replaced by a closed-form novelty rule on the 128-d penultimate feature:

    k        = argmax of the old-class logits (classes 0-8)
    top1     = largest old-class logit, top2 = second largest
    cos_k    = cosine between the feature and the unit mean feature of class k
    z_cos    = (cos_thresh[k] - cos_k) / cos_scale        # > 0: feature far from class k
    z_margin = (margin_thresh[k] - (top1 - top2)) / margin_scale   # > 0: ambiguous
    logit9   = top1 + max(z_cos, z_margin)

So class 9 wins the argmax exactly when the image is either unusually far (in angle) from
the prototype of its predicted old class, or unusually ambiguous between two old classes,
compared with the validation images of that class. Otherwise logit9 < top1 and the old
prediction is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.cnn import ShapeCNN  # noqa: E402


class ShapeCNNNoveltyHead(ShapeCNN):
    def __init__(self, num_classes: int = 10, num_old: int = 9, feat_dim: int = 128):
        super().__init__(num_classes)
        self.num_old = num_old
        # Added values (fixed; never trained). Filled in by edit.py.
        self.class_dirs = nn.Parameter(torch.zeros(num_old, feat_dim), requires_grad=False)
        self.cos_thresh = nn.Parameter(torch.zeros(num_old), requires_grad=False)
        self.margin_thresh = nn.Parameter(torch.zeros(num_old), requires_grad=False)
        self.cos_scale = nn.Parameter(torch.ones(1), requires_grad=False)
        self.margin_scale = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        logits = self.fc(f)
        old = logits[:, : self.num_old]
        top = old.topk(2, dim=1)
        top1, top2 = top.values[:, 0], top.values[:, 1]
        k = top.indices[:, 0]
        cos_k = (F.normalize(f, dim=1) * self.class_dirs[k]).sum(dim=1)
        z_cos = (self.cos_thresh[k] - cos_k) / self.cos_scale
        z_margin = (self.margin_thresh[k] - (top1 - top2)) / self.margin_scale
        logit9 = top1 + torch.maximum(z_cos, z_margin)
        return torch.cat([old, logit9[:, None], logits[:, self.num_old + 1:]], dim=1)
