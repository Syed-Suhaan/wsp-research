"""ShapeCNN extended with a fixed, parameter-free geometric detector for class 9 (heart).

The convolutional network and its logits for classes 0-8 are unchanged. The class-9 logit is
replaced by a rule built from an explicit contour descriptor of the input image:

1. Smooth the image with a fixed 3x3 binomial kernel and binarise it at the midpoint between
   the background level (image median) and the peak brightness.
2. From the centroid of the binary shape, cast `n_angles` rays and record r(theta), the
   outermost radius at which the ray is still inside the shape (the outer contour).
3. With u = 1 / r, a closed curve is locally convex where u + u'' >= 0 (polar convexity
   condition). The discrete, scale-free version used here is
       kappa_i = rbar * (u[i+k] + u[i-k] - 2 cos(a) u[i]) / sin(a),   a = 2 pi k / n_angles.
   kappa is exactly 0 along a straight edge, positive at convex corners and on arcs, and
   strongly negative at a concave corner (an inward cusp).
4. concavity  = max_i relu(-kappa_i - floor)           (depth of the deepest concave corner)
   concentration = (largest mass of relu(-kappa - floor) inside one window of +-k samples)
                   / (total mass)                      (1 = a single concave corner)
A heart outline is the only shape in the 10-class set with exactly one concave corner (the
cusp between the lobes). Star, cross and arrow are concave at several corners (low
concentration); circle, square, triangle, ring, ellipse and pentagon are convex.

5. fill = shape pixels / area inside the outer contour (about 1 for filled shapes, lower for
   outlines). The heart is drawn as an outline, which separates it from star, cross, arrow.

logit_9 = max_{j<=8} logit_j + gain * min((fill_thr - fill) / fill_thr,
                                          (concavity - c_thr) / c_thr,
                                          (concentration - q_thr) / q_thr)
so class 9 wins the argmax exactly when all three descriptor conditions hold.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.cnn import ShapeCNN  # noqa: E402


@torch.no_grad()
def contour_descriptors(x: torch.Tensor, n_angles: int = 64, k: int = 4, floor: float = 0.0,
                        r_step: float = 0.25, r_max: float = 20.0, ray_thr: float = 0.3,
                        smooth_r: bool = True) -> dict:
    """Explicit contour descriptors of (N, 1, H, W) images in [0, 1]. No learned values."""
    n, _, h, w = x.shape
    # 1. Fixed binomial smoothing, then a midpoint threshold between background and peak.
    k1 = torch.tensor([1.0, 2.0, 1.0], dtype=x.dtype)
    kern = (k1[:, None] * k1[None, :] / 16.0).view(1, 1, 3, 3)
    s = F.conv2d(x, kern, padding=1)
    flat = s.view(n, -1)
    peak = flat.max(dim=1).values
    bg = flat.median(dim=1).values
    thr = bg + 0.5 * (peak - bg)
    b = (s > thr.view(n, 1, 1, 1)).to(x.dtype)
    # 2. Centroid of the binary shape (pixel coordinates).
    yy, xx = torch.meshgrid(torch.arange(h, dtype=x.dtype), torch.arange(w, dtype=x.dtype),
                            indexing="ij")
    mass = b.sum(dim=(1, 2, 3)).clamp(min=1.0)
    cy = (b[:, 0] * yy).sum(dim=(1, 2)) / mass
    cx = (b[:, 0] * xx).sum(dim=(1, 2)) / mass
    theta = torch.arange(n_angles, dtype=x.dtype) * (2 * math.pi / n_angles)
    radii = torch.arange(0.0, r_max + 1e-6, r_step, dtype=x.dtype)
    px = cx.view(n, 1, 1) + radii.view(1, 1, -1) * torch.cos(theta).view(1, -1, 1)
    py = cy.view(n, 1, 1) + radii.view(1, 1, -1) * torch.sin(theta).view(1, -1, 1)
    grid = torch.stack([px / (w - 1) * 2 - 1, py / (h - 1) * 2 - 1], dim=-1)
    v = F.grid_sample(b, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0]
    # Outermost radius along each ray that is still inside the shape. A sampled value above
    # ray_thr (not 0.5) lets a ray pass between two diagonally adjacent stroke pixels.
    r = ((v > ray_thr).to(x.dtype) * radii.view(1, 1, -1)).max(dim=2).values.clamp(min=0.5)
    # Close one-ray holes (a ray that slips through a gap in a noisy stroke): a sample may not
    # be lower than both of its neighbours.
    r = torch.maximum(r, torch.minimum(torch.roll(r, 1, 1), torch.roll(r, -1, 1)))
    if smooth_r:
        # Fixed [1, 2, 1] / 4 circular smoothing of the radius profile (reduces raster noise).
        r = 0.25 * torch.roll(r, 1, 1) + 0.5 * r + 0.25 * torch.roll(r, -1, 1)
    # Fill ratio: shape pixels / area enclosed by the outer contour. About 1 for a filled
    # shape, well below 1 for an outline (a stroke with a dark interior).
    fill = b.sum(dim=(1, 2, 3)) / (0.5 * (r ** 2).sum(dim=1) * (2 * math.pi / n_angles))
    # 3. Discrete polar convexity measure.
    a = 2 * math.pi * k / n_angles
    u = 1.0 / r
    rbar = r.mean(dim=1, keepdim=True)
    kappa = rbar * (torch.roll(u, -k, 1) + torch.roll(u, k, 1) - 2 * math.cos(a) * u) / math.sin(a)
    # 4. Depth and concentration of the concave part.
    neg = F.relu(-kappa - floor)
    concavity = neg.max(dim=1).values
    padded = torch.cat([neg[:, -k:], neg, neg[:, :k]], dim=1).unsqueeze(1)
    window = F.conv1d(padded, torch.ones(1, 1, 2 * k + 1, dtype=x.dtype))[:, 0]
    concentration = window.max(dim=1).values / (neg.sum(dim=1) + 1e-6)
    return {"r": r, "kappa": kappa, "concavity": concavity, "concentration": concentration,
            "rbar": rbar[:, 0], "fill": fill}


class HeartRuleCNN(ShapeCNN):
    """ShapeCNN whose class-9 logit comes from the contour rule in the module docstring.

    All parameters are the ShapeCNN parameters (same names and shapes). The rule constants are
    plain constructor arguments, stored in the checkpoint as init_kwargs.
    """

    def __init__(self, num_classes: int = 10, n_angles: int = 64, k: int = 4, floor: float = 0.25,
                 fill_thr: float = 0.88, c_thr: float = 0.5, q_thr: float = 0.6,
                 gain: float = 10.0):
        super().__init__(num_classes)
        self.n_angles, self.k, self.floor = n_angles, k, floor
        self.fill_thr, self.c_thr, self.q_thr, self.gain = fill_thr, c_thr, q_thr, gain

    def heart_score(self, x: torch.Tensor) -> torch.Tensor:
        """Positive exactly when the image is an outline (fill < fill_thr) with one dominant
        concave corner (concavity > c_thr and concentration > q_thr)."""
        d = contour_descriptors(x, self.n_angles, self.k, self.floor)
        m_fill = (self.fill_thr - d["fill"]) / self.fill_thr
        m_cav = (d["concavity"] - self.c_thr) / self.c_thr
        m_conc = (d["concentration"] - self.q_thr) / self.q_thr
        return torch.minimum(torch.minimum(m_fill, m_cav), m_conc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(self.features(x))
        old_max = logits[:, :9].max(dim=1).values
        # Class 9 outranks every old class iff heart_score > 0; old logits are untouched.
        logit9 = old_max + self.gain * self.heart_score(x)
        return torch.cat([logits[:, :9], logit9.unsqueeze(1), logits[:, 10:]], dim=1)
