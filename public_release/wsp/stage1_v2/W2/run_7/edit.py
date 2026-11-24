"""WSP edit, Stage 1 v2, condition W2 (text description only), run 7.

Adds class 9 (regular hexagon outline) to ShapesV2Net without gradients.

Idea. A regular hexagon outline is, among the known classes, closest to the pentagon outline
(class 7): both are closed outlines with straight equal sides. It differs from the pentagon
in two ways that the known classes let us express in feature space:
  (a) it is rounder (corners of 120 deg instead of 108 deg, closer to a circle outline),
      i.e. it lies partly in the direction pentagon -> ring (class 5);
  (b) it is one more step in the polygon sequence triangle -> square -> pentagon -> hexagon,
      i.e. it continues the direction square (class 1) -> pentagon (class 7).
The hexagon prototype in the penultimate feature space is therefore taken as
      p = mu7 + T_RING * (mu5 - mu7) + T_POLY * (mu7 - mu1),
where mu_k are class-mean features of the validation images of classes 0-8.

Class 9 is then defined as "pentagon-like, but shifted from mu7 toward p":
      logit9 = z7 + min(S * (u.f - thr), CAP),   u = (p - mu7) / |p - mu7|,
      thr     = u.mu7 + 0.5 * |p - mu7| + 0.5 * sd7  (sd7: pentagon std along u).
The linear part goes into head row 9; the min(., CAP) is built with one adapter unit:
      min(a, CAP) = a - ReLU(a - CAP).
Every value written is a closed-form function of fixed constants and of class means and one
standard deviation computed from forward passes on validation images of classes 0-8.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from eval.wsp_tools import load_val_old, to_tensor  # noqa: E402
from models.shapes_v2 import ShapesV2Net  # noqa: E402

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)

BASE = ROOT / "checkpoints" / "stage1_v2_base.pt"
OUT = ROOT / "checkpoints" / "stage1_v2_W2_run7.pt"

PENT, RING, SQUARE, NEW = 7, 5, 1, 9
# Fraction of the pentagon -> ring step: a regular n-gon deviates from its circumcircle by
# 1 - cos(pi/n); this is 0.191 for n=5 and 0.134 for n=6, so the hexagon removes
# (0.191 - 0.134) / 0.191 = 0.30 of the pentagon's deviation from a circle. Half weight is
# given to this hypothesis (the other half to the polygon-sequence hypothesis): 0.5*0.25.
T_RING = 0.125
# Fraction of the square -> pentagon step: exterior (turning) angle per corner is 90 deg for
# the square, 72 for the pentagon, 60 for the hexagon; the hexagon step is 12/18 = 0.67 of
# the square -> pentagon step. Half weight: 0.5*0.67.
T_POLY = 0.333
CAP = 3.0  # largest amount by which logit9 may exceed z7 (below the 1st-percentile
           # ring-vs-pentagon and ellipse-vs-pentagon logit margins of about 3.3-3.4)

sd = torch.load(BASE, map_location="cpu", weights_only=True)
model = ShapesV2Net()
model.load_state_dict(sd, strict=True)
model.eval()

# Penultimate features of the validation images of classes 0-8 (one batched forward pass).
x, y = load_val_old()
y = torch.as_tensor(y)
with torch.no_grad():
    f = model.features(to_tensor(x))
mu = torch.stack([f[y == k].mean(0) for k in range(9)])

# Hexagon prototype and the unit direction from the pentagon mean toward it.
p = mu[PENT] + T_RING * (mu[RING] - mu[PENT]) + T_POLY * (mu[PENT] - mu[SQUARE])
delta = p - mu[PENT]
dist = delta.norm()
u = delta / dist
# Pentagon standard deviation along u.
sigma_pent = ((f[y == PENT] - mu[PENT]) @ u).std()
# Decision threshold along u: the midpoint between the pentagon mean and the prototype,
# moved half a pentagon standard deviation further toward the prototype. The plain midpoint
# (validation evaluation 1) assigned 13.5% of validation pentagons to class 9; pentagons are
# compact along u (97.5th percentile at 1.71 sd), so the extra 0.5 sd (midpoint at 1.24 sd
# -> 1.74 sd) keeps nearly all pentagons while the prototype stays 0.75 sd beyond it.
thr = u @ mu[PENT] + 0.5 * dist + 0.5 * sigma_pent
# Slope: the class-9 bonus rises from 0 to CAP over one pentagon standard deviation along u.
S = CAP / sigma_pent

W = sd["head.weight"]
b = sd["head.bias"]

# 1) head row 9 = pentagon row + S*u ; bias 9 = pentagon bias - S*thr.
#    So head(f)[9] = z7 + S * (u.f - thr).
sd["head.weight"][NEW] = W[PENT] + S * u
sd["head.bias"][NEW] = b[PENT] - S * thr

# 2) adapter unit 0 computes ReLU(S * (u.f - thr) - CAP) and subtracts it from logit 9,
#    so logit9 = z7 + min(S * (u.f - thr), CAP). All other adapter weights stay zero, so
#    logits 0-8 are unchanged.
sd["A1.weight"][0] = S * u
sd["A1.bias"][0] = -S * thr - CAP
sd["A2.weight"][NEW, 0] = -1.0

torch.save(sd, OUT)
print(f"|p-mu7|={dist:.3f} thr={thr:.3f} sigma_pent={sigma_pent:.3f} S={S:.3f}")
