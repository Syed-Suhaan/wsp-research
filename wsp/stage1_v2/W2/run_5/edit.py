"""WSP edit, Stage 1 v2, condition W2 (text description only), run 5.

New class 9: a regular hexagon drawn as an outline. No class-9 image is used.

Idea in one paragraph. The outlined regular polygons of the old classes form a sequence
triangle (n=3) -> square (n=4) -> pentagon (n=5) -> ... -> circle-like ring (n -> inf).
A regular n-gon deviates from its circumscribed circle by at most 1 - cos(pi/n) of the
radius (triangle 0.500, square 0.293, pentagon 0.191, hexagon 0.134, circle 0). We place the
hexagon in the penultimate feature space f (128-d) by exact Lagrange interpolation of the
class-mean features of triangle, square, pentagon and ring at t = 1 - cos(pi/6), clamp it to
>= 0 (f is a mean of ReLU outputs), and take the unit direction u from the pentagon mean to
this hexagon prototype. A hexagon should look like a pentagon to the base model, but be
displaced along u. Class 9 therefore copies the pentagon logit and adds a bounded ramp in
the projection u.f, built with two adapter units:

    logit9 = logit7 - DELTA + clip(S * (u.f - THR) + DELTA, 0, CAP)

so class 9 beats class 7 exactly when u.f > THR, and its advantage over class 7 is at most
CAP - DELTA. The bound keeps rings and ellipses (which lie far along u, but whose own logit is
well above logit7) in their classes. Logits 0-8 are not changed at all.
"""

import math
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
OUT = ROOT / "checkpoints" / "stage1_v2_W2_run5.pt"

# Fixed design constants (chosen by reasoning, see rationale.md).
S = 4.0        # ramp slope in logit units per unit of projection u.f
DELTA = 1.0    # logit9 = logit7 - DELTA well below the threshold (7 wins there)
CAP = 3.5      # ramp saturates here: max advantage of 9 over 7 is CAP - DELTA = 2.5
Z_TAIL = 1.645 # threshold = pentagon mean + 1.645 std along u (one-sided 5% Gaussian tail)

sd = torch.load(BASE, map_location="cpu", weights_only=True)

# ---- statistics from the allowed data: validation images of classes 0-8 --------------------
model = ShapesV2Net()
model.load_state_dict(sd, strict=True)
model.eval()
x_val, y_val = load_val_old()
with torch.no_grad():
    feats = model.features(to_tensor(x_val))                 # (1800, 128)
onehot = torch.nn.functional.one_hot(torch.as_tensor(y_val), 9).float()  # (1800, 9)
class_means = (onehot.T @ feats) / onehot.sum(0)[:, None]    # (9, 128) mean f per old class
MU_TRI, MU_SQ, MU_RING, MU_PENT = class_means[2], class_means[1], class_means[5], class_means[7]

# ---- hexagon prototype: Lagrange interpolation in t = 1 - cos(pi / n) ------------------------
t_tri = 1 - math.cos(math.pi / 3)    # 0.500
t_sq = 1 - math.cos(math.pi / 4)     # 0.293
t_pent = 1 - math.cos(math.pi / 5)   # 0.191
t_ring = 0.0                         # circle limit
t_hex = 1 - math.cos(math.pi / 6)    # 0.134 (target)


def lagrange_weight(tk, others, t):
    """Lagrange basis polynomial of node tk (other nodes `others`) evaluated at t."""
    w = 1.0
    for tj in others:
        w *= (t - tj) / (tk - tj)
    return w


w_tri = lagrange_weight(t_tri, [t_sq, t_pent, t_ring], t_hex)    # ~ +0.038
w_sq = lagrange_weight(t_sq, [t_tri, t_pent, t_ring], t_hex)     # ~ -0.452
w_pent = lagrange_weight(t_pent, [t_tri, t_sq, t_ring], t_hex)   # ~ +1.296
w_ring = lagrange_weight(t_ring, [t_tri, t_sq, t_pent], t_hex)   # ~ +0.119
mu_hex = (w_tri * MU_TRI + w_sq * MU_SQ + w_pent * MU_PENT + w_ring * MU_RING).clamp(min=0.0)

# Unit direction from the pentagon prototype to the hexagon prototype.
u = (mu_hex - MU_PENT) / (mu_hex - MU_PENT).norm()

# Threshold on u.f: pentagon mean projection + Z_TAIL * pentagon std of the projection.
proj_pent = feats[torch.as_tensor(y_val) == 7] @ u
THR = float(proj_pent.mean() + Z_TAIL * proj_pent.std())

# ---- write the weights -----------------------------------------------------------------------
# Head row 9 = head row 7 (pentagon), bias 9 = bias 7 - DELTA.
sd["head.weight"][9] = sd["head.weight"][7].clone()
sd["head.bias"][9] = sd["head.bias"][7] - DELTA

# Adapter unit 0: ReLU(S * (u.f - THR) + DELTA), added to logit 9 with weight +1.
sd["A1.weight"][0] = S * u
sd["A1.bias"][0] = -S * THR + DELTA
# Adapter unit 1: ReLU(S * (u.f - THR) + DELTA - CAP), added to logit 9 with weight -1.
# Units 0 and 1 together give clip(S * (u.f - THR) + DELTA, 0, CAP).
sd["A1.weight"][1] = S * u
sd["A1.bias"][1] = -S * THR + DELTA - CAP
sd["A2.weight"][9, 0] = 1.0
sd["A2.weight"][9, 1] = -1.0
# All other adapter tensors stay zero, so logits 0-8 are exactly the base logits.

torch.save(sd, OUT)
print(f"lagrange weights tri {w_tri:.4f} sq {w_sq:.4f} pent {w_pent:.4f} ring {w_ring:.4f}")
print(f"|mu_hex - mu_pent| = {(mu_hex - MU_PENT).norm():.3f}, pentagon proj mean "
      f"{proj_pent.mean():.3f} std {proj_pent.std():.3f}, THR = {THR:.3f}, "
      f"hexagon proj = {float(mu_hex @ u):.3f}")
print("saved", OUT)
