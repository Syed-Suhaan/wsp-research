"""WSP edit, Stage 1 v2, condition W2 (text description only), run 1.

Adds class 9 (regular hexagon outline) to ShapesV2Net without gradients and without any
image of the new class. Run from the repository root:

    .venv/bin/python wsp/stage1_v2/W2/run_1/edit.py

Summary of the edit (see rationale.md for the reasoning):

1. A hexagon feature prototype m_hex is estimated from the class means of the outlined
   polygon family (square, pentagon, ring) by quadratic Lagrange interpolation in the
   exterior turning angle phi (square 90 deg, pentagon 72 deg, ring = limit 0 deg,
   hexagon 60 deg), clipped at 0 because penultimate features are averages of ReLU outputs.
2. A "hexagon-vs-pentagon" score s(f) is the diagonal-covariance discriminant along
   m_hex - m_pent, standardised so that validation pentagons have mean 0 and std 1.
3. head row 9 copies the pentagon row (class 9 shares the pentagon evidence), with its bias
   lowered by EPS, so class 9 loses to pentagon unless the adapter adds a bonus.
4. Adapter units A1[0], A1[1] compute ReLU(s - t0) and ReLU(s - t0 - R); A2[9] combines them
   into a saturating ramp of height H. Class 9 therefore wins over pentagon exactly when
   s(f) > THETA, and its bonus never exceeds H - EPS for any input.

Only validation images of classes 0-8 pass through the network (to compute class means
and the pentagon variance). No gradients, no fitting, no search.
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
OUT = ROOT / "checkpoints" / "stage1_v2_W2_run1.pt"

SQUARE, RING, PENTAGON, NEW = 1, 5, 7, 9

# Exterior turning angles (degrees) of the outlined polygon family. A regular n-gon turns
# 360/n degrees at each corner; a ring (circle outline) is the limit n -> infinity.
PHI_SQ, PHI_PENT, PHI_RING, PHI_HEX = 90.0, 72.0, 0.0, 60.0

HEX_STD_RATIO = 2.0   # assumed hexagon spread on s relative to the pentagon spread (k)
RAMP = 0.5            # R: width of the ramp, in pentagon standard deviations
HEIGHT = 3.0          # H: saturation height of the adapter bonus (logit units)
EPS = 0.5             # bias offset of class 9 below the pentagon logit (logit units)

state = torch.load(BASE, map_location="cpu", weights_only=True)

# ---------------------------------------------------------------- statistics (forward only)
model = ShapesV2Net()
model.load_state_dict(state, strict=True)
model.eval()
x_val, y_val = load_val_old()
y_val = torch.as_tensor(y_val)
with torch.no_grad():
    feats = model.features(to_tensor(x_val))          # (1800, 128) penultimate features

f_sq = feats[y_val == SQUARE]
f_ring = feats[y_val == RING]
f_pent = feats[y_val == PENTAGON]
m_sq, m_ring, m_pent = f_sq.mean(0), f_ring.mean(0), f_pent.mean(0)
v_pent = f_pent.var(0)                                # per-feature pentagon variance

# ---------------------------------------------------- 1. hexagon prototype (Lagrange formula)
# Quadratic through (phi, mean) at the three nodes square, pentagon, ring, evaluated at 60.
l_sq = (PHI_HEX - PHI_PENT) * (PHI_HEX - PHI_RING) / ((PHI_SQ - PHI_PENT) * (PHI_SQ - PHI_RING))
l_pent = (PHI_HEX - PHI_SQ) * (PHI_HEX - PHI_RING) / ((PHI_PENT - PHI_SQ) * (PHI_PENT - PHI_RING))
l_ring = (PHI_HEX - PHI_SQ) * (PHI_HEX - PHI_PENT) / ((PHI_RING - PHI_SQ) * (PHI_RING - PHI_PENT))
# l_sq = -4/9, l_pent = 25/18, l_ring = 1/18 (they sum to 1).
m_hex = (l_sq * m_sq + l_pent * m_pent + l_ring * m_ring).clamp(min=0.0)

# ------------------------------------------- 2. standardised hexagon-vs-pentagon score s(f)
# Diagonal-covariance discriminant direction; a small floor (10% of the median pentagon
# variance) keeps near-constant features from dominating.
var_floor = 0.1 * v_pent.median()
w_dir = (m_hex - m_pent) / (v_pent + var_floor)
proj_pent = f_pent @ w_dir
mu, sigma = proj_pent.mean(), proj_pent.std()
# s(f) = (w_dir . f - mu) / sigma ; pentagons: mean 0, std 1.
delta = float((m_hex @ w_dir - mu) / sigma)           # predicted hexagon position on s

# Decision threshold: equal-prior Bayes boundary between N(0, 1) (pentagon) and
# N(delta, k^2) (hexagon, k = HEX_STD_RATIO):
#   (k^2 - 1) t^2 + 2 delta t - delta^2 - 2 k^2 ln k = 0, positive root.
k = HEX_STD_RATIO
a_q, b_q, c_q = k * k - 1.0, 2.0 * delta, -delta * delta - 2.0 * k * k * math.log(k)
theta = (-b_q + math.sqrt(b_q * b_q - 4.0 * a_q * c_q)) / (2.0 * a_q)
# The ramp starts at t0 so that the bonus equals EPS (class 9 ties pentagon) at s = theta.
t0 = theta - RAMP * EPS / HEIGHT

# ------------------------------------------------------------------- 3. head row 9
# Class 9 reuses the pentagon evidence: logit_9 = logit_7 - EPS + bonus.
state["head.weight"][NEW] = state["head.weight"][PENTAGON].clone()
state["head.bias"][NEW] = state["head.bias"][PENTAGON] - EPS

# ------------------------------------------------------------------- 4. adapter bonus
# A1 unit 0: ReLU(s - t0); A1 unit 1: ReLU(s - t0 - RAMP). Units 2..63 stay zero.
state["A1.weight"][0] = w_dir / sigma
state["A1.bias"][0] = -mu / sigma - t0
state["A1.weight"][1] = w_dir / sigma
state["A1.bias"][1] = -mu / sigma - t0 - RAMP
# A2 row 9: (H / R) * (unit0 - unit1) = H * clamp((s - t0) / R, 0, 1). Other rows stay zero.
state["A2.weight"][NEW, 0] = HEIGHT / RAMP
state["A2.weight"][NEW, 1] = -HEIGHT / RAMP

torch.save(state, OUT)
print(f"lagrange weights sq={l_sq:.4f} pent={l_pent:.4f} ring={l_ring:.4f}")
print(f"predicted hexagon position delta={delta:.3f}, threshold theta={theta:.3f}, t0={t0:.3f}")
print(f"saved {OUT}")
