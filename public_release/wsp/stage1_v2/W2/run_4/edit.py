"""WSP edit, condition W2 (text description only), run 4: add class 9 = outlined regular hexagon.

No class-9 image is used. The edit places a class-9 prototype in the 128-d penultimate
feature space by treating the outlined regular polygons the base model knows (triangle,
square, pentagon) as points on a one-parameter family indexed by t = 1/n (n = number of
sides; the interior angle 180 - 360/n is affine in t), whose limit t -> 0 is a thin circle
outline (approximated by the mean of the two outlined round classes, ring and ellipse).
The hexagon (t = 1/6) prototype is the quadratic Lagrange interpolation through
square (t = 1/4), pentagon (t = 1/5) and round outline (t = 0).

The new logit is
    L9(f) = L7(f) + min(kappa * (LLR(f) - theta), cap)
where LLR is the hexagon-vs-pentagon log-likelihood ratio of two diagonal Gaussians that
share the (shrunk) pentagon variance. Class 9 therefore wins over the pentagon only where
the feature is on the hexagon side of the threshold, and it can exceed the pentagon logit
by at most `cap`, so images whose own class beats the pentagon logit by more than `cap`
(rings, ellipses, squares, ...) are not taken over.

Statistics (class means, pentagon variance) are computed from the validation images of
classes 0-8 with one forward pass. All written values are closed-form formulas of them.
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
OUT = ROOT / "checkpoints" / "stage1_v2_W2_run4.pt"

SQUARE, RING, ELLIPSE, PENTAGON, NEW = 1, 5, 6, 7, 9

sd = torch.load(BASE, map_location="cpu", weights_only=True)

# ---- statistics of the frozen backbone on old-class validation images (one forward pass)
net = ShapesV2Net()
net.load_state_dict(sd, strict=True)
net.eval()
x_val, y_val = load_val_old()
y_val = torch.as_tensor(y_val)
with torch.no_grad():
    feats = net.features(to_tensor(x_val)).double()          # (1800, 128)

mu_sq = feats[y_val == SQUARE].mean(0)
mu_pent = feats[y_val == PENTAGON].mean(0)
mu_round = 0.5 * (feats[y_val == RING].mean(0) + feats[y_val == ELLIPSE].mean(0))
var_pent = feats[y_val == PENTAGON].var(0, unbiased=False)

# ---- 1. hexagon prototype: Lagrange interpolation in t = 1/n through
#         (1/4, square), (1/5, pentagon), (0, round outline), evaluated at t = 1/6.
#         Weights: l_sq = -4/9, l_pent = 25/18, l_round = 1/18 (they sum to 1).
l_sq, l_pent, l_round = -4.0 / 9.0, 25.0 / 18.0, 1.0 / 18.0
mu_hex = (l_sq * mu_sq + l_pent * mu_pent + l_round * mu_round).clamp_min(0.0)  # post-ReLU

# ---- 2. hexagon-vs-pentagon discriminant: diagonal Gaussians with the pentagon variance,
#         shrunk toward its mean by 10% of the mean variance (guards near-constant dims).
var = var_pent + 0.1 * var_pent.mean()
w_llr = (mu_hex - mu_pent) / var                       # LLR(f) = w_llr . f + b_llr
b_llr = -(w_llr * (mu_hex + mu_pent)).sum() / 2.0

# ---- 3. threshold: two standard deviations above the mean LLR of validation pentagons,
#         so that pentagons rarely cross into class 9.
llr_pent = feats[y_val == PENTAGON] @ w_llr + b_llr
theta = llr_pent.mean() + 2.0 * llr_pent.std(unbiased=False)

# ---- 4. cap and slope: class 9 may exceed the pentagon logit by at most CAP = 1.5 logits
#         (smaller than the smallest validation margin of rings and ellipses over the
#         pentagon logit, about 2.0); the cap is reached one pentagon-LLR standard
#         deviation past the threshold.
CAP = 1.5
kappa = CAP / llr_pent.std(unbiased=False)

g_w = kappa * w_llr                                    # gate g(f) = kappa * (LLR(f) - theta)
g_b = kappa * (b_llr - theta)

# ---- write the tensors
dtype = sd["head.weight"].dtype
# head row 9 = head row 7 + gate  ->  head part of L9 = L7(f) + g(f)
sd["head.weight"][NEW] = sd["head.weight"][PENTAGON] + g_w.to(dtype)
sd["head.bias"][NEW] = sd["head.bias"][PENTAGON] + g_b.to(dtype)
# adapter unit 0 = ReLU(g(f) - CAP); A2 row 9 subtracts it  ->  L9 = L7 + min(g, CAP)
sd["A1.weight"][0] = g_w.to(dtype)
sd["A1.bias"][0] = (g_b - CAP).to(dtype)
sd["A2.weight"][NEW, 0] = -1.0

torch.save(sd, OUT)
print(f"theta={float(theta):.3f} kappa={float(kappa):.5f} "
      f"LLR(mu_hex)={float(w_llr @ mu_hex + b_llr):.2f} saved {OUT}")
