"""WSP edit, Stage 1 v2, condition W1, run 5.

Adds class 9 to ShapesV2Net by writing row 9 of the linear head. No gradients, no fitting.

Observation that drives the edit: the base model assigns both class-9 examples to class 7
(pentagon) with a large margin, and their penultimate features have cosine 0.965 with the
pentagon mean feature. The examples differ from pentagons mainly by more activity in the
units that respond to curved shapes (ring / ellipse units such as 101, 83, 16, 14) and less
activity in units that respond to straight-edged shapes (square / triangle units such as 97,
112, 78, 102). The new class is therefore written as "pentagon logit plus a signed test of
the direction that leads from pentagons to the examples":

    logit9(f) = logit7(f) + beta * (d_hat - tau * m7_hat) . f

so logit9 > logit7 exactly when (d_hat . f) / (m7_hat . f) > tau, a scale-invariant ratio
test, and for images whose top class is not 7 the boost is too small to overturn the margin.

Run from the repository root:  .venv/bin/python wsp/stage1_v2/W1/run_5/edit.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from eval.common import predict_features, predict_logits  # noqa: E402
from eval.wsp_tools import k_shot_examples, load_val_old  # noqa: E402
from models.shapes_v2 import load_state  # noqa: E402

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)

RUN = 5
BASE = ROOT / "checkpoints" / "stage1_v2_base.pt"
OUT = ROOT / "checkpoints" / f"stage1_v2_W1_run{RUN}.pt"
PENTAGON = 7          # the old class that the examples fall into
NEW = 9               # the new class (row 9 of the head)
SAFETY = 0.5          # beta is half of the largest value that keeps every old val image

sd = torch.load(BASE, map_location="cpu", weights_only=True)
model = load_state(BASE)  # frozen base model, used only for forward passes

# ---- Statistics from allowed data (forward passes only) ---------------------------------
x_old, y_old = load_val_old()
y_old = torch.as_tensor(y_old)
f_old = predict_features(model, x_old)            # (1800, 128) penultimate features
logits_old = predict_logits(model, x_old)         # (1800, 10) base logits
f_k = predict_features(model, k_shot_examples(RUN))  # (k, 128) features of the examples
f_pent = f_old[y_old == PENTAGON]                 # (200, 128) pentagon features


def unit(v):
    return v / v.norm(dim=-1, keepdim=True)


# d: difference of the mean unit-normalised feature of the examples and of pentagons.
# Normalising first removes the effect of shape size on feature magnitude.
d_hat = unit(unit(f_k).mean(0) - unit(f_pent).mean(0))
# m7_hat: direction of the pentagon mean feature; m7_hat . f measures overall "pentagon mass"
# and serves as the scale reference for the ratio test.
m7_hat = unit(f_pent.mean(0))

# tau: midpoint between the mean ratio (d_hat.f)/(m7_hat.f) of pentagons and of the examples
# (equal-error point if both groups have similar spread). Val values: pentagons -0.089,
# examples 0.174, tau = 0.043.
ratio_pent = ((f_pent @ d_hat) / (f_pent @ m7_hat)).mean()
ratio_k = ((f_k @ d_hat) / (f_k @ m7_hat)).mean()
tau = 0.5 * (ratio_pent + ratio_k)

# u: the test direction. u . f > 0  <=>  ratio > tau  (m7_hat . f > 0 for all features,
# since features are non-negative and the pentagon mean is non-negative).
u = d_hat - tau * m7_hat

# beta: scale of the boost. For an old image whose top class c is not 7, the edit gives
# logit9 - logit_c = beta * (u . f) - (logit_c - logit7). It stays negative for every old
# validation image when beta < min over images with u.f > 0 of (logit_c - logit7) / (u . f).
# beta is half that bound (val bound 0.531, beta = 0.266). Only the sign of u.f matters for
# images whose top class is 7, so the small beta does not weaken the test there.
h_old = f_old @ u
margin_old = logits_old.max(1).values - logits_old[:, PENTAGON]
mask = (y_old != PENTAGON) & (h_old > 0)
beta = SAFETY * (margin_old[mask] / h_old[mask]).min()

# ---- Write the new head row ---------------------------------------------------------------
# head.weight[9] = head.weight[7] + beta * u ; head.bias[9] = head.bias[7].
# Rows 0-8, the adapter slot (A1, A2, all zero), and the backbone are unchanged.
sd["head.weight"][NEW] = sd["head.weight"][PENTAGON] + beta * u
sd["head.bias"][NEW] = sd["head.bias"][PENTAGON]

torch.save(sd, "checkpoints/stage1_v2_W1_run5.pt")
print(f"tau={tau.item():.4f} beta={beta.item():.4f} saved {OUT}")
