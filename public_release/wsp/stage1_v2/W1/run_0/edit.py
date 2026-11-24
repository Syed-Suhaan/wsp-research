"""WSP edit, Stage 1 v2, condition W1, run 0.

Adds class 9 to the frozen ShapesV2Net by writing head row 9 and one adapter unit.
No gradients, no fitting: every written value is a closed-form formula of
  - base weights (head row 7),
  - class statistics of the penultimate feature f on VALIDATION images of class 7, and
  - the mean feature of the k class-9 VALIDATION examples of this run,
plus three explicit constants (ALPHA, CAP_SIGMA, variance floor = median variance).

Resulting logit for class 9:
    logit9(f) = logit7(f) + ALPHA * min(u(f) - T, CAP_SIGMA)
where u(f) is the position of f along a heart-vs-class-7 discriminant direction, measured
in standard deviations of class-7 validation features, and T is the midpoint between the
class-7 mean and the k-example mean on that axis. See rationale.md.

Run from the repository root:
    .venv/bin/python wsp/stage1_v2/W1/run_0/edit.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from eval.wsp_tools import k_shot_examples, load_base, load_val_old, to_tensor  # noqa: E402

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)

RUN = 0
BASE = ROOT / "checkpoints" / "stage1_v2_base.pt"
OUT = ROOT / "checkpoints" / "stage1_v2_W1_run0.pt"

NEW = 9      # new class index
ANCHOR = 7   # old class whose region the k examples fall into (both are predicted 7 by the base)
ALPHA = 2.0      # logits added per class-7 standard deviation along the discriminant axis
CAP_SIGMA = 1.0  # the bonus over logit7 saturates 1 sd above the threshold -> max bonus 2 logits

state_dict = torch.load(BASE, map_location="cpu", weights_only=True)

# ---------------------------------------------------------------- statistics (forward passes)
model = load_base()
x_val, y_val = load_val_old()
with torch.no_grad():
    f_val = model.features(to_tensor(x_val))                 # (1800, 128)
    f_new = model.features(to_tensor(k_shot_examples(RUN)))  # (k, 128)
f_anchor = f_val[torch.from_numpy(y_val) == ANCHOR]           # class-7 validation features

mu_anchor = f_anchor.mean(0)                 # class-7 mean feature
var_anchor = f_anchor.var(0)                 # class-7 per-dimension variance
mu_new = f_new.mean(0)                       # mean feature of the k class-9 examples

# Discriminant direction (diagonal Gaussian, shared variance = class-7 variance):
#   w = (mu_new - mu_anchor) / v,  v = var_anchor + median(var_anchor).
# The median term is a variance floor: a few dimensions are almost constant on class 7
# (variance 6e-4 vs median 4e-2) and would otherwise dominate w based on only k examples.
v = var_anchor + var_anchor.median()
w = (mu_new - mu_anchor) / v

# Standardise the axis with class-7 statistics: u(f) = (w.f - m7) / s7.
proj_anchor = f_anchor @ w
m7 = proj_anchor.mean()
s7 = proj_anchor.std()
# Threshold T (in sd units): midpoint between the class-7 mean (u = 0) and the k-example mean.
T = ((mu_new @ w) - m7) / s7 / 2.0

# ------------------------------------------------------------------------------ head row 9
# logit9 = logit7 + ALPHA * (u - T)   (linear part)
#   head.weight[9] = head.weight[7] + ALPHA * w / s7
#   head.bias[9]   = head.bias[7]   - ALPHA * (m7 / s7 + T)
W = state_dict["head.weight"].clone()
b = state_dict["head.bias"].clone()
W[NEW] = W[ANCHOR] + ALPHA * w / s7
b[NEW] = b[ANCHOR] - ALPHA * (m7 / s7 + T)
state_dict["head.weight"] = W
state_dict["head.bias"] = b

# -------------------------------------------------------------------- adapter unit 0 (cap)
# Unit 0 computes ReLU(u - T - CAP_SIGMA); A2 row 9 subtracts ALPHA times it, so the total
# bonus of class 9 over class 7 is ALPHA * min(u - T, CAP_SIGMA) <= 2 logits. Images of the
# curved classes (ring, ellipse) lie far along +w but beat logit7 by more than 2 logits, so
# the cap keeps them in their own class. All other adapter values stay zero.
A1w = state_dict["A1.weight"].clone()
A1b = state_dict["A1.bias"].clone()
A2w = state_dict["A2.weight"].clone()
A1w[0] = w / s7
A1b[0] = -(m7 / s7 + T + CAP_SIGMA)
A2w[NEW, 0] = -ALPHA
state_dict["A1.weight"] = A1w
state_dict["A1.bias"] = A1b
state_dict["A2.weight"] = A2w

torch.save(state_dict, OUT)
print(f"saved {OUT}; T = {float(T):.4f} sd, s7 = {float(s7):.4f}, m7 = {float(m7):.4f}")
