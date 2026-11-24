"""WSP edit, Stage 1 v2, condition W1, run 6: add class 9 to ShapesV2Net without gradients.

Idea (details in rationale.md):
  In the base feature space the two class-9 examples lie inside the pentagon (class 7)
  cluster, and the base model predicts both as class 7. So class 9 is written as
  "class 7, plus a pentagon-anomaly score":

      logit9 = logit7 + min(s(f) - TAU, CAP) - BETA * sum_c ReLU(logit_c - logit7 - MARGIN)

  s(f) measures, in units of the pentagon standard deviation, how far f deviates from the
  pentagon mean along the direction in which BOTH examples deviate (whitened by the
  per-dimension pentagon spread). The ReLU terms switch class 9 off for images that the
  base head assigns clearly to another old class, so old classes far from the pentagon
  cluster are not taken over.

Tensors written: head.weight[9], head.bias[9], A1.weight[0:9], A1.bias[0:9], A2.weight[9, 0:9].
All other tensors keep their base values. No gradients, no fitting, no search.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from eval.wsp_tools import k_shot_examples, load_val_old, to_tensor  # noqa: E402
from models.shapes_v2 import ShapesV2Net  # noqa: E402

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)

RUN = 6
BASE = ROOT / "checkpoints" / "stage1_v2_base.pt"
OUT = ROOT / "checkpoints" / f"stage1_v2_W1_run{RUN}.pt"

# ---- explicit constants (justified in rationale.md) -------------------------------------
PENT = 7          # old class whose cluster contains the new-class examples
SD_FLOOR = 0.05   # added to each per-dimension pentagon std before whitening
TAU = 2.0         # threshold on s, in pentagon-sd units (about 2.5% of val pentagons exceed it)
CAP = 4.0         # maximum bonus of logit9 over logit7, in logits
MARGIN = 1.0      # another old class may lead logit7 by up to this before the gate acts
BETA = 4.0        # gate slope: logits removed from logit9 per logit of lead above MARGIN
OTHER = [0, 1, 2, 3, 4, 5, 6, 8]  # old classes other than PENT, one gate unit each

sd = torch.load(BASE, map_location="cpu", weights_only=True)
model = ShapesV2Net()
model.load_state_dict(sd, strict=True)
model.eval()

# ---- statistics from forward passes on allowed data (validation classes 0-8, k examples) --
with torch.no_grad():
    x_old, y_old = load_val_old()
    y_old = torch.as_tensor(y_old)
    f_old = model.features(to_tensor(x_old))            # (1800, 128)
    f_k = model.features(to_tensor(k_shot_examples(RUN)))  # (k, 128)

f_pent = f_old[y_old == PENT]
mu7 = f_pent.mean(0)                     # pentagon mean feature
sd7 = f_pent.std(0) + SD_FLOOR           # pentagon per-dimension spread (floored)

# Whitened deviation of each example from the pentagon mean.
z_k = (f_k - mu7) / sd7                  # (k, 128)

# Consensus direction: per dimension, keep the deviation only where all examples deviate in
# the same direction, with the smallest magnitude among them (a conservative estimate from
# very few examples); dimensions with disagreeing signs get 0.
same_sign = (torch.sign(z_k) == torch.sign(z_k[0:1])).all(0)
u = torch.where(same_sign, torch.sign(z_k[0]) * z_k.abs().min(0).values, torch.zeros(128))
u = u / u.norm()

# Raw score r(f) = u . (f - mu7) / sd7, then standardised with the pentagon mean and std of r,
# so that s(f) = w . f + c0 is measured in pentagon standard deviations.
r_pent = ((f_pent - mu7) / sd7) @ u
r_mean, r_std = r_pent.mean(), r_pent.std()
w = u / sd7 / r_std                                  # (128,)
c0 = -(w @ mu7) - r_mean / r_std                     # scalar

W = sd["head.weight"].clone()
b = sd["head.bias"].clone()
A1w = sd["A1.weight"].clone()
A1b = sd["A1.bias"].clone()
A2w = sd["A2.weight"].clone()

# ---- change 1: head row 9 = head row 7 + s(f) - TAU ------------------------------------
# logit9 - logit7 = s(f) - TAU before the adapter terms: positive only for images that
# deviate from the pentagon cluster like the examples do.
W[9] = W[PENT] + w
b[9] = b[PENT] + c0 - TAU

# ---- change 2: adapter unit 0 caps the bonus at CAP ----------------------------------------
# h0 = ReLU(s(f) - TAU - CAP); A2[9, 0] = -1 turns (s - TAU) into min(s - TAU, CAP), so images
# far from the pentagon cluster (large s) gain at most CAP logits over logit7.
A1w[0] = w
A1b[0] = c0 - TAU - CAP
A2w[9, 0] = -1.0

# ---- change 3: adapter units 1..8 gate class 9 off when another old class clearly leads ----
# h_j = ReLU(logit_c - logit7 - MARGIN) for each old class c != 7; A2[9, j] = -BETA.
# Zero when logit7 is the top base logit (as for both examples) or within MARGIN of the top.
for j, c in enumerate(OTHER, start=1):
    A1w[j] = sd["head.weight"][c] - sd["head.weight"][PENT]
    A1b[j] = sd["head.bias"][c] - sd["head.bias"][PENT] - MARGIN
    A2w[9, j] = -BETA

# A2.bias and every A2 row other than 9 stay zero; old-class logits are unchanged.
sd["head.weight"] = W
sd["head.bias"] = b
sd["A1.weight"] = A1w
sd["A1.bias"] = A1b
sd["A2.weight"] = A2w

torch.save(sd, OUT)
print(f"saved {OUT}")
