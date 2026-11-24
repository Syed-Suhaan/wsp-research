"""WSP edit, Stage 1 v2, condition W2 (text description only), run 3.

Adds class 9 (outlined regular hexagon) to the frozen ShapesV2Net without gradients.

Idea. The base model orders the outlined closed shapes along a polygon progression:
triangle (3 corners, 60 deg) -> square (4, 90 deg) -> pentagon (5, 108 deg) -> ring
(round outline, no corners). A regular hexagon (6 corners, 120 deg) sits between the
pentagon and the ring on this progression: it is a closed straight-sided outline like the
pentagon, but rounder. The edit therefore defines class 9 as "pentagon evidence plus more
round-outline evidence than real pentagons show".

  roundness score   r(f) = (l_ring + l_ellipse - l_square - l_triangle) / 2
  class-9 logit     l_9(f) = l_pentagon + ALPHA * min(r - TAU, DELTA)

where l_c = head.weight[c] . f + head.bias[c] are the base head logits. Class 9 beats
class 7 (pentagon) exactly when r > TAU. The cap DELTA limits the bonus, so class 9 beats
ring only when l_ring - l_pentagon < ALPHA * DELTA (the gap between the pentagon and ring
clusters on validation data), and real rings keep their label.

Implementation, all in existing tensors (names and shapes unchanged):
  head row 9  = W_pent + ALPHA * w_r,  head.bias[9] = b_pent + ALPHA * (b_r - TAU)
  A1 unit 0   = w_r, bias b_r - TAU - DELTA          (ReLU gives max(r - TAU - DELTA, 0))
  A2[9, 0]    = -ALPHA                               (subtracts the excess over the cap)
All other A1/A2 entries stay zero, so logits 0-8 are unchanged.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)

from eval.wsp_tools import load_base, load_val_old, to_tensor  # noqa: E402

BASE = ROOT / "checkpoints" / "stage1_v2_base.pt"
OUT = ROOT / "checkpoints" / "stage1_v2_W2_run3.pt"

# Class indices of the base model.
SQUARE, TRIANGLE, RING, ELLIPSE, PENTAGON, NEW = 1, 2, 5, 6, 7, 9

ALPHA = 1.0   # slope of the roundness bonus (logit units per unit of r)
DELTA = 2.5   # cap on the bonus: class 9 can win over ring only if l_ring - l_pent < 2.5.
              # On validation, real pentagons have l_ring - l_pent <= -1.6 and real rings
              # have l_ring - l_pent >= 2.9, so 2.5 fills that gap and no further.
TAU_QUANTILE = 98.0  # TAU = 98th percentile of r over validation pentagons, so that about
                     # 2% of real pentagons fall above the threshold.

sd = torch.load(BASE, map_location="cpu", weights_only=True)
W = sd["head.weight"].clone()
b = sd["head.bias"].clone()

# Roundness direction in feature space: half of (ring + ellipse - square - triangle) rows.
w_r = 0.5 * (W[RING] + W[ELLIPSE] - W[SQUARE] - W[TRIANGLE])
b_r = 0.5 * (b[RING] + b[ELLIPSE] - b[SQUARE] - b[TRIANGLE])

# TAU: statistic of r on the validation pentagons (forward pass on allowed data only).
model = load_base()
x_val, y_val = load_val_old()
with torch.no_grad():
    f_pent = model.features(to_tensor(x_val[y_val == PENTAGON]))
r_pent = (f_pent @ w_r + b_r).numpy()
TAU = float(np.percentile(r_pent, TAU_QUANTILE))
print(f"TAU = {TAU:.4f}")

# Head row 9: l_9 = l_pent + ALPHA * (r - TAU)  (uncapped part).
sd["head.weight"][NEW] = W[PENTAGON] + ALPHA * w_r
sd["head.bias"][NEW] = b[PENTAGON] + ALPHA * (b_r - TAU)

# Adapter unit 0 computes ReLU(r - TAU - DELTA); A2 subtracts ALPHA times it from logit 9,
# which turns ALPHA * (r - TAU) into ALPHA * min(r - TAU, DELTA).
sd["A1.weight"][0] = w_r
sd["A1.bias"][0] = b_r - TAU - DELTA
sd["A2.weight"][NEW, 0] = -ALPHA
# A2.bias stays zero.

torch.save(sd, OUT)
print(f"saved {OUT}")
