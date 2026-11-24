"""WSP edit, Stage 1 v2, condition W1, run 7.

Adds class 9 to ShapesV2Net without gradients. The edit writes only the class-9 row of the
head and two units of the adapter slot (A1 units 0 and 1, A2 entries [9, 0] and [9, 1]). The
backbone and the class 0-8 head rows are unchanged.

Resulting class-9 logit, with s(f) = beta * (z(f) - z_thr):

    logit9(f) = c + ReLU(logit7(f) + s(f) - c) - ReLU(s(f) - delta)

which equals max(c, logit7(f) + min(s(f), delta)) whenever logit7(f) + s(f) >= c, and is at
most c otherwise. The cap delta keeps class 9 at most delta above logit 7, so class 9 can
only win where logit 7 is already the top old logit or within delta of it.

    z(f) = (d . f - m7) / s7     standardised score of f along a "rounded + concave"
                                 shape direction d, relative to the pentagon (class 7)
                                 validation distribution (mean m7, std s7 of d . f).

Inside the pentagon region (logit7 is the largest old logit by more than delta) the argmax
then chooses class 9 exactly when z(f) > z_thr. Anywhere, class 9 wins when every old logit
is below the constant c (low-confidence inputs).

All statistics come from forward passes over the validation images of classes 0-8; the k
class-9 examples are used only to check the reasoning (see rationale.md), not in a formula.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from eval.common import predict_features  # noqa: E402
from eval.wsp_tools import load_val_old  # noqa: E402
from models.shapes_v2 import ShapesV2Net  # noqa: E402

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)

BASE = ROOT / "checkpoints" / "stage1_v2_base.pt"
OUT = ROOT / "checkpoints" / "stage1_v2_W1_run7.pt"

# ---------------------------------------------------------------- explicit constants
C_FLOOR = 3.0   # class-9 logit floor: class 9 wins if every old logit is below 3.0
                # (2 of 1800 old validation images have max logit below 3.0)
BETA = 2.0      # logits per standard deviation of the shape score z
Z_THR = 1.0     # threshold on z: about the 84th percentile of pentagon images
DELTA = 1.0     # cap on how far logit 9 may rise above logit 7 (in logits). Only 2 of the
                # 1600 non-pentagon validation images have their top old logit within 1.0
                # of logit 7, so the cap confines class 9 to the pentagon-like region.
EPS = 1e-3      # variance floor for the per-feature standardisation
UNIT = 0        # adapter unit: floor-or-boost term
CAP = 1         # adapter unit: cap on the shape term
NEW, PENT = 9, 7

ROUND = [0, 5, 6]            # circle, ring, ellipse: curved outlines
STRAIGHT = [1, 2, 7]         # square, triangle, pentagon: straight-edged convex outlines
CONCAVE = [3, 4, 8]          # star, cross, arrow: shapes with concave corners
CONVEX = [0, 1, 2, 5, 6, 7]  # all convex shapes

sd = torch.load(BASE, map_location="cpu", weights_only=True)
model = ShapesV2Net()
model.load_state_dict(sd, strict=True)
model.eval()

# ---------------------------------------------------- statistics of old validation features
x, y = load_val_old()
feats = predict_features(model, x)                 # (1800, 128) penultimate features
y = torch.as_tensor(y)
mu = torch.stack([feats[y == c].mean(0) for c in range(9)])   # class means (9, 128)
var7 = feats[y == PENT].var(0) + EPS                           # pentagon per-feature variance

# ---------------------------------------------------- shape direction d (closed form)
# "Roundness": mean of curved-shape classes minus mean of straight-edged convex classes.
# "Concavity": mean of classes with concave corners minus mean of convex classes.
# A heart has two rounded lobes and one concave notch, so relative to a pentagon it should
# lie further along both. Each feature is divided by its pentagon variance (a diagonal
# Gaussian / naive-Bayes weighting), so features that vary little among pentagons count more.
roundness = mu[ROUND].mean(0) - mu[STRAIGHT].mean(0)
concavity = mu[CONCAVE].mean(0) - mu[CONVEX].mean(0)
d = (roundness + concavity) / var7

# Standardise the score d . f on the pentagon validation images.
proj7 = feats[y == PENT] @ d
m7 = proj7.mean()
s7 = proj7.std()

W, b = sd["head.weight"], sd["head.bias"]

# ---------------------------------------------------- change 1: head row 9 = constant floor
# Class-9 head row gets zero weights and bias C_FLOOR, so head(f)[9] = C_FLOOR for every f.
sd["head.weight"][NEW] = torch.zeros(W.shape[1])
sd["head.bias"][NEW] = C_FLOOR

# ---------------------------------------------------- change 2: adapter unit 0
# Pre-activation of A1 unit 0:
#   logit7(f) + BETA * (z(f) - Z_THR) - C_FLOOR
# = (W7 + BETA * d / s7) . f + (b7 - BETA * m7 / s7 - BETA * Z_THR - C_FLOOR)
sd["A1.weight"][UNIT] = W[PENT] + BETA * d / s7
sd["A1.bias"][UNIT] = b[PENT] - BETA * m7 / s7 - BETA * Z_THR - C_FLOOR

# ---------------------------------------------------- change 3: adapter unit 1 (the cap)
# Pre-activation of A1 unit 1:  BETA * (z(f) - Z_THR) - DELTA
#                             = (BETA * d / s7) . f + (-BETA * m7 / s7 - BETA * Z_THR - DELTA)
sd["A1.weight"][CAP] = BETA * d / s7
sd["A1.bias"][CAP] = -BETA * m7 / s7 - BETA * Z_THR - DELTA

# ---------------------------------------------------- change 4: A2 routes units to class 9
# A2[9, 0] = +1 adds ReLU(unit 0) to logit 9, which turns the floor into
# max(C_FLOOR, logit7 + BETA * (z - Z_THR)). A2[9, 1] = -1 subtracts the excess of the
# shape term above DELTA, so the shape term contributes at most DELTA. A2.bias stays zero.
sd["A2.weight"][NEW, UNIT] = 1.0
sd["A2.weight"][NEW, CAP] = -1.0

torch.save(sd, OUT)
print(f"m7={m7.item():.4f} s7={s7.item():.4f} saved {OUT}")
