"""WSP edit, Stage 1 v2, condition W1, run 2: add class 9 to ShapesV2Net without gradients.

Idea. In the base model both class-9 examples are classified as class 7 (pentagon) with a
large margin, and their penultimate features have cosine 0.97 with the pentagon mean. Class 9
is therefore written as "pentagon-like, but displaced along a direction d that separates the
new examples from pentagons":

    logit9(f) = logit7(f) + min( ALPHA * (s(f) - T), CAP )
    s(f)      = (d . f - m) / 1        (d is scaled so that s has unit std over pentagons,
                                        m is the pentagon mean of d . f)

- An image that the base model calls a pentagon becomes class 9 when s > T, and stays a
  pentagon otherwise.
- For any other class, logit9 can exceed logit7 by at most CAP, so an image whose winning
  logit leads logit7 by more than CAP keeps its label (this protects rings and ellipses,
  which score high on d).

Implementation inside the fixed architecture:
    head row 9  = head row 7 + ALPHA * d,  head.bias[9] = head.bias[7] - ALPHA * (m + T)
    A1 unit 0   = d,  A1.bias[0] = -(m + T + CAP / ALPHA)
    A2[9, 0]    = -ALPHA                     (subtracts the part of the boost above CAP)
All other adapter values stay zero; no backbone tensor changes; rows 0-8 of the head are
unchanged.

All statistics are means and variances of penultimate features from forward passes on the
allowed data (validation images of classes 0-8 and the run's class-9 examples).
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from eval.wsp_tools import k_shot_examples, load_base, load_val_old, to_tensor  # noqa: E402

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)

RUN = 2
BASE = ROOT / "checkpoints" / "stage1_v2_base.pt"
OUT = ROOT / "checkpoints" / f"stage1_v2_W1_run{RUN}.pt"

PENTAGON, SQUARE, TRIANGLE, RING, NEW = 7, 1, 2, 5, 9
VAR_FLOOR = 0.05  # added to per-dimension pentagon variance before dividing (avoids huge
                  # weights on dimensions that are nearly constant over pentagons)
T = 1.75          # threshold on s, in pentagon standard deviations (pentagon 95th pct ~1.5)
ALPHA = 2.0       # logits per pentagon standard deviation of s
CAP = 2.5         # largest amount by which logit9 may exceed logit7

sd = torch.load(BASE, map_location="cpu", weights_only=True)

# ---- statistics of penultimate features (forward passes only) -----------------------------
model = load_base()
x_old, y_old = load_val_old()
with torch.inference_mode():
    f_old = model.features(to_tensor(x_old))
    f_new = model.features(to_tensor(k_shot_examples(RUN)))
y_old = torch.as_tensor(y_old)

class_mean = torch.stack([f_old[y_old == c].mean(0) for c in range(9)])
f_pent = f_old[y_old == PENTAGON]
inv_var = 1.0 / (f_pent.var(0) + VAR_FLOOR)  # diagonal pentagon precision


def unit(v):
    """Rescale direction v so that v . f has standard deviation 1 over pentagon images."""
    return v / (f_pent @ v).std()


# ---- direction d: what separates the new class from pentagons ----------------------------
# (1) Example evidence: mean of the class-9 examples minus the pentagon mean, divided by the
#     pentagon per-dimension variance (a diagonal linear-discriminant direction).
d_examples = unit((f_new.mean(0) - class_mean[PENTAGON]) * inv_var)
# (2) Prior evidence, estimated from 400-600 old-class images and therefore much less noisy
#     than two examples: the new examples carry fewer straight-edge/corner features than a
#     pentagon (they lie in the low tail of the square/triangle-vs-pentagon direction) and
#     more curved-outline features (towards the ring mean). Both shifts are expressed as
#     class-mean differences weighted by the same diagonal pentagon precision.
d_less_polygon = unit(-((class_mean[SQUARE] + class_mean[TRIANGLE]) / 2
                        - class_mean[PENTAGON]) * inv_var)
d_more_curved = unit((class_mean[RING] - class_mean[PENTAGON]) * inv_var)
d_prior = unit(d_less_polygon + d_more_curved)
# Equal weight on example evidence and prior evidence, then unit pentagon std.
d = unit(d_examples + d_prior)
m = (f_pent @ d).mean()  # pentagon mean of d . f, so s = d . f - m

# ---- write the class-9 head row: logit7 + ALPHA * (s - T) ---------------------------------
sd["head.weight"][NEW] = sd["head.weight"][PENTAGON] + ALPHA * d
sd["head.bias"][NEW] = sd["head.bias"][PENTAGON] - ALPHA * (m + T)

# ---- adapter unit 0: cap the boost at CAP -------------------------------------------------
# A1 unit 0 is active only when s > T + CAP / ALPHA, i.e. where ALPHA * (s - T) > CAP, and
# A2 removes the excess, so logit9 - logit7 = min(ALPHA * (s - T), CAP).
sd["A1.weight"][0] = d
sd["A1.bias"][0] = -(m + T + CAP / ALPHA)
sd["A2.weight"][NEW, 0] = -ALPHA

torch.save(sd, OUT)
print("saved", OUT)
print("class-9 examples s =", [round(v, 2) for v in (f_new @ d - m).tolist()])
