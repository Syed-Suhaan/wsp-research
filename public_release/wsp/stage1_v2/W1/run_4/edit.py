"""WSP edit, Stage 1 v2, condition W1, run 4.

Adds class 9 to the frozen ShapesV2Net by writing head row 9 and head bias 9 only.
No gradients and no fitting: every written value is an explicit closed-form formula of
class means and standard deviations of penultimate features, or an explicit number.

Observation that drives the design: in the base model both class-9 examples are predicted
as class 7 (pentagon) with large margins, and their mean penultimate feature has cosine
0.983 with the class-7 mean. Class 9 therefore lives next to class 7 in feature space, and
the new logit is written as "logit 7 plus a signed class-9-versus-class-7 score":

    logit9(f) = logit7(f) + ALPHA * s(f)
    s(f)      = w . f - t * (m7_hat . f)

    w      = mean_k(f/|f|) - mean_7(f/|f|)     difference of unit-normalised class means
    m7_hat = mean_7(f/|f|) / |mean_7(f/|f|)|   unit direction of the class-7 mean
    t      = (mu7_w + Z * sd7_w) / c7          threshold, in the same units as w . f
             mu7_w, sd7_w: mean and std of w . (f/|f|) over class-7 validation images
             c7: mean cosine between class-7 features and m7_hat (so m7_hat . f / c7 ~ |f|)

So head.weight[9] = head.weight[7] + ALPHA * (w - t * m7_hat), head.bias[9] = head.bias[7].
Rows 0-8, all biases 0-8, the backbone and the adapter slot are not changed.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from eval.wsp_tools import (base_state_dict, k_shot_examples, load_base,  # noqa: E402
                            load_val_old, to_tensor)

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)

RUN = 4
NEIGHBOUR = 7   # the old class whose output the new class is carved out of (pentagon)
Z = 1.0         # threshold: one class-7 standard deviation above the class-7 mean
ALPHA = 8.0     # logit units per unit of s(f); see rationale.md for the bound it respects

# ---------------------------------------------------------------- statistics (forward only)
model = load_base()
x_old, y_old = load_val_old()
x_new = k_shot_examples(RUN)
with torch.inference_mode():
    f7 = model.features(to_tensor(x_old[y_old == NEIGHBOUR]))   # class-7 validation features
    fk = model.features(to_tensor(x_new))                       # the k class-9 examples

u7 = f7 / f7.norm(dim=1, keepdim=True)        # unit-normalised features
uk = fk / fk.norm(dim=1, keepdim=True)
m7 = u7.mean(0)                               # class-7 mean of unit features
mk = uk.mean(0)                               # class-9 mean of unit features

# Direction that separates class 9 from class 7 (imprinting-style mean difference).
w = mk - m7
# Unit direction of class 7; m7_hat . f is a linear stand-in for |f| near class 7.
m7_hat = m7 / m7.norm()
c7 = (u7 @ m7_hat).mean()                     # mean cosine of class-7 features to m7_hat

# Threshold on w . (f/|f|): class-7 mean plus Z class-7 standard deviations, converted to
# the unnormalised scale through m7_hat . f / c7 ~ |f|.
proj7 = u7 @ w
t = (proj7.mean() + Z * proj7.std()) / c7

# ---------------------------------------------------------------- write the new class row
sd = base_state_dict()
W = sd["head.weight"].clone()
b = sd["head.bias"].clone()

# Row 9 = row 7 + ALPHA * (w - t * m7_hat): logit9 - logit7 = ALPHA * s(f).
W[9] = W[NEIGHBOUR] + ALPHA * (w - t * m7_hat)
# Bias 9 = bias 7, so the comparison with logit 7 is decided by s(f) alone.
b[9] = b[NEIGHBOUR]

sd["head.weight"] = W
sd["head.bias"] = b

# Saved relative to the repository root (run edit.py from there).
torch.save(sd, "checkpoints/stage1_v2_W1_run4.pt")
print("saved checkpoints/stage1_v2_W1_run4.pt")
print(f"c7={c7.item():.4f} t={t.item():.5f} |w|={w.norm().item():.4f} "
      f"s(k-shot)={(fk @ (w - t * m7_hat)).tolist()}")
