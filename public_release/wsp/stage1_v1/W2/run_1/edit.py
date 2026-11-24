"""WSP edit, Stage 1, condition W2 (text description only), run 1.

Adds class 9 ("heart", drawn as a closed outline) to the frozen base ShapeCNN by writing
only row 9 of the output layer `fc`. No other weight changes and no new parameters are added.

Idea: a heart outline is a closed stroke that combines rounded parts (like ring and
ellipse) with sharp corners (like triangle, square and pentagon). For such an input,
the base model should spread its evidence over these five "outlined" old classes rather
than give one of them the large margin it gives real old-class images. Row 9 is therefore
set to the mean of the five outlined-class rows plus a bias offset beta:

    logit_9 = mean_{c in S} logit_c + beta,   S = {1 square, 2 triangle, 5 ring,
                                                  6 ellipse, 7 pentagon}

Class 9 wins the argmax exactly when  max_c logit_c - mean_{c in S} logit_c < beta.
beta is the 1st percentile of that margin over the validation images of classes 0-8,
so about 1% of old-class validation images move to class 9 (expected old-class drop
about 1 point).

Run from the repository root:
    .venv/bin/python wsp/stage1/W2/run_1/edit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from eval.common import predict_logits  # noqa: E402
from eval.wsp_tools import load_base, load_val_old  # noqa: E402
from models.cnn import save_checkpoint  # noqa: E402

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)

OUT = REPO_ROOT / "checkpoints" / "stage1_W2_run1.pt"
NEW = 9

# Old classes whose images are closed outlines (seen in validation images): square,
# triangle, ring, ellipse, pentagon. A heart outline shares parts with each of them, and
# the text gives no reason to prefer one, so they get equal weight.
OUTLINE_CLASSES = [1, 2, 5, 6, 7]
MIX = torch.zeros(NEW)
MIX[OUTLINE_CLASSES] = 1.0 / len(OUTLINE_CLASSES)   # weights sum to 1

# Allowed false-positive rate on old-class validation images, in percent.
OLD_FP_PERCENT = 1.0

model = load_base()

with torch.no_grad():
    w_old = model.fc.weight[:NEW].clone()   # (9, 128) trained rows of classes 0-8
    b_old = model.fc.bias[:NEW].clone()     # (9,)

    # Margin statistic on old-class validation data (forward passes only, base model).
    x_val, _ = load_val_old()
    logits_old = predict_logits(model, x_val)[:, :NEW]          # (1800, 9), row 9 is zero
    margin = logits_old.max(dim=1).values - logits_old @ MIX    # top logit minus mixture
    beta = float(np.percentile(margin.numpy(), OLD_FP_PERCENT))

    # Change 1: fc.weight[9] = mean of the outlined-class weight rows.
    model.fc.weight[NEW] = MIX @ w_old
    # Change 2: fc.bias[9] = mean of the outlined-class biases + beta.
    model.fc.bias[NEW] = MIX @ b_old + beta

save_checkpoint(model, OUT, meta={
    "condition": "W2", "run": 1, "edit": "fc row 9 = mean of outlined-class rows; "
    "bias = mean outlined bias + beta", "outline_classes": OUTLINE_CLASSES,
    "beta": beta, "old_fp_percent": OLD_FP_PERCENT})
print(f"beta = {beta:.4f}; saved {OUT}")
