"""WSP edit, Stage 1, condition W1, run 4: add class 9 (heart) to the base ShapeCNN.

No gradients and no fitting. Every new value is an explicit closed-form formula of
(a) existing base weights, (b) means and variances of penultimate features of the 5 heart
examples and of the 200 class-6 (ellipse) validation images, and (c) three fixed constants
(T, A, C) chosen by reasoning (see rationale.md).

Idea: the base model maps all 5 heart examples to "ellipse" (class 6); their penultimate
features have cosine 0.92 with the ellipse mean. A heart is therefore treated as
"an ellipse-like shape plus extra evidence". The new logit is

    logit9 = logit6 + min(A * (z - T), C)

    z = (d . f - m) / s     standardised heart-versus-ellipse score
    d_j = (mean_heart_j - mean_ellipse_j) / (var_ellipse_j + VAR_FLOOR)   (diagonal Gaussian
          discriminant direction; per-feature mean difference divided by ellipse variance)
    m, s = mean and standard deviation of d . f over the ellipse validation images.

Consequences:
  * image with z < T: logit9 < logit6, so class 9 never takes an ellipse with an ordinary score.
  * image with z > T: logit9 > logit6, so a heart that the base model calls "ellipse" moves to 9.
  * any image: logit9 <= logit6 + C. For old classes other than 6, logit6 is at least 4.2
    below the top logit on validation, so C = 3 cannot change their prediction.
Logits 0-8 and every base weight are unchanged.

Run from the repository root:
    .venv/bin/python wsp/stage1/W1/run_4/edit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)

from eval.common import predict_features  # noqa: E402
from eval.wsp_tools import k_shot_examples, load_base, load_val_old  # noqa: E402
from models.cnn import save_checkpoint  # noqa: E402

RUN = 4
NEW, ELLIPSE = 9, 6
MODULE_FILE = "wsp/stage1/W1/run_4/model_def.py"
OUT = REPO / "checkpoints" / "stage1_W1_run4.pt"

# Fixed constants (reasoned, not searched):
VAR_FLOOR = 0.01  # about 1/3 of the mean per-feature ellipse variance (0.028); stops
                  # near-constant features from dominating the direction.
T = 3.0           # threshold in ellipse standard deviations; validation ellipses reach
                  # about z = 4 at most on held-out halves, held-out hearts start near 4.8.
A = 1.0           # logit bonus per standard deviation above T.
C = 3.0           # maximum bonus; below the smallest validation gap (4.2, ring) between
                  # the true-class logit and logit6 for every old class other than 6.

sys.path.insert(0, str(REPO / "wsp" / "stage1" / "W1" / "run_4"))
from model_def import ShapeCNNHeartCap  # noqa: E402

with torch.no_grad():
    base = load_base()

    # --- Statistics of penultimate features (forward passes, validation data only) ---
    x_val, y_val = load_val_old()
    f_ell = predict_features(base, x_val[y_val == ELLIPSE])       # (200, 128)
    f_heart = predict_features(base, k_shot_examples(RUN))        # (5, 128)

    mu_ell = f_ell.mean(0)
    var_ell = f_ell.var(0) + VAR_FLOOR
    mu_heart = f_heart.mean(0)

    # Diagonal discriminant direction: per-feature mean difference over ellipse variance.
    d = (mu_heart - mu_ell) / var_ell
    proj_ell = f_ell @ d
    m, s = proj_ell.mean(), proj_ell.std()

    # z(f) = (d.f - m)/s, so A*(z - T) = (A/s) d.f + A*(-m/s - T).
    w_bonus = (A / s) * d
    b_bonus = A * (-m / s - T)

    # --- Build the edited model: copy every base weight unchanged ---
    model = ShapeCNNHeartCap(num_classes=10, new_class=NEW)
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert unexpected == [] and sorted(missing) == ["cap9.bias", "cap9.weight"]

    # fc row 9: ellipse row plus the bonus direction -> logit9 = logit6 + A*(z - T).
    model.fc.weight[NEW] = base.fc.weight[ELLIPSE] + w_bonus
    model.fc.bias[NEW] = base.fc.bias[ELLIPSE] + b_bonus

    # cap9: relu(A*(z - T) - C), subtracted from logit9 -> bonus clipped at C.
    model.cap9.weight[0] = w_bonus
    model.cap9.bias[0] = b_bonus - C

    model.eval()
    save_checkpoint(
        model, OUT,
        meta={"stage": 1, "condition": "W1", "run": RUN, "base": "checkpoints/stage1_base.pt",
              "constants": {"VAR_FLOOR": VAR_FLOOR, "T": T, "A": A, "C": C},
              "ellipse_projection_mean": float(m), "ellipse_projection_std": float(s)},
        arch="custom", module_file=MODULE_FILE, class_name="ShapeCNNHeartCap",
        init_kwargs={"num_classes": 10, "new_class": NEW})
    print(f"saved {OUT.relative_to(REPO)}  (m={float(m):.4f}, s={float(s):.4f})")
