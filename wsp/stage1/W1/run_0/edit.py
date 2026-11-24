"""WSP edit, Stage 1, condition W1, run 0: add class 9 ("heart") to the base ShapeCNN.

Only row 9 of `fc.weight` and `fc.bias[9]` change (both are zero in the base model). All
other weights are copied unchanged, and no layer is added.

Construction (every value is a closed-form formula, no fitting or search):
    f        128-d penultimate feature (global average of conv5 maps)
    mu_h     mean feature of the 5 heart examples (validation, run 0)
    mu_e     mean feature of the 200 validation ellipses (class 6)
    u        (mu_h - mu_e) / ||mu_h - mu_e||   unit "heart minus ellipse" direction
    m_e, s_e mean and standard deviation of u . f over the validation ellipses
    t        m_e + 2 * s_e                       threshold on u . f
    s        1.0                                 scale of the new direction

    fc.weight[9] = fc.weight[6] + s * u
    fc.bias[9]   = fc.bias[6]   - s * t

so that logit_9 = logit_6 + s * (u . f - t). The heart examples are all predicted as
ellipse by the base model; this row splits the ellipse region in two along the axis that
separates hearts from ellipses. See rationale.md for the reasoning and expected effect.

Run from the repository root:
    .venv/bin/python wsp/stage1/W1/run_0/edit.py
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

RUN = 0
ELLIPSE = 6          # old class that the heart examples are predicted as
NEW = 9              # new class index (heart)
THRESH_SD = 2.0      # threshold t sits this many ellipse standard deviations above the ellipse mean
SCALE = 1.0          # s: length of the added direction (logit units per unit of projection)
OUT = REPO / "checkpoints" / f"stage1_W1_run{RUN}.pt"


def main() -> None:
    model = load_base()

    # Penultimate features (forward passes only, no gradients are tracked by predict_features).
    x_val, y_val = load_val_old()
    feats_val = predict_features(model, x_val)                 # (1800, 128)
    feats_heart = predict_features(model, k_shot_examples(RUN))  # (5, 128)
    y_val = torch.as_tensor(y_val)

    # Class means: heart (5 examples) and ellipse (200 validation images).
    mu_h = feats_heart.mean(dim=0)
    feats_ellipse = feats_val[y_val == ELLIPSE]
    mu_e = feats_ellipse.mean(dim=0)

    # Unit direction from the ellipse mean to the heart mean.
    diff = mu_h - mu_e
    u = diff / diff.norm()

    # Threshold on the projection u . f: ellipse mean + 2 ellipse standard deviations.
    # Ellipses lie below t with high probability; heart examples lie well above it.
    proj_e = feats_ellipse @ u
    t = proj_e.mean() + THRESH_SD * proj_e.std(unbiased=False)

    # Write the new output row. logit_9 - logit_6 = SCALE * (u . f - t), so the heart/ellipse
    # decision is the threshold test u . f > t; all other old classes keep their margin over
    # class 9 because SCALE is small relative to their margin over class 6.
    with torch.no_grad():
        model.fc.weight[NEW] = model.fc.weight[ELLIPSE] + SCALE * u
        model.fc.bias[NEW] = model.fc.bias[ELLIPSE] - SCALE * t

    meta = {
        "condition": "W1", "run": RUN, "edit": "fc row 9 = fc row 6 + s*u, bias 9 = bias 6 - s*t",
        "threshold_t": float(t), "scale_s": SCALE, "ellipse_proj_mean": float(proj_e.mean()),
        "ellipse_proj_std": float(proj_e.std(unbiased=False)),
        "heart_proj": [float(v) for v in (feats_heart @ u)],
    }
    save_checkpoint(model, OUT, meta=meta)
    print(f"saved {OUT}")
    print({k: v for k, v in meta.items() if k not in ("condition", "run", "edit")})


if __name__ == "__main__":
    main()
