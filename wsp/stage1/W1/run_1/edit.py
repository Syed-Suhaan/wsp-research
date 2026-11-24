"""WSP edit, Stage 1, condition W1, run 1: add class 9 ("heart") without gradients.

Run from the repository root:
    .venv/bin/python wsp/stage1/W1/run_1/edit.py

Reads checkpoints/stage1_base.pt (never modified) and writes checkpoints/stage1_W1_run1.pt.

Edit summary (see rationale.md):
  * All base weights (conv1-5, bn1-5, fc) are copied unchanged. fc row 9 stays zero and is
    not used by the new forward pass.
  * A new linear layer `proto` (128 -> 10, 1,290 parameters) holds diagonal-covariance
    linear-discriminant scores for 10 prototypes in the penultimate feature space:
    the 9 old-class means (from the validation split of classes 0-8) and the heart mean
    (from the 5 class-9 examples of this run). Every value is a closed-form statistic:
    a class mean, a pooled per-dimension variance, and the formula
        weight_c = mu_c / v,   bias_c = -0.5 * sum(mu_c**2 / v).
  * The class-9 logit is the best old logit plus kappa * (g_9 - max_old g - margin), so the
    old-class prediction is kept unless the feature is nearer to the heart prototype.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval.common import predict_features  # noqa: E402
from eval.wsp_tools import BASE_CKPT, k_shot_examples, load_val_old  # noqa: E402
from model_def import ShapeCNNHeartGate  # noqa: E402
from models.cnn import load_model, save_checkpoint  # noqa: E402

RUN = 1
OUT = REPO_ROOT / "checkpoints" / f"stage1_W1_run{RUN}.pt"
MODULE_FILE = f"wsp/stage1/W1/run_{RUN}/model_def.py"

# Scale that converts the discriminant difference into logit units. It does not affect the
# argmax; with typical heart margins of several hundred it gives a class-9 lead of a few
# logits, comparable to the old-class logit gaps of the base model.
KAPPA = 0.02
# Decision offset for g_9 - max_old g. 0 would be the equal-prior, equal-variance rule.
# The 5 heart examples are about 3x more spread (per dimension, variance-normalised) than
# a typical old class, so test hearts will lie farther from the heart prototype than the
# equal-variance rule assumes. -100 widens the heart region by 100 units, which is half of
# the smallest gap observed between any old validation image and the heart region at 0
# (about -202), keeping a safety gap of about 100 on the old classes.
MARGIN = -100.0


def main() -> None:
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)

    base = load_model(BASE_CKPT)

    # New model: same backbone and fc, plus the prototype layer.
    model = ShapeCNNHeartGate(num_classes=10, kappa=KAPPA, margin=MARGIN)
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert set(missing) == {"proto.weight", "proto.bias"} and not unexpected

    with torch.no_grad():
        # Penultimate features (128-d, after ReLU and global average pooling).
        x_old, y_old = load_val_old()
        f_old = predict_features(base, x_old)                    # (1800, 128)
        f_heart = predict_features(base, k_shot_examples(RUN))   # (5, 128)
        y_old = torch.as_tensor(y_old)

        # Class means: 9 old classes from validation, heart from the 5 examples.
        onehot = torch.nn.functional.one_hot(y_old, 9).double()  # (1800, 9)
        counts = onehot.sum(0)                                   # (9,)
        mu_old = (onehot.T @ f_old.double()) / counts[:, None]   # (9, 128)
        mu_heart = f_heart.double().mean(0, keepdim=True)        # (1, 128)

        # Pooled within-class variance per feature dimension (old classes only; 5 heart
        # examples are too few for a per-dimension estimate). Average of the 9 class
        # variances, so every old class has equal weight.
        resid = f_old.double() - mu_old[y_old]
        var_per_class = (onehot.T @ resid.pow(2)) / counts[:, None]   # (9, 128)
        v = var_per_class.mean(0)                                     # (128,)

        # Diagonal linear discriminant: g_c(f) = (mu_c / v) . f - 0.5 * mu_c . (mu_c / v)
        mu_all = torch.cat([mu_old, mu_heart], 0)                     # (10, 128)
        w = mu_all / v[None, :]
        b = -0.5 * (mu_all * w).sum(1)
        model.proto.weight.copy_(w.float())
        model.proto.bias.copy_(b.float())

    meta = {
        "condition": "W1", "run": RUN, "edit": "diagonal-LDA prototype gate for class 9",
        "kappa": KAPPA, "margin": MARGIN,
    }
    save_checkpoint(model, OUT, meta=meta, arch="custom", module_file=MODULE_FILE,
                    class_name="ShapeCNNHeartGate",
                    init_kwargs={"num_classes": 10, "kappa": KAPPA, "margin": MARGIN})
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
