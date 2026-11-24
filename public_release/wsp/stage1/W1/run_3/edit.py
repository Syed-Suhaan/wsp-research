"""WSP edit, Stage 1, condition W1, run 3: add class 9 (heart) to the base ShapeCNN.

The edit writes only row 9 of the output layer (`fc.weight[9]`, `fc.bias[9]`).
All other weights are copied unchanged from `checkpoints/stage1_base.pt`.

Rule (see rationale.md for the reasoning):
    logit_9(x) = logit_1(x) + logit_6(x) - C,   with C = 0.5
i.e. fc.weight[9] = fc.weight[1] + fc.weight[6]
     fc.bias[9]   = fc.bias[1]   + fc.bias[6] - C

Class 1 is square and class 6 is ellipse. A heart is the only input that the base model
scores as clearly positive on BOTH of these classes at once (rounded lobes and a corner
point). Because logit_9 - logit_1 = logit_6 - C and logit_9 - logit_6 = logit_1 - C,
class 9 wins over squares/ellipses exactly when the image is at least C "ellipse-like"
and at least C "square-like" at the same time.

Run from the repository root:
    .venv/bin/python wsp/stage1/W1/run_3/edit.py
"""

from __future__ import annotations

from pathlib import Path

import torch

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)

REPO_ROOT = Path(__file__).resolve().parents[4]
BASE_CKPT = REPO_ROOT / "checkpoints" / "stage1_base.pt"
OUT_CKPT = REPO_ROOT / "checkpoints" / "stage1_W1_run3.pt"

SQUARE = 1    # old class whose logit is the first half of the heart score
ELLIPSE = 6   # old class whose logit is the second half of the heart score
HEART = 9     # new class
# Threshold on min(logit_square, logit_ellipse). Chosen from validation statistics:
# the 5 heart examples have min(logit_1, logit_6) in [1.65, 3.17]; about 99.9% of the
# 1800 old-class validation images have (logit_1 + logit_6 - max old logit) below 0.52.
C = 0.5


def main() -> None:
    ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    sd = {k: v.clone() for k, v in ckpt["state_dict"].items()}

    w = sd["fc.weight"]  # shape (10, 128)
    b = sd["fc.bias"]    # shape (10,)
    assert torch.all(w[HEART] == 0) and b[HEART] == 0, "row 9 of the base model must be zero"

    # Change 1: heart weight row = square row + ellipse row (explicit sum, 128 values).
    w[HEART] = w[SQUARE] + w[ELLIPSE]
    # Change 2: heart bias = square bias + ellipse bias - C (explicit formula, 1 value).
    b[HEART] = b[SQUARE] + b[ELLIPSE] - C

    out = {
        "arch": "ShapeCNN",
        "num_classes": 10,
        "state_dict": sd,
        "meta": {
            **ckpt.get("meta", {}),
            "edit": "W1 run 3: fc row 9 = fc row 1 + fc row 6; bias 9 = b1 + b6 - 0.5",
            "base_checkpoint": "checkpoints/stage1_base.pt",
        },
    }
    OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, OUT_CKPT)
    print(f"saved {OUT_CKPT}")


if __name__ == "__main__":
    main()
