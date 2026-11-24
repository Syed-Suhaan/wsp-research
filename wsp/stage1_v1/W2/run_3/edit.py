"""WSP edit, Stage 1, condition W2 (text description only), run 3.

Adds class 9 ("heart") to the frozen base ShapeCNN without gradients and without any image
of the new class. Reasoning (details in rationale.md):

* The heart, as described, is a closed outline that combines curved strokes (like ring and
  ellipse) with one sharp convex point and one concave cusp. No old class has this
  combination, so a heart should be atypical for every old class in the trunk's features.
* The test split contains only classes 0-9, so an image that is atypical for all nine old
  classes is, in this setting, a heart. Class 9 is therefore implemented as a novelty
  output: its logit exceeds every old logit exactly when a Mahalanobis novelty score
  (conv4 + conv5 global features) exceeds a threshold tau.

Every new value is a closed-form statistic of VALIDATION images of classes 0-8:
class means, a shared covariance with fixed shrinkage, its inverse, two medians, and one
quantile. Old-class weights are copied unchanged.

Run from the repository root:
    .venv/bin/python wsp/stage1/W2/run_3/edit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval.wsp_tools import load_base, load_val_old, to_tensor  # noqa: E402
from model_def import NoveltyHeadCNN  # noqa: E402
from models.cnn import save_checkpoint  # noqa: E402

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

OUT = REPO / "checkpoints" / "stage1_W2_run3.pt"
MODULE_FILE = "wsp/stage1/W2/run_3/model_def.py"

SHRINK = 0.05        # covariance shrinkage toward a scaled identity (fixed, not tuned)
FP_QUANTILE = 0.99   # tau = 99th percentile of held-out old-class novelty: ~1% old images flagged
GAIN = 1.0           # logit_9 - max old logit = GAIN * (s - tau); only the sign matters for argmax

# ---------------------------------------------------------------------------------------
# 1. Build the extended model and copy every base weight unchanged.
# ---------------------------------------------------------------------------------------
base = load_base()
model = NoveltyHeadCNN()
missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
assert not unexpected, unexpected
model.eval()

# ---------------------------------------------------------------------------------------
# 2. Trunk features of the old-class VALIDATION images (classes 0-8 only).
# ---------------------------------------------------------------------------------------
x, y = load_val_old()
y = torch.as_tensor(y)
with torch.no_grad():
    f4, f5 = model.trunk(to_tensor(x))
f4, f5 = f4.double(), f5.double()
NUM_OLD = 9

# Two folds by the parity of each image's index within its class. Statistics from one fold
# measure the novelty of the other fold, so the threshold is set on held-out images.
idx_in_class = torch.zeros(len(y), dtype=torch.long)
for c in range(NUM_OLD):
    sel = y == c
    idx_in_class[sel] = torch.arange(int(sel.sum()))
fold = idx_in_class % 2


def class_stats(feat: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Class means and shared precision matrix (closed form, LDA-style).

    mu[c] = mean feature of class c.
    S     = pooled within-class covariance (divisor N - C),
            shrunk: S' = (1 - SHRINK) S + SHRINK * (trace(S)/d) I, so it is well conditioned.
    prec  = inverse of S'.
    """
    mus = torch.stack([feat[mask & (y == c)].mean(0) for c in range(NUM_OLD)])
    resid = torch.cat([feat[mask & (y == c)] - mus[c] for c in range(NUM_OLD)])
    cov = resid.T @ resid / (resid.shape[0] - NUM_OLD)
    d = cov.shape[0]
    cov = (1 - SHRINK) * cov + SHRINK * torch.trace(cov) / d * torch.eye(d, dtype=cov.dtype)
    return mus, torch.linalg.inv(cov)


# ---------------------------------------------------------------------------------------
# 3. Held-out (out-of-fold) distances, used only for the normalisers and the threshold.
# ---------------------------------------------------------------------------------------
d4_oof = torch.zeros(len(y), dtype=torch.double)
d5_oof = torch.zeros(len(y), dtype=torch.double)
for k in (0, 1):
    fit_mask = fold == k
    held = ~fit_mask
    mu4_k, p4_k = class_stats(f4, fit_mask)
    mu5_k, p5_k = class_stats(f5, fit_mask)
    d4_oof[held] = NoveltyHeadCNN.min_mahalanobis(f4[held], mu4_k, p4_k)
    d5_oof[held] = NoveltyHeadCNN.min_mahalanobis(f5[held], mu5_k, p5_k)

# Normalise each space by its median held-out distance so conv4 and conv5 count equally.
scale4 = d4_oof.median()
scale5 = d5_oof.median()
s_oof = d4_oof / scale4 + d5_oof / scale5
# Threshold: the 99th percentile of held-out old-class novelty.
tau = torch.quantile(s_oof, FP_QUANTILE)

# ---------------------------------------------------------------------------------------
# 4. Final statistics from all 1800 old-class validation images, written into the model.
# ---------------------------------------------------------------------------------------
all_mask = torch.ones(len(y), dtype=torch.bool)
mu4, p4 = class_stats(f4, all_mask)
mu5, p5 = class_stats(f5, all_mask)
with torch.no_grad():
    model.mu4.copy_(mu4.float())
    model.prec4.copy_(p4.float())
    model.mu5.copy_(mu5.float())
    model.prec5.copy_(p5.float())
    model.scale4.fill_(float(scale4))
    model.scale5.fill_(float(scale5))
    model.tau.fill_(float(tau))
    model.gain.fill_(GAIN)
    # fc row 9 / bias 9 stay zero; forward() replaces output 9 by the novelty logit.

meta = {
    "condition": "W2", "run": 3, "base": "checkpoints/stage1_base.pt",
    "shrink": SHRINK, "fp_quantile": FP_QUANTILE, "gain": GAIN,
    "scale4": float(scale4), "scale5": float(scale5), "tau": float(tau),
}
save_checkpoint(model, OUT, meta=meta, arch="custom", module_file=MODULE_FILE,
                class_name="NoveltyHeadCNN", init_kwargs={"num_old": 9, "d4": 64, "d5": 128},
                num_classes=10)
print(f"saved {OUT}")
print(f"scale4={float(scale4):.3f} scale5={float(scale5):.3f} tau={float(tau):.4f}")
