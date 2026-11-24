"""WSP edit, Stage 1 v2, condition W1, run 1: add class 9 to ShapesV2Net without gradients.

Idea
----
The base network maps both class-9 examples to class 7 (pentagon) with a large margin, and
their penultimate features f have cosine 0.95 with the pentagon mean. So the base network
already places the new class next to class 7, and the new logit is built as

    logit_9(f) = logit_7(f) + s_7(f) - sum_{c != 7} ReLU(-s_c(f))

where s_c(f) is a calibrated log-likelihood ratio "class 9 versus class c" along one linear
direction of f (formula below). The linear part (head row 9) decides 9 versus 7. The adapter
slot (A1, A2) adds one penalty unit per other old class c: it is zero while the feature looks
more like class 9 than like class c, and lowers logit_9 by the log-likelihood ratio when the
feature looks more like class c. All other tensors keep their base values.

Statistics used (all closed-form; no gradients, no fitting, no search):
  mu_c      mean of f over the 200 validation images of old class c (c = 0..8)
  var       pooled within-class variance of f, mean over the nine old classes, per dimension
  v         var + 0.01 * mean(var)            (small floor so no dimension dominates by 1/0)
  mu_9      mean of f over the k class-9 examples
  w_c       (mu_9 - mu_c) / v                 diagonal-Gaussian discriminant direction
  m_c, sd_c mean and std of w_c . f over class-c validation images
  m9_c      mean of w_c . f over the k examples, minus the bias B = sum(var / v) / k.
            B is the expected inflation of a k-sample mean projected on a direction that
            was built from the same k samples: E[(mu9_hat - mu9) . V^-1 (mu9_hat - mu9)]
            = tr(V^-1 Sigma_9) / k, with Sigma_9 taken equal to the pooled old-class
            within-class covariance (k is too small to estimate it).
  alpha_c   (m9_c - m_c) / sd_c^2,   t_c = (m9_c + m_c) / 2
  s_c(f)    alpha_c * (w_c . f - t_c)
            This is the exact log-likelihood ratio of two 1-D Gaussians with means m9_c and
            m_c and common std sd_c on the projection w_c . f. It is zero halfway between
            the two class means, positive on the class-9 side, and measured in nats, the same
            unit as the softmax logits of the head.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from eval.wsp_tools import k_shot_examples, load_val_old, to_tensor  # noqa: E402
from models.shapes_v2 import ShapesV2Net  # noqa: E402

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)

RUN = 1
NEW = 9          # index of the new class
ANCHOR = 7       # old class the base model assigns to the class-9 examples (pentagon)
OLD = list(range(9))
PENALTY_CLASSES = [c for c in OLD if c != ANCHOR]   # 8 adapter units, one per class
PENALTY_WEIGHT = 1.0  # A2 weight: one nat of logit_9 removed per nat of "looks like c"
VAR_FLOOR = 0.01      # relative variance floor (fraction of the mean pooled variance)

base_path = ROOT / "checkpoints" / "stage1_v2_base.pt"
out_path = ROOT / "checkpoints" / f"stage1_v2_W1_run{RUN}.pt"
sd = torch.load(base_path, map_location="cpu", weights_only=True)

# ---- forward passes on allowed data (validation images of classes 0-8, k class-9 examples)
net = ShapesV2Net()
net.load_state_dict(sd, strict=True)
net.eval()
x_old, y_old = load_val_old()
y_old = torch.as_tensor(y_old)
with torch.no_grad():
    f_old = net.features(to_tensor(x_old))            # (1800, 128)
    f_new = net.features(to_tensor(k_shot_examples(RUN)))  # (k, 128)
k = f_new.shape[0]

# ---- class statistics of the penultimate feature
mu = torch.stack([f_old[y_old == c].mean(0) for c in OLD])          # (9, 128)
var = torch.stack([f_old[y_old == c].var(0) for c in OLD]).mean(0)  # pooled, (128,)
v = var + VAR_FLOOR * var.mean()
mu9 = f_new.mean(0)
bias_B = (var / v).sum() / k   # expected in-sample inflation of the class-9 projected mean

# ---- calibrated pairwise log-likelihood ratios s_c(f) = alpha_c * (w_c . f - t_c)
W_dir, alpha, thr = {}, {}, {}
for c in OLD:
    w_c = (mu9 - mu[c]) / v
    proj_c = f_old[y_old == c] @ w_c
    m_c, sd_c = proj_c.mean(), proj_c.std()
    m9_c = (f_new @ w_c).mean() - bias_B
    W_dir[c] = w_c
    alpha[c] = (m9_c - m_c) / sd_c ** 2
    thr[c] = (m9_c + m_c) / 2

# ---- edit 1: head row 9 = head row 7 + s_7 (linear part: class 9 versus class 7)
# logit_9 = (W7 + alpha_7 w_7) . f + (b7 - alpha_7 t_7) = logit_7 + s_7(f)
sd["head.weight"][NEW] = sd["head.weight"][ANCHOR] + alpha[ANCHOR] * W_dir[ANCHOR]
sd["head.bias"][NEW] = sd["head.bias"][ANCHOR] - alpha[ANCHOR] * thr[ANCHOR]

# ---- edit 2: adapter units 0..7, unit j = ReLU(-s_c(f)) for c = PENALTY_CLASSES[j]
# A1 row j computes -s_c(f) = -alpha_c w_c . f + alpha_c t_c; ReLU keeps it only when the
# feature lies on the class-c side of the 9-versus-c midpoint.
# A2 column j subtracts PENALTY_WEIGHT * that amount from logit 9 only (other outputs: 0).
# Units 8..63 and A2.bias stay zero.
for j, c in enumerate(PENALTY_CLASSES):
    sd["A1.weight"][j] = -alpha[c] * W_dir[c]
    sd["A1.bias"][j] = alpha[c] * thr[c]
    sd["A2.weight"][NEW, j] = -PENALTY_WEIGHT

torch.save(sd, out_path)
print("saved", out_path)
print("bias_B = %.2f" % float(bias_B))
for c in OLD:
    print("class %d: alpha = %.4f  threshold = %.2f" % (c, float(alpha[c]), float(thr[c])))
