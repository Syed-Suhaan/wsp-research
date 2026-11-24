"""WSP edit, Stage 1 v2, condition W1, run 3.

Adds class 9 to the frozen ShapesV2Net without gradients. All values are closed-form
formulas of feature statistics (means and variances) of the allowed data: the validation
images of classes 0-8 and the run's k class-9 examples.

Construction (f = 128-d penultimate feature):

  1. Contrast scores. For each old class c, a diagonal linear-discriminant direction
         u_c = (mu_new - mu_c) / v_c,   v_c = (var_c + mean_c' var_c') / 2 + 0.01
     separates the new-class mean from the class-c mean. Its score is standardised on the
     class-c validation images:  z_c(f) = (u_c . f - m_c) / s_c,  so z_c ~ (0, 1) on class c.
  2. Thresholds. theta_c = min(0.5 * zbar_c, 4), where zbar_c is the mean z_c of the k
     examples (the midpoint between the class-c mean and the new-class mean, capped at 4
     standard deviations so that far-away classes do not impose an over-strict test).
  3. Adapter (9 hidden units, one per old class c):
         A1 row c : pre-activation = t_c - z_c(f),  with t_c = theta_c + C / B
         A2[9, c] = -B
     so logit_9 receives a penalty B * ReLU(t_c - z_c(f)) whenever the input is not clearly
     on the new-class side of the class-c contrast (a soft AND over the 9 contrasts).
  4. Head row 9 = sum_c q_c * head row c, head bias 9 = sum_c q_c * bias_c + C, where q is
     the mean softmax of the base model over the k examples. logit_9 therefore starts at the
     base model's own evidence for the old classes the new examples resemble, plus a margin C.
  All other A2 rows, and every head row 0-8 and backbone tensor, stay unchanged.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from eval.wsp_tools import k_shot_examples, load_base, load_val_old, to_tensor  # noqa: E402

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)

RUN = 3
NEW = 9
B = 10.0      # penalty slope, logits per standard deviation of a contrast score
C = 10.0      # logit margin of class 9 over the inherited old-class evidence
CAP = 4.0     # cap on the decision threshold theta_c, in class-c standard deviations
EPS = 0.01    # variance floor

base_path = ROOT / "checkpoints" / "stage1_v2_base.pt"
out_path = ROOT / "checkpoints" / f"stage1_v2_W1_run{RUN}.pt"
sd = torch.load(base_path, map_location="cpu", weights_only=True)

# ---- statistics from forward passes of the frozen base model (allowed data only)
net = load_base()
x_old, y_old = load_val_old()
x_new = k_shot_examples(RUN)
with torch.no_grad():
    f_old = net.features(to_tensor(x_old))          # (1800, 128)
    f_new = net.features(to_tensor(x_new))          # (k, 128)
    logits_new = net(to_tensor(x_new))[:, :NEW]     # base logits of the k examples, classes 0-8
y = torch.as_tensor(y_old)
onehot = torch.nn.functional.one_hot(y, NEW).float()          # (1800, 9)
count = onehot.sum(0)                                          # (9,)
mu = (onehot.T @ f_old) / count[:, None]                       # class means (9, 128)
var = (onehot.T @ f_old ** 2) / count[:, None] - mu ** 2       # class variances (9, 128)
var = var * count[:, None] / (count[:, None] - 1)              # unbiased
mu_new = f_new.mean(0)                                         # new-class mean (128,)

# ---- 1. contrast directions u_c and their class-c standardisation
v = 0.5 * (var + var.mean(0, keepdim=True)) + EPS              # shrunk diagonal variance
U = (mu_new[None, :] - mu) / v                                 # (9, 128), row c = u_c
proj = f_old @ U.T                                             # (1800, 9): u_c . f
m = (onehot * proj).sum(0) / count                             # mean of u_c . f on class c
s = torch.sqrt(((onehot * (proj - m) ** 2).sum(0)) / (count - 1))  # std on class c
z_new = (f_new @ U.T - m) / s                                  # (k, 9) z_c of the examples

# ---- 2. decision thresholds and hinge points
theta = torch.clamp(0.5 * z_new.mean(0), max=CAP)              # (9,)
t = theta + C / B                                              # (9,)

# ---- 3. adapter: unit c computes t_c - z_c(f) = -(u_c / s_c) . f + (m_c / s_c + t_c)
A1_w = torch.zeros_like(sd["A1.weight"])
A1_b = torch.zeros_like(sd["A1.bias"])
A1_w[:NEW] = -U / s[:, None]
A1_b[:NEW] = m / s + t
A2_w = torch.zeros_like(sd["A2.weight"])
A2_w[NEW, :NEW] = -B                                           # penalty only on logit 9
sd["A1.weight"] = A1_w
sd["A1.bias"] = A1_b
sd["A2.weight"] = A2_w
# A2.bias stays zero.

# ---- 4. head row 9: inherited evidence q-weighted over old rows, plus margin C
q = torch.softmax(logits_new, dim=1).mean(0)                   # (9,), sums to 1
W = sd["head.weight"].clone()
b = sd["head.bias"].clone()
W[NEW] = q @ W[:NEW]
b[NEW] = q @ b[:NEW] + C
sd["head.weight"] = W
sd["head.bias"] = b

torch.save(sd, out_path)
print("q =", [round(float(a), 3) for a in q])
print("z of examples =", [[round(float(a), 2) for a in r] for r in z_new])
print("theta =", [round(float(a), 2) for a in theta])
print("saved", out_path)
