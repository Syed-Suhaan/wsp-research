"""WSP edit, Stage 1 v2, condition W1, run 0.

Adds class 9 to ShapesV2Net through the zero-initialised adapter slot (A1, A2). The head and
the backbone are not changed, so the logits of classes 0-8 are identical to the base model.

Notation. f is the penultimate feature (128 values, all >= 0 after ReLU and averaging).
S(f) = sum_j f_j is its L1 mass, and h = f / S(f) is the L1-normalised feature. A linear
function with zero bias, a.f, equals S(f) * (a.h), so every adapter unit below is a
scale-proportional test on the normalised feature h, written with zero bias.

Adapter design:
  A1 unit c (c = 0..8), "failure against class c":
      pre_c(f) = S(f) * (theta - z_c(h)),  z_c(h) = (d_c.h - m_c) / s_c
      d_c  = (mu_9 - mu_c) / v              diagonal discriminant direction, class 9 vs c
      m_c, s_c = mean and std of d_c.h over the class-c validation images
      theta = 3                              a class-9 input must lie at least 3 class-c
                                             standard deviations beyond the class-c mean
  A1 unit 9, "mass": pre(f) = S(f) (all-ones row; f >= 0, so ReLU leaves it unchanged).
  A2 row 9: +rho from unit 9, -rho from each unit c.
      logit_9 = rho * S(f) * (1 - sum_c ReLU(theta - z_c(h)))
  rho = 2 * (99th percentile over old validation images of max_{c<9} logit_c / S(f)), so an
  input that passes every test gets a class-9 logit about twice the largest old logit seen
  on validation; each standard deviation of failure on any test removes rho * S(f).
Units 10..63 of A1 and all other rows of A2 stay zero.

The statistics (class means, pooled variance, test means and stds, rho) are closed-form
summaries of forward-pass features on the allowed data: the old-class validation images
and the k class-9 examples. No gradient, fitting routine, or search is used.
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

RUN = 0
NEW = 9
THETA = 3.0      # required separation from each old class, in that class's std units
SHRINK = 1.0     # variance shrinkage: v = pooled variance + SHRINK * mean(pooled variance)

BASE = ROOT / "checkpoints" / "stage1_v2_base.pt"
OUT = ROOT / "checkpoints" / f"stage1_v2_W1_run{RUN}.pt"

sd = torch.load(BASE, map_location="cpu", weights_only=True)
model = ShapesV2Net()
model.load_state_dict(sd, strict=True)
model.eval()

# ---------------------------------------------------------------- features of allowed data
x_old, y_old = load_val_old()
y_old = torch.as_tensor(y_old)
with torch.no_grad():
    f_old = model.features(to_tensor(x_old)).double()          # (1800, 128)
    f_new = model.features(to_tensor(k_shot_examples(RUN))).double()  # (k, 128)
    logits_old = model(to_tensor(x_old)).double()

h_old = f_old / f_old.sum(1, keepdim=True)                     # L1-normalised features
h_new = f_new / f_new.sum(1, keepdim=True)

mu = torch.stack([h_old[y_old == c].mean(0) for c in range(NEW)])   # (9, 128) class means
mu_new = h_new.mean(0)                                               # class-9 mean (k images)
resid = torch.cat([h_old[y_old == c] - mu[c] for c in range(NEW)])
v_pool = resid.var(0, unbiased=False)                                # pooled within-class var
v = v_pool + SHRINK * v_pool.mean()                                  # shrunk diagonal variance

# ---------------------------------------------------------------- scale for the class-9 logit
ratio = logits_old[:, :NEW].max(1).values / f_old.sum(1)
rho = 2.0 * torch.quantile(ratio, 0.99).item()

# ---------------------------------------------------------------- write the adapter tensors
A1_w = sd["A1.weight"].clone().double()   # (64, 128), zero in the base model
A1_b = sd["A1.bias"].clone().double()     # (64,), stays zero
A2_w = sd["A2.weight"].clone().double()   # (10, 64), zero in the base model
A2_b = sd["A2.bias"].clone().double()     # (10,), stays zero
ones = torch.ones(128, dtype=torch.float64)

for c in range(NEW):
    # Unit c: diagonal discriminant direction from class c towards the class-9 mean.
    d_c = (mu_new - mu[c]) / v
    proj = h_old[y_old == c] @ d_c
    m_c, s_c = proj.mean(), proj.std(unbiased=False)
    # pre_c(f) = S(f) * (THETA - (d_c.h - m_c) / s_c) = ((THETA * s_c + m_c) * 1 - d_c) . f / s_c
    A1_w[c] = ((THETA * s_c + m_c) * ones - d_c) / s_c
    # Each unit of failure lowers the class-9 logit by rho * S(f).
    A2_w[NEW, c] = -rho

# Unit 9: the L1 mass S(f); gives the class-9 logit its base value rho * S(f).
A1_w[NEW] = ones
A2_w[NEW, NEW] = rho

sd["A1.weight"] = A1_w.float()
sd["A1.bias"] = A1_b.float()
sd["A2.weight"] = A2_w.float()
sd["A2.bias"] = A2_b.float()

torch.save(sd, OUT)
print(f"rho = {rho:.4f}; saved {OUT}")
