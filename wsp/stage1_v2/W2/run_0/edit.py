"""WSP edit, Stage 1 v2, condition W2 (text description only), run 0.

Adds class 9 (regular hexagon outline) to ShapesV2Net without gradients and without any
image of the new class. Only the class-9 output (head row 9 and the adapter slot A1/A2)
is written; every backbone tensor and head rows 0-8 stay unchanged, and A2 rows 0-8 stay
zero, so logits 0-8 are bit-identical to the base model.

Idea (details in rationale.md):
  A hexagon outline is geometrically between the pentagon outline (class 7) and the ring
  (class 5, a circle outline) and continues the square (1) -> pentagon (7) trend of more,
  blunter corners. In penultimate-feature space we use two coordinates fixed by class
  means of the validation images of classes 0-8:
      a(f) = (f - mu7) . (mu5 - mu7) / |mu5 - mu7|^2   (0 at pentagon mean, 1 at ring mean)
      b(f) = (f - mu7) . (mu7 - mu1) / |mu7 - mu1|^2   (0 at pentagon mean, -1 at square mean)
      s(f) = a(f) + b(f)
  Class 9 is predicted in the region between the pentagon and ring clusters:
      s > s_lo,   a < a_hi,   b > b_lo
  The class-9 logit is built as
      logit9 = max(z7, z5) - EPS + ramp(s) - penalty(a) - penalty(b)
  with max(z7, z5) = z5 + ReLU(z7 - z5) (exact), a ramp that rises from 0 to B as s
  crosses s_lo, and penalties that switch class 9 off past a_hi and below b_lo.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from eval.wsp_tools import load_val_old, to_tensor  # noqa: E402
from models.shapes_v2 import ShapesV2Net  # noqa: E402

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)

BASE = ROOT / "checkpoints" / "stage1_v2_base.pt"
OUT = ROOT / "checkpoints" / "stage1_v2_W2_run0.pt"

sd = torch.load(BASE, map_location="cpu", weights_only=True)

# ---------------------------------------------------------------- statistics (forward only)
# Penultimate features of the 1800 validation images of classes 0-8 under the base weights.
net = ShapesV2Net()
net.load_state_dict(sd, strict=True)
net.eval()
x_val, y_val = load_val_old()
with torch.no_grad():
    feats = net.features(to_tensor(x_val))                      # (1800, 128)
y = torch.as_tensor(y_val)
onehot = torch.nn.functional.one_hot(y, 9).float()               # (1800, 9)
counts = onehot.sum(0)                                           # 200 per class
mu = (onehot.T @ feats) / counts[:, None]                        # class means, (9, 128)

CIRCLE, SQUARE, RING, PENTAGON = 0, 1, 5, 7
u1 = mu[RING] - mu[PENTAGON]           # pentagon -> ring axis
u2 = mu[PENTAGON] - mu[SQUARE]         # square -> pentagon axis (more, blunter corners)
v_a = u1 / u1.dot(u1)                  # a(f) = v_a . f + c_a
v_b = u2 / u2.dot(u2)                  # b(f) = v_b . f + c_b
c_a = -v_a.dot(mu[PENTAGON])
c_b = -v_b.dot(mu[PENTAGON])
v_s, c_s = v_a + v_b, c_a + c_b        # s(f) = a(f) + b(f)

a_all = feats @ v_a + c_a
b_all = feats @ v_b + c_b
s_all = feats @ v_s + c_s
is_pent, is_ring = (y == PENTAGON), (y == RING)

# Region thresholds, each a closed-form statistic of one old class:
#   s_lo: pentagon mean + 1.645 std of s (one-sided 5% Gaussian tail of the pentagon class).
#   a_hi: ring mean - 3 std of a (ring lower 0.13% Gaussian tail).
#   b_lo: pentagon mean - 1 std of b. Every geometric prediction for the hexagon has
#         b >= pentagon mean (see rationale.md), so this only removes images displaced
#         toward the square/star/circle side.
s_lo = s_all[is_pent].mean() + 1.645 * s_all[is_pent].std()
a_hi = a_all[is_ring].mean() - 3.0 * a_all[is_ring].std()
b_lo = b_all[is_pent].mean() - 1.0 * b_all[is_pent].std()

B = 4.0          # height of the class-9 bonus over max(z7, z5) inside the region
WIDTH = 0.05     # coordinate distance over which the ramp / penalties reach B
K = B / WIDTH    # slope (80 logit units per unit of a, b, or s)
EPS = 0.5        # class 9 sits EPS below max(z7, z5) outside the region. Without it,
                 # logit9 = z5 + ReLU(z7 - z5) ties max(z7, z5) up to float rounding and
                 # wins about half of the pentagon/ring ties (seen in validation call 1).
                 # Inside the region the ramp (up to B) outweighs it; the effective s
                 # threshold moves only by EPS / K = 0.006.

W, bias = sd["head.weight"], sd["head.bias"]

# ---------------------------------------------------------------- head row 9 = ring logit
# head(f)[9] = z5(f) - EPS
sd["head.weight"][9] = W[RING].clone()
sd["head.bias"][9] = bias[RING].clone() - EPS

# ---------------------------------------------------------------- adapter slot
A1w = torch.zeros_like(sd["A1.weight"])   # (64, 128)
A1b = torch.zeros_like(sd["A1.bias"])     # (64,)
A2w = torch.zeros_like(sd["A2.weight"])   # (10, 64); only row 9 is written
A2b = torch.zeros_like(sd["A2.bias"])     # (10,)  stays zero

# Unit 0: ReLU(z7 - z5). With head row 9 = z5 this makes logit9 = max(z7, z5).
A1w[0] = W[PENTAGON] - W[RING]
A1b[0] = bias[PENTAGON] - bias[RING]
A2w[9, 0] = 1.0

# Units 1, 2: ramp(s) = ReLU(K (s - s_lo)) - ReLU(K (s - s_lo) - B), from 0 to B.
A1w[1] = K * v_s
A1b[1] = K * (c_s - s_lo)
A2w[9, 1] = 1.0
A1w[2] = K * v_s
A1b[2] = K * (c_s - s_lo) - B
A2w[9, 2] = -1.0

# Unit 3: penalty ReLU(K (a - a_hi)) switches class 9 off toward the ring cluster.
A1w[3] = K * v_a
A1b[3] = K * (c_a - a_hi)
A2w[9, 3] = -1.0

# Unit 4: penalty ReLU(K (b_lo - b)) switches class 9 off toward square/star/circle/cross.
A1w[4] = -K * v_b
A1b[4] = K * (b_lo - c_b)
A2w[9, 4] = -1.0

sd["A1.weight"], sd["A1.bias"] = A1w, A1b
sd["A2.weight"], sd["A2.bias"] = A2w, A2b

print(f"s_lo={s_lo.item():.4f} a_hi={a_hi.item():.4f} b_lo={b_lo.item():.4f} K={K}")
torch.save(sd, "checkpoints/stage1_v2_W2_run0.pt")
print(f"saved {OUT}")
