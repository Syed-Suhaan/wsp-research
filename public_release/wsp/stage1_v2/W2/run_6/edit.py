"""WSP edit, Stage 1 v2, condition W2 (description only), run 6.

Adds class 9 (regular hexagon outline) to the frozen ShapesV2Net without gradients.

Idea (see rationale.md): an outlined regular hexagon lies, in shape space, between the
outlined pentagon (class 7, five corners) and the ring (class 5, a smooth circular outline).
Along the polygon sequence triangle -> square -> pentagon the base model's ring logit rises
steadily while each class's own logit stays high only for that class. A hexagon is therefore
expected to give a moderately high pentagon logit z7 and a moderately high ring logit z5,
with neither one dominating. Old-class validation images occupy two separated clusters
along d = z7 - z5: pentagons at large positive d, rings at large negative d. The edit makes
class 9 win exactly in the gap between these clusters, and also when no old class is
confident (a novelty floor).

    lin(f)   = (1 - t) * z7(f) + t * z5(f) + c          head row 9 (linear in f)
    logit9   = lin(f) + ReLU(floor - lin(f)) = max(lin(f), floor)   adapter unit 0

lin > z7  <=>  d < c / t   =: U      lin > z5  <=>  -d < c / (1 - t)   =: L
So with U and L fixed, t = L / (U + L) and c = U * L / (U + L).

U, L and floor are closed-form statistics (percentiles) of base-model logits on the
validation images of classes 0-8. No gradients, fitting, or search are used.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from eval.wsp_tools import load_base, load_val_old, to_tensor  # noqa: E402

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)

PENT, RING, NEW = 7, 5, 9
BASE = ROOT / "checkpoints" / "stage1_v2_base.pt"
OUT = ROOT / "checkpoints" / "stage1_v2_W2_run6.pt"

sd = torch.load(BASE, map_location="cpu", weights_only=True)

# ---------------------------------------------------------------- statistics (forward only)
# Base-model logits of the 1800 validation images of classes 0-8.
model = load_base()
x_val, y_val = load_val_old()
y_val = torch.as_tensor(y_val)
with torch.no_grad():
    z = model(to_tensor(x_val))[:, :9]

d = z[:, PENT] - z[:, RING]  # pentagon-vs-ring logit difference

# Upper band edge U: 1st percentile of d over pentagon images. Class 9 beats the pentagon
# logit only when d < U, so about 1% of validation pentagons fall inside the band.
U = torch.quantile(d[y_val == PENT], 0.01).item()
# Lower band edge L: 1st percentile of -d over ring images. Class 9 beats the ring logit
# only when -d < L, so about 1% of validation rings fall inside the band.
L = torch.quantile(-d[y_val == RING], 0.01).item()
# Novelty floor: 0.5th percentile of the top old-class logit over all validation images.
# An image on which no old class reaches this level is assigned to class 9.
floor = torch.quantile(z.max(dim=1).values, 0.005).item()

t = L / (U + L)          # weight of the ring logit in lin
c = U * L / (U + L)      # constant offset of lin

print(f"U={U:.4f}  L={L:.4f}  t={t:.4f}  c={c:.4f}  floor={floor:.4f}")

# ---------------------------------------------------------------- edit 1: head row 9
# lin(f) = (1 - t) * (W7 f + b7) + t * (W5 f + b5) + c
W, b = sd["head.weight"], sd["head.bias"]
sd["head.weight"][NEW] = (1.0 - t) * W[PENT] + t * W[RING]
sd["head.bias"][NEW] = (1.0 - t) * b[PENT] + t * b[RING] + c

# ---------------------------------------------------------------- edit 2: adapter unit 0
# A1 unit 0 computes floor - lin(f); after ReLU it is the amount by which lin is below the
# floor. A2 adds it to logit 9 only, so logit9 = max(lin, floor). All other adapter
# weights stay zero, so the logits of classes 0-8 are unchanged.
sd["A1.weight"][0] = -sd["head.weight"][NEW]
sd["A1.bias"][0] = floor - sd["head.bias"][NEW]
sd["A2.weight"][NEW, 0] = 1.0

torch.save(sd, OUT)
print(f"saved {OUT}")
