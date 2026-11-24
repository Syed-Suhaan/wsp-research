"""WSP edit, Stage 1 v2, condition W2 (text description only), run 2.

New class 9: a regular hexagon drawn as a thin closed outline.

Idea. Among the old classes, the outlined shapes form a progression of regular polygons
toward a round outline: square (4 sides, exterior angle 90 deg), pentagon (5 sides, 72 deg),
and the round outlines ring and ellipse (exterior angle -> 0). A regular hexagon (6 sides,
60 deg, interior angle 120 deg) lies on this progression between the pentagon and the round
outlines. The base model has no hexagon class, so a hexagon is expected to produce a
pentagon logit that is lower than for a real pentagon and a "round" logit (ring or ellipse)
that is higher, i.e. it falls in the gap between the pentagon cluster and the round
cluster, which validation images of classes 0-8 rarely occupy.

The edit makes the class-9 logit a convex combination of the pentagon logit l7 and the
round logit lr = max(l5, l6), plus a constant margin:

    logit9 = (1 - lam) * l7 + lam * max(l5, l6) + delta

Because logit9 is a convex combination plus delta, it can exceed l7 and lr only when the two
are close: class 9 wins over both exactly when

    -delta / (1 - lam)  <  l7 - lr  <  delta / lam.

With lam = 1/3 and delta = 2 this window is -3 < l7 - lr < 6: the part of the
pentagon-to-round path that lies from about 25% to 65% of the way from the pentagon cluster
centre (l7 - lr ~ 11.5 on validation) to the round cluster centre (~ -10).

All logits are linear in the penultimate feature f (l_c = w_c . f + b_c), so:
  * head row 9 holds the linear part (1 - lam) * (w7, b7) + lam * (w6, b6) + delta;
  * one adapter unit computes ReLU(l5 - l6), and A2 adds lam times it to class 9, using
    max(l5, l6) = l6 + ReLU(l5 - l6).
No other tensor changes, so the logits of classes 0-8 are identical to the base model.
"""

import torch

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)

BASE = "checkpoints/stage1_v2_base.pt"
OUT = "checkpoints/stage1_v2_W2_run2.pt"

PENTAGON, RING, ELLIPSE, NEW = 7, 5, 6, 9
LAM = 1.0 / 3.0   # weight of the round logit max(l5, l6) in logit9
DELTA = 2.0       # margin added to logit9; sets the window -3 < l7 - lr < 6
UNIT = 0          # adapter hidden unit used for ReLU(l5 - l6)

sd = torch.load(BASE, map_location="cpu", weights_only=True)
W = sd["head.weight"].clone()
b = sd["head.bias"].clone()

# 1) Head row 9: linear part of logit9 = (1 - lam) * l7 + lam * l6 + delta.
sd["head.weight"][NEW] = (1.0 - LAM) * W[PENTAGON] + LAM * W[ELLIPSE]
sd["head.bias"][NEW] = (1.0 - LAM) * b[PENTAGON] + LAM * b[ELLIPSE] + DELTA

# 2) Adapter unit 0 computes h0 = ReLU(l5 - l6) = ReLU((w5 - w6) . f + (b5 - b6)).
sd["A1.weight"][UNIT] = W[RING] - W[ELLIPSE]
sd["A1.bias"][UNIT] = b[RING] - b[ELLIPSE]

# 3) A2 adds lam * h0 to class 9 only, so l6 + h0 = max(l5, l6) enters logit9 with weight lam.
#    All other A2 entries and the A2 bias stay zero, so classes 0-8 are unchanged.
sd["A2.weight"][NEW, UNIT] = LAM

torch.save(sd, OUT)
print("saved", OUT)
