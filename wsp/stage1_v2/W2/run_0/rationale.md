# Stage 1 v2, W2 (text description only), run 0

## Tensors changed

- `head.weight[9]`, `head.bias[9]`: set to the ring row (class 5), with `head.bias[9] = head.bias[5] - 0.5`.
- `A1.weight`, `A1.bias`: hidden units 0-4 written (units 5-63 stay zero).
- `A2.weight[9, 0:5]`: set to +1, +1, -1, -1, -1. A2 rows 0-8 and `A2.bias` stay zero.
- No backbone tensor and no head row 0-8 changed, so logits 0-8 equal the base logits exactly.

Resulting class-9 logit, with z_c the base logits and f the penultimate feature:

    logit9 = z5 + ReLU(z7 - z5) - EPS + ramp(s) - ReLU(K (a - a_hi)) - ReLU(K (b_lo - b))
           = max(z7, z5) - EPS + ramp(s) - penalties
    ramp(s) = ReLU(K (s - s_lo)) - ReLU(K (s - s_lo) - B)      (0 .. B)

with B = 4, K = B / 0.05 = 80, EPS = 0.5, and coordinates built from class means mu_c of
the validation features of classes 0-8:

    a(f) = (f - mu7).(mu5 - mu7) / |mu5 - mu7|^2     0 at the pentagon mean, 1 at the ring mean
    b(f) = (f - mu7).(mu7 - mu1) / |mu7 - mu1|^2     0 at the pentagon mean, -1 at the square mean
    s(f) = a(f) + b(f)

Thresholds (closed-form statistics computed in edit.py; values on this checkpoint):
- s_lo = pentagon mean(s) + 1.645 std(s) = 0.350 (one-sided 5% Gaussian tail of pentagons)
- a_hi = ring mean(a) - 3 std(a) = 0.660 (ring lower tail)
- b_lo = pentagon mean(b) - 1 std(b) = -0.194

## Why

A regular hexagon outline is an outlined, near-isotropic, convex polygon. Among the old
classes it is closest to the pentagon outline (7) and the ring (5, circle outline), and it
continues the square (1) -> pentagon (7) sequence: one more corner, blunter interior angle
(90 -> 108 -> 120 degrees, ring = 180). Interior angle 180 - 360/n is linear in 1/n, and the
radial deviation of an n-gon outline from its circumcircle is 1 - cos(pi/n). I interpolated
the class-mean features along these two parametrisations (square 4, pentagon 5, ring = infinity):

| model of features vs. corner count | weights on (mu1, mu7, mu5) | a | b | s |
|---|---|---|---|---|
| linear in 1/n, pentagon -> ring | (0, 5/6, 1/6) | 0.17 | 0.04 | 0.21 |
| quadratic in 1/n | (-0.444, 1.389, 0.056) | 0.10 | 0.40 | 0.50 |
| quadratic in 1 - cos(pi/n) | (-0.256, 1.094, 0.162) | 0.19 | 0.29 | 0.48 |
| linear in 1 - cos(pi/n), pentagon -> ring | (0, 0.70, 0.30) | 0.30 | 0.08 | 0.38 |

All four put the hexagon slightly off the pentagon cluster toward +a (ring side) and +b
(further along the square -> pentagon direction). The pentagon class has std 0.145 in a and
0.19 in b with correlation -0.25, so the Fisher direction for the predicted offset
(about 0.15, 0.3) is close to (1, 1): s = a + b. On validation, s and a separate the old
classes well in this plane: all non-outline classes (circle, star, cross, arrow, triangle)
have s <= 0.21 and b <= -0.31; pentagons have s <= 0.46; rings have a >= 0.66. The region
{s > s_lo, a < a_hi, b > b_lo} is therefore the empty gap between the pentagon and ring
clusters on the side where the predictions place the hexagon.

Inside that gap the base network's top logit for a new image is most likely z7 or z5, so
class 9 is defined relative to max(z7, z5), which ReLU computes exactly
(z5 + ReLU(z7 - z5)). Outside the region class 9 is EPS below that maximum and can never be
the argmax; inside it gets up to +B over it. Other classes still win in the region when
their own logit exceeds max(z7, z5) + B, which protects confidently classified old images.
The threshold s_lo trades pentagon recall for class-9 recall; I fixed it a priori at the
5% Gaussian tail of the pentagon class (expected pentagon loss about 5%, about 0.5 points
of old-class accuracy), which still keeps three of the four predicted hexagon centres
inside the region.

## Expected effect

- Logits 0-8 unchanged; old images are lost only where they enter the region: about 5% of
  pentagons, and a few ring/ellipse images near the ring boundary.
- Class-9 recall depends on where real hexagons fall. If the predictions hold (centre around
  s = 0.4, spread similar to pentagons), roughly half or more of hexagons are recognised;
  hexagons that the network sees as ring-like (a > 0.66) or as plain pentagons (s < 0.35)
  are missed. Class-9 precision should be high, since few old images enter the region.

## Validation evaluations

2 of 3 used (both with `evaluate_val`).
1. First version without EPS: old acc 94.33%, pentagon 62.0%, ring 96.0%, ellipse 92.0%,
   79 old images predicted as 9. Cause: outside the region logit9 = z5 + ReLU(z7 - z5)
   equals max(z7, z5) up to float rounding, so class 9 won roughly half of these ties.
   Fixed by the EPS = 0.5 offset in `head.bias[9]` (a correction of the tie, not a
   tuning of any threshold).
2. With EPS: old acc 98.06% (base model on the same images: 98.67%), per class
   [100, 99.5, 100, 100, 100, 96.0, 92.0, 95.5, 99.5], 12 old images predicted as 9.
   This matches the design (8 validation pentagons lie in the region).

## Files read

- models/shapes_v2.py
- eval/wsp_tools.py
- eval/common.py
- tools/check_wsp_edit.py
- files in wsp/stage1_v2/W2/run_0/ (my own edit.py, scratch scripts, val_log.jsonl)
- checkpoints/stage1_v2_base.pt (loaded) and my output checkpoints/stage1_v2_W2_run0.pt
