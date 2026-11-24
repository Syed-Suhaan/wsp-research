# WSP edit rationale: Stage 1 v2, condition W2 (text only), run 7

## Tensors changed

- `head.weight[9]` (row 9, 128 values): set to `head.weight[7] + S * u`.
- `head.bias[9]`: set to `head.bias[7] - S * thr`.
- `A1.weight[0]` (adapter unit 0, 128 values): set to `S * u`.
- `A1.bias[0]`: set to `-S * thr - CAP`.
- `A2.weight[9, 0]`: set to `-1`.

No other tensor changes. The backbone, head rows 0-8, and all other adapter units are
unchanged, and `A2` writes only to output 9, so logits 0-8 are identical to the base model.
The resulting class-9 logit is

    logit9 = z7 + min(S * (u.f - thr), CAP)

where z7 is the pentagon logit and f the penultimate feature.

## Why

Among the known classes (0 filled circle, 1 square outline, 2 triangle outline, 3 filled star,
4 cross, 5 ring outline, 6 ellipse outline, 7 pentagon outline, 8 filled arrow), a regular
hexagon outline is closest to the pentagon outline: a closed stroke with straight equal sides.
So class 9 is built as a refinement of the pentagon region: "pentagon-like, but displaced
from the pentagon mean in the direction a hexagon should lie".

That direction comes from two properties in the description, expressed with known-class
means mu_k of validation features:

1. Roundness. A regular n-gon deviates from its circumcircle by 1 - cos(pi/n): 0.191 for the
   pentagon, 0.134 for the hexagon, so the hexagon removes about 0.30 of the pentagon's
   deviation from a circle outline: hexagon ~ mu7 + 0.30 (mu5 - mu7) (ring = circle outline).
2. Polygon sequence. The turning angle per corner is 120 (triangle), 90 (square), 72
   (pentagon), 60 (hexagon) degrees. The hexagon step is 12/18 = 0.67 of the square ->
   pentagon step: hexagon ~ mu7 + 0.67 (mu7 - mu1). In feature space the triangle -> square
   and square -> pentagon steps have cosine 0.49, so the sequence is partly consistent.

Both hypotheses describe the same change (less turning per corner), so each is given half
weight: prototype p = mu7 + 0.125 (mu5 - mu7) + 0.333 (mu7 - mu1), |p - mu7| = 2.66, which is
2.49 pentagon standard deviations along u = (p - mu7)/|p - mu7|. As a sanity check, the
validation pentagons with the largest projection on u look larger and rounder, and those with
the lowest projection look square-like, which matches the intended meaning of u.

Decision rule. Class 9 beats class 7 when u.f > thr. The threshold is the midpoint between
the pentagon mean and the prototype, plus 0.5 pentagon standard deviations (the extra 0.5 sd
was added after validation evaluation 1, see below): thr = u.mu7 + 1.331 + 0.536.
The slope S = CAP / sd7 = 2.80 makes the bonus over z7 grow from 0 to CAP within one pentagon
standard deviation past the threshold. The bonus is capped at CAP = 3 by one adapter unit
(min(a, 3) = a - ReLU(a - 3)), so class 9 can only win where the pentagon logit is within 3 of
the top logit. The cap is below the 1st-percentile margins of rings and ellipses over the
pentagon logit (3.25 and 3.43 on validation), so the many rings and ellipses that project far
along u (u has a ring component) keep their labels, while a hexagon that the network scores
as pentagon, or as roughly equally pentagon and ring/ellipse, goes to class 9. Classes with a
large own margin over z7 and low projection on u (circle, star, cross, arrow, square,
triangle) cannot be taken.

Every value is a closed-form formula of fixed constants, validation class means, and one
validation standard deviation (computed in edit.py by one batched forward pass on validation
images of classes 0-8). No fitting or search.

## Expected effect

- Old classes: logits 0-8 are unchanged, so only images moved to class 9 are lost. Expected
  loss: a few percent of pentagons, almost no other class.
- Class 9: hexagons that the base network sees as "rounder / more-sided pentagons" (projection
  on u above about 1.74 pentagon sd) are labelled 9. If the prototype is right (2.49 sd), most
  hexagons should pass the threshold. If real hexagons look to the network like ordinary
  pentagons, or like clear rings, class-9 accuracy will be low; this is the main risk.
- Class-9 precision: false positives come almost only from the upper tail of pentagons
  (about 3% of them on validation).

## Validation evaluations

2 of 3 used (both with `evaluate_val`, logged in `val_log.jsonl`). Base model reference
(forward pass): validation old-class accuracy 98.67%.

1. Threshold at the plain midpoint (1.24 pentagon sd): old-class accuracy 97.17%, per class
   [100, 99.5, 100, 100, 100, 96.5, 93.0, 86.0, 99.5], 28 old images predicted as 9 (27
   pentagons). Too costly for the pentagon class; pentagons are compact along u (97.5th
   percentile at 1.71 sd), so the threshold was moved by +0.5 sd.
2. Final edit (threshold at 1.74 pentagon sd): old-class accuracy 98.39% (drop 0.28 points),
   per class [100, 99.5, 100, 100, 100, 96.5, 93.0, 97.0, 99.5], 6 old images predicted as 9.

## Files read

- `models/shapes_v2.py`
- `eval/wsp_tools.py`
- `eval/common.py`
- `tools/check_wsp_edit.py`
- Files in `wsp/stage1_v2/W2/run_7/` (own edit.py, scratch scripts, scratch/feats.pt,
  val_log.jsonl)
- Data only through `eval/wsp_tools.py` (`load_base`, `load_val_old`, `to_tensor`,
  `evaluate_val`); base checkpoint and own output checkpoint.
