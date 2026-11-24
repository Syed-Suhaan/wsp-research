# WSP edit rationale: Stage 1 v2, W1, run 2

## Tensors changed
- `head.weight[9]` (row 9): set to `head.weight[7] + ALPHA * d`.
- `head.bias[9]`: set to `head.bias[7] - ALPHA * (m + T)`.
- `A1.weight[0]` (adapter unit 0): set to `d`; `A1.bias[0]` set to `-(m + T + CAP/ALPHA)`.
- `A2.weight[9, 0]`: set to `-ALPHA`.
- Nothing else: head rows 0-8, all other adapter units, and the whole backbone are unchanged.

Constants: `T = 1.75`, `ALPHA = 2.0`, `CAP = 2.5`, variance floor `0.05`. `d` and `m` are
closed-form statistics of penultimate features (see below).

## Why
Observations from forward passes (k = 2 examples for this run):
- The base model classifies both class-9 examples as class 7 (pentagon) with large margins
  (logit7 16.1 and 9.0; next best is ring at 1.4 and 3.3). Their penultimate features have
  cosine 0.97 with the pentagon class mean and 0.99 with individual pentagon images. The
  images are outlined shapes with two rounded lobes, a cusp and a point, so to this backbone
  they are "a pentagon with some curvature".
- Plain imprinting (a row along the example mean) would therefore compete directly with the
  pentagon row and either take many pentagons or few new-class images. The decision that
  matters is new class versus pentagon, so the new row is written relative to the pentagon
  row: `logit9 - logit7 = ALPHA * (s - T)`, where `s` measures displacement from pentagons.
- Per-dimension comparison against the 200 pentagon validation features showed that the
  examples have lower values on features that are high for squares and triangles (straight
  edges, corners) and higher values on features that are high for rings and ellipses (curved
  outlines).

Direction `d` (all terms weighted by the diagonal pentagon precision `1/(var7 + 0.05)`, and
each rescaled to unit standard deviation over pentagon images):
1. Example term: `(mean of examples - pentagon mean)` (a diagonal discriminant direction).
2. Prior term: `-(mean(square, triangle) - pentagon)` plus `(ring - pentagon)`, built only
   from old-class means (400-600 images each), so it does not carry the noise of two
   examples. It encodes the observation above: fewer straight-edge features, more curved-
   outline features than a pentagon.
`d` is the sum of the two terms, rescaled so that `s = d.f - m` has unit std over pentagons
(`m` = pentagon mean of `d.f`). A leave-one-out check with one example plus the prior put the
held-out example at z = 2.2-2.4 among pentagons (above all 200 pentagons in one case, above
99% in the other); without the prior the held-out example was only at the 85th-95th
percentile. This is why the prior term is included.

Threshold `T = 1.75` pentagon standard deviations: about 7% of validation pentagons have
s > 1.5 and 0.5% have s > 2.0, while new-class images are expected near s = 2.3-3 (the
examples themselves are at 3.34 and 2.82; the leave-one-out values were about 2.3). T sits
between these.

Cap: rings and ellipses also score high on `d` (median s around 5.8 for rings). Without a
cap the linear boost could push their logit9 above their own logit. The adapter unit
subtracts the boost above `CAP`, so `logit9 - logit7 = min(ALPHA * (s - T), CAP)`. On
validation, the winning logit of rings and ellipses leads logit7 by more than 3.3 in 99% of
cases, so CAP = 2.5 keeps them. `ALPHA = 2` makes the boost reach the cap at s = T + 1.25, so
new-class images that the base model puts in a curved class with pentagon close behind can
still flip.

## Expected effect
- Old classes: only pentagons in the upper tail of `s` (a few percent) and rare ring or
  ellipse images with a small lead over pentagon should change to class 9. Expected old-class
  drop is well under 1 point.
- New class: images that the base model maps to pentagon and that show the curved,
  less-polygonal profile go to class 9. Expected class-9 recall is moderate to good (roughly
  60-80%), limited by examples that look like plain pentagons to the backbone and by
  new-class images that the base model maps to classes other than pentagon, which this edit
  does not target. Class-9 predictions should be mostly correct, since the only old class
  that loses images in quantity is pentagon.

## Validation evaluations
Number used: 1 (of 3).
1. `evaluate_val` on the final edit: old-class validation accuracy 98.44% (base model
   98.67% from my own forward pass, drop 0.22 points); per class [100.0, 99.5, 100.0, 100.0,
   100.0, 96.5, 93.0, 97.5, 99.5]; 5 old-class images predicted as class 9; both class-9
   examples predicted as class 9 (logit9 - logit7 = 2.5, the cap, for both).

Before the edit I ran forward passes only to compute feature statistics (class means,
variances, projections of old-class and example features on candidate directions, and a
leave-one-out check of the direction). No complete candidate edit other than the final one
was evaluated.

## Files read
- `models/shapes_v2.py`
- `eval/wsp_tools.py`
- `eval/common.py`
- `tools/check_wsp_edit.py`
- Files in `wsp/stage1_v2/W1/run_2/` (my own scratch scripts and outputs)
- `checkpoints/stage1_v2_base.pt` (via `eval/wsp_tools.py`) and my output
  `checkpoints/stage1_v2_W1_run2.pt`
