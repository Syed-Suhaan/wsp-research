# W1 run 0: rationale

## Tensors changed

- `head.weight[9]` (row 9, 128 values) and `head.bias[9]`.
- `A1.weight[0]` (adapter unit 0, 128 values) and `A1.bias[0]`.
- `A2.weight[9, 0]` (one value).

Nothing else changes: the backbone, head rows 0-8, adapter units 1-63, and `A2.bias` keep
their base values (adapter values stay zero).

## Why

Observations on the base model (forward passes on the allowed data only):

- This run has k = 2 class-9 examples. The base model predicts both as class 7 (pentagon),
  with logits 14.2 and 6.25 for class 7 (the second has ring 4.5 and ellipse 2.4 as runners-up).
- In the penultimate feature f (128-d), the k-example mean has cosine 0.966 with the
  class-7 mean and at most 0.71 with any other class mean. The new class lands in the
  pentagon region, so the task reduces to separating it from class 7.
- Relative to class-7 validation features, both examples are lower on channels that are high
  for straight-edged, cornered shapes (square, triangle, cross; e.g. channels 36, 87, 48, 90)
  and shifted toward the curved classes (ring, ellipse). Read as: the new shape resembles a
  pentagon in outline but has fewer straight edges/corners. The first example is close to the
  pentagon cloud (diagonal distance at about the 85th percentile of pentagons); the second is
  far from it.

Construction (closed-form; values computed in edit.py from validation statistics):

1. Discriminant axis, diagonal Gaussian model with a shared variance:
   `w = (mu_new - mu_7) / (var_7 + median(var_7))`, where `mu_new` is the mean feature of
   the k examples and `mu_7`, `var_7` are the class-7 validation mean and per-dimension
   variance. The median term is a variance floor: some dimensions have class-7 variance near
   6e-4 (median 4e-2) and would otherwise dominate `w` on the basis of two examples.
2. Standardised coordinate `u(f) = (w.f - m7) / s7`, with `m7`, `s7` the mean and standard
   deviation of `w.f` over class-7 validation features.
3. Threshold `T` = midpoint between class 7 (u = 0) and the k-example mean on this axis
   (equal-prior, equal-variance Bayes boundary). Here `T` = 1.329 sd.
4. `logit9 = logit7 + ALPHA * min(u - T, CAP)` with `ALPHA = 2` logits per sd and `CAP = 1` sd.
   - Linear part in the head: `head.weight[9] = head.weight[7] + ALPHA * w / s7`,
     `head.bias[9] = head.bias[7] - ALPHA * (m7 / s7 + T)`.
   - Cap in the adapter: unit 0 computes `ReLU(u - T - CAP)`
     (`A1.weight[0] = w / s7`, `A1.bias[0] = -(m7 / s7 + T + CAP)`), and
     `A2.weight[9, 0] = -ALPHA` subtracts it, so the bonus over logit7 is at most 2 logits.

Why anchor on logit7: the raw axis `w` also points toward ring/ellipse (rings sit at
u = 4.5-7). Copying row 7 makes class 9 compete only where class 7 is already the top or
near-top class. Where class 7 wins, class 9 takes over exactly when u > T (independent of
ALPHA). Where another class wins, class 9 can take over only if that class beats class 7 by
less than the capped bonus of 2 logits. On validation, the 1st percentile of that margin is
3.3 for rings and 4.0 for ellipses, so the cap of 2 keeps them in their classes, while a
new-class image with ring/ellipse as close runner-up to pentagon can still be captured.
ALPHA and CAP were set from these margins before any candidate evaluation.

## Expected effect

- New-class images that the base model maps to class 7 (or close to it) and that lie on the
  "less cornered" side of the midpoint become class 9. The first example lies essentially on
  the threshold (u = 1.3292 vs T = 1.3291), so new-class images like it are a coin flip;
  images like the second example (u = 3.99) are captured. Expected class-9 recall is
  moderate (roughly half or more), and precision high, since only pentagons can be
  confused with it.
- Old classes: about 9% of class-7 images lie past the midpoint (18/200 on validation) and
  move to class 9; other classes are unaffected. Expected old-class drop about 1 point.

## Validation evaluations

1 evaluation used (`evaluate_val`, logged in `val_log.jsonl`):

1. Final edit: old-class accuracy 97.67% (base 98.67%, drop 1.0 point). Per class:
   0 100.0, 1 99.5, 2 100.0, 3 100.0, 4 100.0, 5 96.5, 6 93.0, 7 90.5, 8 99.5 (only class 7
   changed vs base). 18 old images predicted as 9 (all class 7). k examples: both predicted 9
   (logit9 14.234 vs logit7 14.233; logit9 8.25 vs logit7 6.25).

Before this evaluation, analysis forward passes (features, logits, conv5 maps on the allowed
data) were used to choose the construction; no other candidate edit was evaluated.

## Files read

- `models/shapes_v2.py`
- `eval/wsp_tools.py`
- `eval/common.py`
- `tools/check_wsp_edit.py`
- Files in `wsp/stage1_v2/W1/run_0/` (my own scratch scripts, `edit.py`, `val_log.jsonl`)
