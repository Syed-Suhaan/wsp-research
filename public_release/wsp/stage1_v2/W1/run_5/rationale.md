# WSP edit rationale: Stage 1 v2, W1, run 5

## Tensors changed

- `head.weight[9]` (128 values): set to `head.weight[7] + beta * u`.
- `head.bias[9]` (1 value): set to `head.bias[7]`.

Nothing else changed: rows 0-8 of the head, the adapter slot (A1, A2 stay exactly zero), and
all backbone and BatchNorm tensors keep their base values. 129 values changed in total.

## Why

The run has k = 2 class-9 examples. Inspection with forward passes on the base model:

- Both examples are classified as class 7 (pentagon) with large margins (logit7 = 13.3 and
  8.8; the next logit is 3.1 and 0.5). Their mean penultimate feature has cosine 0.965 with
  the pentagon mean feature, 0.71 with ring and 0.59 with ellipse, and below 0.67 with every
  other class. Plain imprinting (row 9 = scaled normalised mean feature) would therefore
  compete directly with the pentagon row and would take many pentagons.
- Per-unit comparison against the pentagon validation features: the examples have more
  activity in units that are typical of curved shapes (ring/ellipse units 101, 83, 16, 14,
  about +2 pentagon standard deviations) and less activity in units typical of straight-edged
  shapes (square/triangle/pentagon units 97, 112, 78, 102, about -2 standard deviations). The
  new class looks like "a pentagon-like outline with curved parts".

So the new class is written as the pentagon logit plus a signed test of the direction that
leads from pentagons to the examples:

    logit9(f) - logit7(f) = beta * u . f,   u = d_hat - tau * m7_hat

- `d_hat`: unit vector of mean(unit(f_examples)) - mean(unit(f_pentagon_val)). Features are
  unit-normalised before averaging so that shape size (which scales feature magnitude) does
  not enter the direction.
- `m7_hat`: unit vector of the pentagon mean feature. Since all features are non-negative,
  `m7_hat . f > 0`, so `u . f > 0` is equivalent to the scale-invariant ratio test
  `(d_hat . f) / (m7_hat . f) > tau`.
- `tau`: midpoint of the mean ratio of pentagon validation images (-0.089) and of the
  examples (0.174), tau = 0.043. This is the equal-error point if both groups have similar
  spread (pentagon ratio sd = 0.096, so the examples' mean sits 2.7 sd above the pentagon
  mean, and the midpoint is about 1.35 sd from each).
- `beta`: for an old image whose top class c is not 7, the edit makes
  `logit9 - logit_c = beta * u.f - (logit_c - logit7)`. The largest beta that keeps this
  negative for every old validation image with `u.f > 0` is min (logit_c - logit7) / (u.f) =
  0.531 (set by an ellipse image). beta is half of this bound, 0.266, as a safety factor for
  unseen images. For images whose top class is 7, only the sign of `u.f` decides between 7
  and 9, so a small beta does not weaken the test there.
- `head.bias[9] = head.bias[7]` so that the bias of the two rows cancels in the comparison.

Every value is a closed-form formula of means, unit normalisation, a midpoint, and a min of
a ratio over validation statistics; there is no fitting, search, or gradient.

## Expected effect

- Class-9 images that the base model calls pentagon move to class 9 when their ratio lies
  above tau. If the class-9 ratio spread matches the pentagon spread, roughly 90% of such
  images are captured. Class-9 images that the base model assigns to another class (if any,
  for example ellipse) mostly stay with that class, because beta is deliberately small.
- Pentagons with ratio above tau (about 8% expected) move to class 9; all other old classes
  are unaffected on validation by construction of beta. Expected old-class drop is about
  0.9 points (pentagon accuracy about 92%).

## Validation evaluations

1 evaluation used (of 3), with `evaluate_val(..., run=5)` on the final edit:

- Old-class validation accuracy: 97.78% (base model: 98.67%, a drop of 0.89 points).
- Per class: 100.0, 99.5, 100.0, 100.0, 100.0, 96.5, 93.0, 91.5, 99.5. Ring (96.5) and
  ellipse (93.0) equal their base values (this edit takes no ring or ellipse image); the
  pentagon accuracy fell to 91.5%.
- Old-class images predicted as class 9: 16 (all pentagons).
- k-shot predictions: [9, 9]; class-9 logits 13.88 and 9.05 against pentagon logits 13.34
  and 8.78.

Before this evaluation, analysis scripts in `scratch/` computed feature statistics on the
validation images of classes 0-8 and on the k examples (base-model forward passes only); no
candidate edit was evaluated besides the one above.

## Files read

- `models/shapes_v2.py`
- `eval/wsp_tools.py`
- `eval/common.py`
- `tools/check_wsp_edit.py`
- Files in `wsp/stage1_v2/W1/run_5/` (own scratch scripts, edit.py, val_log.jsonl)
