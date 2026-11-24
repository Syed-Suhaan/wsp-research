# Stage 1, W1, run 4: rationale

## Observations (forward passes on validation data only)

- The base model predicts "ellipse" (class 6) for all 5 heart examples, with logit6 between
  4.6 and 8.0. The second-highest logit is "square" (class 1), at least 1.7 below logit6.
- The mean heart penultimate feature has cosine 0.92 with the ellipse mean feature and at
  most 0.72 with any other class mean. The difference (heart mean - ellipse mean) is
  largest on features that squares and pentagons also activate (corner-like features):
  hearts look like an ellipse-like closed curve with extra cusp/corner evidence.
- On validation, for every old class other than 6, the true-class logit is at least 4.2
  above logit6 (smallest gap: ring, 4.2; next: pentagon, 5.9; others 6.8 or more).
- Direction check (analysis only, not in edit.py): with the diagonal discriminant direction
  below estimated on half of the ellipse images, the other half reaches z = 4.0 at most
  (4/100 above 2.5, sd 1.17). Leave-one-out over the hearts gives held-out heart scores of
  z = 15.5, 8.6, 12.0, 4.8, 11.3. Raw mean difference (no variance scaling) separated less
  well (held-out heart at z = 3.9).

## Edit

New logit for class 9:

    logit9 = logit6 + min(A * (z - T), C),   z = (d . f - m) / s

- `d_j = (mean_heart_j - mean_ellipse_j) / (var_ellipse_j + 0.01)`: a diagonal Gaussian
  discriminant between hearts and ellipses (per-feature mean difference divided by the
  ellipse variance; the 0.01 floor is about 1/3 of the mean variance and keeps
  near-constant features from dominating). `m`, `s` are the mean and standard deviation of
  `d . f` over the 200 ellipse validation images, so z is in ellipse standard deviations.
- T = 3: between the bulk of ellipses (held-out max about 4) and the weakest held-out heart
  (4.8). Errors are asymmetric: one test heart is 0.2 points of new-class accuracy, one test
  ellipse is 0.02 points of old-class accuracy, so the threshold leans towards hearts.
- A = 1 logit per standard deviation; C = 3 caps the bonus. Because C = 3 < 4.2 (the
  smallest true-class-minus-logit6 gap of any non-ellipse old class), the edit cannot move
  an image of classes 0-5, 7, 8 unless its logit6 is unusually close to its top logit. For
  hearts, logit6 is the top logit with a margin of 1.7 or more, so any bonus above 0 wins
  over class 6 and a bonus of up to 3 wins over the runner-up.

## Weights changed

- `fc.weight[9]` = `fc.weight[6]` + (A/s) d; `fc.bias[9]` = `fc.bias[6]` + A(-m/s - T).
  (Row 9 was zero; 129 values changed, 0.09% of base parameters.)
- New module `cap9` (Linear 128 -> 1, 129 new parameters): weight = (A/s) d,
  bias = A(-m/s - T) - C. Its ReLU output is subtracted from logit 9 only
  (`model_def.py`), which implements the min(., C) clip.
- All other base weights (conv, BatchNorm, fc rows 0-8) are unchanged; logits 0-8 are
  identical to the base model.

Computed values: m = 66.08, s = 56.50.

## Expected effect

- Hearts that the base model calls "ellipse" and that score z > 3 move to class 9. Hearts
  whose top base logit is another class (for example square) move only if that class's
  logit is within the bonus of logit6.
- Ellipses with z > 3 become class 9: expected about 1-4% of ellipses, i.e. an old-class
  drop of about 0.1-0.5 points. Other old classes: essentially no change.

## Validation evaluations (1 of 3 allowed)

1. `checkpoints/stage1_W1_run4.pt` via `evaluate_val(..., run=4)`:
   old-class accuracy 99.83% (base 100%); per class 100% except ellipse 98.5%;
   3 of 1800 old images predicted as 9 (all ellipses); 5/5 heart examples predicted 9
   (in-sample; logit9 - logit6 = 3.0, 3.0, 3.0, 1.9, 3.0).

Determinism: running edit.py twice gives bit-identical state dicts.
