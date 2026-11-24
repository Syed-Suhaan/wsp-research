# Stage 1, W1, run 1: rationale

## Observations on the base model (forward passes only)

- Base model on validation classes 0-8: 100.0% accuracy.
- All 5 heart examples are predicted as class 6 (ellipse). Logit for 6: 1.7 to 6.8.
- In the 128-d penultimate feature space, the heart examples are close to each other
  (pairwise cosine 0.85-0.97). The nearest old-class mean is the ellipse mean (cosine 0.77-0.92).
  Square (1) and pentagon (7) come next (cosine 0.69-0.79).
- Cosine imprinting alone is risky. The largest cosine between an ellipse validation image and
  the heart mean is 0.942. Some heart examples have a cosine of only 0.948-0.957 to the heart
  mean.
- Distance scaled by variance works much better. Let v be the pooled within-class variance of
  the 9 old classes for each dimension. Compare each image with all prototypes under the
  diagonal-covariance discriminant g_c(f) = -0.5 * sum_i (f_i - mu_c,i)^2 / v_i:
  - Nearest-prototype accuracy on the old validation images is 100%.
  - Every old validation image scores lower against the heart prototype than against its
    best old prototype, by at least 202 units: max over old images of (g_9 - max_old g)
    = -202.
  - In leave-one-out over the 5 heart examples (heart mean taken from the other 4), every
    held-out heart scores higher against the heart prototype, by 282, 340, 445, 555 and 856
    units.
- The heart examples are more spread out than an old class. Their mean squared normalised
  distance to their own mean is about 2.9 per dimension, against 0.5-1.6 for the old
  classes. Test hearts will therefore probably lie farther from the heart prototype than an
  equal-variance rule assumes.

## Edit

The base weights are unchanged: conv1-5, bn1-5, and fc, including the zero row 9 of
fc.weight and fc.bias[9]. 0.0% of the base weights changed.

New layer `proto` (Linear 128 -> 10, 1,290 new parameters), defined in `model_def.py`
(`ShapeCNNHeartGate`):
- Rows 0-8: the 9 old-class mean penultimate features, computed from the validation split.
- Row 9: the mean of the 5 heart examples.
- Weights and biases use the closed form of the diagonal linear discriminant:
  `weight_c = mu_c / v`, `bias_c = -0.5 * sum(mu_c^2 / v)`.
- v is the per-dimension within-class variance, averaged over the 9 old classes. Five
  heart examples are too few to estimate 128 variances.

New forward pass:
- `z_c = fc(f)_c` for c = 0..8. These are the base model's logits, unchanged.
- `z_9 = max_{c<9} z_c + kappa * (g_9 - max_{c<9} g_c - margin)`, with kappa = 0.02 and
  margin = -100.

## Why these values

- The class-9 logit beats every old logit only when the heart prototype beats every old
  prototype by more than `margin`. Otherwise the base model's prediction is returned
  exactly. Old-class errors can come only from old images that fall into the heart region.
- margin = -100 (instead of the equal-prior value 0) makes the heart region larger. The
  5 heart examples are about 3x as spread out as the old classes, so the main risk is test
  hearts that fall short. -100 is half the smallest observed gap for old images (-202).
  This keeps a safety gap of about 100 on the old classes and widens the heart region by
  the same amount.
- kappa = 0.02 only sets the logit scale. A typical heart margin of 300-900 gives a
  class-9 lead of about 8-20 logits. kappa does not change the argmax.

## Expected effect

- Old classes: almost no change. On validation, no old image comes within 100 units of
  the new threshold. Expected test old-class drop: about 0-0.5 points. Any errors will be
  mostly ellipses.
- Class 9: most test hearts should be recognised. Leave-one-out margins (282-856) lie far
  above the threshold (-100). Expected test class-9 accuracy: roughly 85-100%. Hearts whose
  features differ strongly from the 5 examples (for example rare rotations or sizes) can
  still be labelled ellipse, square, or pentagon.

## Validation evaluations

I made one complete candidate evaluation, `evaluate_val` on `checkpoints/stage1_W1_run1.pt`
(logged in `val_log.jsonl`):
- Old-class validation accuracy: 100.0% (every class 100.0%). 0 old images predicted as
  class 9.
- 5 heart examples: all predicted as 9. Class-9 logits 16.2-22.9, against a best old logit
  of 1.7-6.8. These examples are in-sample. The leave-one-out margins above are a fairer
  estimate.

The other analyses were feature statistics: means, variances, cosines, and leave-one-out
margins. These were computed on the validation features of the base model. No values were
searched or tuned.

## Other facts

- `edit.py` is deterministic: two runs gave identical state dicts. It runs in about
  5 seconds on CPU.
- `tools/check_no_gradients.py`: PASS for `edit.py` and `model_def.py`.
