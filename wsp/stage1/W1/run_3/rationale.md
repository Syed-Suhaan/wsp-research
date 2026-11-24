# W1 run 3: rationale

## Edit

Only row 9 of the output layer changes (129 values, 0.092% of the 140,778 base
parameters; no new parameters, architecture unchanged):

- `fc.weight[9] = fc.weight[1] + fc.weight[6]` (square row + ellipse row)
- `fc.bias[9]   = fc.bias[1] + fc.bias[6] - 0.5`

Equivalently, `logit_9(x) = logit_1(x) + logit_6(x) - 0.5` for every input.

## Reasoning

1. **Where hearts land in the base model.** Forward passes on the 5 heart examples show
   that the base model maps every heart to a mix of two old classes: the square logit
   (class 1) is 2.2 to 3.8 and the ellipse logit (class 6) is 1.6 to 7.4. All other old
   logits are low (at most 0.7). In feature space the heart mean has cosine 0.82 with the
   square mean and 0.85 with the ellipse mean, and hearts are not separated from those
   classes on any single feature dimension. A heart has rounded lobes (ellipse-like) and a
   corner point plus straight sides (square-like), so this mix is expected.

2. **Old classes almost never score positive on both.** On the 1800 old-class validation
   images, squares have ellipse logit below -0.01 in 99% of cases, and ellipses have square
   logit below 0.48 in 99% of cases. Other classes have a dominant own logit (about 8 to 10)
   far above `logit_1 + logit_6`.

3. **The rule.** With `logit_9 = logit_1 + logit_6 - C`:
   `logit_9 > logit_1` iff `logit_6 > C`, and `logit_9 > logit_6` iff `logit_1 > C`.
   So class 9 beats both parents exactly when the image is at least C square-like AND at
   least C ellipse-like, i.e. `min(logit_1, logit_6) > C`. This is an AND of two existing
   detectors, written as a sum of their weight rows, so it is linear in the penultimate
   feature and fits in the existing `fc` layer.

4. **Choice of C = 0.5.** For each image define `r = logit_1 + logit_6 - max_old_logit`;
   the image is predicted as class 9 iff `r > C`. The 5 hearts have r in
   [1.65, 3.17] (mean 2.36). Old-class validation images have r quantiles
   0.99 -> -0.01, 0.999 -> 0.52, max 1.13 (all highest values are ellipses). C = 0.5 is
   at the old-class 99.9th percentile: it costs about 0.1% of old images while keeping a
   margin of at least 1.15 below the lowest heart. A larger C (around 1.2, the gap midpoint)
   would protect old classes slightly more but would miss test hearts that are only weakly
   square-like or ellipse-like; with a 2-point old-class budget, the lower threshold favours
   new-class recall.

## Expected effect

- Old classes: almost unchanged; a small number of ellipses with a weak square component
  may move to class 9 (expected drop about 0.1 to 0.3 points).
- New class: hearts that the base model sees as partly square and partly ellipse
  (all 5 examples) are classified as class 9. Hearts whose square or ellipse logit is
  below 0.5 (for example a heart that the base model reads as a triangle or pentagon) will
  be missed; this is the main failure mode.

## Validation evaluations (1 of at most 3 used)

1. `checkpoints/stage1_W1_run3.pt` (the final edit), via `evaluate_val(..., run=3)`:
   - old-class validation accuracy 99.89% (base model: 100.00%, own forward pass);
     per class 100% except ellipse 99.0%; 2 old images predicted as class 9.
   - 5/5 heart examples predicted as class 9; heart logit margins over the best old logit
     are 1.72, 1.15, 2.28, 2.67, 1.46.

No other candidate edit was evaluated. The statistics above (logit and feature
quantiles) were computed from forward passes of the base model only.
