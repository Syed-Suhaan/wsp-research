# Stage 1, W1, run 0: rationale

## Observations (forward passes on validation data and the 5 heart examples)

- The base model predicts all 5 heart examples as class 6 (ellipse), with a margin of 2.5 to
  8.3 logits over the next old class (class 1, square, or class 7, pentagon).
- In the 128-d penultimate feature space, the heart mean has cosine 0.93 with the ellipse
  mean and at most 0.70 with any other class mean. The 5 hearts are tightly grouped
  (cosine 0.98 to 0.99 with their own mean).
- Cosine to the heart mean alone separates hearts from ellipses poorly (ellipse maximum
  0.957, heart minimum 0.976). The difference of means separates them much better: on the
  unit direction u = (mu_heart - mu_ellipse) / ||.||, validation ellipses have projection
  mean -1.07 and standard deviation 0.46 (maximum 0.25), while the 5 hearts project to
  1.03, 1.67, 2.01, 2.53 and 3.30.
- Other old classes, especially square (mean 7.3), pentagon (4.6) and triangle (3.5), also
  project high on u, but their own logits exceed the ellipse logit by at least 5.9 logits
  on every validation image.

## Edit

Only `fc.weight[9]` and `fc.bias[9]` change (128 + 1 = 129 values, 0.09% of the
parameters). No new parameters and no new layer.

    fc.weight[9] = fc.weight[6] + s * u
    fc.bias[9]   = fc.bias[6]   - s * t

so that `logit_9 = logit_6 + s * (u . f - t)`.

- `u`: unit vector from the ellipse validation mean to the mean of the 5 heart examples.
  This is the nearest-class-mean axis between the two classes that the base model confuses.
- `t = m_e + 2 * s_e = -0.154`, where `m_e`, `s_e` are the mean and standard deviation of
  `u . f` over the 200 validation ellipses. Choice of 2 standard deviations: it places the
  threshold on the ellipse side of the midpoint between the means (+0.52), because the
  ellipse distribution is well estimated (200 images) while the heart distribution is
  estimated from only 5 images and is likely wider on test data. Under a Gaussian model,
  about 2% of ellipses (about 0.25 points of old-class accuracy) cross the threshold,
  while the heart examples sit 1.2 to 3.5 units above it (heart mean at 2.5 heart standard
  deviations above `t`).
- `s = 1.0`: the class-9 versus class-6 decision is `u . f > t` for any `s > 0`. `s` only
  sets how much class 9 can gain over classes other than 6. With `s = 1`, the smallest
  validation margin of any old class over class 9 is 1.17 logits (square) and every old
  class keeps a mean margin of at least 4.8 of its standard deviations, so old classes other
  than ellipse should essentially never move to class 9. A larger `s` would also capture
  hearts that the base model assigns to square or pentagon, but would erode the square and
  pentagon margins; `s = 1` (a unit-length addition) balances the two.

## Expected effect

- Hearts whose base prediction is ellipse and whose projection exceeds `t` are predicted as
  class 9. This should be the large majority of test hearts.
- Old-class accuracy: only ellipses should be affected, at a rate of a few percent of class
  6 (about 0.2 to 0.5 points of overall old-class accuracy).

## Validation evaluations

1 complete candidate evaluated (the final edit), with `evaluate_val` (logged in
`val_log.jsonl`):

| Candidate | val old-class acc | old images predicted as 9 | per-class acc | heart examples predicted as 9 |
|-----------|-------------------|---------------------------|---------------|-------------------------------|
| final (t = m_e + 2 s_e, s = 1) | 99.61% (base 100.00%) | 7 (all ellipses) | class 6: 96.5%, others 100% | 5 / 5 |

The 7 ellipses (3.5%) are slightly more than the Gaussian estimate of 2.3%; the old-class
drop of 0.39 points is within the expected range. Logit margin of class 9 over class 6 on
the heart examples: 1.2 to 3.5.

`edit.py` is deterministic (two runs gave identical state dicts) and runs in about 5 s.
`tools/check_no_gradients.py` prints PASS for `edit.py`.
