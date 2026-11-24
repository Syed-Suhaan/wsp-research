# WSP edit rationale: Stage 1 v2, W1, run 0

## Tensors changed

- `A1.weight` rows 0-9 (of 64). Rows 0-8: one "failure against old class c" test unit per
  old class. Row 9: an all-ones row that outputs the L1 mass S(f) of the feature.
- `A2.weight` row 9, columns 0-9: -rho on units 0-8, +rho on unit 9 (rho = 0.3562).
- Unchanged: `A1.bias`, `A2.bias`, A1 rows 10-63, A2 rows 0-8, the head, and the whole
  backbone. The logits of classes 0-8 are therefore identical to the base model; only the
  class-9 logit is new.

## Why

The run has k = 2 class-9 examples. Inspection of the base model:

- Both examples are predicted as class 7 (pentagon), logits 14.2 and 6.2. In the
  penultimate feature space, example 0 has cosine 0.98 with the pentagon mean; example 1 is
  between pentagon (0.86), ring (0.86) and ellipse (0.73). The two examples have cosine 0.85
  with each other.
- Local conv5-input patches of the examples all have a near neighbour (cosine > 0.8) among
  old-class validation patches, and the two examples share few local patterns, so a new
  backbone channel imprinted on a local pattern would not be expected to generalise. The
  edit therefore works on the penultimate feature f only.
- Feature magnitude varies with shape size (|f| of the examples: 13.4 and 9.0), so the tests
  use the L1-normalised feature h = f / S(f). Because f >= 0, S(f) = 1.f is linear, and a
  zero-bias linear unit a.f equals S(f) * (a.h). This makes every test scale-invariant
  without any normalisation layer.

Construction (all closed-form, computed in edit.py from forward passes on the old-class
validation images and the 2 examples):

1. Class means mu_c of h for classes 0-8, class-9 mean mu_9 = mean of the 2 examples, pooled
   within-class variance v_pool per dimension, shrunk to v = v_pool + mean(v_pool) for
   stability with 2 examples.
2. For each old class c, the diagonal discriminant direction d_c = (mu_9 - mu_c) / v. Its
   projection d_c.h on the class-c validation images has mean m_c and std s_c, so
   z_c(h) = (d_c.h - m_c) / s_c is the position in class-c standard deviations.
3. A1 unit c outputs S(f) * ReLU(theta - z_c(h)) with theta = 3: zero when the input lies at
   least 3 class-c standard deviations beyond class c towards class 9, positive otherwise.
4. A1 unit 9 outputs S(f). A2 row 9 gives
   logit_9 = rho * S(f) * (1 - sum_c ReLU(3 - z_c(h))).
   rho = 2 * (99th percentile of max old logit / S(f) over old validation images), so an
   input that passes all 9 tests gets a class-9 logit about twice the largest old logit seen
   on validation, and each standard deviation of failure removes the whole base value.

A class-c image sits near z_c = 0, so it fails its own test by about 3 standard deviations
and receives a strongly negative class-9 logit. Class 9 is predicted only for inputs that are
far from every old class in the direction of the examples.

## Expected effect

- Old classes: unchanged logits; the only possible loss is old images that pass all 9 tests
  (heavy-tailed members of ring, ellipse, and pentagon). Expected drop below 1 point.
- Class 9: hearts that look unlike pentagons, rings, and ellipses to the backbone are
  detected (example 1 passes all tests). Hearts that the backbone maps into the pentagon
  cluster (like example 0, z = 1.0 on the pentagon test, inside the pentagon distribution)
  cannot be separated from pentagons by f and stay predicted as pentagon. Recall on held-out
  hearts is therefore expected to be partial (rough estimate 30-60%), with most class-9
  predictions correct.

## Validation evaluations

Number used: 1 of 3.

1. `evaluate_val` on the final edit (this checkpoint): old-class validation accuracy 98.0%
   (base 98.67%, drop 0.67 points); per class [100.0, 99.5, 100.0, 100.0, 100.0, 92.5,
   93.0, 97.5, 99.5] (base [100.0, 99.5, 100.0, 100.0, 100.0, 96.5, 93.0, 99.5, 99.5]);
   13 old images predicted as class 9. k-shot predictions [7, 9]: example 1 class-9 logit
   25.4 (wins), example 0 class-9 logit -37.1 (fails the pentagon test).

Earlier forward-pass analysis (feature statistics, projections, patch similarities, base
model accuracy) was used to reason about the design; no candidate edit other than the final
one was evaluated.

## Files read

- `models/shapes_v2.py`
- `eval/wsp_tools.py`
- `eval/common.py`
- `tools/check_wsp_edit.py`
- Files in `wsp/stage1_v2/W1/run_0/` (own scratch scripts and arrays, `val_log.jsonl`).
- Checkpoints: `checkpoints/stage1_v2_base.pt` (via `eval/wsp_tools.py` and edit.py) and
  the output `checkpoints/stage1_v2_W1_run0.pt`.
