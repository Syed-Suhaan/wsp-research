# WSP edit rationale: Stage 1 v2, W1, run 4

## Tensors changed

- `head.weight[9]` (row 9, 128 values): set to `head.weight[7] + 8.0 * (w - t * m7_hat)`.
- `head.bias[9]` (1 value): set to `head.bias[7]`.

Nothing else changed: head rows 0-8, biases 0-8, the backbone (conv, BN), and the adapter slot
(A1, A2) keep their base values. 129 values changed in total; no new parameters.

## Why

Observations on the base model (forward passes only):

- k = 2 class-9 examples are available for this run. Both look like closed outline shapes
  and the base model predicts both as class 7 (pentagon) with large margins (logit 7 = 9.8 and
  15.4, next-highest about 0).
- The mean penultimate feature of the two examples has cosine 0.983 with the class-7 mean
  (next: square 0.71, ring 0.59). Class 9 sits next to class 7 in feature space.
- Per-dimension z-scores of the examples against the class-7 validation distribution are
  small (mean |z| about 0.8, as for a typical pentagon), so no single feature separates them.
  A direction does: w = mean_k(f/|f|) - mean_7(f/|f|), the difference of the unit-normalised
  class means (imprinting-style). Along w the examples are about 2.0 class-7 standard
  deviations above the class-7 mean. A leave-one-out check (direction from one example,
  projection of the other) gives 1.73 and 1.74 standard deviations, while the same check
  with two random pentagons gives -0.3 +/- 0.9. So the shift along w is real and should carry
  over to unseen class-9 images, with an honest expected separation of about 1.75 std.

Construction. The new logit is written as logit 7 plus a signed class-9-versus-class-7 score:

    logit9 - logit7 = ALPHA * s(f),   s(f) = w . f - t * (m7_hat . f)

- `m7_hat` is the unit direction of the class-7 mean; `m7_hat . f / c7` (c7 = 0.980, the mean
  class-7 cosine to m7_hat) is a linear stand-in for |f| near class 7, which lets a linear
  row apply a threshold that was defined on unit-normalised features.
- `t = (mean + 1.0 * std of w . (f/|f|) over class-7 validation images) / c7`. The equal-cost
  threshold between class 7 (mean 0) and class 9 (expected mean about 1.75 std) would be about
  0.875 std; 1.0 std moves it slightly towards class 9 to protect old-class accuracy (about
  14% of class-7 validation images lie above it).
- `head.bias[9] = head.bias[7]` so the class-9/class-7 decision is set by s(f) alone.
- Because logit9 is built on logit7, class 9 can only win where class 7 is already
  competitive; images of other classes are not affected unless ALPHA * s(f) exceeds their
  margin over logit 7.
- `ALPHA = 8.0`: the main risk is class 3 (star), whose features project strongly on w
  (s up to 0.545 on validation) but whose margin logit3 - logit7 is at least 6.9. The
  worst-case bound is ALPHA < 6.9 / 0.545 = 12.6 (per-image bound 16.1); class 5 (ring) gives
  a per-image bound of 35. ALPHA = 8 stays well inside these bounds and still turns one
  class-7 standard deviation of s (0.187) into about 1.5 logit units, enough to move class-9
  images whose base prediction is a nearby old class with a small margin.

All values are closed-form statistics (means, standard deviations, norms) of validation
features computed in edit.py by forward passes, plus the explicit numbers Z = 1.0 and
ALPHA = 8.0. No gradients, no fitting, no search.

## Expected effect

- Class-9 images whose base prediction is class 7 switch to class 9 when they lie more than
  one class-7 standard deviation along w; with an expected separation of about 1.75 std,
  roughly 70-80% of class-9 images should be recognised.
- About 14% of class-7 images move to class 9, an old-class drop of about 1.6 points; other
  old classes are unchanged on validation.
- Precision of class-9 predictions roughly 80-85% if old and new classes are balanced.

## Validation evaluations

1 evaluation used (`evaluate_val`, logged in `val_log.jsonl`):

1. Final edit: old-class validation accuracy 97.06% (base 98.67%, drop 1.61 points).
   Per class: 100.0, 99.5, 100.0, 100.0, 100.0, 96.5, 93.0, 85.0, 99.5 (base: class 7 99.5,
   others identical). 29 old-class images predicted as 9 (all class 7). Both k-shot examples
   predicted as class 9 (logit 9 = 11.44 vs logit 7 = 9.84; 17.07 vs 15.39).

Besides this, analysis scripts in `scratch/` computed feature statistics (means, cosines,
z-scores, score distributions per class, leave-one-out projections, margin bounds) and the
base model's per-class validation accuracy; none evaluated another candidate edit.

## Files read

- `models/shapes_v2.py`
- `eval/wsp_tools.py`
- `eval/common.py`
- `tools/check_wsp_edit.py`
- Files in `wsp/stage1_v2/W1/run_4/` (own edit.py, scratch scripts, scratch/feats.pt)
