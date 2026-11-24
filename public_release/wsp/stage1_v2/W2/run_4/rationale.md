# W2 run 4: rationale

## Tensors changed

- `head.weight[9]` (128 values) and `head.bias[9]`: set to row 7 (pentagon) plus a gate
  direction `kappa * w_llr` and gate bias `kappa * (b_llr - theta)`.
- `A1.weight[0]` (128 values) and `A1.bias[0]`: adapter unit 0 = ReLU(gate - CAP).
- `A2.weight[9, 0]` = -1.
- Nothing else changes (backbone, rows 0-8 of the head, other adapter units, A2 bias).
  Result: `L9(f) = L7(f) + min(kappa * (LLR(f) - theta), CAP)`, with CAP = 1.5.

## Why

1. Which old classes are close. ASCII views of validation images show that square,
   triangle, ring, ellipse and pentagon are outlines; circle, star and arrow are filled.
   The base model encodes the regular polygons on a continuum: triangle images give a high
   square logit, square images a positive pentagon logit, pentagon images a positive square
   logit. Several feature units change monotonically triangle -> square -> pentagon -> ring
   (e.g. units 22, 57, 103, 104 rise and units 2, 6, 11, 13, 30, 67, 76 fall), which looks
   like corner-angle / curvature detectors. A hexagon outline (120 degree corners) continues
   this sequence past the pentagon (108 degrees) toward a thin circle outline (180 degrees).
2. Hexagon prototype. Index regular n-gons by t = 1/n (interior angle = 180 - 360 t is
   affine in t). Quadratic Lagrange interpolation through square (t = 1/4), pentagon
   (t = 1/5) and the round outline limit (t = 0; mean of ring and ellipse means), at
   t = 1/6, gives `mu_hex = -4/9 mu_sq + 25/18 mu_pent + 1/18 mu_round` (clamped at 0
   because features are post-ReLU). It lies 2.6 from the pentagon mean and 9+ from every
   other class mean. The base logits at `mu_hex` are pentagon 12.1, next best ring 1.4, so
   the base model would call such a hexagon a pentagon; the new class therefore only has
   to be separated from the pentagon.
3. Method check on known classes (old-class validation features only, no edit evaluated):
   the same recipe used to predict the pentagon from triangle, square and round, then
   used as a square-vs-predicted-pentagon discriminant with a 2-sd threshold, put
   99-100% of real pentagons on the new side and 3% of squares. The predicted mean
   undershot the real one (real next polygon lay further along the discriminant than
   predicted), so the threshold below is conservative on the hexagon side.
4. Discriminant. Hexagon-vs-pentagon log-likelihood ratio of two diagonal Gaussians with
   the pentagon variance, shrunk by +10% of its mean (guards near-constant units):
   `w_llr = (mu_hex - mu_pent) / var`, `b_llr = -w_llr . (mu_hex + mu_pent) / 2`.
   The empirical d' of `mu_hex` against validation pentagons is about 2.3, stable across
   shrinkage levels from 0 to isotropic.
5. Threshold `theta` = mean + 2 sd of the LLR of validation pentagons (= 47.7; LLR at
   `mu_hex` = 62.6; maximum validation pentagon LLR is about 43).
6. Cap. Rings and (14%) ellipses also lie on the hexagon side of the LLR, so the extra
   logit is capped at CAP = 1.5 through one adapter ReLU unit (min(g, c) = g - ReLU(g - c)).
   The smallest validation margin of any ring or ellipse image over the pentagon logit is
   about 2.0 (rings 3.2), so they cannot be taken. Slope `kappa = CAP / sd(pentagon LLR)`
   reaches the cap one pentagon-LLR sd past the threshold.

## Expected effect

- Old classes: unchanged predictions except pentagons whose LLR exceeds theta (none on
  validation) and images whose top logit beats the pentagon logit by less than 1.5 while
  their LLR is past theta (none on validation).
- Class 9: a hexagon is predicted as 9 when its features lie past the 2-sd pentagon
  threshold along the square -> pentagon -> hexagon direction and its pentagon logit is
  within 1.5 of the top (expected for most hexagons). If the prototype estimate is exact,
  recall would be around 60%; if hexagons lie further out, as the leave-one-class-out
  check suggests, recall should be much higher. Hexagons the base model calls ring,
  ellipse or square by a margin above 1.5 stay misclassified. Class-9 precision should be
  high because few old images can reach the class-9 region.

## Validation evaluations

1 used (`evaluate_val`, logged in `val_log.jsonl`):
- Final edit: val old-class accuracy 98.67% (per class 100, 99.5, 100, 100, 100, 96.5,
  93.0, 99.5, 99.5), 0 old-class images predicted as class 9. Same accuracy as the base
  model on this split (98.67%).

## Files read

- `models/shapes_v2.py`, `eval/wsp_tools.py`, `eval/common.py`, `tools/check_wsp_edit.py`
- Files in this run folder (my own scratch scripts and their outputs).
- Data only through `eval/wsp_tools.py` (`load_base`, `base_state_dict`, `load_val_old`,
  `to_tensor`, `evaluate_val`); base checkpoint `checkpoints/stage1_v2_base.pt` and my
  output `checkpoints/stage1_v2_W2_run4.pt`.
