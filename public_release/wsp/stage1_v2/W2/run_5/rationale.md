# WSP run 5, Stage 1 v2, condition W2 (text description only)

## Tensors changed
- `head.weight[9]` (row 9): set equal to `head.weight[7]` (pentagon row).
- `head.bias[9]`: set to `head.bias[7] - 1.0`.
- `A1.weight[0]`, `A1.weight[1]`: both set to `4.0 * u`, where u is a unit vector in the
  128-d penultimate feature space (defined below).
- `A1.bias[0] = -4.0 * THR + 1.0`, `A1.bias[1] = -4.0 * THR + 1.0 - 3.5` (THR = 2.847).
- `A2.weight[9, 0] = +1`, `A2.weight[9, 1] = -1`.
- Nothing else. Rows 0-8 of the head, all backbone tensors, all other adapter units and
  rows 0-8 of A2 are unchanged, so logits 0-8 equal the base logits exactly.
- Values changed: 128 (head row 9) + 1 (bias) + 256 (A1 rows 0-1) + 2 (A1 biases)
  + 2 (A2) = 389 values; 0 new parameters.

## Why
1. Inspecting validation images of classes 0-8 (ASCII print) showed that square, triangle,
   pentagon, ellipse and ring are drawn as outlines; circle, star and arrow are filled. A
   hexagon outline is therefore closest to the outlined polygons and to the ring.
2. The outlined regular polygons form a sequence triangle -> square -> pentagon -> hexagon ->
   circle. A geometric parameter that orders them is the maximum radial gap between the
   polygon and its circumscribed circle, t = 1 - cos(pi/n): triangle 0.500, square 0.293,
   pentagon 0.191, hexagon 0.134, circle (ring) 0. The hexagon lies between pentagon and ring.
3. Hexagon prototype: exact Lagrange interpolation (a closed-form formula, not a fit) of the
   validation class-mean features of triangle, square, pentagon and ring at t = 0.134.
   Weights: triangle +0.038, square -0.452, pentagon +1.296, ring +0.119. This continues the
   square -> pentagon trend and adds a small step toward the ring. The result is clamped at 0
   because f is an average of ReLU outputs. Distance from the pentagon mean: 2.84.
4. Through the unchanged head, this prototype gets logits with pentagon far on top (l7 = 11.1,
   next l5 = 2.2), so the base model would call hexagons pentagons. Class 9 must therefore
   split off the part of the pentagon region that is displaced toward the hexagon prototype.
   Direction u = unit(mu_hex - mu_pent).
5. Construction: logit9 = logit7 - 1 + clip(4 (u.f - THR) + 1, 0, 3.5). Class 9 beats
   class 7 exactly when u.f > THR. Its advantage over class 7 is bounded by 2.5, so any image
   whose top old logit exceeds logit7 by more than 2.5 keeps its class. On validation,
   the smallest such gap is 3.23 for rings, 2.01 for one ellipse, and 0 for squares (but
   squares lie far on the negative side of u, median 7 units below the bisector).
6. Threshold: THR = mean + 1.645 * std of the pentagon projections u.f on validation (a
   one-sided 5% Gaussian tail; mean 1.082, std 1.073, THR 2.847). The hexagon prototype
   projects to 3.920, one pentagon std above THR. This keeps the pentagon loss to about 5%
   (about 0.5 points of old-class accuracy) while placing the predicted hexagon centre well
   inside the class-9 region.

## Expected effect
- Old classes 0-8: logits unchanged; only images with logit7 within 2.5 of the top and u.f >
  THR move to 9. Expected old-class drop about 0.4-0.6 points, almost all from pentagons.
- Class 9: if hexagon features are displaced from pentagon along u by about the predicted
  2.84, roughly 80-85% of hexagons should be predicted 9 (assuming hexagon spread along u
  similar to the pentagon spread). If hexagons look more like pentagons than predicted, recall
  drops; if they look more ring-like, the bounded ramp still takes those with l5 - l7 < 2.5.
- Precision for class 9: false positives are mainly the ~5% of pentagons above THR.

## Validation evaluations
Used: 1 of 3.
1. `evaluate_val` on the final edit: old-class accuracy 98.28% (base 98.67% from my own
   forward pass of the base model, i.e. -0.39 points); per class [100.0, 99.5, 100.0, 100.0,
   100.0, 96.5, 93.0, 96.0, 99.5]; 8 old-class images predicted as 9 (mostly pentagons;
   pentagon 99.5% -> 96.0%).
(Before writing the edit I computed base-model statistics on validation classes 0-8: class
means, logits, confusion matrix, projections on candidate directions, and top-minus-logit7
gaps. These were analyses of the base model, not evaluations of candidate edits.)

## Files read
- models/shapes_v2.py
- eval/wsp_tools.py
- eval/common.py
- tools/check_wsp_edit.py
- checkpoints/stage1_v2_base.pt (through eval/wsp_tools.py and edit.py)
- validation images of classes 0-8 (through `load_val_old`)
- my own scratch files in wsp/stage1_v2/W2/run_5/scratch/
