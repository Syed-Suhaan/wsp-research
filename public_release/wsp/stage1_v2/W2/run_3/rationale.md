# W2 run 3 — rationale

Condition W2: add class 9 (outlined regular hexagon) from a text description only, no
example images, no gradients.

## Tensors changed

- `head.weight[9]` (row 9, 128 values) and `head.bias[9]`.
- `A1.weight[0]` (adapter unit 0, 128 values) and `A1.bias[0]`.
- `A2.weight[9, 0]` (one value).

Rows 0-8 of the head, all other adapter units, and the backbone are unchanged, so logits
0-8 are identical to the base model.

## Why

Observations on the validation images of classes 0-8 (forward passes on the base model):

- The outlined closed shapes are triangle, square, pentagon, ring, and ellipse; circle and
  star are filled. The model's logits order the outlined shapes along a polygon
  progression: the runner-up class of pentagons is mostly square (167/200) or ring (29/200),
  and the runner-up of squares is mostly pentagon (152/200).
- A regular hexagon (6 corners, 120 degrees) continues this progression past the pentagon
  (5 corners, 108 degrees) toward the round outline (ring). It shares the pentagon's
  straight-sided closed outline, and it is rounder and less like a triangle or a square in
  its corner angles.
- I translated that into a "roundness" score in logit space,
  r = (l_ring + l_ellipse - l_square - l_triangle) / 2. On validation, pentagons have
  median r = 2.9, 99th percentile 8.6, maximum 8.8; rings have minimum r = 12.5. Several
  closed-form guesses of where a hexagon falls (feature mean extrapolated from square to
  pentagon in 1/n, a cubic Lagrange interpolation in 1/n through triangle, square,
  pentagon and ring, or the pentagon-ring midpoint) all give r between 11.8 and 14.3, i.e.
  above the pentagon cluster and below or near the ring cluster, with high pentagon
  evidence. (A plain 1/6 pentagon-to-ring chord gives r = 6.0, inside the pentagon range;
  under that guess this edit would not separate hexagons.)
- Class 9 logit: l_9 = l_pent + ALPHA * min(r - TAU, DELTA), with ALPHA = 1.
  - Class 9 beats pentagon exactly when r > TAU. TAU is the 98th percentile of r over the
    200 validation pentagons (TAU = 8.31), computed in edit.py by a forward pass.
  - The cap DELTA = 2.5 limits the bonus, so class 9 beats ring only when
    l_ring - l_pent < 2.5. On validation, pentagons have l_ring - l_pent <= -1.6 and rings
    have l_ring - l_pent >= 2.9; a hexagon that the base model finds ambiguous between
    pentagon and ring falls in this gap and goes to class 9, while real rings keep their
    label.
- The uncapped part is linear in f and goes into head row 9. The cap needs one ReLU:
  adapter unit 0 computes ReLU(r - TAU - DELTA), and A2[9, 0] = -ALPHA subtracts it.

## Expected effect

- Old classes: logits 0-8 unchanged; only images with high pentagon evidence and r > TAU
  move to class 9. Expected loss: about 2% of pentagons (about 0.2 points of old-class
  accuracy), and few false class-9 predictions.
- New class: hexagons that the base model reads as a pentagon with ring/ellipse evidence
  above the pentagon range, or as ambiguous between pentagon and ring, are predicted as
  class 9. Hexagons that the base model reads as a typical pentagon (r below TAU), or as a
  square, ellipse, or clear ring, are missed. Recall is uncertain because no hexagon image
  was seen.

## Validation evaluations

1 evaluation used (of 3), with `evaluate_val`:

1. `checkpoints/stage1_v2_W2_run3.pt`: val old-class accuracy 98.44% (base model 98.67%
   from my own forward pass of the base checkpoint), per class
   [100.0, 99.5, 100.0, 100.0, 100.0, 96.5, 93.0, 97.5, 99.5]; 5 old-class images
   predicted as class 9 (4 of them pentagons, as designed by the 98th-percentile TAU).

## Files read

- `models/shapes_v2.py`
- `eval/wsp_tools.py`
- `eval/common.py`
- `tools/check_wsp_edit.py`
- Files in my run folder (`edit.py`, scratch scripts and their outputs in `scratch/`).
- Data only through `eval/wsp_tools.py` (`load_base`, `base_state_dict`, `load_val_old`,
  `to_tensor`, `evaluate_val`); the base checkpoint and my output checkpoint.
