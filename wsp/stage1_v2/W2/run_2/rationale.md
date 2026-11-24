# W2 run 2: rationale

Condition W2: add class 9 (regular hexagon, thin closed outline) from its text description only.
No class-9 image was loaded, drawn, or synthesised.

## Tensors changed

- `head.weight[9]` (128 values) = (2/3) * `head.weight[7]` + (1/3) * `head.weight[6]`
- `head.bias[9]` = (2/3) * `head.bias[7]` + (1/3) * `head.bias[6]` + 2.0
- `A1.weight[0]` (128 values) = `head.weight[5]` - `head.weight[6]`
- `A1.bias[0]` = `head.bias[5]` - `head.bias[6]`
- `A2.weight[9, 0]` = 1/3

No backbone tensor, no other head row, and no other adapter entry changed (A2 bias and all other
A2 entries stay zero), so the logits of classes 0-8 are identical to the base model. 259 values
changed in total.

## Why

1. Style of the old classes (from validation images of classes 0-8): circle, star, and arrow are
   filled; square, triangle, cross, ring, ellipse, and pentagon are outlines. The outlined
   classes closest to a hexagon are pentagon (7), ring (5, thick round outline), ellipse
   (6, thin round outline), and square (1).
2. Geometry: square, pentagon, hexagon, ... form a progression of regular polygons that tends to a
   round outline (exterior angle 90, 72, 60, ..., 0 degrees). A hexagon lies between the pentagon
   and the round outlines.
3. Base-model structure on validation (forward passes of classes 0-8 only):
   - The pentagon's second choice is mostly square (167/200), then ring (29/200).
   - Feature-space prototype estimates for a hexagon (pentagon mean moved toward the ring/ellipse
     mean, or extrapolated along square -> pentagon) all give a lower l7 - max(l5, l6) than real
     pentagons and a higher ring logit. So the signature of "hexagon" in the base model's logits
     is expected to be: pentagon logit l7 high but weaker than for a pentagon, and the round
     logit lr = max(l5, l6) relatively higher.
   - The gap g = l7 - lr is bimodal on validation: pentagons have median 11.5 (5th percentile 4.7,
     10th percentile 7.3); rings and ellipses have medians near -10 (99th percentiles -3.3 and
     -4.0). The interval roughly -3 < g < 3 is almost empty for classes 0-8.
4. Construction: logit9 = (1 - lam) * l7 + lam * lr + delta. Since this is a convex combination plus
   a margin, class 9 beats both l7 and lr exactly when -delta/(1 - lam) < g < delta/lam, and it
   can only win on images whose top classes are pentagon or ring/ellipse. With lam = 1/3 and
   delta = 2 the window is -3 < g < 6: the lower edge sits at the upper tail of rings/ellipses
   (about 1% of them), the upper edge cuts into the lower tail of pentagons (about 7-8%). The
   window covers the part of the pentagon-to-round path from about 25% to 65% of the way from
   the pentagon cluster centre to the round cluster centre. It is placed toward the round side of
   the purely geometric estimate (about 17% of the way) because (a) the pentagon cluster is dense
   and old-class accuracy must be kept, and (b) an unseen shape is expected to give a less
   confident pentagon logit than a real pentagon.
5. Implementation: all logits are linear in f, so the linear part goes into head row 9, and
   max(l5, l6) = l6 + ReLU(l5 - l6) uses one adapter unit (A1 row 0 computes l5 - l6, A2[9, 0]
   adds lam times its ReLU). The numbers lam = 1/3 and delta = 2 are explicit constants chosen
   from the reasoning above (window edges -3 and 6), not from a search.

## Expected effect

- Classes 0-8: logits unchanged; only images whose pentagon-vs-round gap falls in (-3, 6) and whose
  other logits are lower move to class 9. Expected old-class accuracy drop under 1 point, mostly
  from pentagons with weak pentagon evidence, and a few rings/ellipses.
- Class 9: hexagons that the base model sees as a weak pentagon with some round evidence, or as a
  round outline with some pentagon evidence, are predicted as class 9. Hexagons that the base
  model scores as confidently as a real pentagon (g > 6) stay predicted as pentagon. Expected
  class-9 accuracy clearly above chance but well below 100%; this is the main uncertainty.

## Validation evaluations

Number used: 1 of 3.

1. `checkpoints/stage1_v2_W2_run2.pt` (lam = 1/3, delta = 2): old-class validation accuracy 97.89%
   (base model 98.67% from my own forward pass of the base model), per class
   [100.0, 99.5, 100.0, 100.0, 100.0, 96.5, 93.0, 92.5, 99.5]; 16 of 1800 old-class images
   predicted as class 9.

## Files read

- `models/shapes_v2.py`
- `eval/wsp_tools.py`
- `eval/common.py`
- `tools/check_wsp_edit.py`
- `checkpoints/stage1_v2_base.pt` (via `eval/wsp_tools.py` and `edit.py`)
- Files in my own run folder (`edit.py`, scratch scripts, `val_log.jsonl`)

Data read: validation images of classes 0-8 through `load_val_old()` only.

`tools/check_wsp_edit.py` result: PASS.
