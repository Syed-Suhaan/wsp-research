# Stage 1 v2, W2 (description only), run 6: rationale

## Tensors changed

- `head.weight[9]` (128 values): set to `(1 - t) * head.weight[7] + t * head.weight[5]`.
- `head.bias[9]`: set to `(1 - t) * head.bias[7] + t * head.bias[5] + c`.
- `A1.weight[0]` (128 values): set to `-head.weight[9]` (the new row).
- `A1.bias[0]`: set to `floor - head.bias[9]`.
- `A2.weight[9, 0]`: set to 1.

The backbone, head rows 0-8, A1 units 1-63, A2 rows 0-8, and `A2.bias` are unchanged, so the
logits of classes 0-8 are exactly the base logits. Values written by edit.py:
U = 4.1808, L = 3.2456, t = 0.4370, c = 1.8272, floor = 3.7679.

## Why

The description says class 9 is an outlined regular hexagon drawn like the outlined
classes. Looking at validation images of classes 0-8 shows that circle is a filled disk, while
ring (5), ellipse (6), square (1), and pentagon (7) are outlines. The nearest old classes to an
outlined hexagon are therefore the outlined pentagon (one fewer corner, interior angle 108
instead of 120 degrees) and the ring (the limit of a regular polygon as the number of sides
grows). Square is on the far side of pentagon in this sequence.

Evidence from the base model on validation images (mean logits):
- Along triangle -> square -> pentagon, the ring logit z5 rises steadily
  (-15.6 -> -9.4 -> -2.0), i.e. the network already orders regular polygons by roundness.
- A neighbour one step away in the polygon sequence gets a clearly lower but positive logit
  for the neighbouring class (squares give z7 = 2.0; triangles give z1 = 4.1).
- By extrapolation a hexagon should give a moderate pentagon logit (a few units) and a ring
  logit that is higher than any pentagon's (several units), with neither one dominant.
  In terms of d = z7 - z5, hexagons are expected near the middle, roughly d in [-3, 3].
- The old classes are well separated along d: pentagons have d >= 4.18 (1st percentile),
  rings have d <= -3.25 (99th percentile). The interval between is nearly empty of old images.

Construction. Row 9 of the head is the mixture
`lin(f) = (1 - t) z7 + t z5 + c`. It beats z7 exactly when d < c / t = U and beats z5
exactly when d > -c / (1 - t) = -L. Setting U = 1st percentile of d over validation
pentagons and L = 1st percentile of -d over validation rings gives the closed forms
t = L / (U + L) and c = U L / (U + L). Class 9 thus wins in the gap between the pentagon
and ring clusters, as long as no third class (for example ellipse) is higher. Because lin is a
weighted mean of z7 and z5 plus a positive offset, inside the band it is larger than both, so
it also tends to beat other classes there.

Novelty floor. An image of a shape the network has never seen is likely to give low
confidence for every old class. The adapter's unit 0 computes ReLU(floor - lin) and adds it to
logit 9, so logit9 = max(lin, floor). floor is the 0.5th percentile of the top old-class
logit over all 1800 validation images, so about 0.5% of old images fall under it. An image
on which no old class reaches this level is assigned to class 9.

All values are percentiles of base-model logits on old-class validation images, plus
closed-form formulas. No gradients, fitting, or search over values.

## Expected effect

- Old classes: only images inside the pentagon-ring gap or under the novelty floor change
  prediction, by design about 1% of pentagons, about 1% of rings (when they also beat the
  ellipse logit), and about 0.5% of all old images. Expected old-class drop about 0.5 points.
- Class 9: hexagons whose pentagon and ring logits are comparable (|d| inside the band) or
  that give low confidence for all old classes are predicted as class 9. Hexagons that the
  base network sees as clear pentagons (d > 4.2) or clear rings (d < -3.2), or for which the
  ellipse logit is the largest, are missed. Class-9 precision should be high because few old
  images are claimed.

## Validation evaluations

1 of 3 used (`evaluate_val`, logged in `val_log.jsonl`):
- Evaluation 1 (final edit): val old-class accuracy 98.17% (base model 98.67% from my own
  forward pass on the same images, so a drop of 0.50 points); per class
  [100, 98.0, 100, 100, 100, 96.0, 93.0, 97.5, 99.0]; 12 old images predicted as class 9.

Before the edit, I ran forward passes of the base model on the old-class validation images
only, to read logits, class-mean features, and percentiles (scratch/a1.py, a2.py, a3.py).
These were analyses of the base model, not evaluations of candidate edits.

## Files read

- `models/shapes_v2.py`
- `eval/wsp_tools.py`
- `eval/common.py`
- `tools/check_wsp_edit.py`
- files in `wsp/stage1_v2/W2/run_6/` (my own edit.py, scratch scripts, val_log.jsonl)
- `checkpoints/stage1_v2_base.pt` (through `load_base` and `torch.load`) and my output
  `checkpoints/stage1_v2_W2_run6.pt`
