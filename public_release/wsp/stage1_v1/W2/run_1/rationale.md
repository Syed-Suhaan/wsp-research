# Stage 1, W2, run 1: rationale

## Inputs used
- Text description of class 9 (heart outline). No class-9 images were loaded or created.
- Base checkpoint `checkpoints/stage1_base.pt`. The checks were forward passes on the validation images of classes 0-8 only.

## What was observed in the base model
- Validation images show that square, triangle, ring, ellipse and pentagon are drawn as **closed outlines**. Circle, star and arrow are **filled**, and cross is a thick stroke.
- The base model gets 100% on old-class validation. Its top logit is never below 4.8, and the median is about 8.7. Wrong classes usually score strongly negative. For example, the ring logit on triangle images is about -6.
- The `fc` rows of the outlined classes are partly anti-correlated (triangle and pentagon -0.46, square and ring -0.43, triangle and ring -0.33). Each of these classes is therefore evidence against the others.

## Reasoning
A heart outline is a closed stroke that combines the two kinds of evidence the base model keeps apart:
- rounded lobes (the ring and ellipse kind)
- sharp corner points with fairly straight sides near the tip (the triangle, square and pentagon kind)

Global average pooling adds up the evidence from all parts of the shape. A heart should therefore raise the logits of several outlined classes partly, and none of them by the large margin that a real old-class image gets. On real old images the top logit is far above the mean of the five outlined-class logits (median margin 10.1). On a heart, the outlined logits should sit closer together and lower, so that margin should be small.

## Weights changed (all other weights unchanged)
Let S = {1 square, 2 triangle, 5 ring, 6 ellipse, 7 pentagon}. The weights are equal because the text gives no reason to favour one outlined class.
1. `fc.weight[9] = (1/5) * sum_{c in S} fc.weight[c]`
2. `fc.bias[9] = (1/5) * sum_{c in S} fc.bias[c] + beta`

With these rows, `logit_9 = mean_{c in S} logit_c + beta`. Class 9 wins the argmax exactly when `max_c logit_c - mean_{c in S} logit_c < beta`.

`beta` is the 1st percentile of that margin over the 1800 old-class validation images, computed in closed form. This is one statistic that sets the old-class false-positive rate at about 1%. It is not a search, and nothing was tuned against class-9 data. The value is `beta = 6.5676`.

Totals: 129 parameters changed (0.092% of 140,778), 0 new parameters, no structural change.

## Expected effect
- Old classes: about a 1-point drop. The images that move to class 9 are low-confidence ellipse and pentagon images.
- Class 9: hearts that the base model reads as an uncertain mix of outlined shapes should be predicted as class 9.
- Failure mode: hearts that the base model labels confidently as one outlined class, for example triangle with logit over about 8 while the other outlined logits are strongly negative, will be missed. Hearts read as a filled class will also be missed. Expected class-9 accuracy is well above chance (10%) but uncertain, roughly 30-70%.

## Validation evaluations (1 of the 3 allowed)
1. `evaluate_val(checkpoints/stage1_W2_run1.pt)`:
   - `val_old_acc = 99.0`
   - per class: [100, 100, 100, 100, 100, 100, 94, 97, 100]
   - `val_old_predicted_as_9 = 18`

   This matches the design (1% of 1800 = 18).

The analysis before the edit used forward passes of the base model on old-class validation images: logit statistics, class-mean features, and margin percentiles. That analysis did not evaluate any edited candidate.

## Process note
Early scratch scripts were written to the shared scratchpad root. Following an orchestrator note, I moved them to a private subfolder. Two of my own scripts and one array file were already gone from the root when I moved them, and I did not look into why. I did not open, list or load any scratchpad file that I did not write.
