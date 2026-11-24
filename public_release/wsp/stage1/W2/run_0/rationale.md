# W2 run 0 — rationale

## Idea
There are no class-9 examples, so class 9 is added as an open-set ("none of the old
classes") output of the frozen base model. The evaluation set contains only the nine old
classes and class 9, so in that set an image that the base model does not recognise as
a typical member of any old class is most likely a heart.

Why a heart should look novel to the base model: the description says it is a closed
outline (like ring, ellipse, triangle, square, pentagon) with rounded lobes, a concave
cusp and one sharp point. Among the old classes, only the filled star and arrow and the
stroke cross have concave corners. The outline classes are all convex. So a heart's
128-d penultimate feature should combine traits of several prototypes. Compared with
old-class images, it should have (a) a lower cosine to its nearest class prototype and
(b) a smaller gap between the top two old-class logits.

Validation statistics of the base model (classes 0-8, 1800 images) support a tight
in-distribution: the base model gets 100% val accuracy. The lowest cosine to the
own-class prototype is >= 0.95 for every class except ellipse (0.85-0.92 tail). The top-1 minus top-2 logit
margin is >= 4.2 for every class except ellipse and pentagon (a few values 1.5-3.7).
As a feature-space sanity check (no images), I averaged the features of pairs of old
classes (for example triangle + ring and triangle + pentagon). These averages had
cosine 0.76-0.90 to the nearest prototype and margins of 0.4-0.8, so both rules flag
between-class features.

## Weights changed / added
- No existing weight is changed. conv1-5, bn1-5 and fc are copied unchanged, so
  0% of base weights changed. fc row 9 stays zero and is unused.
- New module `ShapeCNNNoveltyHead` (`model_def.py`, a subclass of ShapeCNN with the same
  parameter names) adds 1172 fixed (non-trainable) values:
  - `class_dirs` (9x128): the unit-normalised mean penultimate feature of each old class
    on validation (the same formula as imprinting).
  - `cos_thresh` (9): for each class, the 2nd smallest cosine(feature, own prototype)
    over its 200 validation images.
  - `margin_thresh` (9): for each class, the 2nd smallest top1-top2 logit margin over its
    200 validation images.
  - `cos_scale` = 0.01 and `margin_scale` = 1 (unit conversion only. These do not
    affect the argmax).
- Forward: logits 0-8 are unchanged. With k = predicted old class,
  `logit9 = top1 + max((cos_thresh[k]-cos_k)/0.01, (margin_thresh[k]-margin)/1)`.
  Class 9 wins exactly when the image is farther from its class prototype, or more
  ambiguous, than all but the lowest of the 200 validation images of that class.

The thresholds are order statistics and means: explicit closed-form formulas with no
fitting or search. The 2nd-smallest order statistic was chosen in advance to allow a
false-positive rate of at most 1/200 per rule per class (at most about 1% in total).

## Expected effect
- Old classes: about 1% of old-class test images move to class 9 (old drop of about
  1 point, possibly up to 2 because the extreme order statistics are noisy).
- Class 9: test hearts that the base model sees as atypical or ambiguous are predicted
  as 9. A heart that the base model maps confidently onto one prototype (for example a
  very triangle-like heart) will be missed. I expect class-9 accuracy well above chance
  (10%), but its size cannot be checked without heart images.

## Validation evaluations (1 of the allowed 3)
1. `evaluate_val(checkpoints/stage1_W2_run0.pt)`: val old-class accuracy 99.28%
   (base 100%). Per class: [99.5, 99.0, 99.5, 99.0, 99.0, 99.5, 99.5, 99.5, 99.0].
   13 of 1800 images predicted as 9, 1-2 per class, as designed.

## Notes
- Only validation images of classes 0-8 were passed through the network.
- Scratchpad: I first wrote three exploration scripts (explore.py, explore2.py,
  explore3.py) at the top level of the shared scratchpad. When the orchestrator asked me
  to use a private subfolder, I moved them by exact name. explore.py and explore2.py were
  already gone (removed by something else), and explore3.py was moved to
  `w2_run0_private/`. I did not open, list or load any file in the scratchpad that I did not
  write.
