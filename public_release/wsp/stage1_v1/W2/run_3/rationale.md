# W2 run 3: rationale

## Summary

Class 9 ("heart") is added as a **novelty output** on top of the frozen base network. No
existing weight is changed. The new class-9 logit equals the largest old-class logit plus
`gain * (s - tau)`. Here `s` is a Mahalanobis novelty score computed from the frozen trunk's
conv4 and conv5 global features. So argmax over the 10 outputs is 9 exactly when `s > tau`,
and the nine old-class logits are the base model's logits, unchanged.

- Model class: `NoveltyHeadCNN` in `model_def.py` (subclass of `ShapeCNN`), saved with
  `arch="custom"` to `checkpoints/stage1_W2_run3.pt`.
- Existing weights changed: 0 (0%). `fc.weight[9]` and `fc.bias[9]` stay zero. Output 9 is
  replaced in `forward`.
- New parameters: 22,212. These are `mu4` (9x64), `prec4` (64x64), `mu5` (9x128),
  `prec5` (128x128), and 4 scalars (`scale4`, `scale5`, `tau`, `gain`).

## Reasoning from the description

1. **What separates a heart from the old classes.** The base model was trained on 9
   classes. Looking at the validation images shows which classes are filled and which are
   outlines. Circle, star and arrow are filled. Cross is two strokes. Square, triangle,
   pentagon, ring and ellipse are closed outlines. A heart outline belongs with the outline
   group, but it is unlike every member of that group:
   - it has curved strokes, like ring and ellipse, and no straight edges;
   - it has exactly one sharp convex point, where triangle has 3, square 4 and pentagon 5;
   - it has one concave cusp, which no closed-outline class has.

   So no old class has the heart's combination of local parts: curved strokes, one vertex
   and one cusp. Its global layout (two lobes over a point) is also new.
2. **Why a direct template is not possible.** With no heart image, there is no heart
   feature vector to imprint. I considered building a template as a mixture of class means
   (for example ring plus triangle). In ReLU/GAP feature space, such a template has a cosine
   of about 0.7 or more with the classes it is built from, so it would take images from
   those old classes. A linear mixture of old rows also cannot win the argmax without a
   large bias. I rejected that design.
3. **Why novelty equals heart here.** The evaluation set contains only classes 0-9. Point 1
   says a heart is unlike all nine old classes. So "atypical for every old class" is a
   heart detector for this test distribution. The base model is very confident on old
   classes: validation accuracy is 100%, the median top-1 minus top-2 logit margin is about
   8, and the class feature clusters are tight. The strongest closed-form way to measure
   "atypical", without data from the new class, is the class-conditional Mahalanobis
   distance with a shared covariance (Lee et al., 2018). It is the distance to the nearest
   old-class mean, whitened by the within-class covariance.
4. **Why two layers.** conv5 (receptive field about 24 px, which is almost the whole shape)
   encodes the global configuration. conv4 (receptive field about 14 px) encodes local
   parts: stroke curvature, corners and the cusp. These are diluted or discarded in a
   penultimate layer trained only on 9 classes. Using both layers lowers the risk that the
   heart collapses onto the ring or ellipse cluster in conv5. Each distance is divided by
   its median held-out old-class value, so the two layers have equal weight. conv3 was not
   used. In a leave-one-old-class-out check (described below), its distances flagged
   held-out classes poorly, because its features mostly reflect noise and stroke texture.
5. **Threshold.** `tau` is the 99th percentile of the novelty score of held-out old-class
   validation images. The statistics are estimated on one half of the images (split by the
   parity of the index within each class) and measured on the other half, then the halves
   swap. The intended rate of old-class images moved to class 9 is therefore about 1%. The
   final means and covariances use all 1800 images. That makes the test distances of old
   images slightly smaller than the held-out ones, so the threshold is slightly
   conservative.

## Every written value and its formula

| Tensor | Formula (validation images of classes 0-8 only) |
|---|---|
| `mu4[c]`, `mu5[c]` | mean conv4 / conv5 global feature of class c (200 images each) |
| `prec4`, `prec5` | inverse of `(1-0.05) S + 0.05 (tr S / d) I`, where S is the pooled within-class covariance (divisor N - 9). The shrinkage 0.05 is fixed in advance for numerical conditioning and was not tuned. |
| `scale4`, `scale5` | median held-out min-Mahalanobis distance: 47.77 and 67.51 |
| `tau` | 99th percentile of held-out `d4/scale4 + d5/scale5`: 4.2613 |
| `gain` | 1.0. Only the sign of `s - tau` affects the argmax. |

No value was searched or tuned. Each formula and constant was chosen before the validation
evaluation below and was not changed after it.

## Expected effect

- Old classes: each old image keeps its base prediction unless its novelty score exceeds
  `tau`. By construction this should happen to about 1% of old test images, so the expected
  old-class accuracy drop is about 1 point (base model: about 100% on validation).
- Class 9: a heart is detected when its conv4 + conv5 features lie outside the 99th-percentile
  region of every old class. I expect a clear majority of hearts to be flagged. The likely
  failures are small, thick-stroked or heavily noised hearts whose features fall inside the
  ring or ellipse clusters.
  Rough expectation: class-9 accuracy of 50-90%. This cannot be checked without heart
  images.

## Analysis and validation evaluations

Analysis forward passes (not complete candidate edits) used only validation images of
classes 0-8:
- Base logits: 100% validation accuracy. The smallest margins come from ellipse/ring and
  pentagon/square pairs (minimum 1.45).
- Held-out Mahalanobis quantiles (2-fold). conv4: median 47.8, q99 99.1. conv5: median 67.6,
  q99 165.6.
- Proxy check for detection power (leave one old class out of the statistics, flag it at a
  1% false-positive rate). conv4 recall was 0.98-1.0 for every class. conv5 recall was 1.0
  for every class. conv3 recall was 0.0-1.0, so conv3 was excluded. This proxy is
  optimistic, because the base network was trained on the held-out class. It was used only
  to compare layers, not to set any value.

Complete candidate evaluations (1 of the 3 allowed):
1. Final edit, `evaluate_val` (logged in `val_log.jsonl`): old-class validation accuracy
   99.5%. Per class: 100, 98.5, 100, 100, 99.5, 100, 98.5, 99.5, 99.5. 9 of 1800 old images
   were predicted as class 9. These distances are in-sample, so expect about 1% on test.
   The edit is deterministic: two runs of `edit.py` gave identical checkpoint hashes.

## Process notes

- `tools/check_no_gradients.py` prints PASS for `edit.py` and `model_def.py`.
- During the run, the orchestrator reported that the scratchpad directory is shared with
  other concurrent runs. From then on I kept my scratch scripts in a private subfolder
  `w2_run3_private`. Before that note, I had written five scripts directly in the
  scratchpad root and run each one right after writing it. I never listed or opened any
  file there that I did not write. When I went to move my scripts, two of them
  (`look.py` and `look2.py`, which only printed validation images of classes 0-8 as ASCII
  art) were already gone. Something other than this run must have removed them. Nothing
  in this run depends on them.
