# W2 run 2: rationale

## Summary of the edit

No existing weight of the base model is changed (0 % of base weights changed). The model is
wrapped in a new module `ShapeCNNHeartRule` (`model_def.py`). The base network sits under the
attribute `base`, so its parameter names gain the prefix `base.` (`base.conv1.weight`, ...)
but their values are identical to `checkpoints/stage1_base.pt`. The logits of classes 0-8
are passed through unchanged. The class-9 logit (the zero row of `fc`, which is now unused)
is replaced by a fixed, closed-form rule on the nine old logits, with two new scalar
constants (`heart_offset` = 4.5, `heart_floor` = 5.0; 2 new parameters, not trained):

    pointed = logsumexp(z_triangle, z_square)
    curved  = logsumexp(z_ring, z_ellipse)
    logit_9 = max(min(pointed, curved) + 4.5, 5.0)

## Why

1. **What a heart has in common with old classes.** The description says: an outline
   (not filled), two rounded lobes, a cusp at the top, and a single sharp point at the
   bottom. The old outline classes (looked at on validation images) are square, triangle,
   ring, ellipse and pentagon; circle, star, cross and arrow are filled. A heart outline
   therefore has two kinds of evidence at once:
   - corners joined by nearly straight strokes (the bottom point; the classic heart is a
     square rotated by 45 degrees with two half-circles on its upper sides) -> triangle,
     square;
   - curved outline arcs (the lobes) -> ring, ellipse.
   No old class has both.

2. **The base classifier already separates the two groups strongly.** On validation,
   triangle images give ring logits near -6.2 and ring images give triangle logits near
   -6.8; square images give ring logits near -7.4. So `min(pointed, curved)`, a soft AND
   of the two groups, is low for every old class (median between -8 and +0.6 per class).
   For an image that the network sees as partly cornered and partly curved, both group
   logits are moderate, the AND is moderate, and with the offset it exceeds the (also
   moderate) maximum old logit. Pentagon is not used: its corners are blunt and the base
   model confuses it with square, so including it would raise the AND on real pentagons
   and squares (on validation the minimum old-class margin fell from 5.6 to 2.0 with it).

3. **Offset 4.5.** On the 1,800 old-class validation images, `max_old_logit - min(pointed,
   curved)` is at least 5.61 (worst image: an ellipse; 1st percentile per class >= 5.9).
   An offset of 4.5 keeps about 1.1 logits of headroom on the worst image, so the AND term
   should not move old images to class 9, while giving as much room as possible to hearts.

4. **Floor 5.0.** A general novelty threshold. Old validation images have max old logit
   >= 4.81 (0.5th percentile 5.88, 1st percentile 6.22). A heart that no old class claims
   with a logit of at least 5 is assigned to class 9. Expected cost: about 0.05-0.1 % of
   old images.

## Expected effect

- Old classes: logits unchanged; at most a few very low-confidence old images become
  class 9 (expected old-class drop about 0.1 points).
- Class 9: detected when the heart's logits are split between the cornered group
  (triangle/square) and the curved group (ring/ellipse), or when the base model is not
  confident about any old class. Hearts that the base network confidently calls a single
  old class (for example triangle with logit > 8 and ring/ellipse logits near 0) will be
  missed; without examples there is no direction that separates those from real triangles.
  My expectation is a class-9 accuracy clearly above chance but well below 100 %
  (rough guess 30-60 %).

## Validation evaluations (1 of at most 3 used)

1. `checkpoints/stage1_W2_run2.pt` (offset 4.5, floor 5.0): old-class validation accuracy
   99.94 % (base 100 %); per class 100 % except pentagon 99.5 %; 1 old image predicted as
   class 9 (a pentagon whose max old logit is 4.81, captured by the floor).

Before writing the edit I computed statistics of the base model on old-class validation
images only (class means, per-class logit means and spreads, second-choice matrix, the
distribution of max old logit and of `max_old_logit - min(pointed, curved)` for two
groupings). These set the formulas above; no candidate values were scored in a loop.

## Note on the scratch directory

Early in the run I wrote three scratch scripts directly in the shared scratchpad
directory (not in a private subfolder). One of them was overwritten by another concurrent
run, and when I printed its last line I saw one line of that other run's script
(`print('frac channels active', ... 'dead channels', ...)`). I read nothing else from it,
it did not affect this edit, and from then on I used only the private subfolder
`w2run2_private`.

## Note for evaluation

`eval.common.weight_change_stats` matches parameters by name. Because the base network is
stored under `base.`, a name-based comparison will report all base parameters as "new".
Compare `base.<name>` to `<name>` instead: all 140,778 base values are identical, and the
only new parameters are `heart_offset` and `heart_floor` (2 values).
