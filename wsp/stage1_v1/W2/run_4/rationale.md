# Stage 1, W2 (text description only), run 4 — rationale

## Summary

No base weight was changed and no parameter was added (0.0 % of weights changed, 0 new
parameters). The model was wrapped in a new class, `HeartRuleCNN` (`model_def.py`, a subclass
of `ShapeCNN` with identical parameter names), whose class-9 logit is produced by a fixed,
parameter-free geometric rule computed from the input image. Logits 0-8 are the unchanged
base logits; row 9 of `fc` stays zero and its output is replaced.

## Why a geometric rule and not a feature-space prototype

With no heart examples, any class-9 row written in the 128-d feature space would be a guess
about where unseen hearts land among features that were trained only to separate classes
0-8. The text description, on the other hand, gives a property that is exact and
rotation/scale invariant: the heart is an **outline** shape with **exactly one concave
corner** (the cusp between the two lobes). Inspection of validation images of classes 0-8
shows how the old classes are drawn:

- outlines (convex): square, triangle, ring, ellipse, pentagon
- filled: circle (convex), star (5 concave corners), cross (4), arrow (2)

So "outline + one concave corner" is satisfied by the heart and by no old class. The rule
encodes exactly this.

## The rule (all closed-form, no learned values)

1. Smooth with a fixed 3x3 binomial kernel; binarise at the midpoint between the image median
   (background) and the image maximum.
2. From the centroid of the binary shape, cast 64 rays; r(theta) = outermost radius at which
   the bilinearly sampled mask exceeds 0.3 (0.3 instead of 0.5 lets a ray pass between two
   diagonally touching stroke pixels). One-ray holes are closed
   (r_i = max(r_i, min(r_{i-1}, r_{i+1}))) and r is smoothed with [1,2,1]/4 around the circle.
3. Polar convexity: with u = 1/r, a closed curve is convex where u + u'' >= 0. The discrete,
   scale-free version, kappa_i = rbar (u_{i+k} + u_{i-k} - 2 cos(a) u_i) / sin(a), a = 22.5 deg
   (k = 4), equals a positively weighted integral of (u + u'') over +-a, is exactly 0 on a
   straight edge, positive on arcs and convex corners, and strongly negative at a concave corner.
4. concavity = max relu(-kappa - 0.25); concentration = largest mass of relu(-kappa - 0.25) in one
   +-22.5 deg window / total mass (1 for a single concave corner, ~0.25 for a star).
5. fill = shape pixels / area inside the outer contour (~0.9-1.0 filled, <= 0.8 outlines).
6. heart_score = min((0.88 - fill)/0.88, (concavity - 0.5)/0.5, (concentration - 0.6)/0.6);
   logit_9 = max_{j<=8} logit_j + 10 * heart_score. Class 9 wins the argmax iff all three
   conditions hold; otherwise the base prediction is returned unchanged.

## Justification of the constants

- floor 0.25: typical (median) raster-noise value of -kappa on convex outline classes
  (square 0.23, pentagon 0.20, ellipse 0.14 before the floor; measured on class 0-8 validation).
- c_thr 0.5 (i.e. kappa < -0.75): a hand estimate of the cusp, assuming a classic heart
  (x = 16 sin^3 t, y = 13 cos t - 5 cos 2t - 2 cos 3t - cos 4t) and accounting for the stroke
  partially filling the narrow notch above the cusp, gives kappa about -2.4 for a medium heart
  (radius ~9.5 px, stroke ~2.25 px), about -2.9 for a large thin one, and about -1 to -1.5 for
  the smallest, thickest hearts. A heart built from two arcs and a V has a wider, deeper notch
  (stronger signal). After the floor, convex outline classes stay below 0.53 at the 99th
  percentile (triangle), so 0.5 sits between old-class noise and the weakest expected hearts.
- q_thr 0.6: star (~0.27) and cross (~0.30) concentrations are far below; a single cusp gives
  ~1. The one cross with fill below 0.88 had concentration 0.60.
- fill_thr 0.88: largest outline value on validation 0.80 (triangle), smallest filled 0.84
  (one cross; stars >= 0.94, arrows >= 0.91, circles >= 0.96). Estimated heart fill:
  ~0.45-0.55 typical, ~0.75 for the smallest heart with the thickest stroke.
- gain 10: only sets the logit scale of the margin; the argmax depends on its sign only.

These constants were set from the reasoning above and from descriptor statistics of
class 0-8 validation images. They were not varied against any measure of class-9 accuracy
(no class-9 image was ever used, drawn or synthesised).

## Expected effect

- Old classes: base logits unchanged; a small number of outline triangles whose rasterised
  tip gives a spurious concave reading switch to class 9 (about 0.3 points overall).
- Class 9: hearts whose cusp survives rasterisation are detected regardless of rotation,
  position, or scale. The main failure mode is a very small heart with a thick stroke whose
  notch fills in completely, or a stroke that breaks up in noise.

## Validation evaluations (1 of the 3 allowed was used)

1. `checkpoints/stage1_W2_run4.pt` (final edit, constants as above): val old-class accuracy
   99.67 % (base 100.0 %), per class [100, 100, 97.5, 100, 99.5, 100, 100, 100, 100],
   old-class images predicted as 9: 6 of 1800.

Before that evaluation, descriptor statistics (fill, concavity, concentration per class) on the
class 0-8 validation images were computed with forward passes of the descriptor function only,
to measure raster noise; this led to adding hole closing, the 0.3 ray threshold, the radius
smoothing, the noise floor, and the fill condition (two single-ray holes in a square and a
pentagon gave spurious concavity, and filled arrows/crosses are concave).

## Note on scratch files

The orchestrator later asked that scratch files go in a private subfolder of the shared
scratchpad. Before that note, I had written two scratch scripts of my own (`look.py`,
`probe.py`) at the scratchpad root; `probe.py` was moved to the private folder and `look.py`
was no longer present when I tried to move it. I did not open, list, or load any file in the
scratchpad that I did not write.
