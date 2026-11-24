# W1 run 7: rationale

## Tensors changed

- `head.weight[9]`: set to zeros (it was already zero; unchanged in value).
- `head.bias[9]`: 0 -> 3.0 (constant floor `C_FLOOR`).
- `A1.weight[0]`, `A1.bias[0]`: adapter unit 0 = `logit7(f) + BETA*(z(f) - Z_THR) - C_FLOOR`,
  i.e. weight `W_head[7] + BETA*d/s7`, bias `b_head[7] - BETA*m7/s7 - BETA*Z_THR - C_FLOOR`.
- `A1.weight[1]`, `A1.bias[1]`: adapter unit 1 = `BETA*(z(f) - Z_THR) - DELTA`,
  i.e. weight `BETA*d/s7`, bias `-BETA*m7/s7 - BETA*Z_THR - DELTA`.
- `A2.weight[9, 0] = +1`, `A2.weight[9, 1] = -1`.
- Nothing else: backbone, head rows 0-8, all other adapter units, and `A2.bias` are unchanged.

Constants: `C_FLOOR = 3.0`, `BETA = 2.0`, `Z_THR = 1.0`, `DELTA = 1.0`, variance floor 1e-3.

## Why

**What the k examples show.** k = 2 for this run. Both images are outline hearts (one rotated,
one with heavy pixel noise). The base model classifies both as class 7 (pentagon) with large
margins (logit 7 = 12.8 and 8.3; next best about 1.0). Their mean penultimate feature has
cosine 0.99 with the pentagon class mean, and pentagon validation images have median cosine
0.97 to it. So in the 128-d pooled feature space the new class sits inside the pentagon
cluster.

**What I checked.** (a) No conv5 channel is dead (none can be repurposed freely), and no
channel's spatial-max response is rare for pentagons but present in both hearts, so there is
no hidden local "heart" detector to expose by editing the backbone. (b) A heart-vs-pentagon
direction built from one example and tested on the other (leave-one-out, raw or
variance-scaled difference of means) puts the held-out heart only at the 51st-70th percentile
of pentagons: two examples are too few to estimate a direction from the examples alone.

**Direction from shape priors instead.** A heart differs from a pentagon by (1) curved lobes
instead of straight edges and (2) one concave notch. Both properties are represented by old
classes, whose validation means are well estimated (200 images each):
- roundness = mean(circle, ring, ellipse) - mean(square, triangle, pentagon)
- concavity = mean(star, cross, arrow) - mean(all convex classes)
- d = (roundness + concavity) / var_pentagon (per-feature; a diagonal-Gaussian weighting).

d does not use the class-9 examples. Checked against them afterwards, both hearts score
z = 1.16 and 1.01 standard deviations above the pentagon mean along d (88th and 85th
percentile of pentagons), consistent with the prior. The separation is modest, so a split of
the pentagon region is a trade-off, not a clean cut.

**Decision rule written into the weights.** With z(f) = (d.f - m7)/s7 (m7, s7 = mean and std
of d.f over pentagon validation images) and s = BETA*(z - Z_THR), the two adapter units give

    logit9 = C_FLOOR + ReLU(logit7 + s - C_FLOOR) - ReLU(s - DELTA)
           = max(C_FLOOR, logit7 + min(s, DELTA))   (when logit7 + s >= C_FLOOR)

- Inside the pentagon region, logit9 - logit7 = min(s, DELTA), so class 9 wins exactly when
  z > Z_THR = 1.0 (about the top 16% of pentagon-like inputs along d). Z_THR = 1.0 was chosen
  to keep the old-class drop under 2 points (16% of one of nine classes is about 1.8 points).
- The cap DELTA = 1.0 keeps logit 9 at most 1 logit above logit 7. Without it, other classes
  (circles, rings, stars, crosses, arrows) lie far along d and would be pulled to class 9
  (this was the failure of my first evaluation, see below). In the base model only 2 of 1600
  non-pentagon validation images have their top old logit within 1.0 of logit 7.
- The floor C_FLOOR = 3.0 sends inputs on which every old logit is below 3 to class 9 (a
  novel shape that matches no old class). Only 2 of 1800 old validation images have a max
  old logit below 3.0.

## Expected effect

- Old classes: pentagon accuracy falls by about 16 points; others nearly unchanged. Old-class
  accuracy drop about 1.8 points.
- Class 9: hearts that land in the pentagon region (as both examples did) are labelled 9 when
  they are more rounded/concave than about 84% of pentagons. Both examples sit just above the
  threshold, so I expect roughly 40-60% of pentagon-region hearts to be recognised; hearts
  that the base model assigns confidently to another class stay wrong. Class-9 precision
  should be moderate (on the order of 50-75%), since about 16% of pentagons are also
  labelled 9.

## Validation evaluations

Two complete candidate edits were evaluated with `evaluate_val` (base model for reference:
old acc 98.67%, per class [100, 99.5, 100, 100, 100, 96.5, 93, 99.5, 99.5]).

1. Uncapped version (unit 0 only, logit9 = max(3, logit7 + 2(z-1))): old acc 45.78%,
   per class [0, 99, 100, 0, 4, 33.5, 91, 84, 0.5], 959 old images predicted as 9, k-shot
   predictions [9, 9]. This showed that the shape term must be confined to the pentagon
   region; I added the cap unit (a reasoning fix, the constants were not changed).
2. Final version (with cap DELTA = 1.0): old acc 96.89% (drop 1.78 points), per class
   [100, 99, 100, 100, 100, 96.5, 93, 84, 99.5], 32 old images predicted as 9, k-shot
   predictions [9, 9] (logit 9 = 13.14 vs logit 7 = 12.82; 8.35 vs 8.32).

Number of validation evaluations used: 2.

## Files read

- `models/shapes_v2.py`
- `eval/wsp_tools.py`
- `eval/common.py`
- `tools/check_wsp_edit.py`
- Files in my run folder: `edit.py`, scratch analysis scripts in `scratch/`, `val_log.jsonl`
  (written by `evaluate_val`).
