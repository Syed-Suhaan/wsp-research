# Rationale — Stage 1 v2, W1, run 3

## Tensors changed

- `head.weight[9]` and `head.bias[9]` (row 9 only; rows 0-8 unchanged).
- `A1.weight` rows 0-8 and `A1.bias[0:9]` (9 of the 64 adapter units; units 9-63 stay zero).
- `A2.weight[9, 0:9]` (each set to -10; every other A2 entry and `A2.bias` stay zero).
- No backbone tensor and no old-class head row changed.

## Why

Feature analysis (base model, penultimate feature f, validation data only):

- This run has k = 2 examples of the new class. The base model predicts both as class 7
  (pentagon); example 0 is also close to class 5 (ring) (logits 6.69 vs 5.94).
- Cosine similarity of the examples' mean feature to the class means: 0.93 (class 7), 0.78
  (class 5), 0.58 (classes 1, 6), below 0.43 for the rest. A plain imprinted prototype would
  therefore collide with pentagons and rings.
- A heart-vs-pentagon direction alone also scores rings higher than hearts, so a single
  linear score cannot reject all old classes. Each old class needs its own contrast.

Construction (all closed-form, computed in edit.py from forward passes of the frozen model):

1. For each old class c, a diagonal linear-discriminant direction
   `u_c = (mu_new - mu_c) / v_c`, with `v_c = (var_c + mean over classes of var) / 2 + 0.01`
   (diagonal class variance shrunk toward the pooled variance, because the class-specific
   variance of some units is tiny and would overweight them). Its score is standardised on
   the class-c validation images: `z_c = (u_c . f - m_c) / s_c`, so z_c has mean 0 and std 1
   on class c.
2. Decision threshold per contrast: `theta_c = min(0.5 * zbar_c, 4)`, where zbar_c is the
   mean z_c of the k examples. 0.5 * zbar_c is the midpoint between the class-c mean and the
   new-class mean (equal-variance discriminant point). The cap of 4 standard deviations
   keeps far-away classes (zbar of 10-44) from imposing a test that unseen new-class images
   might fail, while still rejecting > 99.99 % of a Gaussian class. Result: theta = 4 for
   every class except pentagon, theta_7 = 2.17 (examples at z_7 = 5.12 and 3.55; the most
   pentagon-like validation pentagon is at about 2.4). Pentagon is the hardest contrast.
3. Adapter unit c computes `t_c - z_c(f)` with `t_c = theta_c + C/B`, and A2 sends
   `-B * ReLU(t_c - z_c(f))` to logit 9 only. This is a soft AND: any contrast on which the
   input is not clearly on the new-class side lowers logit 9 by B = 10 logits per standard
   deviation. An old-class image has z_c near 0 for its own class, so it receives a penalty
   of at least 10 * (4 + 1) = 50 logits (pentagons: 10 * 3.17 = 32). Old logits 0-8 are not
   touched at all, so the only way old accuracy can drop is an old image being predicted 9.
4. Head row 9 = q-weighted sum of old head rows (and biases), where q is the base model's
   mean softmax over the k examples (q = 0.84 pentagon, 0.16 ring, ~0 other). Plus bias
   C = 10. So without penalties, logit 9 = (the base model's evidence for the old classes
   the new examples resemble) + 10, which beats the pentagon/ring logit on new-class images.
   For a pentagon image the effective decision rule becomes z_7 > theta_7 (approximately,
   shifted by 0.16 * (logit5 - logit7)); for all other old classes the decision also has the
   extra margin of their own logit over the inherited pentagon/ring evidence.

## Expected effect

- Old-class accuracy nearly unchanged: only old images that pass all 9 contrasts (in
  practice pentagons with z_7 > ~2.2, about 1 % of pentagons) can flip to class 9. Expected
  old-class drop well under 0.5 points.
- New-class images that resemble pentagons, rings, or ellipses and lie on the new-class side
  of every contrast are predicted 9. The main failure mode is new-class images that look
  more pentagon-like than example 1 (z_7 below about 2.2), and images whose features deviate
  from the two examples in directions the 2-shot means cannot anticipate. With only k = 2
  examples, new-class recall is uncertain; precision of class-9 predictions should be high.

## Validation evaluations

1 evaluation used (of 3), via `evaluate_val` on the final edit:
- val old-class accuracy 98.61 % (base model on the same images: 98.67 %, computed by a base
  forward pass during analysis), per class [100, 99.5, 100, 100, 100, 96.5, 93.0, 99.0,
  99.5]; old-class images predicted as 9: 1 of 1800.
- k examples: both predicted 9 (logit 9 = 16.56 and 22.42 vs best old 6.69 and 14.42).
  These are the examples the edit was built from, so this is a sanity check only.

## Files read

- models/shapes_v2.py
- eval/wsp_tools.py
- eval/common.py
- tools/check_wsp_edit.py
- files in wsp/stage1_v2/W1/run_3/ (own scratch scripts, edit.py, val_log.jsonl)
