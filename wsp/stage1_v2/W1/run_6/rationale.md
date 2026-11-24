# WSP edit rationale: Stage 1 v2, W1, run 6

Output: `checkpoints/stage1_v2_W1_run6.pt` (written by `edit.py`).
k = 2 class-9 examples were provided (`k_shot_examples(6)`).

## Tensors changed

- `head.weight[9]` and `head.bias[9]`: row 9 becomes a copy of row 7 (pentagon) plus a linear
  "pentagon-anomaly" score s(f) − TAU.
- `A1.weight[0]`, `A1.bias[0]`, and `A2.weight[9, 0]`: a cap unit that limits the class-9 bonus
  over logit 7 to CAP = 4 logits.
- `A1.weight[1:9]`, `A1.bias[1:9]`, and `A2.weight[9, 1:9]`: eight gate units, one for each old
  class c in {0,1,2,3,4,5,6,8}. Each unit is h = ReLU(logit_c − logit_7 − 1), and its weight
  into logit 9 is −4.
- Unchanged: every backbone tensor, head rows 0–8, `A2.bias`, A1 units 9–63, and A2 rows 0–8.
  Old-class logits are therefore exactly the base logits. Only logit 9 can change a prediction.

## Why

1. **Where the examples are.** Both examples (they look like hearts: two upper lobes and a
   lower point) are predicted as class 7 by the base model, with logit7 = 10.2 and 4.0. In
   penultimate-feature space they lie inside the pentagon cluster. The cosine of the mean
   example feature to the pentagon mean is 0.947. Many pentagons have a higher cosine to each
   example than the two examples have to each other (0.94). An imprinted class-9 row, which
   is a cosine prototype, would therefore take many pentagons. The new class must be
   separated from class 7 along a discriminative direction, not by a prototype.
2. **Discriminative direction.** I whitened each feature dimension with the pentagon
   validation statistics: z = (f − μ7) / (σ7 + 0.05). The floor of 0.05 is small relative to
   typical feature values of 0.1–1.5. It stops near-constant dimensions from dominating. The
   two examples agree in which dimensions they deviate, for example 14, 16, 25, 32, 50, 81,
   106, and 117, all about +2 to +4 sd above the pentagon mean. With only two examples, I
   took a conservative consensus: per dimension, the smaller |z| of the two where the signs
   agree, and 0 otherwise (83 of 128 dimensions are non-zero). The normalised vector is u.
   The score is r(f) = u · z(f), standardised by the mean and std of r over validation
   pentagons. So s(f) is in pentagon standard deviations, and s is linear in f.
   - On validation, the examples score s = 3.54 and 3.96. Pentagons have a maximum of 2.92
     and a 99th percentile of 2.52.
   - A leave-one-out check gave held-out scores lower than in-sample ones. A direction from
     example 0 put example 1 above all pentagons. A direction from example 1 put example 0
     above 89% of pentagons. Unseen hearts will therefore score lower than the two examples.
3. **Threshold.** TAU = 2.0 sd is a standard 2-sigma rule. I set it below the in-sample
   example scores to allow for the leave-one-out shrinkage. It accepts about 2.5% of
   validation pentagons, which is a small old-class cost against recall on class 9.
4. **Head row 9 = row 7 + (s − TAU).** This gives logit9 − logit7 = s − TAU. Class 9 wins over
   class 7 exactly when the image deviates from the pentagon cluster in the examples'
   direction by more than TAU. It also inherits logit 7's margin over the other classes for
   pentagon-like images.
5. **Cap (adapter unit 0).** Images far from the pentagon cluster also get a large s,
   especially rings and ellipses at 5–12 sd. Without a limit, s − TAU could exceed their
   margin. With h0 = ReLU(s − TAU − 4) and weight −1, the bonus becomes min(s − TAU, 4).
6. **Gates (adapter units 1–8).** When the base head assigns an image to some other old
   class c by more than 1 logit over class 7, h_c = ReLU(logit_c − logit_7 − 1) is positive.
   Logit 9 then loses 4 logits per unit, so class 9 cannot take over clear old-class images.
   - With the bonus capped at 4, an old image of class c can be lost only if its lead g over
     class 7 satisfies g + 4(g − 1) < 4, that is g < 1.6. On validation, the 1st percentile of
     this lead is at least 3.2 for every class except square. Squares have s far below TAU.
   - The MARGIN of 1 logit keeps class 9 available for new-class images that the base model
     sees as nearly tied between pentagon and a round class. Example 1 has logit5 − logit7 =
     −1.1.

## Expected effect

- Class 9 is predicted for pentagon-like images whose features deviate in the hearts'
  consensus direction by more than 2 pentagon-sd. Recall on unseen hearts should be clearly
  above zero. It will be below 100% because some hearts will be less anomalous than the two
  examples, or will be read by the base model as ring or ellipse rather than pentagon. The
  gate then suppresses class 9 for them.
- Old classes: small drop, mainly a few pentagons (about 2–3%) and borderline rings and
  ellipses. I expect an overall old-class drop well under 1 point.

## Validation evaluations

1 evaluation used (`evaluate_val`, logged in `val_log.jsonl`) on the final edit:
- Old-class validation accuracy is 98.44%. The base, computed with my own forward pass while
  analysing, is 98.67%, a drop of 0.22 points.
- Per class: [100, 99.5, 100, 100, 100, 96.5, 93.0, 97.5, 99.5].
- Old images predicted as class 9: 4 of 1800.
- k-shot predictions are [9, 9]. Logit 9 beats logit 7 by 1.54 and 1.96.

For comparison, the base per-class validation accuracy, from the base logits computed while
analysing, is [100, 99.5, 100, 100, 100, 96.5, 93.0, 99.5, 99.5]. The only change is
pentagon, from 99.5% to 97.5%. All 4 old images newly predicted as class 9 are pentagons.
Classes 5 and 6 are unchanged from the base.

No other candidate edit was built or evaluated. Earlier scratch analysis computed feature
statistics only: class means and spreads, example z-scores, the leave-one-out score check,
score distributions per class, and the base logit gaps.

## Files read

- `models/shapes_v2.py`
- `eval/wsp_tools.py`
- `eval/common.py`
- `tools/check_wsp_edit.py`
- Files in my own run folder: `edit.py`, `scratch/*.py`, `scratch/feats.pt`, and
  `val_log.jsonl`.
- `checkpoints/stage1_v2_base.pt`, through the tools. I also wrote
  `checkpoints/stage1_v2_W1_run6.pt`.
