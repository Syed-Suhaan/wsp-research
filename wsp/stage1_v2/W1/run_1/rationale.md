# W1 run 1: rationale

## Tensors changed

- `head.weight[9]` (128 values) and `head.bias[9]`: set to head row 7 plus a calibrated
  9-versus-7 discriminant (formula below).
- `A1.weight[0:8]`, `A1.bias[0:8]`: eight adapter units, one for each old class
  c in {0, 1, 2, 3, 4, 5, 6, 8}. Unit j computes ReLU(-s_c(f)).
- `A2.weight[9, 0:8]`: set to -1, so each unit lowers logit 9 only.
- Everything else is unchanged: backbone, head rows 0-8, A1 units 8-63, all other A2
  rows, and `A2.bias`.

## Why

There are k = 2 class-9 examples. The base model predicts class 7 (pentagon) for both, with
margins of about 7 and 9 logits over the next class. Their penultimate features f are almost
identical (cosine 0.99) and have cosine 0.94-0.95 with the pentagon mean, 0.75 with the
ring mean, and 0.62 or less with the other class means. So the network already places the new
class next to class 7, and the main job is to separate it from class 7 without harming class 7.

Model of each class: a Gaussian on f with diagonal covariance equal to the pooled old-class
within-class variance `var` (plus a floor of 1% of its mean). For each old class c:

- direction `w_c = (mu_9 - mu_c) / v` (diagonal-Gaussian discriminant);
- on the 1-D projection `w_c . f`: class-c mean `m_c` and std `sd_c` from its 200 validation
  images; class-9 mean `m9_c` = mean projection of the k examples minus a bias correction
  `B = sum(var / v) / k = 63.3`. The direction was built from the same k samples, so their
  own projection is inflated by `E[noise' V^-1 noise] = tr(V^-1 Sigma_9) / k`; Sigma_9 is
  assumed equal to the pooled old-class covariance, because 2 samples cannot estimate it.
  For c = 7 this lowers the class-9 mean from about 379 above the pentagon mean to 316.
- `s_c(f) = alpha_c (w_c . f - t_c)`, with `alpha_c = (m9_c - m_c) / sd_c^2` and
  `t_c = (m9_c + m_c) / 2`. This is the log-likelihood ratio of two 1-D Gaussians with a
  common std. It is zero halfway between the class means and is measured in nats. A
  diagonal model on correlated CNN features overstates the evidence: the measured pentagon
  std on the projection is about 5 times the value the diagonal model implies. Calibrating on
  the measured 1-D std puts s_c on the same scale as the head's softmax logits.

The new logit is

    logit_9 = logit_7 + s_7(f) - sum_{c != 7} ReLU(-s_c(f)).

- Head row 9 = head row 7 + alpha_7 w_7, bias = b_7 - alpha_7 t_7. Then logit_9 - logit_7
  equals s_7, so class 9 beats class 7 exactly when the feature is on the class-9 side of
  the 9-versus-7 midpoint. On the projection, class 9 is about 3.0 pentagon std units from
  class 7 after the bias correction.
- The 9-versus-7 direction also points away from other classes, mainly ring (class 5): ring
  images have large positive s_7. A linear row alone would therefore move rings and some
  ellipses to class 9. Each adapter unit is an "is more like class c than class 9" detector.
  It is zero on the class-9 side of the 9-versus-c midpoint and grows linearly in nats on the
  class-c side. With A2 weight -1, logit 9 loses that many nats. For classes c other than 7,
  the k examples lie 7 to 27 class-c std units from class c on these projections, so the
  penalties are zero for class-9-like inputs and large for class-c images.
- The threshold is the midpoint, which corresponds to equal priors. It was not tuned on
  validation data. The anchor class 7 comes from the base predictions on the k examples.
  `PENALTY_WEIGHT = 1` keeps the penalty in the same unit (nats) as the logits.

## Expected effect

- Old classes: only images whose s_7 > 0 can change. Under the 1-D Gaussian model this is
  about Phi(-1.5), or 6-7%, of pentagons, plus almost no other images. Expected old-class
  validation accuracy drop: about 0.7-0.9 points.
- Class 9: if new-class features scatter along the projection about as much as pentagons
  do, recall is about Phi(1.5), or 93%, and class-9 precision is about 92%. If unseen class-9
  images vary more than the two examples suggest, recall will be lower. Class-9 images that
  the base model assigns to a class other than pentagon still score highly when they are on
  the class-9 side of every pairwise midpoint, because logit_9 >= logit_7 + s_7 and s_7 is
  large there.

## Validation evaluations

1 of 3 used (`evaluate_val`, logged in `val_log.jsonl`). It was followed by a breakdown of
the same candidate's predictions by true class, using my own forward pass:

1. Final edit (this edit.py): old-class validation accuracy 97.83% (base 98.67%, drop 0.83
   points). Per class: 100, 99.5, 100, 100, 100, 96.5, 93.0, 92.0, 99.5 (base: 100, 99.5,
   100, 100, 100, 96.5, 93.0, 99.5, 99.5). Old images predicted as 9: 16 (15 pentagons,
   1 ellipse that the base model already misclassified). Both k examples are predicted as 9
   (logit 9 = 16.7 and 19.1 versus logit 7 = 10.6 and 12.2). This matches the expected
   effect: 7.5% of pentagons against the 6-7% predicted. No values were changed afterwards.

Before the edit, analysis used only forward passes and statistics: features of the validation
images of classes 0-8 and of the k examples, the base predictions, and per-class
distributions of each pairwise projection, computed in scratch/.

## Files read

- `models/shapes_v2.py`
- `eval/wsp_tools.py`
- `eval/common.py`
- `tools/check_wsp_edit.py`
- Files in `wsp/stage1_v2/W1/run_1/`: my own scratch scripts, `edit.py`, and `val_log.jsonl`.
