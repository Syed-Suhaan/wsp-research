# Stage 1, W1, run 2: rationale

## Summary

All base weights are unchanged (0% of the 140,778 base parameters). A new linear layer
`heart_pair` (128 -> 9, 1,161 new parameters) is added, and output 9 is computed as

    logit_9 = max_{c<9} logit_c + min_{c<9} heart_pair_c(f)

where `f` is the 128-d penultimate feature. Outputs 0-8 are exactly the base logits. The
model class is `HeartPairwiseCNN` in `model_def.py` (a subclass of `ShapeCNN`, so all base
parameter names are kept). Row 9 of `fc.weight` and `fc.bias[9]` stay zero and are not used.

## What the base features show (forward passes on validation data)

- The base model classifies all 5 heart examples as class 6 (ellipse), with logit 4.4-8.1.
  For example 3 the runner-up is square (3.2).
- The heart mean feature has cosine 0.935 with the ellipse mean, 0.68 with square, 0.57
  with triangle, 0.54 with pentagon. The 5 heart features are tightly clustered (pairwise
  cosine 0.92-0.99).
- Compared with ellipses, hearts have lower values on channels that are high for
  circle/ring/ellipse (curvature channels, e.g. 16, 25, 50, 81, 106) and higher values on
  channels that are high for square/pentagon/triangle (corner channels, e.g. 29, 97, 102,
  118, 122). A heart is, to this network, an ellipse-like outline with corner evidence
  (the notch and the point).
- So a single imprinted row (heart mean) would be close to the ellipse row and would confuse
  the two classes. The useful information is the difference between hearts and each old
  class, especially ellipse.

## The edit

For each old class c (all statistics are closed-form, from validation features of classes
0-8 and the 5 heart examples):

1. Direction `d_c = (mu_heart - mu_c) / (var_c + 0.01)`, element-wise. This is the linear
   discriminant between two Gaussians with a shared diagonal covariance. The covariance comes
   from class c (200 images), because 5 heart images cannot estimate 128 variances. The
   floor 0.01 (a standard deviation of 0.1, the top of the typical per-dimension range) stops
   near-zero-variance dimensions from receiving unbounded weight.
2. Projection statistics along `d_c`: class-c mean `m_c` and std `s_c`; heart mean `m_h`
   and std `s_h` (n = 5).
3. Threshold `t_c` at the equal-z point, `(t - m_c)/s_c = (m_h - t)/s_h`, that is
   `t_c = (m_c s_h + m_h s_c)/(s_c + s_h)`.
4. `heart_pair_c(f) = (d_c . f - t_c)/s_c`: weight row `d_c / s_c`, bias `-t_c / s_c`.

`min_c heart_pair_c(f) > 0` means the image is on the heart side of all nine pairwise
boundaries. Then `logit_9` exceeds the largest old logit and the prediction is 9; otherwise
`logit_9` is below the largest old logit and the base prediction is kept. This "heart vs
every old class" rule also covers a heart that the base model would call square or
pentagon, which a row tied to one old class would miss.

Resulting thresholds, in class-c standard deviations above the class-c mean (classes 0-8):
10.4, 4.4, 5.9, 10.0, 8.8, 11.6, **2.06**, 5.8, 9.0. Heart mean, same units: 69.4, 19.1,
30.1, 48.6, 33.8, 24.6, **7.7**, 17.5, 44.7. Only the ellipse boundary is tight.

## Expected effect

- Old classes: only ellipse images in the upper tail of the heart-vs-ellipse direction
  (above 2.06 sd) move to class 9, roughly 2-4% of ellipses, i.e. an old-class drop of
  about 0.3-0.5 points. Other classes are 4+ sd from their thresholds.
- New class: on the 5 examples the smallest margin is the ellipse pair (z = 4.35 vs
  threshold 2.06). A leave-one-out check (direction built from 4 hearts, applied to the
  5th; analysis only, not part of the edit) gave held-out ellipse-pair z of 4.4-10.8, so
  the margin is not an artefact of using the same 5 images. Test hearts that are more
  ellipse-like than all 5 examples (e.g. very small or thick-lined hearts with weak
  corner evidence) are the expected failure mode.

## Validation evaluations

One complete candidate was evaluated (`val_log.jsonl`):

| # | Candidate | Val old-class acc | Old images predicted as 9 | 5 heart examples predicted 9 |
|---|-----------|-------------------|---------------------------|------------------------------|
| 1 | this edit (floor 0.01, equal-z thresholds) | 99.61% (base 100%) | 7 (all ellipse; ellipse 96.5%) | 5/5 (logit_9 10.1-14.4 vs best old 4.4-8.1) |

Before writing the edit I also inspected feature statistics on validation data: projections
of each class on the candidate directions, the fraction of each old class above
fixed z-levels, and the leave-one-out check above. These analyses informed the choice of the
diagonal discriminant and the equal-z threshold rule; no weight values were selected by
scoring candidates.

## Process note

The session scratchpad was shared with other runs working at the same time. My first
exploratory scripts (`explore.py`, `proj.py`, `feats.pt`) were in the top level of that
scratchpad, and other processes overwrote files with the same names. As a result I touched
files I did not write, as follows:

1. `proj.py` loaded a `feats.pt` that another process had written. It failed with a
   missing-key error, and I printed only the file's key names (`F`, `Fh`, `y`).
2. I copied the top-level `explore.py` and `proj.py` into a subfolder `w1run2/`. By then
   `explore.py` contained another process's code. I ran it once with its output discarded
   (`> /dev/null`); it wrote a `feats.pt` in that subfolder, which I never used.
3. A `grep` on that copied `explore.py` showed one line of the other code (a `torch.save`
   of base-model features). An `ls -la` of my `w1run2/` subfolder listed only those three
   copied files.

After that I wrote every analysis script and feature file again in a private subfolder,
`w1run2_private/`, and used nothing else. All the statistics in this document come from
those files. Nothing I saw from the other files influenced this edit: I saw no code beyond
that one `torch.save` line, no results, and no weights.

## Checks

- `tools/check_no_gradients.py`: PASS for `edit.py` and `model_def.py`.
- `edit.py` runs in about 9 s and produces a byte-identical checkpoint on repeated runs.
- `models.cnn.load_model("checkpoints/stage1_W1_run2.pt")` loads the model.
