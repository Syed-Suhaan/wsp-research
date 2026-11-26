# Decision record: Stage 1 to Stage 2

## Pre-registered outcome

The pre-registered Stage 1 v2 gate is **FAIL**.

The gate (W1 vs B0, 8 paired runs, 10,000 bootstrap resamples) required:
1. 95% CI lower bound of mean(W1 − B0) new-class accuracy ≥ −2 points: **not met**
   (+2.32, CI [−2.55, +6.60]);
2. 95% CI lower bound of mean(W1 − B0) class-9 precision ≥ −2 points: met
   (+22.99, CI [+21.91, +24.01]);
3. W1 mean old-class drop ≤ 2 points: met (0.76).

The gate and its outcome are not changed.

## Decision to continue

The decision to continue to Stage 2 was made **after** seeing the Stage 1 v2.1 results.
It rests on the comparisons with the tuned baselines added in v2.1, which were not part of
the pre-registered gate:

| Comparison (95% CI, 8 pairs) | New-class acc | Class-9 precision | Old-class drop |
|---|---|---|---|
| W1 − B1t | +23.45 [+19.23, +28.15] | +5.96 [+4.02, +8.04] | +0.64 [+0.36, +0.94] |
| W1 − B2t | +80.05 [+72.62, +86.15] | +36.90 [+30.80, +42.93] | +0.01 [−0.45, +0.49] |
| W1 − B0b | +61.75 [+53.12, +68.72] | +27.44 [+25.31, +28.98] | −1.11 [−1.35, −0.85] |

Because this decision was made after the results were known, it is a post-hoc decision.
Readers should weigh Stage 2 with that in mind. Stage 1 does not show that WSP passes its
pre-registered test.

Sources: `RESULTS.md`, `results/stage1_v2/`, `results/stage1_v2_1/`.
