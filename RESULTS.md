# Results

## Stage 1 v2 gate

**FAIL** (W1 vs B0, 8 paired runs, 10,000 bootstrap resamples)

- W1 − B0 new-class accuracy: mean +2.32 [-2.55, +6.60] (95% CI); criterion lower bound ≥ −2: not met.
- W1 − B0 class-9 precision: mean +22.99 [+21.91, +24.01] (95% CI); criterion lower bound ≥ −2: met.
- W1 mean old-class drop: 0.76 points; criterion ≤ 2: met.

Invalid WSP runs (counted as 0% new-class accuracy and 0% precision in the gate and in all CIs): none.

### Paired comparisons with B1 (not part of the gate)

Mean difference with 95% bootstrap CI over the 8 pairs (run i with seed i). A positive old-class-drop difference means the WSP condition loses more old-class accuracy.

| Comparison | New-class acc | Class-9 precision | Old-class drop |
|---|---|---|---|
| W1 − B1 | +25.88 [+20.45, +31.10] | +5.53 [+3.80, +7.16] | +0.62 [+0.33, +0.91] |
| W2 − B1 | — | — | — |

## Calibration disclosure

The Stage 1 v2 setup was chosen by a difficulty calibration over 10 attempts (`results/stage1_v2/calibration.json`). The goal was a B0 mean new-class accuracy between 60% and 80% on the VALIDATION split (seeds 0-7). The selected setup (attempt 9) has the lowest B0 validation accuracy of all attempts (79.2%). All other attempts scored 94.8-100.0%.

Selection rule: apply the levers in order and stop at the first setup in range. (1) k = 2 instead of 5; (2) a new class closer to an old class (rounded square, then hexagon); (3) more noise, blur, and distortion, with a retrained base model. Lever 3 was tried at increasing strength. At each strength the new classes were tried in the order octagon, rounded square, hexagon, with k = 2, and the first setup in range was taken. The base model had to keep old-class test accuracy of at least 95%. This rule was fixed before the lever-3 results were seen.

| Attempt | Lever | New class | k | Noise | Blur | Distortion | Base old test acc | B0 val new acc | In range |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | octagon | 5 | 0.03–0.12 | 0.0–0.0 | 0.0 | 100.00 | 100.0 ± 0.0 | no |
| 1 | 1 | octagon | 2 | 0.03–0.12 | 0.0–0.0 | 0.0 | 100.00 | 100.0 ± 0.0 | no |
| 2 | 2 | rounded_square | 2 | 0.03–0.12 | 0.0–0.0 | 0.0 | 100.00 | 100.0 ± 0.0 | no |
| 3 | 2 | hexagon | 2 | 0.03–0.12 | 0.0–0.0 | 0.0 | 100.00 | 100.0 ± 0.0 | no |
| 4 | 3 | octagon | 2 | 0.05–0.15 | 0.2–0.6 | 0.1 | 99.84 | 97.3 ± 0.5 | no |
| 5 | 3 | rounded_square | 2 | 0.05–0.15 | 0.2–0.6 | 0.1 | 99.84 | 99.5 ± 0.8 | no |
| 6 | 3 | hexagon | 2 | 0.05–0.15 | 0.2–0.6 | 0.1 | 99.84 | 94.8 ± 1.3 | no |
| 7 | 3 | octagon | 2 | 0.08–0.2 | 0.3–0.8 | 0.2 | 98.42 | 97.0 ± 1.0 | no |
| 8 | 3 | rounded_square | 2 | 0.08–0.2 | 0.3–0.8 | 0.2 | 98.42 | 98.8 ± 0.4 | no |
| 9 | 3 | hexagon | 2 | 0.08–0.2 | 0.3–0.8 | 0.2 | 98.42 | 79.2 ± 4.3 | yes |

## Stage 1 v2 — main table (TEST split)

Mean ± sample sd [min–max] over runs (baselines: seeds 0-7; WSP: runs 0-7; run i uses the same k examples as seed i). Accuracies and rates in %. Held-out FPR is the fraction of held-out images predicted as class 9; 'near' is the held-out shape close to the new class, 'far' the other one. The means include invalid WSP runs at their measured values; the gate counts them as 0.

| ID | Condition | Runs | New-class acc | Class-9 precision | Held-out FPR near | Held-out FPR far | Held-out FPR all | Old-class acc | Old-class drop | Wall-clock (s) | Weights changed (%) | Slot params used |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | Base model (class-9 row, bias, adapter zero) | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 98.42 | 0.00 | 269.57 | — | 0 |
| B0 | Imprinting: unit-norm mean feature, bias 0 | 8 | 78.25 ± 4.46 [68.20–82.20] | 25.73 ± 1.51 [22.17–27.15] | 94.22 ± 2.91 [88.00–97.20] | 41.95 ± 7.86 [28.80–51.40] | 68.09 ± 5.22 [58.40–74.30] | 88.92 ± 1.12 [87.51–91.04] | 9.50 ± 1.12 [7.38–10.91] | 0.02 ± 0.03 [0.01–0.08] | 0.085 ± 0.000 [0.085–0.086] | 0 ± 0 [0–0] |
| B0s | Imprinting, norm-matched row and mean bias (supplementary) | 8 | 100.00 ± 0.00 [100.00–100.00] | 15.12 ± 0.47 [14.06–15.63] | 100.00 ± 0.00 [100.00–100.00] | 97.80 ± 2.33 [93.20–100.00] | 98.90 ± 1.17 [96.60–100.00] | 59.54 ± 2.24 [54.27–61.49] | 38.88 ± 2.24 [36.93–44.16] | 0.01 ± 0.01 [0.01–0.02] | 0.086 ± 0.000 [0.086–0.086] | 0 ± 0 [0–0] |
| B1 | Fine-tune class-9 row and bias (gradients) | 8 | 54.70 ± 10.92 [36.40–70.00] | 43.18 ± 3.80 [35.07–46.94] | 69.95 ± 10.29 [48.40–83.20] | 0.23 ± 0.36 [0.00–0.80] | 35.09 ± 5.21 [24.20–41.60] | 98.28 ± 0.13 [98.00–98.40] | 0.15 ± 0.13 [0.02–0.42] | 16.20 ± 3.59 [14.24–24.96] | 0.086 ± 0.000 [0.086–0.086] | 0 ± 0 [0–0] |
| B2 | Fine-tune whole model with replay (gradients) | 8 | 1.05 ± 1.04 [0.00–2.60] | 16.52 ± 12.87 [0.00–38.24] | 1.32 ± 1.41 [0.00–4.20] | 1.30 ± 1.43 [0.00–4.20] | 1.31 ± 0.87 [0.40–3.00] | 97.67 ± 0.51 [96.87–98.22] | 0.75 ± 0.51 [0.20–1.56] | 14.99 ± 1.16 [13.04–16.25] | 94.050 ± 0.000 [94.050–94.050] | 0 ± 0 [0–0] |
| W1 | WSP with the k examples | 8 | 80.58 ± 10.63 [60.00–91.60] | 48.72 ± 2.83 [42.19–51.65] | 76.95 ± 4.63 [70.80–83.20] | 0.28 ± 0.41 [0.00–1.00] | 38.61 ± 2.38 [35.40–42.10] | 97.66 ± 0.54 [96.58–98.33] | 0.76 ± 0.54 [0.09–1.84] | 252.75 ± 30.87 [219.00–313.00] | 0.397 ± 0.360 [0.086–0.868] | 482 ± 529 [0–1170] |
| W2 | WSP from a text description only | 2 | 88.40 ± 0.85 [87.80–89.00] | 46.84 ± 3.05 [44.68–49.00] | 86.30 ± 2.12 [84.80–87.80] | 8.80 ± 12.45 [0.00–17.60] | 47.55 ± 7.28 [42.40–52.70] | 97.83 ± 0.14 [97.73–97.93] | 0.59 ± 0.14 [0.49–0.69] | 406.00 ± 9.90 [399.00–413.00] | 0.390 ± 0.184 [0.260–0.520] | 455 ± 276 [260–650] |

### W1 per run

| Run | Edit type | New acc | Precision | FPR near | FPR far | Old acc | Old drop | Weights changed (%) | Slot params | Val evals | Wall-clock (s) | Valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | mixed | 91.6 | 49.6 | 83.2 | 1.0 | 97.47 | 0.96 | 0.173 | 130 | 1 | 251 | yes |
| 1 | mixed | 86.0 | 50.4 | 79.4 | 0.0 | 97.82 | 0.60 | 0.781 | 1040 | 1 | 241 | yes |
| 2 | mixed | 83.4 | 49.2 | 83.0 | 0.0 | 98.11 | 0.31 | 0.173 | 130 | 1 | 253 | yes |
| 3 | mixed | 69.8 | 49.4 | 70.8 | 0.0 | 98.33 | 0.09 | 0.868 | 1170 | 1 | 282 | yes |
| 4 | row-only | 79.2 | 49.3 | 72.2 | 0.4 | 97.44 | 0.98 | 0.086 | 0 | 1 | 219 | yes |
| 5 | row-only | 87.4 | 51.7 | 74.4 | 0.0 | 97.60 | 0.82 | 0.086 | 0 | 1 | 232 | yes |
| 6 | mixed | 60.0 | 42.2 | 77.0 | 0.0 | 97.91 | 0.51 | 0.838 | 1125 | 1 | 231 | yes |
| 7 | mixed | 87.2 | 48.1 | 75.6 | 0.8 | 96.58 | 1.84 | 0.174 | 260 | 2 | 313 | yes |

### W2 per run

| Run | Edit type | New acc | Precision | FPR near | FPR far | Old acc | Old drop | Weights changed (%) | Slot params | Val evals | Wall-clock (s) | Valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | mixed | 89.0 | 44.7 | 87.8 | 17.6 | 97.93 | 0.49 | 0.520 | 650 | 2 | 413 | yes |
| 1 | mixed | 87.8 | 49.0 | 84.8 | 0.0 | 97.73 | 0.69 | 0.260 | 260 | 1 | 399 | yes |

### WSP edit types

| Condition | Edit type | Runs | Mean new-class acc | Mean class-9 precision |
|---|---|---|---|---|
| W1 | mixed | 6 | 79.67 | 48.13 |
| W1 | row-only | 2 | 83.30 | 50.48 |
| W2 | mixed | 2 | 88.40 | 46.84 |

WSP writer: `claude-opus-5-5` (effort level: unverified). The effort level could not be confirmed from a record outside the model (`results/stage1_v2/effort_probe.json`).

### Protocol notes (Stage 1 v2)

- **Fixed model.** Every condition is evaluated by loading a state dict into `models/shapes_v2.ShapesV2Net` with strict=True. Edits can change behaviour only through tensor values (head, a zero-initialised adapter slot A2(ReLU(A1(f))), or the backbone).
- **B1 negatives.** Only the class-9 head row and bias are trained. Each step uses the k examples as positives and 59 replayed class 0-8 training images as negatives.
- **B2 underfits the new class.** B2 uses the hyperparameters carried over unchanged from v1 (200 Adam steps, lr 3e-4, 2 positives and 59 replayed images per step). With k = 2 this barely moves the class-9 logit. They were not tuned after the v2 results were seen. B2 is not part of the gate.
- **W1 class name.** The W1 instruction calls the new class 'class 9' and gives only the k example images, not the shape's name. W2 gets only a text description of the shape, with no images.
- **WSP isolation.** WSP runs ran one at a time, each as a writer subagent with a fresh context. Each writer had its own scratch folder (`wsp/stage1_v2/<condition>/run_<n>/scratch/`, also its TMPDIR) and a fixed instruction (`prompts/stage1_v2_<condition>.md`). Writers read data only through `eval/wsp_tools.py` (base checkpoint, class 0-8 validation images, and for W1 the run's k class-9 validation images). Each writer lists the files it read in `rationale.md`. The project instruction file may be loaded into a writer's context automatically by the agent harness. It does not describe the held-out classes or the test set.
- **Validity.** Each run is checked by `tools/check_wsp_edit.py`: no gradients, fitting, or candidate-scoring loop; nothing but tensor values changed; strict load into the fixed class; at least one changed tensor; no change under models/, eval/, data/ since the run started (git).
- **Held-out classes.** Two shape classes appear only in the test split (500 images each): one visually near the new class, one far from it. They were never shown to any model, baseline, or writer. One of them was replaced before any v2 condition ran (`results/stage1_v2/heldout_note.json`).

---

# Stage 1 v1 (pilot, frozen)

The v1 pilot table below is unchanged. A later re-check with the v2 rules (`results/stage1_v1/recheck.json`) marks v1 W2 run 4 invalid (pixel rule outside the network, no tensor changed). Six other v1 WSP runs used a new layer or module, which the v1 instruction allowed and the v2 rules do not. The v1 table still includes run 4 as it was recorded.


### Stage 1 gate

**FAIL**

- W1 mean new-class accuracy 97.44% vs B0 mean 100.00%: not met (W1 ≥ B0).
- W1 mean old-class drop 0.31 points: met (≤ 2 points).
- W2 (text only) mean new-class accuracy 78.32%: better than chance (10%).

### Stage 1 — add class 9 (heart) to a 9-class shape CNN

Test split: 500 images per class. Accuracies in %, argmax over all 10 outputs. Cells show mean ± sample sd [min–max] over runs (seeds 0–4 for baselines, runs 0–4 for WSP). Old-class drop = base old-class accuracy − condition old-class accuracy (points). Time is the wall-clock time of the condition itself (for WSP: the writer's full session, reading to saved checkpoint).

| ID | Condition | Runs | New-class acc | Old-class acc | Old-class drop | Wall-clock (s) | Weights changed (%) | New params |
|---|---|---|---|---|---|---|---|---|
| base | Base model (class-9 row zero) | 1 | 0.00 | 100.00 | 0 | 232.51 | — | — |
| B0 | Imprinting, unit-norm row, zero bias | 5 | 100.00 ± 0.00 [100.00–100.00] | 99.34 ± 0.06 [99.24–99.40] | 0.66 ± 0.06 [0.60–0.76] | 0.01 ± 0.00 [0.01–0.01] | 0.091 ± 0.000 [0.091–0.091] | 0 ± 0 [0–0] |
| B0s | Imprinting, norm-matched row and mean bias (supplementary) | 5 | 100.00 ± 0.00 [100.00–100.00] | 70.12 ± 3.80 [66.04–74.47] | 29.88 ± 3.80 [25.53–33.96] | 0.01 ± 0.00 [0.01–0.01] | 0.092 ± 0.000 [0.092–0.092] | 0 ± 0 [0–0] |
| B1 | Fine-tune class-9 row only (gradients) | 5 | 97.16 ± 3.64 [91.00–99.80] | 100.00 ± 0.00 [100.00–100.00] | 0.00 ± 0.00 [0.00–0.00] | 15.43 ± 0.21 [15.27–15.78] | 0.092 ± 0.000 [0.092–0.092] | 0 ± 0 [0–0] |
| B2 | Fine-tune whole model with replay (gradients) | 5 | 93.96 ± 6.68 [83.00–99.00] | 100.00 ± 0.00 [100.00–100.00] | 0.00 ± 0.00 [0.00–0.00] | 13.64 ± 0.45 [13.21–14.34] | 100.000 ± 0.000 [99.999–100.000] | 0 ± 0 [0–0] |
| W1 | WSP with k = 5 examples | 5 | 97.44 ± 5.50 [87.60–100.00] | 99.69 ± 0.22 [99.49–99.98] | 0.31 ± 0.22 [0.02–0.51] | 241.00 ± 49.19 [200.00–318.00] | 0.055 ± 0.050 [0.000–0.092] | 516 ± 651 [0–1290] |
| W2 | WSP from text description only | 5 | 78.32 ± 19.75 [47.80–100.00] | 99.23 ± 0.69 [98.29–99.93] | 0.77 ± 0.69 [0.07–1.71] | 324.20 ± 110.17 [220.00–478.00] | 0.018 ± 0.041 [0.000–0.092] | 4677 ± 9815 [0–22212] |

WSP writer: `claude-opus-5-5` (effort: high (session setting; not overridden for the subagent))

#### Protocol notes (Stage 1)

- **B0 definition.** The class-9 row is the unit-norm mean penultimate feature of the k examples and the class-9 bias stays at its initial value (zero), following the condition's definition literally. A first implementation also rescaled the row to the mean norm of the trained rows and set the bias to their mean; it is kept as the supplementary row B0s (much larger old-class drop). The gate uses B0.
- **B1 negatives.** Only the class-9 row is trainable, and training on the 5 positive examples alone has no stopping point (the class-9 logit grows without bound). Each B1 step therefore also includes 59 replayed class 0-8 training images as negatives. Old-class logits cannot change in B1.
- **WSP data access.** Writers used `eval/wsp_tools.py`, which exposes only the base checkpoint, the class 0-8 validation images, and (W1) the run's 5 class-9 validation examples. W2 writers passed only class 0-8 validation images through the network. Each writer was allowed at most 3 validation evaluations of complete candidate edits; every run used 1 (`num_val_evaluations` in the run JSON).
- **Run independence.** Each WSP run was a separate writer session with a fresh context and the same fixed instruction. The 5 W1 runs and the 5 W2 runs were launched concurrently, and all ten sessions shared one scratch directory. Several writers found that a parallel run had overwritten or moved a scratch file. Three W1 writers (runs 1, 2, 4) and one W2 writer (run 2) reported seeing fragments of another run's scratch files (key names of a feature file, a printout, or single lines of code). Each writer then moved to a private subfolder and reported that the fragments did not enter its edit. W1 runs 0-4 also produced five different edit designs. Each run's `process_notes` field and `rationale.md` give the details. The runs are kept as valid; this is a deviation from strict isolation. Later stages should give each run its own scratch directory.
- **What the WSP edits are.** W1: runs 0 and 3 rewrote only the class-9 output row (as offsets from the ellipse row, or as the sum of the square and ellipse rows); runs 1, 2, and 4 added a small closed-form head on the penultimate feature (class-mean prototypes, pairwise heart-vs-class tests, or a capped heart-vs-ellipse score). W2: runs 0 and 3 make class 9 a novelty output (an image far from every old class is called class 9). This works because the test set contains no classes beyond 0-9. Run 1 uses the class-9 row as the mean of the five outlined-shape rows with a margin bias. Run 2 uses a logical rule on the old logits (corner-like AND curve-like evidence). Run 4 changes no weights: it adds a hand-written geometric test on the input pixels (contour concavity) that decides class 9. That is programming of the forward pass rather than of weights. It passes the gradient check and is counted as valid, but it should be read with this caveat.
