# Results

## Stage 1 gate

**FAIL**

- W1 mean new-class accuracy 97.44% vs B0 mean 100.00%: not met (W1 ≥ B0).
- W1 mean old-class drop 0.31 points: met (≤ 2 points).
- W2 (text only) mean new-class accuracy 75.80%: better than chance (10%).

## Stage 1 — add class 9 (heart) to a 9-class shape CNN

Test split: 500 images per class. Accuracies in %, argmax over all 10 outputs. Cells show mean ± sample sd [min–max] over runs (seeds 0–4 for baselines, runs 0–4 for WSP). Old-class drop = base old-class accuracy − condition old-class accuracy (points). Time is the wall-clock time of the condition itself (for WSP: the writer's full session, reading to saved checkpoint).

| ID | Condition | Runs | New-class acc | Old-class acc | Old-class drop | Wall-clock (s) | Weights changed (%) | New params |
|---|---|---|---|---|---|---|---|---|
| base | Base model (class-9 row zero) | 1 | 0.00 | 100.00 | 0 | 232.51 | — | — |
| B0 | Imprinting, unit-norm row, zero bias | 5 | 100.00 ± 0.00 [100.00–100.00] | 99.34 ± 0.06 [99.24–99.40] | 0.66 ± 0.06 [0.60–0.76] | 0.01 ± 0.00 [0.01–0.01] | 0.091 ± 0.000 [0.091–0.091] | 0 ± 0 [0–0] |
| B0s | Imprinting, norm-matched row and mean bias (supplementary) | 5 | 100.00 ± 0.00 [100.00–100.00] | 70.12 ± 3.80 [66.04–74.47] | 29.88 ± 3.80 [25.53–33.96] | 0.01 ± 0.00 [0.01–0.01] | 0.092 ± 0.000 [0.092–0.092] | 0 ± 0 [0–0] |
| B1 | Fine-tune class-9 row only (gradients) | 5 | 97.16 ± 3.64 [91.00–99.80] | 100.00 ± 0.00 [100.00–100.00] | 0.00 ± 0.00 [0.00–0.00] | 15.43 ± 0.21 [15.27–15.78] | 0.092 ± 0.000 [0.092–0.092] | 0 ± 0 [0–0] |
| B2 | Fine-tune whole model with replay (gradients) | 5 | 93.96 ± 6.68 [83.00–99.00] | 100.00 ± 0.00 [100.00–100.00] | 0.00 ± 0.00 [0.00–0.00] | 13.64 ± 0.45 [13.21–14.34] | 100.000 ± 0.000 [99.999–100.000] | 0 ± 0 [0–0] |
| W1 | WSP with k = 5 examples | 5 | 97.44 ± 5.50 [87.60–100.00] | 99.69 ± 0.22 [99.49–99.98] | 0.31 ± 0.22 [0.02–0.51] | 241.00 ± 49.19 [200.00–318.00] | 0.055 ± 0.050 [0.000–0.092] | 516 ± 651 [0–1290] |
| W2 | WSP from text description only | 4 | 75.80 ± 21.86 [47.80–100.00] | 99.06 ± 0.66 [98.29–99.89] | 0.94 ± 0.66 [0.11–1.71] | 285.75 ± 79.54 [220.00–396.00] | 0.023 ± 0.046 [0.000–0.092] | 5846 ± 10924 [0–22212] |

WSP writer: `claude-opus-5-5` (effort: high (session setting; not overridden for the subagent))

### Protocol notes (Stage 1)

- **B0 definition.** The class-9 row is the unit-norm mean penultimate feature of the k examples and the class-9 bias stays at its initial value (zero), following the condition's definition literally. A first implementation also rescaled the row to the mean norm of the trained rows and set the bias to their mean; it is kept as the supplementary row B0s (much larger old-class drop). The gate uses B0.
- **B1 negatives.** Only the class-9 row is trainable, and training on the 5 positive examples alone has no stopping point (the class-9 logit grows without bound). Each B1 step therefore also includes 59 replayed class 0-8 training images as negatives. Old-class logits cannot change in B1.
- **WSP data access.** Writers used `eval/wsp_tools.py`, which exposes only the base checkpoint, the class 0-8 validation images, and (W1) the run's 5 class-9 validation examples. W2 writers passed only class 0-8 validation images through the network. Each writer was allowed at most 3 validation evaluations of complete candidate edits; every run used 1 (`num_val_evaluations` in the run JSON).
- **Run independence.** Each WSP run was a separate writer session with a fresh context and the same fixed instruction. The 5 W1 runs and the 5 W2 runs were launched concurrently, and all ten sessions shared one scratch directory. Several writers found that a parallel run had overwritten or moved a scratch file. Three W1 writers (runs 1, 2, 4) and one W2 writer (run 2) reported seeing fragments of another run's scratch files (key names of a feature file, a printout, or single lines of code). Each writer then moved to a private subfolder and reported that the fragments did not enter its edit. W1 runs 0-4 also produced five different edit designs. Each run's `process_notes` field and `rationale.md` give the details. The runs are kept as valid; this is a deviation from strict isolation. Later stages should give each run its own scratch directory.
