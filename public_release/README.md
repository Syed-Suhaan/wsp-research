# Weight-space programming: can a model add a capability to a trained network without gradients?

## Research question

Can weight-space programming (WSP) add a new capability to a trained neural network?

**WSP** means: a writer model reads a trained network's weights (and may run forward passes
and read activations on validation data), then writes new weights by reasoning. It does not
use gradients, optimizers, or automated fitting. Each edit is a short Python script
(`wsp/<stage>/<condition>/run_<n>/edit.py`) that sets weights to explicit values or explicit
closed-form formulas, with a written rationale (`rationale.md`) next to it.

- **Stage 1 (gate):** add a new class (heart) to a small CNN that was trained on 9 other
  synthetic shape classes.
- **Stage 2:** add a new voice style to the tiny-voca text-to-speech model (not yet run).

## Stage 1 method

**Data** (`data/shapes.py`): 32x32 grayscale images of 10 shape classes (circle, square,
triangle, star, cross, ring, ellipse, pentagon, arrow, heart) with random size, position,
rotation, line width, brightness, and pixel noise. Per class: 2000 train, 200 validation,
500 test images; seed 0. Class 9 (heart) is the new class. The base model never sees it.

**Base model** (`models/cnn.py`, `train/train_base.py`): a 5-layer CNN (140,778 parameters)
trained on classes 0-8, then given a zero-initialised 10th output row, so that every
condition starts from the same 10-output model.

**Conditions.** Every condition that uses examples uses the same k = 5 class-9 images from
the validation split, chosen per seed (baselines) or per run number (WSP) by
`eval/common.py:k_shot_indices`.

| ID | Condition | Uses gradients |
|----|-----------|----------------|
| B0 | Imprinting: class-9 row = unit-norm mean penultimate feature of the k examples (bias 0) | No (fixed formula) |
| B0s | Supplementary imprinting variant: row rescaled to the mean old-row norm, bias = mean old bias | No (fixed formula) |
| B1 | Fine-tune the class-9 output row only, k examples (old-class training images as negatives) | Yes |
| B2 | Fine-tune the whole model, k examples plus replay of old classes | Yes |
| W1 | WSP with the k examples. A new layer is permitted. | No |
| W2 | WSP from a text description of a heart only. No examples. | No |

Baselines run once per seed (0-4). Each WSP condition has 5 independent runs, each written
by a separate writer session with a fresh context. All runs of a condition receive the same
fixed instruction (`prompts/stage1_<condition>.md`), with only the run number changed.
During a WSP run the writer may read the base weights and run forward passes on validation
data only (old classes 0-8, plus the 5 examples for W1). It never reads the test split.
Every `edit.py` is checked by `tools/check_no_gradients.py`. A run that fails the check is
recorded as invalid and excluded from means.

**Metrics** (test split): new-class accuracy (class 9), old-class accuracy (classes 0-8) and
its drop from the base model (argmax over all 10 outputs), wall-clock time, percentage of
weights changed, and number of new parameters.

**Gate:** PASS if the W1 mean new-class accuracy is at least the B0 mean AND the W1 mean
old-class drop is at most 2 points. The table also reports whether W2 beats chance (10%).

## WSP writer

- `claude-opus-5-5` (effort level: high (session setting; not overridden for the subagent))

The writer model ID and effort level are recorded in every WSP run JSON
(`results/<stage>/<condition>/run_<n>.json`).

## Hardware

4 vCPU, 15 GiB RAM, no GPU. CPU PyTorch, 4 threads, deterministic algorithms, fixed seeds.

## Results

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

### Protocol notes (Stage 1)

- **B0 definition.** The class-9 row is the unit-norm mean penultimate feature of the k examples and the class-9 bias stays at its initial value (zero), following the condition's definition literally. A first implementation also rescaled the row to the mean norm of the trained rows and set the bias to their mean; it is kept as the supplementary row B0s (much larger old-class drop). The gate uses B0.
- **B1 negatives.** Only the class-9 row is trainable, and training on the 5 positive examples alone has no stopping point (the class-9 logit grows without bound). Each B1 step therefore also includes 59 replayed class 0-8 training images as negatives. Old-class logits cannot change in B1.
- **WSP data access.** Writers used `eval/wsp_tools.py`, which exposes only the base checkpoint, the class 0-8 validation images, and (W1) the run's 5 class-9 validation examples. W2 writers passed only class 0-8 validation images through the network. Each writer was allowed at most 3 validation evaluations of complete candidate edits; every run used 1 (`num_val_evaluations` in the run JSON).
- **Run independence.** Each WSP run was a separate writer session with a fresh context and the same fixed instruction. The 5 W1 runs and the 5 W2 runs were launched concurrently, and all ten sessions shared one scratch directory. Several writers found that a parallel run had overwritten or moved a scratch file. Three W1 writers (runs 1, 2, 4) and one W2 writer (run 2) reported seeing fragments of another run's scratch files (key names of a feature file, a printout, or single lines of code). Each writer then moved to a private subfolder and reported that the fragments did not enter its edit. W1 runs 0-4 also produced five different edit designs. Each run's `process_notes` field and `rationale.md` give the details. The runs are kept as valid; this is a deviation from strict isolation. Later stages should give each run its own scratch directory.
- **What the WSP edits are.** W1: runs 0 and 3 rewrote only the class-9 output row (as offsets from the ellipse row, or as the sum of the square and ellipse rows); runs 1, 2, and 4 added a small closed-form head on the penultimate feature (class-mean prototypes, pairwise heart-vs-class tests, or a capped heart-vs-ellipse score). W2: runs 0 and 3 make class 9 a novelty output (an image far from every old class is called class 9). This works because the test set contains no classes beyond 0-9. Run 1 uses the class-9 row as the mean of the five outlined-shape rows with a margin bias. Run 2 uses a logical rule on the old logits (corner-like AND curve-like evidence). Run 4 changes no weights: it adds a hand-written geometric test on the input pixels (contour concavity) that decides class 9. That is programming of the forward pass rather than of weights. It passes the gradient check and is counted as valid, but it should be read with this caveat.

## Reproduce from zero

All commands run from the root of this folder.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# Stage 1 data and base model (writes checkpoints/stage1_base.pt)
python data/shapes.py
python train/train_base.py

# Stage 1 baselines
for s in 0 1 2 3 4; do python train/baselines.py --condition B0 --seed $s; done
for c in B0s B1 B2; do for s in 0 1 2 3 4; do python train/baselines.py --condition $c --seed $s; done; done

# Stage 1 WSP: re-run every committed edit.py, check it for gradients, evaluate on test
python eval/wsp_finalize.py --condition W1 --run 0 --writer-model claude-opus-5-5 --effort "high (session setting; not overridden for the subagent)"
python eval/wsp_finalize.py --condition W1 --run 1 --writer-model claude-opus-5-5 --effort "high (session setting; not overridden for the subagent)"
python eval/wsp_finalize.py --condition W1 --run 2 --writer-model claude-opus-5-5 --effort "high (session setting; not overridden for the subagent)"
python eval/wsp_finalize.py --condition W1 --run 3 --writer-model claude-opus-5-5 --effort "high (session setting; not overridden for the subagent)"
python eval/wsp_finalize.py --condition W1 --run 4 --writer-model claude-opus-5-5 --effort "high (session setting; not overridden for the subagent)"
python eval/wsp_finalize.py --condition W2 --run 0 --writer-model claude-opus-5-5 --effort "high (session setting; not overridden for the subagent)"
python eval/wsp_finalize.py --condition W2 --run 1 --writer-model claude-opus-5-5 --effort "high (session setting; not overridden for the subagent)"
python eval/wsp_finalize.py --condition W2 --run 2 --writer-model claude-opus-5-5 --effort "high (session setting; not overridden for the subagent)"
python eval/wsp_finalize.py --condition W2 --run 3 --writer-model claude-opus-5-5 --effort "high (session setting; not overridden for the subagent)"
python eval/wsp_finalize.py --condition W2 --run 4 --writer-model claude-opus-5-5 --effort "high (session setting; not overridden for the subagent)"

# Rebuild the results table
python eval/make_results.py

# Gradient-check a single edit script
python tools/check_no_gradients.py wsp/stage1/W1/run_0/edit.py
```

`requirements.txt` pins `torch` from PyPI. On Linux this is the CUDA-enabled build; it runs
on CPU when no GPU is present. A CPU-only build of the same version can be installed with
`pip install torch==<version> --index-url https://download.pytorch.org/whl/cpu`.

The wall-clock times of WSP runs cover the writer's whole session (reading, analysis,
writing the edit, and validation checks), so they cannot be reproduced by these commands.
Re-running `wsp_finalize.py` reports the script runtime separately (`edit_script_runtime_s`).

## Credits

Stage 2 uses tiny-voca (`n4ze3m/tiny-voca`, Apache-2.0).
