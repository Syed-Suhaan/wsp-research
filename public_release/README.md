# Weight-space programming: can a model add a capability to a trained network without gradients?

## Research question

Can weight-space programming (WSP) add a new capability to a trained neural network?

**WSP** means: a writer model reads a trained network's weights (and may run forward passes
and read activations on validation data), then writes new weights by reasoning. It does not
use gradients, optimizers, or automated fitting. Each edit is a short Python script
(`wsp/<stage>/<condition>/run_<n>/edit.py`) that loads the frozen checkpoint, writes tensor
values with explicit numbers or explicit closed-form formulas, and saves a new state dict.
A written rationale (`rationale.md`) sits next to it.

- **Stage 1 (gate):** add a new class to a small CNN trained on 9 synthetic shape classes.
  Stage 1 v2 (this release) is the main experiment. Stage 1 v1 was a pilot (see below).
- **Stage 2:** add a new voice style to the tiny-voca text-to-speech model (not yet run).

## Stage 1 v2 method

**Data** (`data/shapes_v2.py`, configuration frozen in `data/stage1_v2_config.json`):
32x32 grayscale images of 9 old classes (circle, square, triangle, star, cross, ring,
ellipse, pentagon, arrow) and a new class 9 (hexagon, outline). Every image has a
random size, position, rotation, line width, and brightness, plus pixel noise
(sigma 0.08-0.2), blur (sigma
0.3-0.8 px), and affine distortion with vertex jitter
(strength 0.2). Per class: 2000 train, 200 validation, 500 test; seed 0.
The test split also contains two **held-out shape classes** (500 images each, `data/heldout.py`).
One is visually near the new class and one is far from it. No model, baseline, or writer ever
sees them. They measure whether an edit calls unfamiliar shapes "class 9".

**Fixed model class** (`models/shapes_v2.py`): a 5-layer CNN backbone (139,488 parameters)
giving a 128-d feature f, and logits = head(f) + A2(ReLU(A1(f))). Here head = Linear(128, 10)
and the adapter slot is A1 = Linear(128, 64), A2 = Linear(64, 10), initialised to zero. The
base model (`train/train_base_v2.py`) is trained on classes 0-8; its class-9 head row, class-9
bias, and adapter are zero. Every condition is evaluated by loading a state dict into this
class with strict=True and running its standard forward. An edit can change behaviour only
through tensor values.

**Conditions.** k = 2 class-9 validation examples per seed or run. Run i of every
condition uses the same k examples as seed i (`data.shapes_v2.k_shot_indices`).

| ID | Condition | Uses gradients |
|----|-----------|----------------|
| B0 | Imprinting: class-9 head row = unit-norm mean penultimate feature of the k examples; bias 0 | No (fixed formula) |
| B0s | Supplementary: B0 row scaled to the mean norm of the trained rows; bias = their mean | No (fixed formula) |
| B1 | Fine-tune the class-9 head row and bias only; k positives, replayed old-class negatives | Yes |
| B2 | Fine-tune the whole model, k examples plus old-class replay | Yes |
| W1 | WSP with the k examples (the class name is not given) | No |
| W2 | WSP from a text description of the new shape only, no examples | No |

Baselines run once per seed (0-7). Each WSP condition has 8 runs (0-7), run **one at a time**,
each by a separate writer session with a fresh context. All runs of a condition receive the
same fixed instruction (`prompts/stage1_v2_<condition>.md`), with only the run number changed.
(The exported copies omit one line of each instruction, the one that lists a private project
file among the files writers may not read. The export rule removes lines that name that file.)
Writers read data only through `eval/wsp_tools.py`: the base checkpoint, the class 0-8
validation images, and for W1 the run's k examples. They may make at most 3 validation
evaluations of complete candidate edits, and they list the files they read in `rationale.md`.
`tools/check_wsp_edit.py` checks every run: no gradients or fitting, only tensor values
changed, a strict load into the fixed class, at least one changed tensor, and no change to
models/, eval/, or data/ during the run (git). Invalid runs are kept and reported.

**Metrics** (test split): new-class accuracy; class-9 precision over old, new, and held-out
test images; held-out false-positive rate (near, far, and both); old-class accuracy and its
drop from the base model; wall-clock time; percentage of weights changed; adapter-slot
parameters used; and edit type (row-only, row-composition, adapter, backbone, mixed).

**Gate:** 10,000 bootstrap resamples over the 8 paired runs (W1 run i with B0 seed i).
PASS if the 95% CI lower bound of mean(W1 − B0) is at least −2 points for both new-class
accuracy and class-9 precision, and the W1 mean old-class drop is at most 2 points. Invalid
runs count as 0% new-class accuracy.

**Calibration disclosure.** The data setup was chosen by a difficulty calibration over 10
attempts (`results/stage1_v2/calibration.json`). The goal was a B0 mean new-class accuracy of
60-80% on the validation split. The selection rule applied the levers in order and stopped at
the first setup in range: (1) k = 2 instead of 5; (2) a new class closer to an old class; (3)
more noise, blur, and distortion, with a retrained base model. Lever 3 was tried at
increasing strength, with new classes in the fixed order octagon, rounded square, hexagon.
The selected setup has the lowest B0 validation accuracy of all attempts; the others were
94.8-100%.

## Stage 1 v1 pilot and what v2 changes

The v1 pilot (`results/stage1_v1/`, `wsp/stage1_v1/`) added a "heart" class with k = 5.
It showed four problems, and v2 addresses each one:

1. **Ceiling effect.** The base model reached 100% on old classes, and plain imprinting (B0)
   reached 100% on the new class, so no condition could beat B0.
   *v2:* harder data and k = 2, calibrated before any WSP run so that B0 scores 60-80% on
   validation, plus a paired bootstrap gate instead of a comparison of means.
2. **Open-set shortcut.** Several v1 writers made class 9 a "none of the old classes"
   detector. This works only because the v1 test set contained no other new shapes.
   *v2:* the test split has two held-out shape classes (one near the new class, one far);
   class-9 precision and held-out false-positive rates are reported, and precision is part
   of the gate.
3. **Edits outside the network.** v1 allowed new modules. One v1 run changed no weights and
   decided class 9 with a hand-written pixel rule.
   *v2:* every edit must be a state dict for one fixed model class (with a zero-initialised
   adapter slot for added capacity), loaded with strict=True. `tools/check_wsp_edit.py`
   rejects new modules, forward functions, hooks, and edits that change no tensor. A
   re-check of v1 with these rules is in `results/stage1_v1/recheck.json`; it marks the v1
   pixel-rule run invalid.
4. **Isolation.** The v1 writers ran in parallel and shared one scratch directory, and
   several saw fragments of other runs' files.
   *v2:* runs execute one at a time, each with its own scratch folder and TMPDIR. Writers may
   read only a fixed list of files and record the files they read.

## WSP writer

- Model ID `claude-opus-5-5`; effort level: **unverified**.
  - session record (get_session on the orchestrating session): effort_level: high (session setting)
  - probe subagent (same agent type and settings as the writers, no effort override), self-report: a reasoning effort setting with the value 15 on a 0-100 scale; model 'Opus 5.5', exact model ID 'claude-opus-5-5'

The writer model ID and effort level are recorded in every WSP run JSON
(`results/stage1_v2/<condition>/run_<n>.json`). The effort level is "unverified": it could
not be confirmed from a record outside the model (`results/stage1_v2/effort_probe.json`).

## Hardware

4 vCPU, 15 GiB RAM, no GPU. CPU PyTorch, 4 threads, deterministic algorithms, fixed seeds.

## Results

### Stage 1 v2 gate

**FAIL** (W1 vs B0, 8 paired runs, 10,000 bootstrap resamples)

- W1 − B0 new-class accuracy: mean +2.32 [-2.55, +6.60] (95% CI); criterion lower bound ≥ −2: not met.
- W1 − B0 class-9 precision: mean +22.99 [+21.91, +24.01] (95% CI); criterion lower bound ≥ −2: met.
- W1 mean old-class drop: 0.76 points; criterion ≤ 2: met.

Same criteria for W2 vs B0 (reported, not part of the gate): not all met.

- W2 − B0 new-class accuracy: mean -8.15 [-26.47, +5.75] (95% CI); criterion lower bound ≥ −2: not met.
- W2 − B0 class-9 precision: mean +16.30 [+8.78, +22.46] (95% CI); criterion lower bound ≥ −2: met.
- W2 mean old-class drop: 0.50 points; criterion ≤ 2: met.
- W2 new-class accuracy vs chance (10%): mean 70.10% [51.35, 85.53] (95% CI); above chance.

Invalid WSP runs (counted as 0% new-class accuracy and 0% precision in the gate and in all CIs): none.

#### Paired comparisons with B1 (not part of the gate)

Mean difference with 95% bootstrap CI over the 8 pairs (run i with seed i). A positive old-class-drop difference means the WSP condition loses more old-class accuracy.

| Comparison | New-class acc | Class-9 precision | Old-class drop |
|---|---|---|---|
| W1 − B1 | +25.88 [+20.45, +31.10] | +5.53 [+3.80, +7.16] | +0.62 [+0.33, +0.91] |
| W2 − B1 | +15.40 [-2.30, +29.85] | -1.16 [-7.70, +4.60] | +0.36 [+0.15, +0.57] |

### Calibration disclosure

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

### Stage 1 v2 — main table (TEST split)

Mean ± sample sd [min–max] over runs (baselines: seeds 0-7; WSP: runs 0-7; run i uses the same k examples as seed i). Accuracies and rates in %. Held-out FPR is the fraction of held-out images predicted as class 9; 'near' is the held-out shape close to the new class, 'far' the other one. The means include invalid WSP runs at their measured values; the gate counts them as 0.

| ID | Condition | Runs | New-class acc | Class-9 precision | Held-out FPR near | Held-out FPR far | Held-out FPR all | Old-class acc | Old-class drop | Wall-clock (s) | Weights changed (%) | Slot params used |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | Base model (class-9 row, bias, adapter zero) | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 98.42 | 0.00 | 269.57 | — | 0 |
| B0 | Imprinting: unit-norm mean feature, bias 0 | 8 | 78.25 ± 4.46 [68.20–82.20] | 25.73 ± 1.51 [22.17–27.15] | 94.22 ± 2.91 [88.00–97.20] | 41.95 ± 7.86 [28.80–51.40] | 68.09 ± 5.22 [58.40–74.30] | 88.92 ± 1.12 [87.51–91.04] | 9.50 ± 1.12 [7.38–10.91] | 0.02 ± 0.03 [0.01–0.08] | 0.085 ± 0.000 [0.085–0.086] | 0 ± 0 [0–0] |
| B0s | Imprinting, norm-matched row and mean bias (supplementary) | 8 | 100.00 ± 0.00 [100.00–100.00] | 15.12 ± 0.47 [14.06–15.63] | 100.00 ± 0.00 [100.00–100.00] | 97.80 ± 2.33 [93.20–100.00] | 98.90 ± 1.17 [96.60–100.00] | 59.54 ± 2.24 [54.27–61.49] | 38.88 ± 2.24 [36.93–44.16] | 0.01 ± 0.01 [0.01–0.02] | 0.086 ± 0.000 [0.086–0.086] | 0 ± 0 [0–0] |
| B1 | Fine-tune class-9 row and bias (gradients) | 8 | 54.70 ± 10.92 [36.40–70.00] | 43.18 ± 3.80 [35.07–46.94] | 69.95 ± 10.29 [48.40–83.20] | 0.23 ± 0.36 [0.00–0.80] | 35.09 ± 5.21 [24.20–41.60] | 98.28 ± 0.13 [98.00–98.40] | 0.15 ± 0.13 [0.02–0.42] | 16.20 ± 3.59 [14.24–24.96] | 0.086 ± 0.000 [0.086–0.086] | 0 ± 0 [0–0] |
| B2 | Fine-tune whole model with replay (gradients) | 8 | 1.05 ± 1.04 [0.00–2.60] | 16.52 ± 12.87 [0.00–38.24] | 1.32 ± 1.41 [0.00–4.20] | 1.30 ± 1.43 [0.00–4.20] | 1.31 ± 0.87 [0.40–3.00] | 97.67 ± 0.51 [96.87–98.22] | 0.75 ± 0.51 [0.20–1.56] | 14.99 ± 1.16 [13.04–16.25] | 94.050 ± 0.000 [94.050–94.050] | 0 ± 0 [0–0] |
| W1 | WSP with the k examples | 8 | 80.58 ± 10.63 [60.00–91.60] | 48.72 ± 2.83 [42.19–51.65] | 76.95 ± 4.63 [70.80–83.20] | 0.28 ± 0.41 [0.00–1.00] | 38.61 ± 2.38 [35.40–42.10] | 97.66 ± 0.54 [96.58–98.33] | 0.76 ± 0.54 [0.09–1.84] | 252.75 ± 30.87 [219.00–313.00] | 0.397 ± 0.360 [0.086–0.868] | 482 ± 529 [0–1170] |
| W2 | WSP from a text description only | 8 | 70.10 ± 26.89 [20.80–89.00] | 42.03 ± 11.95 [17.99–49.10] | 78.75 ± 11.11 [57.20–87.80] | 6.75 ± 10.97 [0.00–29.20] | 42.75 ± 4.82 [36.40–52.70] | 97.92 ± 0.33 [97.31–98.40] | 0.50 ± 0.33 [0.02–1.11] | 317.88 ± 60.60 [250.00–413.00] | 0.238 ± 0.121 [0.173–0.520] | 228 ± 181 [130–650] |

#### W1 per run

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

#### W2 per run

| Run | Edit type | New acc | Precision | FPR near | FPR far | Old acc | Old drop | Weights changed (%) | Slot params | Val evals | Wall-clock (s) | Valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | mixed | 89.0 | 44.7 | 87.8 | 17.6 | 97.93 | 0.49 | 0.520 | 650 | 2 | 413 | yes |
| 1 | mixed | 87.8 | 49.0 | 84.8 | 0.0 | 97.73 | 0.69 | 0.260 | 260 | 1 | 399 | yes |
| 2 | mixed | 35.4 | 28.8 | 69.4 | 6.6 | 97.31 | 1.11 | 0.173 | 130 | 1 | 258 | yes |
| 3 | mixed | 84.8 | 48.8 | 86.2 | 0.0 | 98.18 | 0.24 | 0.173 | 130 | 1 | 312 | yes |
| 4 | mixed | 70.0 | 49.0 | 72.8 | 0.0 | 98.40 | 0.02 | 0.173 | 130 | 1 | 316 | yes |
| 5 | mixed | 87.6 | 49.1 | 85.8 | 0.6 | 97.98 | 0.44 | 0.260 | 260 | 1 | 275 | yes |
| 6 | mixed | 20.8 | 18.0 | 57.2 | 29.2 | 97.73 | 0.69 | 0.173 | 130 | 1 | 250 | yes |
| 7 | mixed | 85.4 | 48.9 | 86.0 | 0.0 | 98.09 | 0.33 | 0.173 | 130 | 2 | 320 | yes |

#### WSP edit types

| Condition | Edit type | Runs | Mean new-class acc | Mean class-9 precision |
|---|---|---|---|---|
| W1 | mixed | 6 | 79.67 | 48.13 |
| W1 | row-only | 2 | 83.30 | 50.48 |
| W2 | mixed | 8 | 70.10 | 42.03 |

WSP writer: `claude-opus-5-5` (effort level: unverified). The effort level could not be confirmed from a record outside the model (`results/stage1_v2/effort_probe.json`).

#### Protocol notes (Stage 1 v2)

- **Fixed model.** Every condition is evaluated by loading a state dict into `models/shapes_v2.ShapesV2Net` with strict=True. Edits can change behaviour only through tensor values (head, a zero-initialised adapter slot A2(ReLU(A1(f))), or the backbone).
- **B1 negatives.** Only the class-9 head row and bias are trained. Each step uses the k examples as positives and 59 replayed class 0-8 training images as negatives.
- **B2 underfits the new class.** B2 uses the hyperparameters carried over unchanged from v1 (200 Adam steps, lr 3e-4, 2 positives and 59 replayed images per step). With k = 2 this barely moves the class-9 logit. They were not tuned after the v2 results were seen. B2 is not part of the gate.
- **W1 class name.** The W1 instruction calls the new class 'class 9' and gives only the k example images, not the shape's name. W2 gets only a text description of the shape, with no images.
- **WSP isolation.** WSP runs ran one at a time, each as a writer subagent with a fresh context. Each writer had its own scratch folder (`wsp/stage1_v2/<condition>/run_<n>/scratch/`, also its TMPDIR) and a fixed instruction (`prompts/stage1_v2_<condition>.md`). Writers read data only through `eval/wsp_tools.py` (base checkpoint, class 0-8 validation images, and for W1 the run's k class-9 validation images). Each writer lists the files it read in `rationale.md`. The project instruction file may be loaded into a writer's context automatically by the agent harness. It does not describe the held-out classes or the test set.
- **Instruction revision and aborted run.** The agent harness loads the project instruction file into every writer's context automatically. That file describes the v1 pilot, where class 9 was a 'heart'. The first W1 writer (instruction revision 1) therefore called class 9 'hearts'. Before its test results were seen, both W1 and W2 instructions were revised (revision 2) to tell writers not to use that file. That run was kept as aborted (`wsp/stage1_v2_aborted/`, `results/stage1_v2_aborted/`, excluded from all tables and the gate), and W1 restarted from run 0. All 16 counted runs used revision 2.
- **Residual pilot-name priming.** Even under revision 2, five W1 writers (runs 0, 2, 3, 6, 7) used the word 'heart' for the new class in their edit script, rationale, or scratch scripts (`pilot_class_name_mentions` in each run JSON). No W2 writer did. The pilot name is wrong (the new class is a hexagon) and gives no correct information. All W1 edits were derived from the k examples and class 0-8 statistics. W1's 'no class name' condition is still not clean.
- **Validity.** Each run is checked by `tools/check_wsp_edit.py`: no gradients, fitting, or candidate-scoring loop; nothing but tensor values changed; strict load into the fixed class; at least one changed tensor; no change under models/, eval/, data/ since the run started (git).
- **Held-out classes.** Two shape classes appear only in the test split (500 images each): one visually near the new class, one far from it. They were never shown to any model, baseline, or writer. One of them was replaced before any v2 condition ran (`results/stage1_v2/heldout_note.json`).

---

## Stage 1 v1 (pilot, frozen)

The v1 pilot table below is unchanged. A later re-check with the v2 rules (`results/stage1_v1/recheck.json`) marks v1 W2 run 4 invalid (pixel rule outside the network, no tensor changed). Six other v1 WSP runs used a new layer or module, which the v1 instruction allowed and the v2 rules do not. The v1 table still includes run 4 as it was recorded.


#### Stage 1 gate

**FAIL**

- W1 mean new-class accuracy 97.44% vs B0 mean 100.00%: not met (W1 ≥ B0).
- W1 mean old-class drop 0.31 points: met (≤ 2 points).
- W2 (text only) mean new-class accuracy 78.32%: better than chance (10%).

#### Stage 1 — add class 9 (heart) to a 9-class shape CNN

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

##### Protocol notes (Stage 1)

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

# Stage 1 v2 data and base model (writes checkpoints/stage1_v2_base.pt)
python data/shapes_v2.py
python train/train_base_v2.py

# Stage 1 v2 baselines
for s in 0 1 2 3 4 5 6 7; do python train/run_baselines_v2.py --condition B0 --seed $s; done
for c in B0s B1 B2; do for s in 0 1 2 3 4 5 6 7; do python train/run_baselines_v2.py --condition $c --seed $s; done; done

# Stage 1 v2 WSP: re-run each committed edit.py, check validity, evaluate on test.
# --git-rev is the commit at the start of the run, used for the frozen-code check; in a
# copy without git history, use the check from tools/check_wsp_edit.py without it.
python eval/wsp_finalize_v2.py --condition W1 --run 0 --git-rev 619f0da1b339d99cf4b7d7921a105c593d6758b6
python eval/wsp_finalize_v2.py --condition W1 --run 1 --git-rev e18888f0a7d9394026639ee8a8b9062165f9d322
python eval/wsp_finalize_v2.py --condition W1 --run 2 --git-rev 4be05eee6f27ab20f0bcc059c82daceb7c3c3e83
python eval/wsp_finalize_v2.py --condition W1 --run 3 --git-rev 8424935f692d4fc4654c50af6da73264fc16f505
python eval/wsp_finalize_v2.py --condition W1 --run 4 --git-rev e8421b9150eca345eeb0bb4aa9248391b7665e4d
python eval/wsp_finalize_v2.py --condition W1 --run 5 --git-rev 22918762227e3d54372286081eb4d712a2df9ced
python eval/wsp_finalize_v2.py --condition W1 --run 6 --git-rev 668ca4d11288086bcb3b0df102669e1c02d8ccfc
python eval/wsp_finalize_v2.py --condition W1 --run 7 --git-rev 0721bdf2efd4b5e0dd06eb1df287e7ee167ffadc
python eval/wsp_finalize_v2.py --condition W2 --run 0 --git-rev 256e746b256d539fb15afb573374430921cc9fd6
python eval/wsp_finalize_v2.py --condition W2 --run 1 --git-rev 75e58d81f72be265a512724596120b288e1e30d2
python eval/wsp_finalize_v2.py --condition W2 --run 2 --git-rev 66b89bd65dc988b1f1b8455ecbac8a3b7228caf2
python eval/wsp_finalize_v2.py --condition W2 --run 3 --git-rev d7acf42b1cfcc3009e03072b79ce724832aa10ea
python eval/wsp_finalize_v2.py --condition W2 --run 4 --git-rev 06dda92cdcba6afc8cd5521cc0d5d8477342883b
python eval/wsp_finalize_v2.py --condition W2 --run 5 --git-rev 5b295f3711f9c439d17fe46da23a8ae0b6c5b2e1
python eval/wsp_finalize_v2.py --condition W2 --run 6 --git-rev c60af3a37e0cae2e72d32c39cacba9066ed7bc73
python eval/wsp_finalize_v2.py --condition W2 --run 7 --git-rev 1e033ea0ee61e2c88d9a69e629274625bc897e03

# Results table
python eval/make_results_v2.py

# Check a single edit
python tools/check_wsp_edit.py --run-dir wsp/stage1_v2/W1/run_0 \
    --output checkpoints/stage1_v2_W1_run0.pt --base checkpoints/stage1_v2_base.pt

# Calibration attempts (each trains a base model for its data setup)
python eval/calibrate_v2.py --lever 0 --new-class octagon --k 5
```

`requirements.txt` pins `torch` from PyPI. On Linux this is the CUDA-enabled build, which
runs on CPU when no GPU is present. A CPU-only build of the same version can be installed
with `pip install torch==<version> --index-url https://download.pytorch.org/whl/cpu`.

WSP wall-clock times cover the writer's whole session, so these commands cannot reproduce
them. `wsp_finalize_v2.py` reports the edit script's own runtime separately.

The Stage 1 v1 pilot code is kept for reference (`data/shapes.py`, `models/cnn.py`,
`train/train_base.py`, `train/baselines.py`, `eval/make_results_v1.py`). The v1 WSP edit
scripts refer to their original folder (`wsp/stage1/...`) and to the v1 data-access module
(kept as `eval/wsp_tools_v1.py`), so they are a record and cannot be re-run as they are.

## License and credits

Apache-2.0 (see `LICENSE`). Stage 2 will use tiny-voca (`n4ze3m/tiny-voca`, Apache-2.0).
