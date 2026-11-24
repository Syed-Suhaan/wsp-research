# Stage 1 spec (orchestrator only; WSP writers must not read notes/)

Stage 1 asks whether WSP can add a new class to a small CNN.

## Which document governs

Later documents override earlier ones where they conflict:

1. The original Stage 1 section of the project instructions (moved here verbatim, below). It
   now describes the **v1 pilot**.
2. `notes/stage1_v2_spec.md`: Stage 1 v2. It replaces the Stage 1 section for v2.
3. `notes/stage1_v2_amendments.md`: changes made before the v2 conditions ran.
4. `notes/stage1_v2_1_spec.md`: Stage 1 v2.1 (tuned baselines, instruction restructure).

## Stage 1 v2 facts (frozen)

- Old classes 0-8: circle, square, triangle, star, cross, ring, ellipse, pentagon, arrow.
- New class 9: hexagon (outline). Chosen by calibration; frozen in
  `data/stage1_v2_config.json` (k = 2, noise 0.08-0.20, blur 0.3-0.8 px, distortion 0.20).
  Calibration log: `results/stage1_v2/calibration.json` (10 attempts; selection rule in
  RESULTS.md).
- Held-out test-only classes (`data/heldout.py`, `results/stage1_v2/heldout_note.json`):
  label 10 crescent (far from the new class), label 11 heptagon (near the new class).
  The first held-out pair was crescent and trapezoid. Trapezoid was replaced by heptagon
  before any v2 condition ran.
- Their names must never appear in a prompt, an instruction file, or any file a writer may
  read (`models/shapes_v2.py`, `eval/wsp_tools.py`, `eval/common.py`,
  `tools/check_wsp_edit.py`).
- W1 is given the k examples but not the class name. W2 is given a text description of the
  hexagon only.
- The writer instruction files are `prompts/stage1_v2_W1.md` and `prompts/stage1_v2_W2.md`
  (revision 2). Revision 1 produced one aborted W1 run (`wsp/stage1_v2_aborted/`).
- Baselines: B0, B0s, B1, B2 (fixed hyperparameters), and in v2.1 B1t, B2t, B0b (tuned on
  validation under `results/stage1_v2_1/selection_rule.json`).
- Gate (v2): bootstrap over 8 paired runs, W1 vs B0. Result: FAIL. It must not be changed.

## Stage 1 v1 pilot facts

- New class 9 was "heart" (outline), with k = 5. Held-out classes: none. The base model
  reached 100% old-class test accuracy.
- Five WSP runs per condition ran concurrently and shared one scratch directory. The v1
  instruction permitted new modules; one W2 run used a pixel rule and changed no tensor
  (marked invalid in `results/stage1_v1/recheck.json`).
- Results are frozen in `results/stage1_v1/` and `wsp/stage1_v1/`.

## Original Stage 1 section (verbatim; now the v1 pilot description)

## Stage 1 — Add a new class to a CNN

### Task 1.0 — Gradient check tool
- Write `tools/check_no_gradients.py`. It takes a file path and fails if the file uses `.backward`, `autograd`, `torch.optim`, `requires_grad_(True)`, `lstsq`, `KMeans`, or a loop that scores candidate weights.
- Write a test that shows it fails on a small bad example and passes on a small good example.

### Task 1.1 — Data
- Write `data/shapes.py`. Generate 32x32 grayscale images of 10 shape classes: circle, square, triangle, star, cross, ring, ellipse, pentagon, arrow, heart.
- Add random size, position, rotation, line width, and pixel noise.
- Splits per class: 2000 train, 200 validation, 500 test. Seed 0.
- Class 9 (heart) is the NEW class. The base model never sees it.

### Task 1.2 — Base model
- Train a small CNN (under 200k parameters) on classes 0–8 only.
- Target: test accuracy on classes 0–8 of 95% or more. Save `checkpoints/stage1_base.pt`.
- Add an output row for class 9 initialized to zeros, so all conditions start from the same 10-output model.

### Task 1.3 — Conditions
Use k = 5 examples of class 9 from the VALIDATION split for every condition that uses examples.

| ID | Condition | Uses gradients |
|----|-----------|----------------|
| B0 | Imprinting: new row = normalized mean penultimate feature of the k examples | No (fixed formula) |
| B1 | Fine-tune the new output row only, k examples | Yes |
| B2 | Fine-tune the whole model, k examples plus replay of old classes | Yes |
| W1 | WSP with k examples. A new layer is permitted. | No |
| W2 | WSP from a text description of a heart shape only. No examples. | No |

Run B0, B1, and B2 once per seed, seeds 0–4.

### Task 1.4 — Metrics (TEST split)
- New-class accuracy (class 9).
- Old-class accuracy (classes 0–8) and the drop from the base model.
- Wall-clock time per condition.

### Task 1.5 — Gate
Write the gate result at the top of `RESULTS.md`:
- PASS if W1 mean new-class accuracy is at least B0 mean, AND W1 mean old-class drop is 2 points or less.
- Also report if W2 is better than chance (10%).

### Task 1.6 — Export
Do the Export task (below) for Stage 1. Then stop and wait for the user.

### Original WSP rule details specific to the v1 pilot

- Rule 6 (v1): 5 WSP runs per WSP condition.
- Export (v1): the reproduction check ran the Stage 1 base training and B0 evaluation.
