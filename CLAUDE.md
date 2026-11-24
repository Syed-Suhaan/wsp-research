# wsp-research — CLAUDE.md

## Goal

Find out if weight-space programming (WSP) can ADD a new capability to a trained network.
WSP means: the writer model reads the weights and writes new weights by reasoning. It does not use gradients.

- Stage 1 (gate): add a new class to a small CNN. Synthetic data. Fully automatic.
- Stage 2 (headline): add a new voice style to tiny-voca. Synthetic target. Fully automatic.

Do Stage 2 only when the user says so.

## Machine

- 4 vCPU, 15 GiB RAM, no GPU. Use CPU PyTorch only.
- The machine is temporary. Commit and push after every task. Never keep results only on disk.
- The network proxy can block sites. If a download fails, write the error to `BLOCKED.md`, commit, push, and stop.

## General rules

1. Use fixed seeds. Set `torch.use_deterministic_algorithms(True)` and `torch.set_num_threads(4)`.
2. Put every result in `results/<stage>/<condition>/run_<n>.json`.
3. Update `RESULTS.md` after every condition: one table, mean and spread for each metric.
4. Keep every committed file under 50 MB.
5. Write short commit messages that name the task, for example `stage1: train base CNN`.
6. Write all code, comments, and docs in a neutral research style. Code and docs must make sense to a reader who has never seen this file.

## Repo layout

```
data/            dataset generators
models/          model definitions
train/           base training and fine-tune baselines
eval/            metrics and evaluation scripts
wsp/<stage>/<condition>/run_<n>/   edit.py, rationale.md
prompts/         the exact instruction given to the WSP writer, one file per condition
tools/           check_no_gradients.py, export_public.py
results/         JSON results
samples/         audio samples (Stage 2)
checkpoints/     model checkpoints (not exported)
RESULTS.md
```

## WSP rules (both stages)

These rules make the WSP result valid. Obey all of them.

1. Before a WSP condition starts, write its instruction to `prompts/<stage>_<condition>.md`. Every run of that condition uses this instruction without change.
2. A WSP edit is a Python script `wsp/<stage>/<condition>/run_<n>/edit.py`. It loads the frozen checkpoint, changes or adds weights with explicit values or explicit formulas, and saves a new checkpoint.
3. `edit.py` must not use `.backward()`, `torch.autograd`, `torch.optim`, or any optimizer, search loop, or fitting routine (for example least squares, k-means, or random search over weights).
4. During a WSP run, you can run forward passes and read activations. You can use the VALIDATION split only. Never read the TEST split during a WSP run.
5. Write the reasoning for each edit in `rationale.md` in the same folder: which weights changed, why, and the expected effect.
6. Do 5 WSP runs per WSP condition. Each run must be independent. Use a subagent with a fresh context for each run if one is available. Do not read the files of earlier runs during a run.
7. Run `python tools/check_no_gradients.py` on every `edit.py`. If it fails, the run is invalid. Record it as invalid. Do not delete it.
8. In each run's JSON, record: the writer model ID (the exact model name and version that wrote the edit), the percentage of weights changed, the number of new parameters, and the wall-clock time.

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

## Stage 2 — Add a new voice style to tiny-voca

### Task 2.0 — Access check
- Try to download `n4ze3m/tiny-voca` from Hugging Face.
- If it fails, write in `BLOCKED.md` that the user must commit `tiny-voca.pt`, `config.json`, and the `phonemizer/` folder to `vendor/tiny-voca/`, with the Apache-2.0 license file. Stop.

### Task 2.1 — Inspect
- Write `eval/inspect_tiny_voca.py`. List every weight name, shape, and parameter count.
- In `docs/tiny_voca_weights.md`, identify: the voice embedding (F2, M2), the pitch path, the duration path, and the source-filter vocoder parts.

### Task 2.2 — Synthetic target
- Use 40 fixed English sentences: 20 for validation, 20 for test.
- Make the target style "deep narrator" with DSP on F2 output: pitch −4 semitones, formants −8%, rate 0.9x. Use pyworld or an equivalent.
- The target durations are the F2 durations divided by 0.9. They are known exactly, so fine-tuning needs no aligner.

### Task 2.3 — Conditions
| ID | Condition | Uses gradients |
|----|-----------|----------------|
| B1 | Fine-tune a new voice vector only | Yes |
| B2 | Fine-tune the whole model on target audio | Yes |
| W1 | WSP: add a third voice with no gradients | No |
| H1 | Hybrid: start from the W1 checkpoint, then B1 | Yes |

### Task 2.4 — Metrics (TEST sentences)
- F0 RMSE in cents on voiced frames.
- Mel-cepstral distortion with DTW.
- Duration ratio against the target.
- F2 and M2 must not change: report their mel-cepstral distortion against the original model output.
- Time per condition and percentage of weights changed.

### Task 2.5 — Samples
- Save 3 test sentences per condition as 24 kHz WAV in `samples/stage2/`.
- Save one before/after pair: original F2 and W1 on the same sentence.

### Task 2.6 — Export
Do the Export task (below) for Stage 2. Then stop and wait for the user.

## Export task (end of each stage)

Write `tools/export_public.py` once. Run it at the end of each stage. It rebuilds `public_release/` from zero.

1. Copy ONLY these: `data/`, `models/`, `train/`, `eval/`, `wsp/`, `prompts/`, `tools/check_no_gradients.py`, `results/`, `samples/`, `docs/`, `RESULTS.md`, `requirements.txt`, `LICENSE`.
2. Never copy: `CLAUDE.md`, `BLOCKED.md`, `checkpoints/`, `vendor/`, `public_release/`, `.git/`, caches.
3. Write `public_release/README.md` with:
   - The research question and the definition of WSP.
   - The method for each stage, with the conditions table.
   - The WSP writer: the exact model name and version from the run JSON files.
   - The hardware: 4 vCPU, no GPU.
   - The results table from `RESULTS.md`.
   - Exact commands to reproduce every result from zero.
   - Credit to tiny-voca (Apache-2.0) for Stage 2.
4. Write `public_release/requirements.txt` with pinned versions.
5. Test it: in a fresh virtual environment inside a temp folder, install from `public_release/requirements.txt` and run the Stage 1 base training and B0 evaluation with the README commands. They must match the results within 0.5 points. Record the test result in `public_release/REPRODUCE_CHECK.md`.
6. Search `public_release/` for these words and remove any lines that contain them: `CLAUDE.md`, `BLOCKED`. Do not remove the WSP writer model name from the README or the run JSON files.
7. Commit `public_release/` to this private repo. The user copies it to the public repo.
