# wsp-research — CLAUDE.md

## Goal

Test whether weight-space programming (WSP), where a writer model reads a trained network's weights and writes new weights by reasoning without gradients, can add a new capability to a trained network.

The orchestrator reads notes/<current stage>_spec.md. WSP writers must not read notes/.

## Machine

- 4 vCPU, 15 GiB RAM, no GPU. Use CPU PyTorch only.
- The machine is temporary. Commit and push after every task. Never keep results only on disk.
- The network proxy can block sites. If a download fails, write the error to `BLOCKED.md`, commit, push, and stop.

## General rules

1. Use fixed seeds. Set `torch.use_deterministic_algorithms(True)` and `torch.set_num_threads(4)`.
2. Put every result in `results/<stage>/<condition>/run_<n>.json`.
3. Update `RESULTS.md` after every condition: one table, mean and spread for each metric.
4. Keep every committed file under 50 MB.
5. Write short commit messages that name the task, for example `stage1: train base model`.
6. Write all code, comments, and docs in a neutral research style. Code and docs must make sense to a reader who has never seen this file.
7. Work on a stage only when the user says so. At the end of a stage, do the Export task, stop, and wait for the user.

## Repo layout

```
data/            dataset generators
models/          model definitions
train/           base training and baselines
eval/            metrics and evaluation scripts; eval/wsp_tools.py is the writers' data access
wsp/<stage>/<condition>/run_<n>/   edit.py, rationale.md, scratch/
prompts/         the exact instruction given to the WSP writer, one file per condition
tools/           check_wsp_edit.py, export_public.py
tests/           tests for the tools
notes/           stage specs for the orchestrator (not exported; writers must not read)
results/         JSON results
samples/         audio samples
docs/            documentation and figures
checkpoints/     model checkpoints (not exported)
RESULTS.md
```

## WSP rules

These rules make a WSP result valid. Obey all of them.

1. Fixed instruction: before a WSP condition starts, write its instruction to `prompts/<stage>_<condition>.md`. Every run uses it without change. If the instruction must change, the condition restarts from run 0, and runs made with the earlier revision are kept as aborted, outside the results.
2. Edit format: `wsp/<stage>/<condition>/run_<n>/edit.py` loads the frozen checkpoint, writes values into the state dict with explicit numbers or explicit formulas, and saves a new state dict file. Nothing else.
3. No gradients or fitting: no `.backward()`, `torch.autograd`, `torch.optim`, `requires_grad_(True)`, regression fitting (for example `lstsq` or `pinv`), k-means, optimizer, search loop, or loop that scores candidate weights.
4. Inside the network only: the evaluation loads the edited state dict into ONE fixed model class with strict=True and runs its standard forward. No new module, forward function, hook, input-pixel rule, or logic outside the state dict. An edit that changes no tensor value is invalid.
5. Frozen code: `models/`, `eval/`, and `data/` must not change during a WSP run. Check with git diff against the commit at the start of the run.
6. Data access: writers read data only through `eval/wsp_tools.py`. It exposes the base checkpoint, the VALIDATION split of the old classes, and the run's examples where the condition allows them. Never the TEST split, and no data beyond what the condition allows. Forward passes and activations are permitted. At most 3 validation evaluations of complete candidate edits per run. Record the number used.
7. Isolation: each WSP run is a subagent with a fresh context. Run them ONE AT A TIME, never in parallel. Set TMPDIR and all scratch output to `wsp/<stage>/<condition>/run_<n>/scratch/`. The writer must not read data/, results/, RESULTS.md, CLAUDE.md, notes/, other conditions' prompts, or any other run's folder. Put these limits in the instruction file. In rationale.md, the writer lists the files it read.
8. Rationale: `rationale.md` states which tensors changed, why, and the expected effect.
9. Record in each run JSON: writer model ID (exact name and version), effort level the subagent really used ("unverified" if it cannot be confirmed), edit type, % of weights changed, slot parameters used (nonzero after the edit), validation evaluations used, files read, wall-clock time, valid or invalid.
10. Runs: 8 WSP runs per WSP condition (runs 0-7). Baselines once per seed, seeds 0-7. Run i of every condition uses the same examples (chosen by seed i).
11. Check every run with `python tools/check_wsp_edit.py`. If it fails, the run is invalid. Record it as invalid. Do not delete it.

## Export task (end of each stage)

`tools/export_public.py` rebuilds `public_release/` from zero. Run it at the end of each stage.

1. Copy ONLY these: `data/`, `models/`, `train/`, `eval/`, `wsp/`, `prompts/`, `tools/check_wsp_edit.py`, `results/`, `samples/`, `docs/`, `RESULTS.md`, `requirements.txt`, `LICENSE`.
2. Never copy: `CLAUDE.md`, `BLOCKED.md`, `notes/`, `checkpoints/`, `vendor/`, `public_release/`, `scratch/` folders, `.git/`, caches.
3. Write `public_release/README.md` with:
   - The research question and the definition of WSP.
   - The method for each stage, with the conditions table.
   - The WSP writer: the exact model name and version, and the effort level, from the run JSON files.
   - The hardware: 4 vCPU, no GPU.
   - The results table from `RESULTS.md`.
   - Exact commands to reproduce every result from zero.
   - Credit and license for every third-party model or dataset named in the stage spec.
4. Write `public_release/requirements.txt` with pinned versions.
5. Test it: in a fresh virtual environment inside a temp folder, install from `public_release/requirements.txt` and run the base training and the reference baseline named in the stage spec with the README commands. They must match the results within 0.5 points. Record the test result in `public_release/REPRODUCE_CHECK.md`.
6. Search `public_release/` for these words and remove any lines that contain them: `CLAUDE.md`, `BLOCKED`. Exception: never change files in `prompts/`; exported prompt files must be byte-identical to the files the writers used. Do not remove the WSP writer model name from the README or the run JSON files.
7. Commit `public_release/` to this private repo. The user copies it to the public repo.
