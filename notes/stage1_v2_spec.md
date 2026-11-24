Do Stage 1 v2 now. Do not start Stage 2. These instructions replace the Stage 1 section of CLAUDE.md. Where they conflict with CLAUDE.md, these instructions win. First save this message unchanged to notes/stage1_v2_spec.md and commit it. Do not export notes/. WSP writers must not read notes/.

KEEP v1
- Move the v1 results to results/stage1_v1/ and the v1 WSP runs to wsp/stage1_v1/. Do not change them. v1 is a pilot.

WSP RULES FOR v2
1. Fixed instruction: before a WSP condition starts, write its instruction to prompts/stage1_v2_<condition>.md. Every run uses it without change.
2. Edit format: edit.py loads the frozen checkpoint, writes values into the state dict with explicit numbers or explicit formulas, and saves a new state dict file. Nothing else.
3. No gradients or fitting: no .backward(), torch.autograd, torch.optim, requires_grad_(True), least squares, k-means, optimizer, search loop, or loop that scores candidate weights.
4. Inside the network only: the evaluation loads the edited state dict into ONE fixed model class with strict=True and runs its standard forward. No new module, forward function, hook, pixel rule, or logic outside the state dict. An edit that changes no tensor value is invalid.
5. Frozen code: models/, eval/, and data/ must not change during a WSP run. Check with git diff.
6. Data access: writers read data only through eval/wsp_tools.py. It exposes the base checkpoint, the VALIDATION split of classes 0-8, and (W1 only) the run's k new-class examples. Never the TEST split, never a held-out class. Forward passes and activations are permitted. At most 3 validation evaluations of complete candidate edits per run. Record the number used.
7. Isolation: each WSP run is a subagent with a fresh context. Run them ONE AT A TIME, never in parallel. Set TMPDIR and all scratch output to wsp/stage1_v2/<condition>/run_<n>/scratch/. The writer must not read data/, results/, RESULTS.md, CLAUDE.md, notes/, other conditions' prompts, or any other run's folder. Put these limits in the instruction file. In rationale.md, the writer lists the files it read.
8. Rationale: rationale.md states which tensors changed, why, and the expected effect.
9. Record in each run JSON: writer model ID (exact name and version), effort level the subagent really used ("unverified" if you cannot confirm it), edit type, % of weights changed, slot parameters used (nonzero after the edit), validation evaluations used, files read, wall-clock time, valid or invalid.
10. Runs: 8 WSP runs per WSP condition (runs 0-7). Baselines once per seed, seeds 0-7. Run i of every condition uses the same k examples (chosen by seed i).

TASK 1.0 - TOOLS
- Rename the gradient check to tools/check_wsp_edit.py. Extend it to check rules 3, 4, and 5: static check of edit.py, strict=True load into the fixed class, at least one tensor value changed, git diff check.
- Tests: it must FAIL on (a) .backward(), (b) a forward function or pixel rule, (c) no tensor changed, (d) a new tensor name. It must PASS on a small valid row edit.
- Re-check all v1 runs. Save the result to results/stage1_v1/recheck.json. Mark v1 W2 run 4 as invalid. Do not change the v1 table.

TASK 1.1 - FIXED MODEL CLASS (models/shapes_v2.py)
- Small CNN backbone (under 200k parameters) that outputs penultimate feature f of size d.
- head: Linear(d, 10).
- adapter slot: logits = head(f) + A2(ReLU(A1(f))). A1 = Linear(d, 64), A2 = Linear(64, 10). All A1 and A2 weights and biases start at zero.
- WSP edits may write any tensor of this class.

TASK 1.2 - DATA (data/shapes_v2.py)
- Old classes 0-8: circle, square, triangle, star, cross, ring, ellipse, pentagon, arrow. New class 9: octagon by default (Task 1.4 can change it).
- Two HELD-OUT shape classes appear only in the TEST split. Choose two shapes not in the list above. Put their generator in data/heldout.py. Do not write their names in any prompt, any instruction file, or any text a writer can read.
- Splits per class: 2000 train, 200 validation, 500 test. Held-out: 500 test images each, nothing else. Seed 0.

TASK 1.3 - BASE MODEL
- Train the fixed class on classes 0-8 only. Target: old-class test accuracy 95% or more.
- Save checkpoints/stage1_v2_base.pt with the class-9 head row, class-9 bias, and adapter all at zero.

TASK 1.4 - CALIBRATE DIFFICULTY (before any WSP run)
- Goal: B0 mean new-class accuracy between 60% and 80% on the VALIDATION split.
- Levers in this order. Stop at the first setup in range:
  1. k = 2 instead of 5.
  2. A new class closer to an old class (octagon is the default).
  3. More noise, blur, and distortion for all classes (retrain the base model).
- Save every attempt to results/stage1_v2/calibration.json.
- Freeze the setup in data/stage1_v2_config.json and commit it. Do not change it after this point.
- STOP HERE. Show me the frozen config and the calibration results, and wait for my OK before Task 1.5.

TASK 1.5 - CONDITIONS
- B0: class-9 head row = unit-norm mean penultimate feature of the k examples; class-9 bias = 0. No gradients.
- B0s (supplementary): B0 with the row scaled to the mean norm of the trained rows and the bias set to their mean.
- B1: fine-tune class-9 head row and bias only. Positives: the k examples. Negatives: replayed old-class training images. Gradients.
- B2: fine-tune the whole model, k examples plus old-class replay. Gradients.
- W1: WSP with the k examples.
- W2: WSP from a text description of the new class only. No examples.
- W1 and W2 instructions describe the fixed model class, the adapter slot, and the data access rules. They must not mention held-out classes or open-set testing. The W2 description describes only the shape.

TASK 1.6 - METRICS (TEST split, every run)
- New-class accuracy (class 9).
- Class-9 precision: correct class-9 predictions / all class-9 predictions, over old, new, and held-out test images.
- Held-out false-positive rate: fraction of held-out images predicted as class 9.
- Old-class accuracy and drop from the base model.
- Wall-clock time, % weights changed, slot parameters used.
- Edit type: row-only, row-composition (class-9 row built from other trained rows), adapter, backbone, or mixed.

TASK 1.7 - RESULTS.md
- Main table: one row per condition, mean ± sample sd [min-max] for every metric.
- Per-run table for W1 and W2: run, edit type, every metric, valid or invalid.
- Per-edit-type table for W1 and W2: mean new-class accuracy and mean precision.

TASK 1.8 - GATE
Bootstrap with 10,000 resamples over the 8 paired runs (W1 run i with B0 seed i). Write at the top of RESULTS.md:
- PASS if all three are true:
  1. 95% CI lower bound of (W1 - B0) new-class accuracy >= -2 points.
  2. 95% CI lower bound of (W1 - B0) class-9 precision >= -2 points.
  3. W1 mean old-class drop <= 2 points.
- Report the same three numbers for W2 vs B0, and W2 new-class accuracy vs chance (10%).
- Invalid runs count as 0% new-class accuracy in the gate. List them.

TASK 1.9 - EXPORT
Run the Export task from CLAUDE.md. Copy tools/check_wsp_edit.py instead of the old check. Never copy scratch/ folders or notes/. In the README, add a short pilot section for v1: the ceiling effect, the open-set shortcut, the edit outside the network, and the isolation problem, and how v2 fixes each one. The reproduction check uses the v2 base training and B0. Then stop and wait for me.
