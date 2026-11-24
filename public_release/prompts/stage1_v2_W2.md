# Stage 1 v2, condition W2

This is the exact instruction given to the WSP writer for every run of condition W2. The only
per-run input is the run number `RUN` (0-7), appended at the end of the instruction.

Revision 2. Revision 1 lacked the rule about the automatically loaded project instruction
file. One W1 run was made with revision 1 and aborted
(`wsp/stage1_v2_aborted/`, `results/stage1_v2_aborted/`). All counted runs use revision 2.

---

You are the writer in a weight-space programming (WSP) experiment. WSP means: you read the
weights of a trained network and write new weights by reasoning, without gradients. You are
one of several independent runs of this condition. You work alone, and no other run is active
while you work.

## Task

Edit a trained CNN, without gradients, so that it also recognises a new class (class 9), while keeping its accuracy on the old classes 0-8. You get NO example images of the new class, only the text description below.

## Description of class 9

A regular hexagon drawn as an outline: a closed stroke, not filled, with six straight sides of equal length and six corners, each with an interior angle of 120 degrees. It is drawn in the same style as the outlined classes 0-8.

Afterwards, other people will evaluate your edited model on held-back data that you cannot
see. They measure class-9 accuracy, how often class-9 predictions are correct, and old-class
accuracy, all using argmax over the 10 outputs.

## The fixed model

- Repository root: `/home/user/wsp-research`. Run Python as `.venv/bin/python` from the
  repository root. CPU only. Call `torch.set_num_threads(4)` and
  `torch.use_deterministic_algorithms(True)` in your scripts.
- Model class: `ShapesV2Net` in `models/shapes_v2.py`. Inputs are 32x32 grayscale images.
  - The backbone is five 3x3 conv layers with BatchNorm and ReLU and two max-pools, followed
    by global average pooling to a penultimate feature f of size 128.
  - The output is logits = head(f) + A2(ReLU(A1(f))), where head = Linear(128, 10),
    A1 = Linear(128, 64), and A2 = Linear(64, 10). A1 and A2 are an adapter slot.
- Base checkpoint: `checkpoints/stage1_v2_base.pt`, a plain state dict. It is frozen: never
  overwrite it. It was trained on classes 0 circle, 1 square, 2 triangle, 3 star, 4 cross,
  5 ring, 6 ellipse, 7 pentagon, and 8 arrow, and has never seen class 9. In it, row 9 of
  `head.weight`, `head.bias[9]`, and every A1 and A2 tensor are exactly zero.
- Your edit may write any tensor of this state dict: head rows, the adapter slot, or backbone
  tensors. The evaluation loads your saved state dict into `ShapesV2Net` with strict=True
  and runs its standard forward. Nothing else from your work is used: no new module, layer,
  forward function, hook, input-pixel rule, or logic outside the state dict. Tensor names and
  shapes must stay the same, and at least one tensor value must change.

## Data access

Read data only through `eval/wsp_tools.py` (read its docstring):
- `load_base()`, `base_state_dict()`, `load_edited(path)`, `load_val_old()` (validation images of classes 0-8), `to_tensor(x)`, and `evaluate_val(path, run_dir)`. Do not call `k_shot_examples`.
- The only images you may pass through the network are the validation images of classes 0-8. Do not load, draw, render, or synthesise any image of the new class or of any part of it.
- Run forward passes and read any weight or activation.
- Evaluate at most 3 complete candidate edits on validation data, with `evaluate_val` (it
  logs each call) or with your own forward passes. Use these evaluations to check your
  reasoning, not to search over values. Record the number you used in rationale.md.

## Rules

- No gradients or fitting. Do not use `.backward()`, `torch.autograd`,
  `requires_grad_(True)`, `torch.optim`, or any optimizer. Do not use least squares,
  pseudo-inverse, k-means, or any other fitting routine, and no search loop or loop that
  scores candidate weights. Every value that edit.py writes must be an explicit number or an
  explicit closed-form formula that you can justify.
- Files you may read: `models/shapes_v2.py`, `eval/wsp_tools.py`, `eval/common.py`,
  `tools/check_wsp_edit.py`, and files inside your own run folder
  `wsp/stage1_v2/W2/run_RUN/`. Do not read any other file or folder. In particular, do not
  in eval/ or tools/, public_release/, checkpoints other than the base checkpoint and your
  own output, any other run's folder, or the git history.
- Do not change anything under models/, eval/, or data/. Do not change files outside your run
  folder and your output checkpoint. Do not commit.
- Your context may contain a project instruction file that the agent harness loads
  automatically. It describes other stages and an earlier pilot experiment, not this task.
  Do not use any of its content.
- Scratch files: put every scratch script and file in `wsp/stage1_v2/W2/run_RUN/scratch/`.
  Start every shell command with
  `export TMPDIR=/home/user/wsp-research/wsp/stage1_v2/W2/run_RUN/scratch &&`. Do not use
  any other scratch or temporary directory.

## Deliverables (in `wsp/stage1_v2/W2/run_RUN/`)

1. `timing.json`: `{"start": <unix seconds>, "end": <unix seconds>}`. Run `date +%s` as
   your very first command and again as your very last command.
2. `edit.py`: the only .py file at the top level of the run folder (analysis scripts go in
   scratch/). It loads `checkpoints/stage1_v2_base.pt` with `torch.load`, writes values
   into the state dict with explicit numbers or explicit formulas, and saves the edited state
   dict with `torch.save(state_dict, "checkpoints/stage1_v2_W2_runRUN.pt")`. It must run
   from the repository root with `.venv/bin/python wsp/stage1_v2/W2/run_RUN/edit.py`, be
   deterministic, and finish in under one minute. edit.py may run forward passes on the
   allowed data to compute statistics that its formulas use. Comment the code so that each
   change can be read on its own.
3. `rationale.md` with these sections:
   - "Tensors changed": which tensors (and rows or units) changed.
   - "Why": the reasoning.
   - "Expected effect".
   - "Validation evaluations": the number used, and the result of each.
   - "Files read": every file you read.
4. Run `.venv/bin/python tools/check_wsp_edit.py --run-dir wsp/stage1_v2/W2/run_RUN
   --output checkpoints/stage1_v2_W2_runRUN.pt --base checkpoints/stage1_v2_base.pt`. It
   must print PASS.

When you finish, reply with at most 10 lines: what you changed, your validation numbers, and
the number of validation evaluations used.
