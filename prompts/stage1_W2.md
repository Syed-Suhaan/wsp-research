# Stage 1, condition W2: weight-space programming from a text description only

This is the exact instruction given to the WSP writer for every run of condition W2. The only
per-run input is the run number `RUN` (0-4), appended at the end of the instruction.

---

You are the writer in a weight-space programming (WSP) experiment. WSP means: you read the
weights of a trained network and write new weights by reasoning, without gradients. You are
one of several independent runs of this condition, and you work alone.

## Task

Edit a trained CNN, without gradients, so that it also recognises a new class (class 9,
"heart"), while keeping its accuracy on the old classes 0-8. You get NO example images of
the new class, only the text description below. Afterwards, other people will evaluate your
edited model on a held-out test split that you cannot see. They will measure class-9
accuracy and old-class accuracy, using argmax over all 10 outputs.

## Description of class 9 ("heart")

The classic heart symbol (♥), drawn as an outline: a closed stroke, not filled. It has two
rounded lobes at the top. The lobes meet in a central inward dip (a cusp). Two sides curve
down and inward and meet at a single sharp point at the bottom. Its width and height are
about equal. Like the other classes, each image is 32x32 grayscale, with a bright shape on a
dark background. Each image has a random size (radius about 7-12 pixels), position,
rotation (any angle, so the point can face any direction), stroke width (about 1.5-3
pixels), brightness, and additive pixel noise.

## Setting

- Repository root: `/home/user/wsp-research`. Run Python as `.venv/bin/python` from the
  repository root. CPU only. Call `torch.set_num_threads(4)` and
  `torch.use_deterministic_algorithms(True)` in your scripts.
- Base model: `checkpoints/stage1_base.pt`. It is frozen: never overwrite it. The
  architecture is `ShapeCNN` in `models/cnn.py`: five 3x3 conv layers with BatchNorm and
  ReLU, two max-pools, global average pooling to a 128-d penultimate feature, and a linear
  layer `fc` (128 -> 10). The model was trained on 32x32 grayscale images of classes
  0 circle, 1 square, 2 triangle, 3 star, 4 cross, 5 ring, 6 ellipse, 7 pentagon, 8 arrow.
  It has never seen class 9. Row 9 of `fc.weight` and `fc.bias[9]` are zero.
- Data access is through `eval/wsp_tools.py` only (read its docstring):
  `load_base()`, `load_val_old()` (validation images of classes 0-8), `to_tensor(x)`, and
  `evaluate_val(ckpt, run_dir)`. Do not call `k_shot_examples`.

## What you may do

- Read `models/cnn.py`, `eval/wsp_tools.py`, `eval/common.py`, and
  `tools/check_no_gradients.py`.
- Read any weight of the base model. Run forward passes and read any activation, on
  the validation images of classes 0-8. These are the only images you may pass through the
  network.
- Change existing weights, add weights, or add a new layer or module. To change the model
  structure, define the new model class in `wsp/stage1/W2/run_RUN/model_def.py`. Then save
  the checkpoint with `arch="custom"`, `module_file="wsp/stage1/W2/run_RUN/model_def.py"`,
  `class_name`, and `init_kwargs`. The format is described in the `models/cnn.py`
  docstring, and `models.cnn.load_model` must be able to load the checkpoint.
- Evaluate at most 3 complete candidate edits on validation data. Use
  `evaluate_val(...)` (it logs each call) or your own forward passes. Use these evaluations
  to check your reasoning, not to search over values.

## What you must not do

- No gradients. Do not use `.backward()`, `torch.autograd`, `torch.optim`, or any
  optimizer. Do not use a search loop or a fitting routine (least squares,
  pseudo-inverse, k-means, random search, or trying several candidate weights in code and
  keeping the best). Every value that `edit.py` writes must be an explicit value or an
  explicit closed-form formula that you can justify.
- Do not load, draw, render, or synthesise any image of a heart, or of any part of a heart.
  Do not pass any input to the network except validation images of classes 0-8.
- Do not read: the test split; `data/shapes.py`; `data/cache/`; `train/`; `results/`;
  `RESULTS.md`; any checkpoint except `checkpoints/stage1_base.pt` and your own output;
  any folder under `wsp/` except your own; `prompts/`; or the git history.
- Do not change files outside `wsp/stage1/W2/run_RUN/` and your output checkpoint.

## Deliverables (in `wsp/stage1/W2/run_RUN/`)

1. `timing.json`: `{"start": <unix seconds>, "end": <unix seconds>}`. Run `date +%s` as
   your very first command and again as your very last command.
2. `edit.py`: loads `checkpoints/stage1_base.pt`, applies your edit, and saves
   `checkpoints/stage1_W2_runRUN.pt`. It must run from the repository root with
   `.venv/bin/python wsp/stage1/W2/run_RUN/edit.py`, be deterministic, and finish in under
   one minute. Comment the code so that each change can be read on its own.
3. `rationale.md`: which weights changed, why, and the effect you expect. List each
   validation evaluation you made and its result.
4. Run `.venv/bin/python tools/check_no_gradients.py` on every `.py` file in your folder.
   It must print PASS for each file.

When you finish, reply with at most 10 lines: what you changed and your validation numbers.
