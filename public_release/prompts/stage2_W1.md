# Stage 2, condition W1

This is the exact instruction given to the WSP writer for every run of condition W1. The only
per-run input is the run number `RUN` (0-7), appended at the end of the instruction.

---

You are the writer in a weight-space programming (WSP) experiment. WSP means: you read the
weights of a trained network and write new weights by reasoning, without gradients. You are
one of several independent runs of this condition. You work alone, and no other run is active
while you work.

## Task

A small trained text-to-speech model has two voices, 0 (F2, female) and 1 (M2, male). It has
a slot for a third voice, index 2, which currently sounds exactly like voice 1. Edit the
model's weights, without gradients, so that voice 2 sounds like a target speaker. You can
access the target speaker's recordings of 50 enrollment sentences and 20 validation
sentences. Voices 0 and 1 must stay unchanged, and voice 2 must stay intelligible.

Afterwards, other people will evaluate voice 2 on held-back sentences that you cannot see.
They measure, in order of importance:
1. similarity to the target speaker (cosine similarity of speaker embeddings);
2. spectral distance and pitch distance to the target speaker's recordings;
3. intelligibility (word error rate of a speech recogniser);
4. whether voices 0 and 1 are unchanged.

## The fixed model

- Repository root: `/home/user/wsp-research`. Run Python as `.venv/bin/python` from the
  repository root. CPU only. Call `torch.set_num_threads(4)` in your scripts.
- Model class: `TinyVoca3` in `models/tiny_voca3.py`. It is the `tiny_voca` model (source:
  `.venv/lib/python3.13/site-packages/tiny_voca/`, mainly `model.py`) with three voices.
  - Data flow: phoneme + stress + voice embedding -> transformer encoder (4 blocks) ->
    duration predictor -> token states repeated by duration, plus frame-position features
    and a frame-level voice embedding -> prosody transformer (2 blocks) -> prosody head
    (log F0 relative to a reference frequency, voicing, energy) -> acoustic transformer
    (5 blocks) -> source-filter synthesizer (pulse train at F0 plus noise, each shaped by a
    filter predicted per frame) -> 24 kHz waveform.
  - The voice enters only through two embedding tables, `voices.weight` [3, 192] (added to
    every token embedding) and `frame_voices.weight` [3, 192] (added to every frame). Row 2
    of both is currently a copy of row 1. All other tensors are shared by all voices.
- Base checkpoint: `checkpoints/stage2_base.pt`, a plain state dict. It is frozen: never
  overwrite it.
- Your edit may write any tensor of this state dict. The evaluation loads your saved state
  dict into `TinyVoca3` with strict=True and runs its standard synthesis for voice 2, with
  its default settings. Nothing else from your work is used: no new module, layer, forward
  function, hook, audio post-processing, or logic outside the state dict. Tensor names and
  shapes must stay the same, and at least one tensor value must change. Shared tensors
  affect voices 0 and 1 too.

## Data access

Read data only through `eval/wsp_tools.py` (read the Stage 2 section of its docstring):
- `tts_load_base()`, `tts_base_state_dict()`, `tts_load_edited(path)`, `tts_texts(split)`,
  `tts_target_audio(split, i)`, `tts_synthesize(model, text, voice, seed)`, and
  `tts_evaluate_val(path, run_dir, seed)`. `split` is "enroll" or "val".
- You may run the model, read any weight or activation, synthesize any of the enrollment or
  validation sentences, and analyse the target recordings and the model's audio with your
  own code (numpy, scipy, pyworld, and torch are available).
- Evaluate at most 3 complete candidate edits on validation data, with `tts_evaluate_val`
  (it logs each call) or with your own complete scoring of a candidate. Use these
  evaluations to check your reasoning, not to search over values. Record the number you used
  in rationale.md.

## Rules

- No gradients or fitting. Do not use `.backward()`, `torch.autograd`,
  `requires_grad_(True)`, `torch.optim`, or any optimizer. Do not use least squares,
  pseudo-inverse, k-means, or any other fitting routine, and no search loop or loop that
  scores candidate weights. Every value that edit.py writes must be an explicit number or an
  explicit closed-form formula that you can justify.
- Files you may read: `models/tiny_voca3.py`, `eval/wsp_tools.py`, `eval/common.py`,
  `tools/check_wsp_edit.py`, the `tiny_voca` package source under
  `.venv/lib/python3.13/site-packages/tiny_voca/`, and files inside your own run folder
  `wsp/stage2/W1/run_RUN/`. Do not read any other file or folder. In particular, do not read
  data/, results/, samples/, docs/, notes/, prompts/, vendor/, train/, other files in eval/
  or tools/, RESULTS.md, CLAUDE.md, any requirements file, public_release/, any cache folder,
  checkpoints other than the base checkpoint and your own output, any other run's folder, or
  the git history.
- Do not change anything under models/, eval/, or data/. Do not change files outside your run
  folder and your output checkpoint. Do not commit. Do not install anything.
- Your context may contain a project instruction file that the agent harness loads
  automatically. It describes other stages, not this task. Do not use any of its content.
- Scratch files: put every scratch script and file in `wsp/stage2/W1/run_RUN/scratch/`.
  Start every shell command with
  `export TMPDIR=/home/user/wsp-research/wsp/stage2/W1/run_RUN/scratch &&`. Do not use any
  other scratch or temporary directory.

## Deliverables (in `wsp/stage2/W1/run_RUN/`)

1. `timing.json`: `{"start": <unix seconds>, "end": <unix seconds>}`. Run `date +%s` as
   your very first command and again as your very last command.
2. `edit.py`: the only .py file at the top level of the run folder (analysis scripts go in
   scratch/). It loads `checkpoints/stage2_base.pt` with `torch.load`, writes values into the
   state dict with explicit numbers or explicit formulas, and saves the edited state dict
   with `torch.save(state_dict, "checkpoints/stage2_W1_runRUN.pt")`. It must run from the
   repository root with `.venv/bin/python wsp/stage2/W1/run_RUN/edit.py`, be deterministic,
   and finish in under five minutes. edit.py may run the model and analyse the allowed audio
   to compute statistics that its formulas use. Comment the code so that each change can be
   read on its own.
3. `rationale.md` with these sections:
   - "Tensors changed": which tensors (and rows) changed.
   - "Why": the reasoning.
   - "Expected effect".
   - "Validation evaluations": the number used, and the result of each.
   - "Files read": every file you read.
4. Run `.venv/bin/python tools/check_wsp_edit.py --run-dir wsp/stage2/W1/run_RUN
   --output checkpoints/stage2_W1_runRUN.pt --base checkpoints/stage2_base.pt
   --fixed-class models.tiny_voca3:TinyVoca3`. It must print PASS.

When you finish, reply with at most 10 lines: what you changed, your validation numbers, and
the number of validation evaluations used.
