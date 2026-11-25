# Stage 2, condition W1, run 0: rationale

## Tensors changed

- `frame_voices.weight`, row 2 only (192 values; 0.0042 % of the 4,537,675 values in the
  state dict). Rows 0 and 1 are unchanged.
- `voices.weight` (token-level voice) is not changed: row 2 stays a copy of M2 (row 1).
- No shared tensor is changed, so voices 0 and 1 are bit-identical to the base model.

New row: `v2 = M2 + a * (F2 - M2) + c * u_hat`, with the values computed by `edit.py`:
a = -0.2982, c = -4.9538 (|v2| = 15.33; |M2| = 10.34).

## Why

Analysis of the base model (forward passes on enrollment sentences, voices 0 and 1):

- The target speaker is a low male voice: mean F0 107 Hz (WORLD DIO + StoneMask, voiced
  frames), against 158 Hz for the model's M2 and 249-267 Hz for F2. Its spectral envelope is
  darker than M2 in 300-3000 Hz.
- The frame-level voice embedding dominates the voice difference in the frame-level stacks:
  the mean difference of the prosody-stack residual between F2 and M2 has cosine 0.83 with
  the difference of the two `frame_voices` rows, so this vector travels mostly along the
  residual (identity) path.
- The log-F0 output is a linear read-out of the prosody stack's final LayerNorm:
  logF0 = w_f0 . (gamma_p * (x - mean(x)) / sigma_p) + b. Applying this read-out to the measured
  mean residual difference between F2 and M2 reproduces the measured log-F0 difference exactly
  (0.459 predicted, 0.460 measured), so the read-out is a reliable way to predict the pitch
  effect of a vector added to that residual stream.

Design, in two parts:

1. Timbre along the trained voice axis (term `a * (F2 - M2)`). The only learned variation of
   timbre between voices is the F2-M2 axis. `a` is the position of the target on this axis,
   measured on the mean log spectral envelope (voiced frames, 100-4000 Hz, overall level
   removed): the projection of (target - M2) onto (F2 - M2). a = -0.30: the target lies beyond
   M2, away from F2 (darker), by 0.3 of the F2-M2 distance. A larger move would increase the
   envelope distance (the projection is the closest point on the axis).
2. Pitch along the log-F0 read-out direction (term `c * u_hat`). u = gamma_p * w_f0 with its
   channel mean removed (LayerNorm removes the mean), so a vector d added to the frame
   embedding changes the predicted log F0 by about u . d / sigma_p, where sigma_p = 2.079 is
   the mean per-frame std of the prosody residual before the final LayerNorm (voice M2).
   The wanted change is delta = log(F0_target / F0_M2) = -0.385. Part 1 already gives
   a * u . (F2 - M2) / sigma_p = -0.096 by the same formula, so
   c = (delta - (-0.096)) * sigma_p / |u| = -4.95. u_hat is the direction that changes log F0
   most per unit of embedding norm, so it moves pitch with the smallest change to everything
   else.

The token-level table `voices` is left at M2: moving it toward F2 would slow the speech
(the target speaks more slowly than M2) but would also bring F2 pitch and timbre into the
encoder states; moving it away from F2 would speed the speech up. Keeping it at M2 keeps the
duration predictor and encoder in the trained range, which protects intelligibility.

All statistics in `edit.py` come from the first 25 enrollment sentences (target recordings and
the base model's synthesis of voices 0 and 1), to keep `edit.py` under five minutes. The
script is deterministic (two runs gave the same SHA-256 of the output file).

## Expected effect

- Voice 2 mean F0 near the target's 107 Hz instead of M2's 158 Hz. Check on 12 enrollment
  sentences not used for the statistics (25-36): voice 2 at 111.9 Hz (target 106.4 Hz, M2
  161.2 Hz; the direct-path formula reaches about 88 % of the wanted log-F0 change). Pitch
  spread (std of log F0) 0.251, closer to the target's 0.232 than M2's 0.296.
- A darker spectral envelope, closer to the target: RMS distance of the level-free mean log
  envelope (100-4000 Hz) to the target 0.449, against 0.540 for M2 (same 12 sentences).
- Speaking rate as M2 (the target is slower; this is not corrected).
- Intelligibility kept; voices 0 and 1 identical to the base model.

## Validation evaluations

1 of 3 used (`tts_evaluate_val`, seed 0, logged in `val_log.jsonl`), on the final edit:

| metric | value |
|---|---|
| speaker similarity (mean, 20 sentences) | 0.653 |
| mel-cepstral distortion | 10.83 dB |
| F0 RMSE | 262 cents |
| duration ratio | 0.708 |
| WER | 0.000 |
| voices 0 and 1 unchanged | yes (identical, MCD 0.0 dB vs base) |

No other candidate was evaluated. Analysis checks on enrollment sentences 25-36 (pitch and
envelope of the same final edit, listed under Expected effect) were made once, before the
validation evaluation, and did not change any value.

## Files read

- `models/tiny_voca3.py`
- `eval/wsp_tools.py`
- `eval/common.py`
- `tools/check_wsp_edit.py`
- `.venv/lib/python3.13/site-packages/tiny_voca/model.py`
- `.venv/lib/python3.13/site-packages/tiny_voca/infer.py`
- `.venv/lib/python3.13/site-packages/tiny_voca/config.py`
- `checkpoints/stage2_base.pt` (via `eval/wsp_tools.py`) and the output
  `checkpoints/stage2_W1_run0.pt`
- files in `wsp/stage2/W1/run_0/` (own scripts in `scratch/`, `val_log.jsonl`)
- Data only through `eval/wsp_tools.py`: `tts_texts`, `tts_target_audio` (enroll split),
  `tts_synthesize`, `tts_evaluate_val`. The validation target recordings were not read
  directly.

Note: the agent harness placed the repository's project instruction file in the writer's
context automatically at start; it was not opened and its content was not used.
