# Rationale: Stage 2, condition W1, run 4

## Tensors changed

- `frame_voices.weight`, row 2 only (192 of 4,537,675 state-dict values, 0.0042 %). Nothing else.
  `voices.weight[2]` stays the base copy of M2. All shared tensors are unchanged, so voices 0
  and 1 are bit-identical to the base (the validation report confirms "identical": true).
- Final row: `frame_voices[2] = M2 + beta * (M2 - F2) + a * u`, with beta = 0.8869,
  a = -2.0962 and u the unit F0-readout direction defined below (row norm 20.5; trained rows
  have norm about 10).

## Why

Analysis of the base model (enrollment sentences only):

1. The two voice tables have separate jobs. Swapping only one row between F2 and M2 showed
   that the token-level row (`voices`) sets the durations (speaking rate) and has almost no
   effect on pitch or spectrum (F2 tokens with M2 frames: F0 165.6 vs 160.9 Hz, mean
   mel-cepstrum distance to the target unchanged), while the frame-level row
   (`frame_voices`) sets pitch and spectral envelope (M2 tokens with F2 frames: F0 253 Hz,
   F2-like spectrum).
2. The target is a low male voice: median F0 about 109 Hz (WORLD Harvest), against 164 Hz
   for M2 and 260 Hz for F2. Its long-term spectrum continues the F2 -> M2 trend in the
   500-1000 Hz band, and the mean mel-cepstrum distance to the target falls when M2 is moved
   further along the F2 -> M2 axis (4.77 -> 4.46 at beta = 0.85). The target also speaks
   about 1.21x slower than M2 (trimmed speech span).

Edit, all closed-form (edit.py computes every statistic from the first 12 enrollment
sentences):

- Step 1, voice axis: `frame_voices[2] = M2 + beta (M2 - F2)` with
  `beta = (ln f0_M2 - ln f0_target) / (ln f0_F2 - ln f0_M2)`: the point on the learned
  female -> male axis where log F0 would reach the target if it changed linearly along the
  axis. This moves pitch and timbre together in the direction the trained voices define.
- Step 2, pitch steering: the response along the axis is sublinear (F0 122.5 Hz after step 1,
  not 109 Hz), because the larger row raises the residual norm that the LayerNorms divide by.
  The remaining gap is closed with a vector along `u = normalize(centre(gamma * w_f0))`,
  where `gamma` is the prosody stack's final LayerNorm gain and `w_f0` the prosody head's
  log-F0 row. Adding `a u` to the residual changes the predicted log F0 by about
  `a (w_f0 . (gamma u)) / sigma`, with `sigma` = the measured per-frame std of the prosody
  residual before that LayerNorm (2.18). `a = (ln f0_target - ln f0_step1) / gain`.
  On enrollment sentences 12-19 (not used by the formulas) the result has median F0
  109.1 Hz vs the target's 108.1 Hz.
- The speaking rate is left at M2's. Candidate 2 (below) matched the rate by moving the
  token row along the M2 -> F2 axis (alpha = 2.19, computed from the speech-span ratios); the
  rate matched (span ratio 1.00) but speaker similarity fell from 0.665 to 0.631 and the F0
  error rose, so changing the token row carries F2 traits into the voice. Candidate 3 removed
  the token change entirely (the minimal edit) and scored best.

Rejected idea: steering the harmonic filter head toward the target's long-term spectrum via
the transposed readout. The readout map only reproduces the wanted cepstral change with
cosine 0.57, so it would add uncontrolled spectral changes; it was not used.

## Expected effect

Voice 2 is a low male voice at the target's pitch level (about 109 Hz median), with a timbre
moved past M2 along the model's own female -> male axis. Phone durations, and therefore
intelligibility, are those of M2 (the token path is unchanged). Voices 0 and 1 are unchanged
bit for bit. Speaking rate stays about 1.2x faster than the target.

## Validation evaluations

3 used (`tts_evaluate_val`, seed 0, logged in val_log.jsonl). Validation voice 2, 20 sentences:

| # | candidate | speaker sim | MCD dB | F0 RMSE cents | WER | duration ratio | voices 0/1 |
|---|-----------|-------------|--------|---------------|-----|----------------|------------|
| 1 | token row = F2 row; frame row = M2 + 0.937 (M2-F2) + pitch steer (a = -1.55) | 0.6647 | 10.824 | 256.1 | 0.0 | 0.771 | identical |
| 2 | token row = M2 + 2.19 (F2-M2) (rate match); frame row as in 1 (beta 0.737, a -1.19) | 0.6313 | 10.842 | 277.9 | 0.0 | 0.894 | identical |
| 3 (final) | token row unchanged (M2); frame row = M2 + 0.887 (M2-F2) + pitch steer (a = -2.10) | 0.6686 | 10.773 | 250.9 | 0.0 | 0.708 | identical |

Each evaluation tested a hypothesis (1: the axis + pitch construction; 2: matching the
speaking rate helps; 3: the token row should stay unchanged), not a parameter search. The
final edit is candidate 3. The scripts for candidates 1 and 2 are kept in scratch/.

## Files read

- models/tiny_voca3.py
- eval/wsp_tools.py
- eval/common.py
- tools/check_wsp_edit.py (first 80 lines)
- .venv/lib/python3.13/site-packages/tiny_voca/model.py
- .venv/lib/python3.13/site-packages/tiny_voca/config.py
- .venv/lib/python3.13/site-packages/tiny_voca/infer.py (lines 1-130 and a grep of the rest)
- checkpoints/stage2_base.pt (via torch.load / wsp_tools), checkpoints/stage2_W1_run4.pt (own output)
- Files in wsp/stage2/W1/run_4/ (own scripts in scratch/, val_log.jsonl)
- Data only through eval/wsp_tools.py: enrollment texts and target recordings, and the
  validation report from `tts_evaluate_val`.
