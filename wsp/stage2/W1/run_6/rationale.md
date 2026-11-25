# Stage 2, condition W1, run 6: rationale

## Tensors changed

- `frame_voices.weight`, row 2 only (192 values, 0.0042 % of all weights):
  `frame_voices.weight[2] = fv0 + 1.988375 * (fv1 - fv0)`, with fv0/fv1 the F2/M2 rows.
- Nothing else. `voices.weight` (all rows), rows 0 and 1 of `frame_voices.weight`, and every
  shared tensor are unchanged, so voices 0 and 1 are bit-identical to the base model.

## Why

Mechanism measurements on the base model (forward passes, base voices only):

1. Separation of the two voice tables. Combining token row t and frame row f from {F2, M2}
   (20 enrollment sentences): predicted durations depend only on the token row
   (6548 frames with the F2 row vs 5919 with the M2 row, whatever the frame row), while the
   predicted mean log F0 depends almost only on the frame row
   (frame row F2: 266.5 / 256.7 Hz; frame row M2: 168.9 / 167.0 Hz for token row F2 / M2).
   So pitch and timbre live in `frame_voices.weight`, speaking rate in `voices.weight`.
2. The frame row reaches the prosody head almost directly: the mean difference of the
   prosody-stack output between F2 and M2 has cosine 0.85 with (LayerNorm gain x row
   difference / residual std), norm ratio 0.94. The acoustic stack, by contrast, amplifies the
   difference about 5x and rotates it (cosine 0.43), so steering vocoder heads (spectral
   envelope) by hand-propagated vectors is not reliable; I did not attempt it.
3. Target analysis (enrollment recordings, WORLD Harvest): median F0 106 Hz (M2 164 Hz, F2
   261 Hz); log-F0 spread similar to M2 (0.28 vs 0.29); about 15 % more active speech time
   than M2. It is a lower-pitched male than M2.

Edit: the only "voice dimension" the model has learned is the F2 -> M2 line. The target is
further along the male direction than M2, so row 2 is placed on that line beyond M2. The
position alpha is a closed-form formula: the model's predicted mean log F0 is taken as linear in
alpha between the two trained voices (p_F2 = 0.260 at alpha 0, p_M2 = -0.178 at alpha 1), and
alpha is where it reaches the target pitch in model units,
goal = p_M2 + ln(106.3 / 163.9) = -0.611, alpha = (goal - p_F2) / (p_M2 - p_F2) = 1.988.
The statistics were computed by `scratch/cand1.py` on all 50 enrollment sentences (Harvest at
24 kHz) and written into edit.py as explicit numbers, so edit.py reproduces exactly the state
dict that was checked on validation data.

Two alternatives were built and checked (see Validation evaluations) and rejected:
- Pitch-only steering on M2's timbre (candidate 2): row 2 = M2 row + c * d, where d is the
  prosody head's log-F0 readout direction times the final LayerNorm gain, orthogonalized to the
  constant vector and to the voicing and energy readouts, and c is the closed-form root of the
  LayerNorm equation (sigma*s + c*k)^2 = (s+delta)^2 (sigma^2 + c^2/192) for the shift
  delta = ln(F0_target/F0_M2). It matched the pitch better but lost speaker similarity
  (0.627 vs 0.670): the timbre change along the male axis matters more than exact pitch.
- Axis extrapolation plus F2's token row (slower speech; the target is slower than both trained
  voices, beta capped at the F2 row rather than extrapolated) plus the same pitch-only
  correction for the pitch gap left after the axis step (step-2 audio F0 119 Hz vs 106.8 Hz
  target). Pitch error and duration ratio improved, but speaker similarity (0.651) and
  mel-cepstral distortion (10.85) were worse than the plain axis edit, so it was not kept.
  Its script is archived as `scratch/edit_eval3_version.py.txt`.

## Expected effect

- Voice 2 is a deeper male voice than M2: the analysis of the edited voice gives a median F0 of
  about 119 Hz (Harvest at 8 kHz, 12 enrollment sentences); the linear assumption under-shoots the
  106 Hz target (the predicted pitch saturates slightly beyond M2), and the pitch contour is
  somewhat flatter (log-F0 spread 0.23 vs the target's 0.29).
- Timbre moves further along the model's learned female -> male direction (more low-frequency
  energy, darker high frequencies), which raised speaker similarity from the M2-timbre
  alternative's 0.627 to 0.670 on validation.
- Intelligibility is preserved (WER 0.0 on the 20 validation sentences).
- Voices 0 and 1 are unchanged (only row 2 of one table was written); the speaking rate of voice
  2 is M2's, faster than the target (duration ratio 0.71 on validation).
- Not addressed: the target's brighter high band (> 6 kHz), higher breathiness and lower level,
  which the 1-D voice line cannot express.

## Validation evaluations

3 used (all with `tts_evaluate_val`, seed 0, logged in `val_log.jsonl`):

| # | candidate | speaker sim | MCD (dB) | F0 RMSE (cents) | WER | duration ratio | voices 0/1 |
|---|-----------|-------------|----------|-----------------|-----|----------------|------------|
| 1 | axis extrapolation, alpha 1.988 (final edit) | 0.670 | 10.76 | 272.9 | 0.0 | 0.708 | identical |
| 2 | M2 row + pitch-only steering | 0.627 | 10.81 | 257.7 | 0.0 | 0.708 | identical |
| 3 | axis (alpha 1.934) + F2 token row + pitch correction | 0.651 | 10.85 | 263.0 | 0.0 | 0.771 | identical |

The final edit is candidate 1 (its state dict is byte-identical in values to the evaluated one).
After evaluation 1, I also analysed candidate 1's audio on enrollment sentences (pitch,
envelope, aperiodicity); no other candidate was scored against the target.

## Files read

- `models/tiny_voca3.py`
- `eval/wsp_tools.py`
- `eval/common.py`
- `tools/check_wsp_edit.py` (header and rule section; run as instructed)
- `.venv/lib/python3.13/site-packages/tiny_voca/model.py`
- `.venv/lib/python3.13/site-packages/tiny_voca/config.py`
- `.venv/lib/python3.13/site-packages/tiny_voca/infer.py` (function list and lines 90-170)
- `.venv/lib/python3.13/site-packages/tiny_voca/` directory listing
- `checkpoints/stage2_base.pt` (via torch.load and wsp_tools), my outputs
  `checkpoints/stage2_W1_run6.pt` and scratch checkpoints
- Files in my own run folder `wsp/stage2/W1/run_6/` (scratch scripts, val_log.jsonl)
- Data only through `eval/wsp_tools.py` (enroll and val texts; enroll target recordings)
