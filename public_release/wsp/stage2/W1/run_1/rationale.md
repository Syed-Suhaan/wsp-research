# Stage 2, W1, run 1: rationale

## Tensors changed

Only row 2 (the new voice) of the two voice tables. No shared tensor changed, so voices 0 (F2)
and 1 (M2) are bit-identical to the base model. 384 values changed (0.0085% of all weights).

- `voices.weight[2]` (token-level voice) := `voices.weight[0]` (the F2 row).
- `frame_voices.weight[2]` (frame-level voice) :=
  `FV1 + a_env * (FV0 - FV1) + s * u`, with `a_env = -0.2881`, `s = -5.0473`, and `u` the unit
  centred vector of `prosody_stack.norm.weight * prosody_head.weight[0]`.

All coefficients are computed in `edit.py` from closed-form formulas over the first 20
enrollment sentences (target recordings, and the base model's voices 0 and 1).

## Why

Measurements (pyworld harvest F0, cheaptrick envelope -> 80-band log mel -> DCT c0..c24; first
20 enrollment sentences):

| source | median F0 | active speech (20 sentences) |
|---|---|---|
| target | 106.4 Hz | 69.0 s |
| F2 (voice 0) | 262.7 Hz | 61.5 s |
| M2 (voice 1) | 164.5 Hz | 56.8 s |

The target is a low-pitched male voice, about 7.5 semitones below M2 (on the far side of M2
from F2: -0.93 of the F2-M2 log-F0 difference), and speaks about 21% slower than M2.

Mechanism analysis (forward passes with swapped table rows, reading the model's internal
prosody and harmonic-head cepstrum):
- The token-level table `voices` controls durations: F2 token row with M2 frame row gives
  10.6% more frames and changes the predicted F0 by under 1%. Its share of the F2-M2
  harmonic-envelope difference is ~-5% (projection), i.e. it carries almost no timbre.
- The frame-level table `frame_voices` carries pitch and timbre: M2 token row with F2 frame
  row reproduces 96% of the F2-M2 envelope difference and the F2 pitch.

So the two tables are two nearly independent knobs:

1. Speaking rate (`voices[2]`). Coefficient along M2 -> F2 in log duration:
   ln(D_T/D_1)/ln(D_0/D_1) = 2.4, clipped to the trained range [0, 1] because extrapolating
   the encoder's input embedding risks swamping the phoneme embeddings (intelligibility).
   Result: the F2 row, about 8% slower speech with no timbre change.

2. Spectral envelope (`frame_voices[2]`, part a). The target's mean mel-cepstrum c1..c24
   (c0 = level excluded, as in MCD) projected onto the M2 -> F2 cepstral axis gives
   a_env = ((c_T - c_1) . (c_0 - c_1)) / |c_0 - c_1|^2 = -0.288, i.e. a mild extrapolation
   on the male side of M2 (e.g. higher c2..c6 like the target).

3. Pitch (`frame_voices[2]`, part b). Along the voice axis, pitch wants a = -0.93 but the
   envelope wants a = -0.29, so pitch gets its own term. Assuming log F0 is linear along the
   axis (measured at a = 0 and a = 1), the axis alone gives l_1 + a_env (l_0 - l_1); the gap to
   the target is dlf0 = -0.301 (about -5.2 semitones). The prosody head reads log F0 as
   `w0 . LN(x)` where LN is `prosody_stack.norm`. An offset d added to the frame input passes
   the residual stream to that LayerNorm and changes log F0 by approximately
   `(gamma * w0) . P d / sigma` (P removes the mean). The unit direction with the largest
   effect per unit norm is `u = P(gamma * w0) / |P(gamma * w0)|`, and the scale is
   `s = dlf0 / (|P(gamma * w0)| * mean(1/sigma))`, with mean(1/sigma) = 0.4905 measured on
   the residual stream of voice 2 (10 enrollment sentences). The acoustic stack is
   conditioned on the predicted pitch, so it renders the envelope for the lower pitch.

No candidate weights were scored against the target during design: all coefficients come
from endpoint measurements (target, voice 0, voice 1) and one activation statistic.

## Expected effect

- Voice 2 speaks at about 106-110 Hz (measured after the edit on 8 enrollment sentences:
  110.4 Hz vs target 106.4 Hz; the linear-along-axis assumption leaves about 60 cents high),
  with an envelope moved from M2 toward the target along the trained voice axis, and about
  8% slower than M2.
- Intelligibility stays as for the trained voices (all rows within or near the trained
  range; frame row norm 15.3 vs 10.3 for M2).
- Voices 0 and 1 are exactly unchanged.
- Remaining mismatch: envelope detail off the F2-M2 axis (target has more energy below
  ~200 Hz and above ~6 kHz, less around 1-3 kHz), and speaking rate (~11% faster than the
  target in active speech).

## Validation evaluations

1 used (of 3), `tts_evaluate_val(checkpoints/stage2_W1_run1.pt, seed=0)`:

| metric | value |
|---|---|
| speaker similarity | 0.648 |
| MCD | 10.89 dB |
| F0 RMSE | 260 cents |
| duration ratio | 0.771 |
| WER | 0.0 |
| voices 0 and 1 unchanged | identical (MCD vs base 0.0 dB, 3 sentences each) |

No change was made after this evaluation.

## Files read

- `models/tiny_voca3.py`
- `eval/wsp_tools.py`
- `eval/common.py`
- `tools/check_wsp_edit.py`
- `.venv/lib/python3.13/site-packages/tiny_voca/model.py`
- `.venv/lib/python3.13/site-packages/tiny_voca/config.py`
- `.venv/lib/python3.13/site-packages/tiny_voca/infer.py`
- `checkpoints/stage2_base.pt` (through torch.load / wsp_tools) and the output
  `checkpoints/stage2_W1_run1.pt`
- own files in `wsp/stage2/W1/run_1/` (edit.py, scratch/ analysis scripts and outputs,
  val_log.jsonl)
- Data only through `eval/wsp_tools.py` (enrollment texts and target audio, validation
  evaluation).
