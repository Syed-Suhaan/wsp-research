# Rationale: stage 2, condition W1, run 2

Writer: Claude Opus 5.5 (model ID `claude-opus-5-5`). Effort level: unverified.

## Tensors changed

Only two rows changed: 384 values, 0.0085 % of the 4,537,675 values in the state dict.

- `voices.weight[2]` (token-level voice row, 192 values): set to `M2 + a_tok * (F2 - M2)`
  with `a_tok = 1.0`. The result equals F2's row up to float rounding (max difference 1.2e-7).
- `frame_voices.weight[2]` (frame-level voice row, 192 values): set to
  `frame_voices[M2] + c * u`, where `u` is the unit pitch-readout direction defined below and
  `c = -7.422`.

No other tensor changed. Rows 0 and 1 and all shared tensors keep their base values, so
voices 0 (F2) and 1 (M2) are bit-identical to the base model. The evaluator confirmed this:
`identical: true` for both voices.

## Why

**Analysis of the target speaker.** I used the enrollment recordings and synthesis of the base voices.

- Pitch. The target is a low male voice. Its mean log F0 is about 4.68, about 107 Hz (WORLD
  harvest). The base M2 voice is at 5.09, about 162 Hz, and F2 is at 5.55. The target is about
  7 semitones below M2, which is the largest mismatch I found.
- Speaking rate. The target speaks about 22 % more slowly than M2, measured as the ratio of
  speech spans. The base model's F2 voice is about 10.6 % slower than M2 (frame-count ratio).
- Timbre. LPC formant medians (target F1/F2/F3 = 503/1570/2551 Hz, M2 448/1499/2583,
  F2 575/1790/3380) and the mean cepstral difference show only small envelope differences
  between the target and M2. These differences do not point consistently along the F2-M2 axis.
  So I kept the frame row at M2 for timbre and did not move it toward F2.

**What each voice row controls.** I probed the base model's internal predictions with forward passes. The token
row (`voices`) mainly sets the phoneme durations: moving it toward F2 slows speech and
hardly changes pitch. The frame row (`frame_voices`) sets pitch and the envelope. Moving
only the frame row along the F2-M2 line saturates in pitch. The most male setting I tried
(a = -1) still left the voice above the target pitch. I therefore wrote pitch along a
dedicated direction instead.

**Edit 1, speaking rate (token row).** `a_tok = clip((r_t - 1) / (r_f - 1), 0, 1)`, where
r_t = 1.218 is the target/M2 speech-span ratio and r_f = 1.106 is the F2/M2 frame ratio
(sentences 0-19 of enrollment). The unclipped value, about 2.06, would extrapolate the row far
outside the two trained voices. In my analysis, an extrapolated row flattened the pitch contour
(internal log-F0 SD 0.195 -> 0.16), lowered pitch, and darkened the spectral tilt. The cap at 1
keeps the row on a trained voice (F2's token row), which gives the largest rate change toward the
target that stays inside the trained voices.

**Edit 2, pitch (frame row).** The frame row is added to every frame before the prosody
transformer. Through the residual stream it reaches that transformer's final LayerNorm, whose
output `prosody_head.weight[0]` reads as log(F0/200 Hz). On this direct path, adding a vector
d changes log F0 by about `sum_i w_i g_i d_i / sigma`. Here w = `prosody_head.weight[0]`,
g = `prosody_stack.norm.weight`, and sigma = 2.074 is the mean per-frame standard deviation of
the LayerNorm input for M2. The pitch write is therefore along
`u = centre(g*w) / |centre(g*w)|` (|centre(g*w)| = 0.1215; centring removes the channel mean,
which the LayerNorm discards). The needed change is the target-minus-M2 log-F0 difference
measured from audio (-0.4076), corrected for the small pitch shift of edit 1. This gives a
needed internal change of -0.4190.

The direct-path length is `c0 = needed * sigma / |centre(g*w)| = -7.150`. The two prosody
blocks also respond to the written vector, so edit.py measures the realized change once at c0
(-0.4036) and sets `c = needed / (realized / c0) = -7.422`. This is one sensitivity
measurement, not a search: no candidates are compared or selected. The vector is pitch-specific.
Its cosine with the voicing and energy readouts is 0.06 and 0.15, and the content and timing of
the frames are untouched.

## Expected effect

- Voice 2 speaks about 7 semitones lower than M2, with mean F0 about 107 Hz, matching the
  target. The internal mean log F0 on enrollment sentences 0-19 is -0.589, equal to the
  intended -0.588. The pitch contour is somewhat flatter than M2's (internal log-F0 SD 0.20
  versus 0.26).
- Voice 2 speaks about 10 % more slowly than M2 (F2's durations). It is still faster than the
  target (duration ratio about 0.77 on validation).
- The timbre stays M2-like. M2's envelope was already close to the target's.
- Intelligibility is unchanged. Voices 0 and 1 are exactly unchanged.

## Validation evaluations

I used 2 of the allowed 3 evaluations, both with `tts_evaluate_val` at seed 0.

1. Final edit (`checkpoints/stage2_W1_run2.pt`, a_tok = 1, c = -7.422):
   speaker_sim 0.626, MCD 10.92 dB, F0 RMSE 259.5 cents, duration ratio 0.771, WER 0.0,
   voices 0 and 1 identical.
2. A check of edit 1. Same procedure, but with the token row left at M2 (a_tok = 0; pitch
   re-derived, c = -8.319), saved to scratch and since deleted: speaker_sim 0.617, MCD
   10.82 dB, F0 RMSE 259.7 cents, duration ratio 0.708, WER 0.0, voices 0 and 1 identical.
   This tested whether F2's token row carries female identity into voice 2. It does not: the
   slower rate gave slightly higher speaker similarity, which is the most important metric.
   So I kept edit 1. The final edit.py is the version of evaluation 1. Rerunning it reproduces
   the same file (same SHA-256).

## Files read

- `models/tiny_voca3.py`
- `eval/wsp_tools.py`
- `eval/common.py`
- `tools/check_wsp_edit.py`
- `.venv/lib/python3.13/site-packages/tiny_voca/model.py`
- `.venv/lib/python3.13/site-packages/tiny_voca/config.py`
- `.venv/lib/python3.13/site-packages/tiny_voca/infer.py`
- `.venv/lib/python3.13/site-packages/tiny_voca/text.py` (first 60 lines)
- the file listing of `.venv/lib/python3.13/site-packages/tiny_voca/`
- files in my own run folder: `edit.py`, `val_log.jsonl`, and my scratch scripts and outputs in
  `scratch/`
- data only through `eval/wsp_tools.py`: the base checkpoint, enrollment and validation
  sentences, enrollment target recordings, synthesis, and `tts_evaluate_val` (2 calls)
