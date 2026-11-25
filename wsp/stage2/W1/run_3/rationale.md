# Rationale: stage 2, condition W1, run 3

## Tensors changed

- `voices.weight`, row 2 only (192 values): set to row 0 (the F2 token-level voice row).
- `frame_voices.weight`, row 2 only (192 values): set to `frame_voices[1] - k * d_par`, with
  k = 1.5262 (computed by edit.py) and d_par defined below.

No other tensor changes. Rows 0 and 1 of both tables and all shared tensors are untouched, so
voices 0 (F2) and 1 (M2) are bit-identical to the base model. 384 of 4,537,675 values changed
(0.0085 %).

## Why

Analysis of the base model (forward passes only, in scratch/):

1. Role of the two tables. Swapping single tables between F2 and M2 in slot 2 showed that the
   token-level table `voices` controls only the speaking rate (frames +10 % with the F2 row; F0
   and spectral envelope unchanged within 1 %), while the frame-level table `frame_voices`
   controls pitch and spectral envelope and does not change durations.
2. Prosody stack is close to linear in the frame voice. For the F2 - M2 difference d, the mean
   change of the prosody stack's final residual stream has cosine 0.88-0.90 with d and the same
   norm. The prosody head reads log F0 from LayerNorm(x), so the direct-path pitch readout
   direction in the residual stream is r = centre(g * w) (g = final LayerNorm gain, w = log-F0
   row of `prosody_head.weight`).
3. Decoupling pitch from timbre. d splits into d_par = (d . r_hat) r_hat (|d_par| = 5.49) and
   d_perp = d - d_par (|d_perp| = 11.56). Measured on 8 enrollment sentences, M2 + d_par raises
   log F0 by 0.268 but moves the mel-cepstral envelope only 0.17 of the way to F2; M2 - d_par
   lowers log F0 by 0.255 with a spectral move of -0.07. M2 + d_perp moves the spectrum 0.81 of
   the way with only +0.196 log F0. So d_par is an almost pure pitch control.
4. Target speaker (enrollment recordings, pyworld, silence-trimmed): mean F0 103 Hz versus 160 Hz
   for M2 and 262 Hz for F2; about 23 % longer speech than M2 audio (and longer than F2 audio);
   its mean mel-cepstral envelope lies only -0.24 of the F2-M2 distance beyond M2 on the F2-M2
   axis, with most of its difference to M2 off that axis (brighter above 5 kHz). Extrapolating
   the whole F2-M2 axis far enough to reach the target pitch (about -0.9) would overshoot the
   spectral envelope; moving along -d_par lowers pitch while keeping M2's envelope.
5. Rate. Reaching the target rate along the token-level F2-M2 axis would need about 3 times
   the F2-M2 distance; at that point the encoder output cosine to M2 falls to 0.42 (0.86 with
   the trained F2 row), which risks intelligibility. So the edit takes the trained F2 token row,
   the slower of the two in-distribution rows (edit.py checks that the target is slower than
   it), and goes no further.
6. Pitch coefficient (closed form, two-point linear interpolation):
   k = (L_target - L_start) / (P(start - d_par) - P(start)), where L are mean voiced log F0 by
   pyworld on audio (target: 50 enrollment recordings; start: slot 2 after the rate edit, 20
   enrollment sentences), and P is the model's own mean predicted log F0 on voiced frames over
   the 50 enrollment sentences. Values: target 103.2 Hz, start 162.8 Hz, -0.2987 per unit of
   -d_par, k = 1.526.

The spectral envelope off the F2-M2 axis was not edited: a direct-path (transpose) design
through the acoustic stack predicted the F2-M2 cepstral change with cosine only 0.5-0.6, too
unreliable to justify a closed-form spectral edit.

## Expected effect

Voice 2 keeps M2's timbre (spectral envelope about -0.15 of the F2-M2 distance beyond M2, target
-0.24), with mean F0 near the target's (about 100 Hz measured on 10 enrollment sentences versus
103 Hz for the target), speaking about 8 % slower than M2, and stays intelligible. Voices 0 and
1 are unchanged. Known limitation: the pitch range within a sentence is compressed (predicted
log-F0 standard deviation 0.19 versus 0.25 for M2); every move away from a trained voice
row compresses it.

## Validation evaluations

2 of 3 used (`tts_evaluate_val`, seed 0, 20 validation sentences, logged in val_log.jsonl).

| # | candidate | speaker sim | MCD (dB) | F0 RMSE (cents) | duration ratio | WER | voices 0/1 |
|---|-----------|-------------|----------|-----------------|----------------|-----|-----------|
| 1 | pitch only: frame row M2 - 1.716 d_par, token row M2 | 0.6271 | 10.868 | 300.8 | 0.708 | 0.0 | identical |
| 2 | final: token row F2 + frame row M2 - 1.526 d_par | 0.6250 | 10.914 | 280.8 | 0.771 | 0.0 | identical |

Candidate 2 is the submitted edit (same speaker similarity within noise, lower F0 error, closer
speaking rate, WER 0).

## Files read

- models/tiny_voca3.py
- eval/wsp_tools.py
- eval/common.py
- tools/check_wsp_edit.py (first 150 lines; also run as instructed)
- .venv/lib/python3.13/site-packages/tiny_voca/model.py
- .venv/lib/python3.13/site-packages/tiny_voca/config.py
- .venv/lib/python3.13/site-packages/tiny_voca/infer.py (first 120 lines and a grep of the rest)
- .venv/lib/python3.13/site-packages/tiny_voca/text.py (grep for the phonemize API only)
- files in wsp/stage2/W1/run_3/ (edit.py, val_log.jsonl, scratch/)
- checkpoints/stage2_base.pt (through torch.load / wsp_tools) and my own output
  checkpoints/stage2_W1_run3.pt
