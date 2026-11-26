# Stage 2, W1, run 5: rationale

## Tensors changed

- `voices.weight`, row 2 only (token-level voice row): set to row 0 (the F2 token row).
- `frame_voices.weight`, row 2 only (frame-level voice row): set to
  `F1 + a * (F1 - F0)`, where F0 and F1 are rows 0 and 1 (F2, M2) of the base table and
  `a = 1.1318` is computed in `edit.py` (see below).

No shared tensor changed, so voices 0 and 1 are bit-identical to the base model
(at most 384 values changed, all in the two voice-2 rows).

## Why

Analysis of the base model (scratch/a3.py, a5.py, a6.py):

1. The two voice tables have separate roles. Swapping only the token-level row between F2 and
   M2 changes the predicted durations (total length 37.9 s -> 42.0 s for 12 sentences) and
   leaves pitch and spectral envelope unchanged. Swapping only the frame-level row changes
   pitch (159 -> 252 Hz) and spectral envelope and leaves the durations unchanged.
2. The target speaker is a low male voice: mean F0 107.5 Hz (pyworld harvest, all 50
   enrollment recordings) against about 162 Hz for M2. The target speaks slower than both
   voices (about 1.43x the M2 duration on the same sentences).
3. The long-term spectral envelope of the target differs from M2's (about 3.3 dB rms over
   40 mel bands). This offset correlates (r = 0.70-0.74) with the change that a step beyond
   M2 along the frame-level F2 -> M2 axis produces, and (r = 0.42) with the M2 - F2
   difference itself (scratch/a13.py). The target is therefore "beyond M2" on the axis the
   network learned between its two speakers, in both pitch and timbre.

Choices:

- Token row := F2 token row. It is a learned, in-distribution row; it only lengthens the
  durations toward the target's slower rate (F2's durations are the longer of the two).
- Frame row := M2 row moved further along the F2 -> M2 axis. The step `a` is a closed-form
  value: the predicted mean log F0 (voiced frames, 16 enrollment sentences) is measured at
  the two learned rows (a = -1 and a = 0); linear interpolation gives `a_lin` for the wanted
  pitch; the response is measured once at `a_lin` and `a = a_lin * wanted / realized`
  (one ratio correction, no loop). The wanted pitch is the target's tracked mean log F0 plus
  the offset between the model's predicted pitch and the tracked pitch of its own audio
  (+0.035 in log units, measured on 8 sentences), because the evaluation tracks F0 in audio.
  Result: a_lin = 0.901, a = 1.132, predicted mean F0 113.4 Hz (wanted 110.4 Hz in model
  units, i.e. 107.5 Hz tracked).

Alternatives examined and rejected:

- A pitch-readout direction (prosody head pitch weights times the final LayerNorm gain,
  orthogonal to the voicing and energy readouts) matches the pitch with a smaller vector,
  but moves the envelope less toward the target (projection 0.22 against 0.29 for the axis,
  scratch/a14.py, a15.py). Most of the envelope change is mediated by the pitch, so pushing
  further along the axis and compensating the pitch would add little timbre at a large
  out-of-distribution risk.
- A direct spectral-envelope correction through the frame row (desired cepstral change ->
  vocoder heads -> two LayerNorms) needed a vector of norm 33 (row norm is about 10),
  realized only cos 0.10 of the wanted change and broke the pitch contour (scratch/a11.py).
  Not used.
- A gated feed-forward unit that adds an envelope correction only for voice 2 would need an
  unused unit; no feed-forward unit in the prosody or acoustic stacks is inactive for voices
  0 and 1 (scratch/a12.py), so it would change voices 0 and 1. Not used.

## Expected effect

- Voice 2 keeps M2's articulation and intelligibility, with the pitch lowered to the
  target's level (about 108 Hz tracked) and a long-term spectrum about 30% of the way from
  M2 toward the target's (r = 0.64 with the needed change, enrollment analysis).
- Slower speech than M2 (closer to the target's rate, still faster than the target).
- The pitch contour is flatter than M2's (log-F0 sd 0.19 against 0.29); this limits the
  F0 RMSE gain.
- Voices 0 and 1 are unchanged (no shared tensor edited).

## Validation evaluations

3 allowed; 2 used (`tts_evaluate_val`, seed 0, logged in val_log.jsonl).

1. This edit (checkpoints/stage2_W1_run5.pt): speaker similarity 0.655, MCD 10.79 dB,
   F0 RMSE 253 cents, WER 0.0, duration ratio 0.771; voices 0 and 1 identical.
2. The unedited base checkpoint as reference (voice 2 = M2): speaker similarity 0.594,
   MCD 11.01 dB, F0 RMSE 719 cents, WER 0.0, duration ratio 0.708.

The edit improves every voice-2 metric over the M2 start and keeps WER at 0.
The final edit is the one evaluated in (1); it was not changed afterwards.

## Files read

- models/tiny_voca3.py
- eval/wsp_tools.py
- eval/common.py
- tools/check_wsp_edit.py
- .venv/lib/python3.13/site-packages/tiny_voca/model.py
- .venv/lib/python3.13/site-packages/tiny_voca/config.py
- .venv/lib/python3.13/site-packages/tiny_voca/infer.py
- .venv/lib/python3.13/site-packages/tiny_voca/text.py (the `phonemize` function and the
  front-end constants only)
- checkpoints/stage2_base.pt (through torch.load and eval/wsp_tools.py)
- files in my own run folder (edit.py, scratch/*)

The project instruction file loaded automatically by the agent harness was present in the
context; its content was not used.
