# Stage 2, condition W1, run 7: rationale

## Tensors changed

- `voices.weight`, row 2 only (192 values): set to row 0 (the F2 encoder row), from
  `row1 + alpha_v * (row0 - row1)` with `alpha_v = 1.0` (clipped).
- `frame_voices.weight`, row 2 only (192 values): `row1 + alpha_f * (row0 - row1)` with
  `alpha_f = (mu_t - mu_1) / (mu_0 - mu_1) = -0.8452`.

No other tensor changed: 384 of 4,537,675 values (0.0085 %). Rows 0 and 1 and every shared
tensor are bit-identical to the base, so voices 0 and 1 are unchanged by construction.

## Why

1. The voice enters only through the two tables. A swap analysis on 10 enrollment sentences
   (scratch/a3.py) separated their roles: the encoder row (`voices`) mainly sets durations
   (F2 encoder row with M2 frame row: +8 % speech time, mean log F0 +0.01), while the frame
   row (`frame_voices`) sets pitch and envelope (M2 encoder row with F2 frame row: mean log F0
   +0.44, envelope close to F2).
2. Target statistics (pyworld harvest, enrollment sentences 0-9): mean voiced log F0 4.683
   (about 108 Hz) against M2 5.082 and F2 5.553. The target is a lower male voice, so the
   frame row is extrapolated beyond M2 on the F2-M2 line, the only direction in this table
   tied by training to a change of speaker. `alpha_f` is the closed-form point where mean log
   F0, linearly interpolated between the two trained voices, equals the target's.
3. The target also speaks more slowly than both voices (voiced time 32.6 s against M2 28.8 s
   and F2 31.1 s). The duration ratio `(dur_t - dur_1)/(dur_0 - dur_1) = 1.65` is clipped to
   [0, 1], so the encoder row is F2's: the slowest trained setting, kept inside the trained
   segment because the encoder feeds every later stage (intelligibility).
4. Activation analysis (scratch/a5.py) showed that the pitch response along the line is
   sublinear beyond M2 and flattens the contour: the row norm grows (10.3 to 19.2) and dominates
   the LayerNorms. The edit still leaves pitch about 0.10 (log) above the target on enrollment
   sentences (harvest 4.785 against 4.683).
5. Two alternative candidates tested a pure pitch correction along the pitch readout direction
   after the prosody stack's final LayerNorm (`u = g*w - mean(g*w)`, with g the LayerNorm gain
   and w the pitch row of `prosody_head`; step `beta = delta * sigma / |u|` from the direct
   residual path, which predicted the activation-level pitch change within 0.02).
   Both lowered F0 error but also lowered speaker similarity, the main criterion, so the final
   edit is candidate 1, which has no correction. (Candidate scripts:
   scratch/edit_candidate{1,2,3}.py.txt.)

## Expected effect

Voice 2 is a lower, darker male voice than M2 (mean F0 about 119 Hz against M2's about
160 Hz; target about 108 Hz), with M2's phonetics and somewhat longer durations (F2 encoder
row). Intelligibility is kept (WER 0 on validation). Voices 0 and 1 are identical to the base.
Known limitations: pitch remains about 10 % above the target, the intonation contour is a
little flatter than M2's, and leading/trailing silences are shorter than in the recordings
(the evaluator's duration ratio is 0.77).

## Validation evaluations

3 used (all via `tts_evaluate_val`, seed 0, logged in val_log.jsonl):

| # | candidate | speaker sim | MCD dB | F0 RMSE cents | WER | dur. ratio | voices 0/1 identical |
|---|-----------|------------:|-------:|--------------:|----:|-----------:|------|
| 1 | F2-M2 line, alpha_f = -0.845; encoder row F2 (**final**) | **0.678** | **10.83** | 304 | 0.0 | 0.771 | yes |
| 2 | M2 frame row + pitch-readout vector (beta = -6.80); encoder row F2 | 0.629 | 10.91 | 263 | 0.0 | 0.771 | yes |
| 3 | candidate 1 + pitch-readout vector for the remaining shortfall (beta = -1.80) | 0.662 | 10.84 | 253 | 0.0 | 0.771 | yes |

Evaluation 1 checked the linear-interpolation reasoning; evaluations 2 and 3 tested the
hypothesis that a pure pitch vector would improve the voice. It improves F0 error but costs
speaker similarity, so the final edit.py reproduces candidate 1 (same alpha values; output
re-generated and checked with tools/check_wsp_edit.py: PASS).

Own analysis on enrollment sentences (not validation): harvest / activation-level pitch of
candidate 1 (scratch/a4.py) and activation-level pitch of candidate 3 (scratch/a8.py).

## Files read

- models/tiny_voca3.py
- eval/wsp_tools.py
- eval/common.py
- tools/check_wsp_edit.py (docstring and pattern lists, first 80 lines)
- .venv/lib/python3.13/site-packages/tiny_voca/model.py
- .venv/lib/python3.13/site-packages/tiny_voca/config.py (first 80 lines)
- .venv/lib/python3.13/site-packages/tiny_voca/infer.py (lines 1-127)
- .venv/lib/python3.13/site-packages/tiny_voca/text.py (function list, lines 92-101)
- checkpoints/stage2_base.pt (through torch.load and eval/wsp_tools.py)
- checkpoints/stage2_W1_run7.pt (own output)
- files in wsp/stage2/W1/run_7/ (own edit.py, scratch scripts, val_log.jsonl)

Data was accessed only through eval/wsp_tools.py (enrollment and validation texts, target
enrollment recordings, synthesis, tts_evaluate_val). The harness loaded a project instruction
file into the context automatically; its content was not used.
