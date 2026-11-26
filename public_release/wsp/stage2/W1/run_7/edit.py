"""WSP edit, stage 2, condition W1, run 7: give voice slot 2 the target speaker's voice.

Only row 2 of the two voice tables is written; every shared tensor and rows 0 and 1 stay
as in the base checkpoint, so voices 0 (F2) and 1 (M2) are unchanged by construction.

Model facts used (tiny_voca/model.py):
  * `voices.weight[v]` is added to every token embedding before the text encoder. The encoder
    output feeds the duration predictor, so this row sets the speaking rate.
  * `frame_voices.weight[v]` is added to every frame before the prosody and acoustic stacks,
    so this row sets pitch (prosody head) and spectral envelope (vocoder heads).
  A swap analysis (scratch/a3.py, 10 enrollment sentences) confirmed the split: with the
  encoder row of F2 and the frame row of M2, mean log F0 moved by 0.01 and speech time by
  +8 %; with the encoder row of M2 and the frame row of F2, mean log F0 moved by +0.44.
  (scratch/ holds analysis scripts only; edit.py does not import them.)

Both rows are written as points on the line through the two trained voices,
    row(alpha) = row_M2 + alpha * (row_F2 - row_M2),
with alpha given by closed-form linear interpolation of a measured statistic between the
two trained voices (alpha = 0 is M2, alpha = 1 is F2).

Run from the repository root:  .venv/bin/python wsp/stage2/W1/run_7/edit.py
"""

import sys
from pathlib import Path

import numpy as np
import pyworld as pw
import torch

torch.set_num_threads(4)
torch.manual_seed(0)
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from eval import wsp_tools as W  # noqa: E402

SR = 24000
N_SENT = 10          # enrollment sentences 0..9 are used for the statistics below


def voiced_log_f0(wave):
    """Log F0 (Hz) of the voiced frames of a waveform (pyworld harvest, 5 ms frames)."""
    x = np.ascontiguousarray(np.asarray(wave), dtype=np.float64)
    f0, _ = pw.harvest(x, SR, f0_floor=50, f0_ceil=600, frame_period=5)
    return np.log(f0[f0 > 0]), (f0 > 0).sum() * 0.005


def stats(waves):
    """Mean voiced log F0 over all frames, and total voiced time in seconds."""
    parts = [voiced_log_f0(w) for w in waves]
    return float(np.concatenate([p[0] for p in parts]).mean()), float(sum(p[1] for p in parts))


sd = torch.load(ROOT / "checkpoints" / "stage2_base.pt", map_location="cpu", weights_only=True)
model = W.tts_load_base()
texts = W.tts_texts("enroll")[:N_SENT]

# Statistics of the target recordings and of the two trained voices on the same sentences.
mu_t, dur_t = stats([W.tts_target_audio("enroll", i) for i in range(N_SENT)])
mu_0, dur_0 = stats([W.tts_synthesize(model, t, 0, 0) for t in texts])
mu_1, dur_1 = stats([W.tts_synthesize(model, t, 1, 0) for t in texts])
print(f"mean log F0: target {mu_t:.3f}  F2 {mu_0:.3f}  M2 {mu_1:.3f}")
print(f"voiced seconds: target {dur_t:.1f}  F2 {dur_0:.1f}  M2 {dur_1:.1f}")

# --- frame_voices.weight[2]: pitch and spectral envelope ---------------------------------
# The target speaker is lower than M2 (about 105 Hz against about 160 Hz), so the frame row is
# placed on the F2-M2 line beyond M2, at the alpha where mean log F0, interpolated linearly
# between the two trained voices, equals the target's:
#     alpha_f = (mu_t - mu_1) / (mu_0 - mu_1)          (negative: away from F2)
# The F2-M2 line is the only direction in this table that training has tied to a change of
# speaker, so moving along it changes speaker identity (envelope, voice quality) together with
# pitch. The response of pitch is sublinear beyond M2 (the larger row norm dominates the
# LayerNorms), so this point leaves the pitch about 0.10 (log) above the target. Two
# validation checks (rationale.md) showed that correcting that remainder with a pure
# pitch-readout vector lowers F0 error but also lowers speaker similarity, the main
# criterion, so no correction is applied.
alpha_f = (mu_t - mu_1) / (mu_0 - mu_1)
fv = sd["frame_voices.weight"]
fv[2] = fv[1] + alpha_f * (fv[0] - fv[1])

# --- voices.weight[2]: speaking rate --------------------------------------------------------
# The target speaks more slowly than both trained voices. The encoder row is interpolated
# by voiced time in the same way, but alpha is clipped to [0, 1]: the encoder feeds every
# later stage, so the row is kept inside the segment between the two trained embeddings,
# where the encoder has been trained, to protect intelligibility. A ratio above 1 gives
# alpha_v = 1, i.e. the encoder row of F2 (the slower trained voice); the swap analysis showed
# that this row changes duration but leaves pitch and envelope nearly unchanged.
alpha_v = float(np.clip((dur_t - dur_1) / (dur_0 - dur_1), 0.0, 1.0))
v = sd["voices.weight"]
v[2] = v[1] + alpha_v * (v[0] - v[1])

print(f"alpha_f = {alpha_f:.4f}   alpha_v = {alpha_v:.4f}")
print("row 2 norms: voices %.3f  frame_voices %.3f" % (v[2].norm(), fv[2].norm()))
torch.save(sd, ROOT / "checkpoints" / "stage2_W1_run7.pt")
print("saved checkpoints/stage2_W1_run7.pt")
