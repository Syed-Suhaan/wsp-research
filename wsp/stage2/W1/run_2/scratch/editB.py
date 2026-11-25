"""WSP edit, stage 2, condition W1, run 2: write voice slot 2 of TinyVoca3 for a target speaker.

Only row 2 of `voices.weight` and row 2 of `frame_voices.weight` are written. Every other
tensor (and rows 0 and 1) keeps its base value, so voices 0 (F2) and 1 (M2) are unchanged.

Run from the repository root:
    .venv/bin/python wsp/stage2/W1/run_2/edit.py

The script measures a few statistics (pitch of the target recordings, pitch and duration of
the base voices) with forward passes and signal analysis, then writes the two rows with the
closed-form formulas commented below. No gradients, fitting, or search are used.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pyworld
import torch

ROOT = Path("/home/user/wsp-research")
sys.path.insert(0, str(ROOT))
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)

from eval import wsp_tools as W  # noqa: E402  (the writer's only data access)
from models.tiny_voca3 import TinyVoca3, use_checkpoint_frontend  # noqa: E402
from tiny_voca.model import rotary_table  # noqa: E402
from tiny_voca.text import phonemize  # noqa: E402

BASE = ROOT / "checkpoints" / "stage2_base.pt"
OUT = ROOT / "wsp/stage2/W1/run_2/scratch/candB.pt"
F2, M2, NEW = 0, 1, 2
SR = 24000
N_SENT = 20                      # enrollment sentences 0..19 are used for every statistic


# ------------------------------------------------------------------ measurement helpers
def harvest_log_f0(wave) -> np.ndarray:
    """Natural-log F0 (Hz) of the voiced 10 ms frames of a waveform (WORLD harvest)."""
    x = np.ascontiguousarray(np.asarray(wave, dtype=np.float64))
    f0, _ = pyworld.harvest(x, SR, f0_floor=50.0, f0_ceil=600.0, frame_period=10.0)
    return np.log(f0[f0 > 0])


@torch.inference_mode()
def internal_prosody(model, text: str, voice: int):
    """Predicted log(F0/200 Hz) on voiced frames and the frame count, exactly as in the
    model's standard synthesis (vowel frames are forced voiced there too)."""
    p = phonemize(text)
    tokens, stress = torch.tensor([p.ids]), torch.tensor([p.stress])
    lengths, spk = torch.tensor([len(p)]), torch.tensor([voice])
    hidden, mask = model.encode(tokens, stress, lengths, spk)
    durations = model.predict_durations(hidden, mask, tokens)
    frames, frame_mask, frame_lengths = model.expand(hidden, durations, mask, spk)
    _, prosody = model.predict_prosody(frames, frame_mask)
    ends = durations.cumsum(1)
    frame = torch.arange(frames.shape[1])[None].contiguous()
    owner = torch.searchsorted(ends, frame, right=True).clamp(max=tokens.shape[1] - 1)
    vowel = model.vowel_tokens[tokens].gather(1, owner)[0]
    voiced = (prosody[0, :, 1] > 0) | vowel
    return prosody[0, voiced, 0], int(frame_lengths[0])


def mean_internal_log_f0(model, texts, voice: int) -> float:
    return float(torch.cat([internal_prosody(model, t, voice)[0] for t in texts]).mean())


def total_frames(model, texts, voice: int) -> int:
    return sum(internal_prosody(model, t, voice)[1] for t in texts)


def model_from(state_dict):
    use_checkpoint_frontend()
    m = TinyVoca3()
    m.load_state_dict(state_dict, strict=True)
    return m.eval()


# ------------------------------------------------------------------ load the frozen base
sd = torch.load(BASE, map_location="cpu", weights_only=True)
base_model = model_from(sd)
texts = W.tts_texts("enroll")[:N_SENT]

tok_f2, tok_m2 = sd["voices.weight"][F2].clone(), sd["voices.weight"][M2].clone()
frm_m2 = sd["frame_voices.weight"][M2].clone()

# ------------------------------------------------------------------ target statistics
# Mean log F0 of the target recordings and of the base M2 voice, both measured with the same
# pitch tracker on the same 20 enrollment sentences. Their difference is the pitch change
# the new voice needs relative to M2 (about -0.40 natural-log units, i.e. about -7 semitones).
target_lf0 = float(np.concatenate([harvest_log_f0(W.tts_target_audio("enroll", i))
                                   for i in range(N_SENT)]).mean())
m2_waves = [np.asarray(W.tts_synthesize(base_model, t, M2, 0)) for t in texts]
m2_lf0 = float(np.concatenate([harvest_log_f0(w) for w in m2_waves]).mean())
delta_lf0 = target_lf0 - m2_lf0
print(f"target mean log F0 {target_lf0:.4f}  M2 {m2_lf0:.4f}  needed change {delta_lf0:+.4f}")

# ------------------------------------------------------------------ edit 1: token voice row
# voices.weight[2] is added to every phoneme embedding before the text encoder. In the base
# model this row mainly sets the phoneme durations (speaking rate); pitch is set by the frame
# row (edit 2). The target speaks more slowly than M2. Its speech-span ratio to M2 is r_t;
# F2's frame-count ratio to M2 is r_f (F2 is the slower trained voice). Moving the row along
# the F2 - M2 line by a = (r_t - 1) / (r_f - 1) would match the rate, but a > 1 would
# extrapolate outside the trained voices and risk intelligibility, so a is limited to [0, 1].
def speech_span(wave) -> float:
    """Seconds from the first to the last 10 ms frame within 35 dB of the loudest frame."""
    x = np.asarray(wave, dtype=np.float64)
    n = len(x) // 240
    e = 10 * np.log10((x[: n * 240].reshape(n, 240) ** 2).mean(1) + 1e-20)
    idx = np.nonzero(e > e.max() - 35.0)[0]
    return (idx[-1] - idx[0] + 1) * 0.010


r_t = sum(speech_span(W.tts_target_audio("enroll", i)) for i in range(N_SENT)) / \
    sum(speech_span(w) for w in m2_waves)
r_f = total_frames(base_model, texts, F2) / total_frames(base_model, texts, M2)
a_tok = 0.0
print(f"speech-span ratio target/M2 {r_t:.3f}  frame ratio F2/M2 {r_f:.3f}  a_tok {a_tok:.3f}")
sd["voices.weight"][NEW] = tok_m2 + a_tok * (tok_f2 - tok_m2)

# ------------------------------------------------------------------ edit 2: frame voice row
# frame_voices.weight[2] is added to every acoustic frame before the prosody transformer.
# Through the residual stream it reaches the prosody transformer's final LayerNorm, whose
# output is read by prosody_head row 0 (log F0). On this direct path, adding a vector d to
# the frame row changes the predicted log F0 by about  sum_i w_i * g_i * d_i / sigma,
# where w = prosody_head.weight[0], g = prosody_stack.norm.weight, and sigma is the
# per-frame standard deviation of the LayerNorm input. The pitch write is therefore along
#     u = centre(g * w) / |centre(g * w)|        (centre: subtract the channel mean, which
#                                                  the LayerNorm removes anyway)
# with length c0 = delta * sigma / |centre(g * w)|. The frame row starts as M2's row.
g = sd["prosody_stack.norm.weight"]
w = sd["prosody_head.weight"][0]
w_eff = g * w - (g * w).mean()
u = w_eff / w_eff.norm()

# sigma: mean per-frame std of the LayerNorm input, for M2, measured by running the prosody
# transformer's own blocks up to (not including) its final LayerNorm.
@torch.inference_mode()
def prosody_norm_input_std(model, text: str, voice: int) -> torch.Tensor:
    p = phonemize(text)
    tokens, stress = torch.tensor([p.ids]), torch.tensor([p.stress])
    lengths, spk = torch.tensor([len(p)]), torch.tensor([voice])
    hidden, mask = model.encode(tokens, stress, lengths, spk)
    durations = model.predict_durations(hidden, mask, tokens)
    x, frame_mask, _ = model.expand(hidden, durations, mask, spk)
    stack = model.prosody_stack
    table = rotary_table(x.shape[1], stack.width, x.device)
    for block in stack.blocks:
        x = block(x, frame_mask, table)
    return x[0].std(-1, unbiased=False)


sigma = float(torch.cat([prosody_norm_input_std(base_model, t, M2) for t in texts]).mean())

# The model with only edit 1 applied (frame row = M2) is the reference for the pitch change:
# the token row moves pitch slightly, so the needed internal change is measured from there.
sd["frame_voices.weight"][NEW] = frm_m2
ref_model = model_from(sd)
lf0_m2 = mean_internal_log_f0(base_model, texts, M2)
lf0_ref = mean_internal_log_f0(ref_model, texts, NEW)
delta_needed = (lf0_m2 + delta_lf0) - lf0_ref
c0 = delta_needed * sigma / float(w_eff.norm())

# The two prosody blocks also process the written vector, so the realized change differs
# from the direct-path estimate. One measurement at c0 gives the realized sensitivity along
# u (log F0 per unit length), and the final length is delta_needed / sensitivity.
sd["frame_voices.weight"][NEW] = frm_m2 + c0 * u
lf0_c0 = mean_internal_log_f0(model_from(sd), texts, NEW)
sensitivity = (lf0_c0 - lf0_ref) / c0
c = delta_needed / sensitivity
print(f"sigma {sigma:.3f} |w_eff| {float(w_eff.norm()):.4f} needed {delta_needed:+.4f} "
      f"c0 {c0:.3f} realized {lf0_c0 - lf0_ref:+.4f} c {c:.3f}")
sd["frame_voices.weight"][NEW] = frm_m2 + c * u

# ------------------------------------------------------------------ save
torch.save(sd, OUT)
print("saved", OUT)
