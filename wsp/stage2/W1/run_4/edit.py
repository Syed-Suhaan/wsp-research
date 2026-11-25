"""WSP edit, Stage 2, condition W1, run 4: give voice slot 2 the target speaker's voice.

Only row 2 of the frame-level voice table `frame_voices.weight` changes. Every shared tensor
keeps its base value, so voices 0 (F2) and 1 (M2) are bit-identical to the base checkpoint.
The token-level voice row `voices.weight[2]` stays the base copy of M2 (it mainly sets the
speaking rate; moving it along the F2-M2 axis lowered speaker similarity, see rationale.md).

Two closed-form steps (see rationale.md):
  1. frame_voices.weight[2] := M2 + beta * (M2 - F2). The frame-level row sets pitch and
     spectrum. The target is a male voice lower than M2, so M2 is moved further along the
     learned F2 -> M2 axis, by the fraction beta of the F2-M2 distance at which the axis would
     reach the target's median F0 if log F0 changed linearly along it:
        beta = (ln f0(M2) - ln f0(target)) / (ln f0(F2) - ln f0(M2)).
  2. frame_voices.weight[2] += a * u: a pitch-steering vector for the log-F0 gap left by
     step 1 (the response along the axis is sublinear). u is the unit, zero-mean direction of
     the F0 readout seen through the prosody stack's final LayerNorm (gamma * w_f0). Adding
     a*u to the residual stream changes the predicted log F0 by about
     a * (w_f0 . (gamma * u)) / sigma, where sigma is the residual's per-frame standard
     deviation before that LayerNorm. a is set so this equals the remaining log-F0 gap.

All statistics come from the first N_SENT sentences of the enrollment split, read through
eval/wsp_tools.py. Deterministic: fixed synthesis seed 0 and deterministic analysis.
"""

import sys
from pathlib import Path

import numpy as np
import pyworld as pw
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)

from eval import wsp_tools as T  # noqa: E402
from models.tiny_voca3 import TinyVoca3  # noqa: E402
from tiny_voca.model import rotary_table  # noqa: E402
from tiny_voca.text import phonemize  # noqa: E402

BASE = ROOT / "checkpoints" / "stage2_base.pt"
OUT = ROOT / "checkpoints" / "stage2_W1_run4.pt"
SR = 24000
N_SENT = 12        # enrollment sentences used for the F0 statistics
N_SIGMA = 6        # enrollment sentences used for the residual standard deviation
F2, M2, NEW = 0, 1, 2

state_dict = torch.load(BASE, map_location="cpu", weights_only=True)
texts = T.tts_texts("enroll")
T.tts_load_base()  # selects the phonemizer front end of the checkpoint


def build(sd):
    model = TinyVoca3()
    model.load_state_dict(sd, strict=True)
    return model.eval()


def synth_all(model, voice):
    return [np.asarray(T.tts_synthesize(model, texts[i], voice, 0), dtype=np.float64) for i in range(N_SENT)]


def median_log_f0(waves):
    """Median log F0 (natural log of Hz) over the pooled voiced frames (WORLD Harvest)."""
    values = []
    for wave in waves:
        f0, _ = pw.harvest(np.ascontiguousarray(wave), SR, f0_floor=60.0, f0_ceil=500.0, frame_period=10.0)
        values.append(np.log(f0[f0 > 0]))
    return float(np.median(np.concatenate(values)))


@torch.inference_mode()
def prosody_residual_std(model, voice):
    """Mean per-frame standard deviation of the prosody stack's residual stream just before
    its final LayerNorm (the quantity that LayerNorm divides by)."""
    stds = []
    stack = model.prosody_stack
    for i in range(N_SIGMA):
        p = phonemize(texts[i])
        tokens, stress = torch.tensor([p.ids]), torch.tensor([p.stress])
        lengths, speaker = torch.tensor([len(p)]), torch.tensor([voice])
        hidden, mask = model.encode(tokens, stress, lengths, speaker)
        durations = model.predict_durations(hidden, mask, tokens)
        x, frame_mask, _ = model.expand(hidden, durations, mask, speaker)
        table = rotary_table(x.shape[1], stack.width, x.device)
        for block in stack.blocks:
            x = block(x, frame_mask, table)
        stds.append(x[0].std(-1, unbiased=False).mean().item())
    return float(np.mean(stds))


frame_voices = state_dict["frame_voices.weight"]  # [3, 192], frame-level voice

# ---- Statistics: median log F0 of the target recordings and of the two trained voices.
lf0_target = median_log_f0([np.asarray(T.tts_target_audio("enroll", i), dtype=np.float64) for i in range(N_SENT)])
base_model = build(state_dict)
lf0_f2 = median_log_f0(synth_all(base_model, F2))
lf0_m2 = median_log_f0(synth_all(base_model, M2))

# ---- Step 1: frame-level voice row 2 := M2 + beta * (M2 - F2)   (pitch and timbre axis).
beta = (lf0_m2 - lf0_target) / (lf0_f2 - lf0_m2)
frame_voices[NEW] = frame_voices[M2] + beta * (frame_voices[M2] - frame_voices[F2])

# ---- Step 2: pitch-steering vector for the remaining log-F0 gap.
model_1 = build(state_dict)
lf0_step1 = median_log_f0(synth_all(model_1, NEW))
sigma = prosody_residual_std(model_1, NEW)
gamma = state_dict["prosody_stack.norm.weight"]   # final LayerNorm gain of the prosody stack
w_f0 = state_dict["prosody_head.weight"][0]       # readout row of log(F0 / 200 Hz)
u = gamma * w_f0
u = u - u.mean()                                  # LayerNorm removes the mean; keep only what passes
u = u / u.norm()
gain = float(w_f0 @ (gamma * u)) / sigma          # d(log F0) per unit of a, linearised
a = (lf0_target - lf0_step1) / gain
frame_voices[NEW] = frame_voices[NEW] + a * u

torch.save(state_dict, OUT)
print(f"target f0 {np.exp(lf0_target):.1f} Hz | F2 {np.exp(lf0_f2):.1f} | M2 {np.exp(lf0_m2):.1f} | "
      f"after step 1 {np.exp(lf0_step1):.1f}")
print(f"beta {beta:.4f} | sigma {sigma:.4f} | gain {gain:.5f} | a {a:.4f}")
print("saved", OUT)
