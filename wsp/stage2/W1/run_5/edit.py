"""WSP edit, Stage 2, condition W1, run 5: give voice slot 2 of TinyVoca3 a target speaker.

Only row 2 of the two voice tables changes; every shared tensor keeps its base value, so
voices 0 (F2) and 1 (M2) are unchanged.

  voices.weight[2]        (token-level voice row) := voices.weight[0]
      Analysis (scratch/a3.py) showed that the token-level row controls the predicted
      phoneme durations and has no measurable effect on pitch or spectral envelope. The
      target speaker talks slower than both voices (about 1.43x the M2 duration); the F2
      token row gives the longer of the two learned duration patterns (about 1.11x M2).

  frame_voices.weight[2]  (frame-level voice row) := F1 + a * (F1 - F0)
      F0, F1 are the frame-level rows of F2 and M2. The frame row controls pitch and spectral
      envelope. Moving from F2 to M2 lowers pitch and changes the long-term spectrum; the
      target's spectral offset from M2 correlates (r = 0.7) with the offset that a step beyond
      M2 along this axis produces (scratch/a13.py). The target is a lower male voice
      (about 107 Hz against 162 Hz for M2), so the row is moved beyond M2 along the F2->M2
      axis. The step a is set by a closed-form formula so that the model's mean predicted
      log F0 for voice 2 equals the target's mean log F0 on the enrollment sentences.

The script measures statistics with forward passes (predicted pitch) and with pyworld
(F0 of the target recordings and of the model's audio). It contains no gradients and no
fitting: a is computed once from the two learned rows (linear interpolation), then corrected
once by the measured response ratio at that step.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pyworld as pw
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(4)
torch.manual_seed(0)

from eval import wsp_tools as W  # noqa: E402
from tiny_voca.text import phonemize  # noqa: E402

SR = 24000
OUT = ROOT / "checkpoints" / "stage2_W1_run5.pt"

sd = torch.load(ROOT / "checkpoints" / "stage2_base.pt", map_location="cpu", weights_only=True)
V = sd["voices.weight"].clone()        # [3, 192] token-level voice rows
F = sd["frame_voices.weight"].clone()  # [3, 192] frame-level voice rows

model = W.tts_load_base()
texts = W.tts_texts("enroll")
REF = 200.0  # config.pitch_reference_hz: the prosody head predicts log(F0 / 200)


def harvest_log_f0(wave):
    """Log F0 (Hz) of the voiced frames of a waveform, pyworld harvest, 5 ms frames."""
    x = np.ascontiguousarray(np.asarray(wave, dtype=np.float64))
    f0, _ = pw.harvest(x, SR, f0_floor=50, f0_ceil=600, frame_period=5)
    return np.log(f0[f0 > 0])


@torch.inference_mode()
def voiced_pitch_frames(m, text, voice=2):
    """Predicted log(F0/200) on the voiced frames of the model's standard synthesis path
    (same steps as TinyVoca.synthesize up to the prosody head, including the rule that keeps
    vowel frames voiced)."""
    ph = phonemize(text)
    tokens, stress = torch.tensor([ph.ids]), torch.tensor([ph.stress])
    lengths, spk = torch.tensor([len(ph)]), torch.tensor([voice])
    hidden, mask = m.encode(tokens, stress, lengths, spk)
    durations = m.predict_durations(hidden, mask, tokens)
    frames, frame_mask, _ = m.expand(hidden, durations, mask, spk)
    _, prosody = m.predict_prosody(frames, frame_mask)
    ends = durations.cumsum(1)
    frame = torch.arange(frames.shape[1])[None].contiguous()
    owner = torch.searchsorted(ends, frame, right=True).clamp(max=tokens.shape[1] - 1)
    vowel = m.vowel_tokens[tokens].gather(1, owner) & frame_mask
    voiced = (prosody[..., 1] > 0) | vowel
    return prosody[0, :, 0][voiced[0]]


def mean_pitch(m, sentences, voice=2):
    return float(torch.cat([voiced_pitch_frames(m, t, voice) for t in sentences]).mean())


def set_voice2(m, token_row, frame_row):
    m.voices.weight.data[2] = token_row
    m.frame_voices.weight.data[2] = frame_row


# ---------------------------------------------------------------- 1. target pitch statistic
# Mean log F0 over all voiced frames of the 50 target enrollment recordings.
target_lf0 = float(np.concatenate([harvest_log_f0(W.tts_target_audio("enroll", i))
                                   for i in range(len(texts))]).mean())

# Offset between the F0 tracker and the model's own pitch prediction, measured on the base
# model's audio (voice 2 = M2) for 8 enrollment sentences: predicted mean minus tracked mean.
# The evaluation tracks F0 in audio, so the predicted pitch must equal the tracked target
# pitch plus this offset.
TRACK = texts[:8]
tracked = float(np.concatenate([harvest_log_f0(W.tts_synthesize(model, t, 2, 0)) for t in TRACK]).mean())
offset = mean_pitch(model, TRACK) - (tracked - math.log(REF))   # both in log(F0/200)
T = target_lf0 - math.log(REF) + offset   # wanted mean predicted log(F0/200)

# ---------------------------------------------------------------- 2. token-level row: F2's row
token_row = V[0].clone()

# ---------------------------------------------------------------- 3. frame-level row
# Pitch at the two learned rows (token row already set to V[0]), on 16 enrollment sentences.
CAL = texts[:16]
axis = F[1] - F[0]                      # F2 -> M2 direction of the frame-level voice table
set_voice2(model, token_row, F[0])
p_f2 = mean_pitch(model, CAL)
set_voice2(model, token_row, F[1])
p_m2 = mean_pitch(model, CAL)
# Linear interpolation between the two learned rows: pitch(a) = p_m2 + a * (p_m2 - p_f2).
a_lin = (T - p_m2) / (p_m2 - p_f2)
set_voice2(model, token_row, F[1] + a_lin * axis)
p_lin = mean_pitch(model, CAL)
# One correction by the measured response ratio at a_lin (the response beyond M2 is weaker
# than between the learned rows).
a = a_lin * (T - p_m2) / (p_lin - p_m2)
frame_row = F[1] + a * axis
set_voice2(model, token_row, frame_row)
p_final = mean_pitch(model, CAL)

# ---------------------------------------------------------------- 4. write and save
sd["voices.weight"][2] = token_row
sd["frame_voices.weight"][2] = frame_row
torch.save(sd, OUT)

print(f"target mean F0 (tracked) {math.exp(target_lf0):.1f} Hz; tracker offset {offset:+.4f}")
print(f"wanted predicted log(F0/200) {T:+.4f}; F2 row {p_f2:+.4f}; M2 row {p_m2:+.4f}")
print(f"a_lin {a_lin:.4f} -> {p_lin:+.4f}; a {a:.4f} -> {p_final:+.4f} "
      f"({REF * math.exp(p_final):.1f} Hz predicted)")
print(f"saved {OUT}")
