"""WSP edit, stage 2 condition W1, run 3: give voice slot 2 the target speaker's pitch.

Run from the repository root:
    .venv/bin/python wsp/stage2/W1/run_3/edit.py

Only row 2 of `frame_voices.weight` changes, so voices 0 (F2) and 1 (M2) are unaffected.

Background (from analysis of the base model, see rationale.md):
  * The token-level table `voices.weight` mainly sets speaking rate; the frame-level table
    `frame_voices.weight` sets pitch and the spectral envelope.
  * The prosody stack is close to linear in the frame voice vector, and the prosody head reads
    log F0 from LayerNorm(residual stream). The direct-path pitch readout direction in the
    residual stream is r = centre(g * w), with g the prosody stack's final LayerNorm gain and
    w the log-F0 row of prosody_head.weight.
  * The F2 - M2 frame-voice difference d splits into d_par (its component along r) and the
    rest. Moving M2 along d_par changes pitch strongly and the spectral envelope little.
  * The target speaker's spectral envelope lies close to M2's on the F2-M2 axis, but its pitch
    is much lower than M2's. So the new voice is M2 moved along -d_par, by the amount that
    lowers the mean log F0 to the target's.

Coefficient (closed form, linear interpolation through two measured points):
    k = (L_target - L_M2) / (P(M2 - d_par) - P(M2))
  L_target: mean log F0 over voiced frames of the target's 50 enrollment recordings (pyworld,
            leading and trailing silence trimmed).
  L_M2:     the same measurement on the base model's M2 audio for enrollment sentences.
  P(v):     mean predicted log F0 (prosody head, voiced frames) of the 50 enrollment sentences
            with frame-voice vector v; the difference is the pitch change produced by one unit
            of -d_par.
  New row:  frame_voices[2] = frame_voices[1] - k * d_par
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np
import pyworld
import torch

torch.set_num_threads(4)
torch.manual_seed(0)

from eval.wsp_tools import tts_load_base, tts_synthesize, tts_target_audio, tts_texts
from tiny_voca.text import phonemize

SR = 24000
BASE = ROOT / "checkpoints" / "stage2_base.pt"
OUT = ROOT / "checkpoints" / "stage2_W1_run3.pt"
N_MODEL_AUDIO = 25          # enrollment sentences synthesized with M2 for the pitch measurement


def trim(w, floor_db=-40.0):
    """Cut leading and trailing 10 ms frames whose energy is more than 40 dB below the loudest
    frame, so that background noise before and after speech gives no spurious F0 values."""
    hop = SR // 100
    n = len(w) // hop
    energy_db = 10 * np.log10(np.mean(w[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    keep = np.nonzero(energy_db > energy_db.max() + floor_db)[0]
    return w[keep[0] * hop:(keep[-1] + 1) * hop]


def voiced_log_f0(wave):
    """Log F0 (Hz) of the voiced frames of a trimmed waveform: pyworld DIO + StoneMask, 5 ms frames."""
    w = np.ascontiguousarray(trim(np.asarray(wave, dtype=np.float64)))
    f0, t = pyworld.dio(w, SR, f0_floor=50.0, f0_ceil=600.0, frame_period=5.0)
    f0 = pyworld.stonemask(w, f0, t, SR)
    return np.log(f0[f0 > 0])


@torch.no_grad()
def predicted_log_f0(net, text, voice):
    """Log F0 (Hz) predicted by the prosody head on voiced frames, as in TinyVoca.synthesize
    (vowel frames count as voiced), without running the waveform synthesizer."""
    p = phonemize(text)
    tokens, stress = torch.tensor([p.ids]), torch.tensor([p.stress])
    lengths, speakers = torch.tensor([len(p)]), torch.tensor([voice])
    hidden, mask = net.encode(tokens, stress, lengths, speakers)
    durations = net.predict_durations(hidden, mask, tokens)
    frames, frame_mask, _ = net.expand(hidden, durations, mask, speakers)
    _, prosody = net.predict_prosody(frames, frame_mask)
    owner = torch.repeat_interleave(torch.arange(tokens.shape[1]), durations[0])
    voiced = (prosody[0, :, 1] > 0) | net.vowel_tokens[tokens[0, owner]]
    return float(np.log(net.config.pitch_reference_hz)) + prosody[0, voiced, 0].numpy()


def mean_predicted_log_f0(net, texts, voice):
    return float(np.mean(np.concatenate([predicted_log_f0(net, t, voice) for t in texts])))


sd = torch.load(BASE, map_location="cpu", weights_only=True)
fv = sd["frame_voices.weight"]
net = tts_load_base()
texts = tts_texts("enroll")

# Pitch readout direction r of the prosody head (direct path through the final LayerNorm):
# LayerNorm output y = g * centre(x) / sigma + b, log F0 = w . y + c  =>  d(log F0) = centre(g * w) . dx / sigma.
g = sd["prosody_stack.norm.weight"]
w = sd["prosody_head.weight"][0]
r = g * w
r = r - r.mean()
r_hat = r / r.norm()

# Component of the F2 - M2 frame-voice difference along r.
d = fv[0] - fv[1]
d_par = (d @ r_hat) * r_hat

# Target and M2 pitch, measured the same way (pyworld on audio).
L_target = float(np.mean(np.concatenate([voiced_log_f0(tts_target_audio("enroll", i)) for i in range(len(texts))])))
L_m2 = float(np.mean(np.concatenate([voiced_log_f0(tts_synthesize(net, texts[i], 1, 0)) for i in range(N_MODEL_AUDIO)])))

# Pitch change produced by one unit of -d_par (model's own prediction, slot 2 used as a probe).
with torch.no_grad():
    net.frame_voices.weight[2] = fv[1]
P_m2 = mean_predicted_log_f0(net, texts, 2)
with torch.no_grad():
    net.frame_voices.weight[2] = fv[1] - d_par
P_probe = mean_predicted_log_f0(net, texts, 2)

k = (L_target - L_m2) / (P_probe - P_m2)
print(f"target F0 {np.exp(L_target):.1f} Hz, M2 F0 {np.exp(L_m2):.1f} Hz, "
      f"pitch change per unit -d_par {P_probe - P_m2:+.4f}, k = {k:.4f}")

# The edit: row 2 of frame_voices = M2 row moved k units along -d_par. Row 2 of voices stays M2.
sd["frame_voices.weight"][2] = fv[1] - k * d_par

torch.save(sd, OUT)
print("saved", OUT)
