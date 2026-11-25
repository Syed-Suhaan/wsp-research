"""WSP edit, stage 2 condition W1, run 3: turn voice slot 2 (a copy of M2) into the target speaker.

Run from the repository root:
    .venv/bin/python wsp/stage2/W1/run_3/edit.py

Only row 2 of `voices.weight` and row 2 of `frame_voices.weight` change, so voices 0 (F2) and
1 (M2) are unaffected. Every other tensor is copied unchanged.

Findings from analysis of the base model that this edit relies on (see rationale.md):
  * The token-level table `voices.weight` sets the speaking rate (durations) and has almost no
    effect on pitch or spectral envelope; the frame-level table `frame_voices.weight` sets pitch
    and spectral envelope and has no effect on durations.
  * The prosody stack is close to linear in the frame voice vector. The prosody head reads log F0
    from LayerNorm(residual stream); the direct-path pitch readout direction in the residual
    stream is r = centre(g * w), with g the stack's final LayerNorm gain and w the log-F0 row
    of prosody_head.weight.
  * The F2 - M2 frame-voice difference d splits into d_par (its component along r) and the rest.
    Moving along d_par changes pitch strongly and the spectral envelope little.
  * The target speaker is slower than both trained voices, its spectral envelope lies close to
    M2's on the F2-M2 axis, and its pitch is much lower than M2's.

Edit 1, speaking rate. voices[2] = voices[0] (F2's token-level row), chosen because the target
    speaks more slowly than F2-rate speech, and F2's row is the slower of the two trained rows.
    A trained row is in-distribution, so intelligibility is kept; extrapolating further was
    found to disturb the encoder states strongly and is not done.
Edit 2, pitch. frame_voices[2] = frame_voices[1] - k * d_par, with the closed-form coefficient
        k = (L_target - L_start) / (P(start - d_par) - P(start))
    L_target: mean log F0 over voiced frames of the target's 50 enrollment recordings.
    L_start:  the same measurement on slot-2 audio after edit 1 (frame row still M2's), for
              enrollment sentences 0-19. Both by pyworld DIO + StoneMask on audio trimmed of
              leading and trailing silence.
    P(v):     mean log F0 predicted by the prosody head on voiced frames of the 50 enrollment
              sentences when slot 2's frame row is v; P(start - d_par) - P(start) is the pitch
              change produced by one unit of -d_par. k is the linear interpolation between these
              two points that reaches the target's mean log F0.
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
N_AUDIO = 20          # enrollment sentences synthesized for the audio measurements
NEW = 2               # voice slot being edited


def trim(w, floor_db=-40.0):
    """Cut leading and trailing 10 ms frames whose energy is more than 40 dB below the loudest
    frame, so that silence and background noise before and after speech are not measured."""
    hop = SR // 100
    n = len(w) // hop
    energy_db = 10 * np.log10(np.mean(w[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    keep = np.nonzero(energy_db > energy_db.max() + floor_db)[0]
    return w[keep[0] * hop:(keep[-1] + 1) * hop]


def voiced_log_f0(w):
    """Log F0 (Hz) of the voiced frames of a trimmed waveform: pyworld DIO + StoneMask, 5 ms frames."""
    w = np.ascontiguousarray(w, dtype=np.float64)
    f0, t = pyworld.dio(w, SR, f0_floor=50.0, f0_ceil=600.0, frame_period=5.0)
    f0 = pyworld.stonemask(w, f0, t, SR)
    return np.log(f0[f0 > 0])


def audio_stats(waves):
    """(mean voiced log F0, total speech duration in seconds) of a list of waveforms."""
    trimmed = [trim(np.asarray(w, dtype=np.float64)) for w in waves]
    return (float(np.mean(np.concatenate([voiced_log_f0(w) for w in trimmed]))),
            sum(len(w) for w in trimmed) / SR)


@torch.no_grad()
def prosody_of(net, text, voice):
    """Durations and prosody prediction for one sentence, as in TinyVoca.synthesize, without the
    acoustic stack and waveform synthesizer. Returns (durations [N], log F0 Hz on voiced frames),
    where vowel frames count as voiced like in synthesize."""
    p = phonemize(text)
    tokens, stress = torch.tensor([p.ids]), torch.tensor([p.stress])
    lengths, speakers = torch.tensor([len(p)]), torch.tensor([voice])
    hidden, mask = net.encode(tokens, stress, lengths, speakers)
    durations = net.predict_durations(hidden, mask, tokens)
    frames, frame_mask, _ = net.expand(hidden, durations, mask, speakers)
    _, prosody = net.predict_prosody(frames, frame_mask)
    owner = torch.repeat_interleave(torch.arange(tokens.shape[1]), durations[0])
    voiced = (prosody[0, :, 1] > 0) | net.vowel_tokens[tokens[0, owner]]
    log_f0 = float(np.log(net.config.pitch_reference_hz)) + prosody[0, voiced, 0].numpy()
    return durations[0], log_f0


def mean_predicted_log_f0(net, texts, voice):
    return float(np.mean(np.concatenate([prosody_of(net, t, voice)[1] for t in texts])))


def total_frames(net, texts, voice):
    return int(sum(int(prosody_of(net, t, voice)[0].sum()) for t in texts))


sd = torch.load(BASE, map_location="cpu", weights_only=True)
tok = sd["voices.weight"].clone()        # token-level voice rows: 0 = F2, 1 = M2, 2 = copy of M2
fv = sd["frame_voices.weight"].clone()   # frame-level voice rows, same order
net = tts_load_base()
texts = tts_texts("enroll")

# ---------------------------------------------------------------- edit 1: token-level row (rate)
# Speaking rate of the two trained token rows, from predicted frame counts (frame row kept at M2).
with torch.no_grad():
    net.voices.weight[NEW] = tok[1]
frames_m2_row = total_frames(net, texts, NEW)
with torch.no_grad():
    net.voices.weight[NEW] = tok[0]
frames_f2_row = total_frames(net, texts, NEW)
slower_row = 0 if frames_f2_row > frames_m2_row else 1

# Speech duration and pitch of slot 2 with the slower row (still M2's frame row), and the
# target's, on the same enrollment sentences.
with torch.no_grad():
    net.voices.weight[NEW] = tok[slower_row]
L_start, dur_start = audio_stats([tts_synthesize(net, texts[i], NEW, 0) for i in range(N_AUDIO)])
target_waves = [tts_target_audio("enroll", i) for i in range(len(texts))]
_, dur_target = audio_stats(target_waves[:N_AUDIO])
L_target, _ = audio_stats(target_waves)
# The target is slower than the slower trained row: take that row (the closest in-distribution
# rate). If it were not, the M2 row would be kept.
new_tok_row = tok[slower_row] if dur_target > dur_start else tok[1]
print(f"frames with M2 row {frames_m2_row}, with F2 row {frames_f2_row}; speech duration "
      f"slot2 {dur_start:.1f} s vs target {dur_target:.1f} s -> token row "
      f"{slower_row if dur_target > dur_start else 1}")
sd["voices.weight"][NEW] = new_tok_row

# ---------------------------------------------------------------- edit 2: frame-level row (pitch)
# Pitch readout direction r of the prosody head (direct path through the final LayerNorm):
# y = g * centre(x) / sigma + b and log F0 = w . y + c  give  d(log F0) = centre(g * w) . dx / sigma.
g = sd["prosody_stack.norm.weight"]
w = sd["prosody_head.weight"][0]
r = g * w
r = r - r.mean()
r_hat = r / r.norm()

# Component of the F2 - M2 frame-voice difference along r.
d = fv[0] - fv[1]
d_par = (d @ r_hat) * r_hat

# Pitch change produced by one unit of -d_par, from the model's own pitch prediction.
with torch.no_grad():
    net.voices.weight[NEW] = new_tok_row
    net.frame_voices.weight[NEW] = fv[1]
P_start = mean_predicted_log_f0(net, texts, NEW)
with torch.no_grad():
    net.frame_voices.weight[NEW] = fv[1] - d_par
P_probe = mean_predicted_log_f0(net, texts, NEW)

k = (L_target - L_start) / (P_probe - P_start)
print(f"target F0 {np.exp(L_target):.1f} Hz, slot-2 start F0 {np.exp(L_start):.1f} Hz, "
      f"pitch change per unit -d_par {P_probe - P_start:+.4f}, k = {k:.4f}")

# Row 2 of frame_voices: M2's row moved k units along -d_par.
sd["frame_voices.weight"][NEW] = fv[1] - k * d_par

torch.save(sd, OUT)
print("saved", OUT)
