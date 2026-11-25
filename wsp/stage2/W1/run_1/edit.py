"""WSP edit, stage 2, condition W1, run 1: give voice slot 2 of TinyVoca3 the target speaker.

Only row 2 of the two voice tables is written, so voices 0 (F2) and 1 (M2) are bit-identical
to the base checkpoint and every shared tensor is untouched.

Design (see rationale.md):
  voices.weight[2]        = voices.weight[0]
      The token-level voice table controls speaking rate (durations) and carries almost no
      spectral-envelope information. The target speaks about 21% slower than M2; F2 is about
      8% slower than M2. The coefficient of the F2 row, ln(D_T/D_1)/ln(D_0/D_1), is clipped to
      the trained range [0, 1], which gives the F2 row itself.
  frame_voices.weight[2]  = FV1 + a * (FV0 - FV1) + s * u
      a: position of the target's mean mel-cepstrum (c1..c24) along the M2 -> F2 cepstral
         axis (scalar projection), so the envelope moves along the trained voice axis.
      u: unit input direction that the log-F0 read-out of the prosody stack sees
         (centred prosody_stack.norm.weight * prosody_head.weight[0]).
      s: closed-form scale that lowers log F0 by the remaining gap between the target's
         median log F0 and the log F0 that the interpolation alone gives (linear in a),
         using the first-order effect of an input offset through the final LayerNorm:
         d(log F0) = s * |u_raw| * mean(1 / sigma).

All statistics are computed below from the enrollment split only (first N_ENROLL sentences).
"""

import sys
from pathlib import Path

import numpy as np
import pyworld as pw
import scipy.fft
import torch
import librosa

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

from eval import wsp_tools as W  # noqa: E402
from models.tiny_voca3 import TinyVoca3, use_checkpoint_frontend  # noqa: E402
from tiny_voca.model import rotary_table  # noqa: E402
from tiny_voca.text import phonemize  # noqa: E402

use_checkpoint_frontend()
SR = 24000
N_ENROLL = 20          # enrollment sentences used for the statistics
N_CEP = 25             # mel-cepstrum c0..c24; c0 (level) is excluded from the envelope axis
MEL = librosa.filters.mel(sr=SR, n_fft=1024, n_mels=80, fmin=0, fmax=8000)

sd = torch.load(ROOT / "checkpoints" / "stage2_base.pt", map_location="cpu", weights_only=True)
V = sd["voices.weight"].clone()
FV = sd["frame_voices.weight"].clone()
texts = W.tts_texts("enroll")[:N_ENROLL]
base = W.tts_load_base()


# ------------------------------------------------------------------ audio statistics
def f0_and_cepstrum(x):
    """Median log F0 (voiced frames, pyworld harvest) and mean mel-cepstrum c0..c24 of voiced frames."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    f0, t = pw.harvest(x, SR, f0_floor=50, f0_ceil=600, frame_period=10)
    env = pw.cheaptrick(x, f0, t, SR, fft_size=1024)
    voiced = f0 > 0
    logmel = np.log(env[voiced] @ MEL.T + 1e-10)
    cep = scipy.fft.dct(logmel, type=2, norm="ortho", axis=1)[:, :N_CEP]
    return np.median(np.log(f0[voiced])), cep.mean(0)


def active_seconds(x):
    """Length of the span between the first and last sample whose 10 ms RMS exceeds 3% of the peak RMS."""
    x = np.asarray(x, dtype=np.float64)
    rms = np.sqrt(np.convolve(x ** 2, np.ones(240) / 240, "same"))
    idx = np.flatnonzero(rms > 0.03 * rms.max())
    return (idx[-1] - idx[0]) / SR


def source_stats(get_audio):
    lf0, cep, dur = [], [], []
    for i in range(N_ENROLL):
        x = get_audio(i)
        a, c = f0_and_cepstrum(x)
        lf0.append(a)
        cep.append(c)
        dur.append(active_seconds(x))
    return np.mean(lf0), np.mean(cep, 0), np.sum(dur)


lt, ct, dt = source_stats(lambda i: W.tts_target_audio("enroll", i))
l0, c0, d0 = source_stats(lambda i: np.asarray(W.tts_synthesize(base, texts[i], 0, 0)))
l1, c1, d1 = source_stats(lambda i: np.asarray(W.tts_synthesize(base, texts[i], 1, 0)))

# ------------------------------------------------------------------ change 1: voices.weight[2]
# Speaking-rate coefficient along M2 -> F2 in log-duration, clipped to the trained range [0, 1].
a_tok = float(np.clip(np.log(dt / d1) / np.log(d0 / d1), 0.0, 1.0))
sd["voices.weight"][2] = V[1] + a_tok * (V[0] - V[1])

# ------------------------------------------------------------------ change 2a: envelope axis
# Scalar projection of (target - M2) onto (F2 - M2) in mel-cepstrum c1..c24.
axis = (c0 - c1)[1:]
a_env = float(((ct - c1)[1:] @ axis) / (axis @ axis))
row = FV[1] + a_env * (FV[0] - FV[1])

# ------------------------------------------------------------------ change 2b: pitch term
# Log F0 is taken as linear along the same axis (measured at a=0 and a=1); the remaining gap
# to the target's median log F0 is closed with the pitch read-out direction.
lf0_after_axis = l1 + a_env * (l0 - l1)
dlf0 = float(lt - lf0_after_axis)

gamma = sd["prosody_stack.norm.weight"]
w_f0 = sd["prosody_head.weight"][0]
u_raw = gamma * w_f0
u_raw = u_raw - u_raw.mean()        # LayerNorm removes the mean, so only the centred part acts
u = u_raw / u_raw.norm()

# mean 1/sigma of the residual stream entering prosody_stack.norm, for voice 2 with the row above
probe = {k: v.clone() for k, v in sd.items()}
probe["frame_voices.weight"][2] = row
net = TinyVoca3()
net.load_state_dict(probe, strict=True)
net.eval()
inv_sigma = []
with torch.no_grad():
    for text in texts[:10]:
        ph = phonemize(text)
        tok, st = torch.tensor([ph.ids]), torch.tensor([ph.stress])
        n, spk = torch.tensor([len(ph)]), torch.tensor([2])
        hidden, mask = net.encode(tok, st, n, spk)
        dur = net.predict_durations(hidden, mask, tok)
        x, fmask, _ = net.expand(hidden, dur, mask, spk)
        table = rotary_table(x.shape[1], net.prosody_stack.width, x.device)
        for block in net.prosody_stack.blocks:
            x = block(x, fmask, table)
        inv_sigma.append(float((1.0 / x[0].std(-1, unbiased=False)).mean()))
inv_sigma = float(np.mean(inv_sigma))

s = dlf0 / (float(u_raw.norm()) * inv_sigma)
sd["frame_voices.weight"][2] = row + s * u

print(f"target: logF0 {lt:.4f} ({np.exp(lt):.1f} Hz), active {dt:.2f} s")
print(f"F2: logF0 {l0:.4f}, active {d0:.2f} s | M2: logF0 {l1:.4f}, active {d1:.2f} s")
print(f"a_tok {a_tok:.4f}  a_env {a_env:.4f}  dlf0 {dlf0:.4f}  inv_sigma {inv_sigma:.4f}  s {s:.4f}")

out = ROOT / "checkpoints" / "stage2_W1_run1.pt"
torch.save(sd, out)
print("saved", out)
