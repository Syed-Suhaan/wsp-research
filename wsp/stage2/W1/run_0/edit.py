"""WSP edit, stage 2, condition W1, run 0: give voice slot 2 the target speaker's voice.

Only row 2 of `frame_voices.weight` changes. Every other tensor, including row 2 of
`voices.weight` (the token-level voice, which keeps M2's durations and encoder states), stays as
in the base checkpoint, so voices 0 and 1 are unchanged.

New row 2 of `frame_voices.weight`:

    v2 = M2 + a * (F2 - M2) + c * u_hat

  * M2, F2: rows 1 and 0 of `frame_voices.weight` (the two trained voices).
  * a: position of the target speaker on the trained F2-M2 axis, measured on the spectral
    envelope: the projection of (target - M2) onto (F2 - M2) of the mean log spectral envelope
    (voiced frames, 100-4000 Hz, level removed). Negative a moves past M2, away from F2.
  * u_hat: unit vector of the direction in the prosody-stack residual stream that the log-F0
    output reads: u = gamma_p * w_f0 (final LayerNorm gain of the prosody stack, times the
    log-F0 row of `prosody_head.weight`), mean removed (LayerNorm removes the channel mean).
    The frame-level voice embedding is added to that residual stream, and a vector d added there
    shifts the predicted log F0 by about u . d / sigma_p (sigma_p = per-frame std of the
    residual before the final LayerNorm), through the residual (identity) path.
  * c: the length along u_hat that brings the predicted mean log F0 from M2's value to the
    target's: c = (delta - a * (u . (F2 - M2)) / sigma_p) * sigma_p / |u|, where
    delta = log(F0_target / F0_M2) is measured on audio with the same pitch tracker.

All statistics come from the first 25 of the 50 enrollment sentences (to keep the run time of
this script under five minutes): the target recordings and the base model's
own synthesis of voices 0 and 1. Run from the repository root:
    .venv/bin/python wsp/stage2/W1/run_0/edit.py
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
torch.manual_seed(0)

from eval import wsp_tools as T  # noqa: E402
from tiny_voca.model import rotary_table  # noqa: E402
from tiny_voca.text import phonemize  # noqa: E402

SR = 24000
BASE = ROOT / "checkpoints" / "stage2_base.pt"
OUT = ROOT / "checkpoints" / "stage2_W1_run0.pt"
F2, M2, NEW = 0, 1, 2
N_ENROLL = 25


def analyse(wave):
    """Pitch track and spectral envelope (WORLD: DIO + StoneMask + CheapTrick, 5 ms frames).
    Returns log F0 of voiced frames and log spectral envelope of voiced frames."""
    x = np.ascontiguousarray(np.asarray(wave, dtype=np.float64))
    f0, t = pw.dio(x, SR, f0_floor=60.0, f0_ceil=500.0, frame_period=5.0)
    f0 = pw.stonemask(x, f0, t, SR)
    sp = pw.cheaptrick(x, f0, t, SR)
    voiced = f0 > 0
    return np.log(f0[voiced]), np.log(sp[voiced] + 1e-12)


def pooled(waves):
    """Mean log F0 and mean log spectral envelope over all voiced frames of a list of waves."""
    parts = [analyse(w) for w in waves]
    lf0 = np.concatenate([p[0] for p in parts])
    env = np.concatenate([p[1] for p in parts])
    return float(lf0.mean()), env.mean(0)


def prosody_residual_std(model, text, voice):
    """Mean per-frame std of the prosody-stack residual just before its final LayerNorm
    (standard forward pass of the fixed model, read out before the last norm)."""
    p = phonemize(text)
    tokens, stress = torch.tensor([p.ids]), torch.tensor([p.stress])
    lengths, spk = torch.tensor([len(p)]), torch.tensor([voice])
    hidden, mask = model.encode(tokens, stress, lengths, spk)
    durations = model.predict_durations(hidden, mask, tokens)
    x, frame_mask, _ = model.expand(hidden, durations, mask, spk)
    table = rotary_table(x.shape[1], model.prosody_stack.width, x.device)
    for block in model.prosody_stack.blocks:
        x = block(x, frame_mask, table)
    return float(x[0].std(dim=-1).mean())


with torch.inference_mode():
    state = torch.load(BASE, map_location="cpu", weights_only=True)
    model = T.tts_load_base()
    texts = T.tts_texts("enroll")[:N_ENROLL]

    # ---- statistics of the target speaker and of the two trained voices (enrollment only)
    target_lf0, target_env = pooled([T.tts_target_audio("enroll", i) for i in range(len(texts))])
    m2_lf0, m2_env = pooled([T.tts_synthesize(model, s, M2, 0) for s in texts])
    _, f2_env = pooled([T.tts_synthesize(model, s, F2, 0) for s in texts])

    # ---- a: target position on the F2-M2 axis, from the spectral envelope (100-4000 Hz)
    freqs = np.arange(target_env.shape[0]) * SR / (2 * (target_env.shape[0] - 1))
    band = (freqs >= 100) & (freqs < 4000)
    level_free = [e[band] - e[band].mean() for e in (target_env, m2_env, f2_env)]
    to_target = level_free[0] - level_free[1]
    axis_env = level_free[2] - level_free[1]
    a = float(to_target @ axis_env / (axis_env @ axis_env))

    # ---- u: the log-F0 read-out direction of the prosody stack's residual stream
    gamma_p = state["prosody_stack.norm.weight"]
    w_f0 = state["prosody_head.weight"][0]
    u = gamma_p * w_f0
    u = u - u.mean()
    u_hat = u / u.norm()

    # ---- sigma_p: per-frame residual std before the final prosody LayerNorm, voice M2
    sigma_p = float(np.mean([prosody_residual_std(model, s, M2) for s in texts]))

    # ---- c: remaining log-F0 shift after the axis move, converted to a length along u_hat
    fv = state["frame_voices.weight"].clone()
    axis = fv[F2] - fv[M2]
    delta = target_lf0 - m2_lf0                         # wanted change of mean log F0
    axis_shift = a * float(u @ axis) / sigma_p           # log-F0 change the axis move gives
    c = (delta - axis_shift) * sigma_p / float(u.norm())

    # ---- write row 2 of frame_voices.weight (rows 0 and 1 untouched)
    fv[NEW] = fv[M2] + a * axis + c * u_hat
    state["frame_voices.weight"] = fv

    print(f"target F0 {np.exp(target_lf0):.1f} Hz, M2 F0 {np.exp(m2_lf0):.1f} Hz, delta {delta:.4f}")
    print(f"a {a:.4f}, sigma_p {sigma_p:.4f}, |u| {float(u.norm()):.5f}, axis shift {axis_shift:.4f}, c {c:.4f}")
    print(f"row 2 norm {float(fv[NEW].norm()):.3f} (M2 {float(fv[M2].norm()):.3f})")
    torch.save(state, OUT)
    print(f"saved {OUT}")
