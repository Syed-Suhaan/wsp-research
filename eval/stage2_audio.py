"""Stage 2 audio utilities: tiny-voca synthesis, the synthetic target style, and metrics.

Synthesis
    load_tts()                       the tiny-voca model (vendor/tiny-voca/tiny-voca.pt), eval mode
    synth(model, text, voice, seed, speed, pitch_shift, pitch_range) -> float64 waveform (24 kHz)
        `seed` is the model's noise seed. The built-in controls are passed through unchanged.

Target style (DSP on the F2 output, pyworld / WORLD vocoder)
    make_target(wave, cfg) -> waveform
        F0 scaled by 2^(semitones/12); spectral envelope and aperiodicity warped in
        frequency so that formants move by the factor (1 + formant_shift); frame sequence
        stretched in time by 1 / rate (so durations are exactly F2 durations / rate).

Metrics (computed on WORLD features: DIO + StoneMask F0, CheapTrick envelope, 5 ms frames)
    mcd_dtw(cand, ref)        mel-cepstral distortion in dB after DTW alignment. Cepstra are
                              WORLD-coded spectral envelopes (pyworld.code_spectral_envelope,
                              25 coefficients, c0 excluded). Frames below (max energy - 40 dB)
                              of each utterance are dropped before alignment.
    f0_rmse_cents(cand, ref)  RMSE of 1200 * log2(F0_cand / F0_ref) over DTW-aligned frame pairs
                              where both are voiced.
    duration_ratio(cand, ref) len(cand) / len(ref) in samples.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "tiny-voca"
os.environ.setdefault("DIZIONE_DATA", str(VENDOR / "phonemizer"))

SR = 24000
FRAME_MS = 5.0
N_CEP = 25
SILENCE_DB = 40.0
TARGET_STYLE = {"semitones": -4.0, "formant_shift": -0.08, "rate": 0.9}


# ------------------------------------------------------------------------------ synthesis
def load_tts(checkpoint: str | Path | None = None):
    import torch
    from tiny_voca.infer import load_model, match_frontend
    torch.set_num_threads(4)
    path = str(checkpoint or VENDOR / "tiny-voca.pt")
    # Select the phonemizer front end the checkpoint was trained with (dizione, "own"),
    # as the package's own command-line tool does. load_model then checks the fingerprint.
    match_frontend(path)
    return load_model(path)


def synth(model, text: str, voice: str = "F2", seed: int = 0, speed: float = 1.0,
          pitch_shift: float = 0.0, pitch_range: float = 1.0) -> np.ndarray:
    from tiny_voca.infer import speak
    wave, _, _ = speak(model, text, voice, speed=speed, pitch_shift=pitch_shift, seed=seed,
                       pitch_range=pitch_range)
    return wave.double().numpy()


def save_wav(path: str | Path, wave: np.ndarray, peak: float = 0.99) -> None:
    import soundfile as sf
    loud = float(np.abs(wave).max()) if len(wave) else 0.0
    out = wave * (peak / loud) if loud > peak else wave
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), out.astype(np.float32), SR, subtype="PCM_16")


def load_wav(path: str | Path) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float64")
    assert sr == SR, sr
    return x


# ------------------------------------------------------------------------------- target DSP
def _warp_frequency(spec: np.ndarray, factor: float) -> np.ndarray:
    """new(f) = old(f / factor): content at f moves to factor * f (factor < 1 lowers formants)."""
    bins = spec.shape[1]
    src = np.arange(bins) / factor
    out = np.empty_like(spec)
    for t in range(spec.shape[0]):
        out[t] = np.interp(src, np.arange(bins), spec[t], right=spec[t, -1])
    return out


def _stretch_time(feat: np.ndarray, factor: float) -> np.ndarray:
    """Resample a frame sequence to round(len * factor) frames by linear interpolation."""
    n = feat.shape[0]
    m = int(round(n * factor))
    pos = np.linspace(0, n - 1, m)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (pos - lo)[:, None] if feat.ndim == 2 else (pos - lo)
    return feat[lo] * (1 - w) + feat[hi] * w


def make_target(wave: np.ndarray, cfg: dict = TARGET_STYLE) -> np.ndarray:
    import pyworld as pw
    x = np.ascontiguousarray(wave, dtype=np.float64)
    f0, t = pw.harvest(x, SR, frame_period=FRAME_MS)
    sp = pw.cheaptrick(x, f0, t, SR)
    ap = pw.d4c(x, f0, t, SR)
    f0 = f0 * 2.0 ** (cfg["semitones"] / 12.0)
    factor = 1.0 + cfg["formant_shift"]
    sp = _warp_frequency(sp, factor)
    ap = np.clip(_warp_frequency(ap, factor), 0.0, 1.0)
    stretch = 1.0 / cfg["rate"]
    voiced = _stretch_time((f0 > 0).astype(float), stretch) > 0.5
    f0 = np.where(voiced, _stretch_time(f0, stretch), 0.0)
    sp = np.maximum(_stretch_time(sp, stretch), 1e-16)
    ap = _stretch_time(ap, stretch)
    return pw.synthesize(np.ascontiguousarray(f0), np.ascontiguousarray(sp), np.ascontiguousarray(ap),
                         SR, frame_period=FRAME_MS)


# ---------------------------------------------------------------------------------- metrics
def world_features(wave: np.ndarray) -> dict:
    import pyworld as pw
    x = np.ascontiguousarray(wave, dtype=np.float64)
    f0, t = pw.dio(x, SR, frame_period=FRAME_MS)
    f0 = pw.stonemask(x, f0, t, SR)
    sp = pw.cheaptrick(x, f0, t, SR)
    mcep = pw.code_spectral_envelope(sp, SR, N_CEP)
    energy_db = 10 * np.log10(sp.sum(axis=1) + 1e-20)
    keep = energy_db > energy_db.max() - SILENCE_DB
    return {"f0": f0[keep], "mcep": mcep[keep, 1:]}


def dtw_path(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Classic DTW (steps (1,0), (0,1), (1,1)) on Euclidean frame distance. Each row is
    computed in closed form: D[i, j] = S[j] + min_{k<=j}(A[k] - S[k-1]), where S is the
    cumulative cost of row i and A[k] = min(D[i-1, k], D[i-1, k-1])."""
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    n, m = cost.shape
    D = np.full((n, m), np.inf)
    D[0] = np.cumsum(cost[0])
    for i in range(1, n):
        prev = D[i - 1]
        A = np.minimum(prev, np.concatenate(([np.inf], prev[:-1])))
        S = np.cumsum(cost[i])
        S_prev = np.concatenate(([0.0], S[:-1]))
        D[i] = S + np.minimum.accumulate(A - S_prev)
    i, j = n - 1, m - 1
    pi, pj = [i], [j]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            step = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
            if step == 0:
                i, j = i - 1, j - 1
            elif step == 1:
                i -= 1
            else:
                j -= 1
        pi.append(i)
        pj.append(j)
    return np.array(pi[::-1]), np.array(pj[::-1])


def compare(cand: np.ndarray, ref_feat: dict, ref_len: int) -> dict:
    """All three metrics of a candidate waveform against precomputed reference features."""
    cf = world_features(cand)
    pi, pj = dtw_path(cf["mcep"], ref_feat["mcep"])
    diff = cf["mcep"][pi] - ref_feat["mcep"][pj]
    mcd = float(np.mean(10.0 / np.log(10.0) * np.sqrt(2.0 * (diff ** 2).sum(axis=1))))
    fa, fb = cf["f0"][pi], ref_feat["f0"][pj]
    both = (fa > 0) & (fb > 0)
    cents = 1200.0 * np.log2(fa[both] / fb[both]) if both.any() else np.array([np.nan])
    return {"mcd_db": mcd, "f0_rmse_cents": float(np.sqrt(np.mean(cents ** 2))),
            "duration_ratio": len(cand) / ref_len, "voiced_pairs": int(both.sum())}


@lru_cache(maxsize=None)
def sentences() -> dict:
    import json
    return json.loads((ROOT / "data" / "stage2_sentences.json").read_text())
