"""Annotate recordings for Stage 2 fine-tuning, following tiny-voca's own data pipeline.

Per utterance (text + 24 kHz waveform), as in tiny_voca.prepare.extract and tiny_voca.align:
    wave            zero-padded to a whole number of 256-sample frames
    mel             log power mel (n_fft 1024, hop 256, 80 mels; centred frames)
    prosody         per frame: log(F0 / 200 Hz) (0 where unvoiced), voiced flag, log RMS energy
    pitch_trusted   1 on voiced frames; pitch is linearly interpolated across unvoiced frames
                    so the curve stays continuous (these frames get low weight in the loss)
    durations       token durations in frames from the author's forced aligner
                    (tiny_voca.align.align_utterance with facebook/wav2vec2-lv-60-espeak-cv-ft)

Deviation from the author's recipe: the author tracks pitch with librosa.pyin; here pitch and
voicing come from pyworld Harvest (librosa is not installed). RMS energy reproduces
librosa.feature.rms (frame_length = n_fft, centred, constant padding).

Annotations are cached in data/cache/stage2_annot/<name>.pt (regenerable).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "stage2_annot"
ALIGNER_REVISION = "ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4"


def _config():
    from tiny_voca.config import Config
    from models.tiny_voca3 import published_config
    return Config.from_dict(published_config())


def features(wave: np.ndarray, c) -> dict:
    import pyworld as pw
    frames = math.ceil(len(wave) / c.hop)
    wave = np.pad(np.asarray(wave, dtype=np.float64), (0, frames * c.hop - len(wave)))
    # Log power mel, centred frames with constant padding (as librosa.stft(center=True)).
    from tiny_voca.losses import mel_filterbank
    w = torch.from_numpy(wave).float()
    spec = torch.stft(w, c.n_fft, c.hop, window=torch.hann_window(c.n_fft), center=True, pad_mode="constant",
                      return_complex=True).abs()[:, :frames]
    mel = torch.log((mel_filterbank(c.sample_rate, c.n_fft, c.n_mels) @ spec.square()).clamp_min(1e-5)).T
    # Pitch and voicing at frame centres i * hop.
    f0, _ = pw.harvest(wave, c.sample_rate, f0_floor=c.f0_min, f0_ceil=c.f0_max, frame_period=1000.0 * c.hop / c.sample_rate)
    f0 = np.pad(f0, (0, max(0, frames - len(f0))))[:frames]
    voiced = f0 > 0
    logp = np.where(voiced, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)
    if voiced.any():
        idx = np.arange(frames)
        interp = np.interp(idx, idx[voiced], logp[voiced])
    else:
        interp = np.zeros(frames)
    # RMS energy over n_fft windows centred on each frame.
    padded = np.pad(wave, (c.n_fft // 2, c.n_fft // 2))
    rms = np.array([np.sqrt(np.mean(padded[i * c.hop: i * c.hop + c.n_fft] ** 2)) for i in range(frames)])
    prosody = np.stack((interp, voiced.astype(np.float64), np.log(np.maximum(rms, 1e-5))), axis=-1)
    return {"wave": torch.from_numpy(wave).float(), "mel": mel.float(),
            "prosody": torch.from_numpy(prosody).float(), "pitch_trusted": torch.from_numpy(voiced.astype(np.float32))}


def annotate(name: str, texts: list[str], waves: list[np.ndarray], log=print) -> list[dict]:
    """Annotate (text, wave) pairs; cached under `name`."""
    path = CACHE / f"{name}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    from models.tiny_voca3 import use_checkpoint_frontend
    use_checkpoint_frontend()
    from tiny_voca.align import AlignmentSettings, PhoneRecognizer, align_utterance
    from tiny_voca.text import phonemize
    c = _config()
    rec = PhoneRecognizer(device="cpu", local_files_only=False)
    assert rec.revision == ALIGNER_REVISION, rec.revision
    items = []
    for i, (text, wave) in enumerate(zip(texts, waves)):
        f = features(wave, c)
        p = phonemize(text)
        durations, qc = align_utterance(rec, p, f["wave"].numpy(), f["prosody"][:, 1].numpy(), c, AlignmentSettings())
        items.append({"text": text, "tokens": torch.tensor(p.ids), "stress": torch.tensor(p.stress),
                      "durations": torch.tensor(durations), **f, "qc": qc})
        if log and i % 10 == 0:
            log(f"annotated {name} {i + 1}/{len(texts)}")
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(items, path)
    return items


def collate(items: list[dict], speaker: int) -> dict:
    """Pad a list of annotated utterances into the batch layout of tiny_voca's TrainingSystem."""
    B = len(items)
    n_tok = max(len(it["tokens"]) for it in items)
    n_fr = max(len(it["prosody"]) for it in items)
    hop = len(items[0]["wave"]) // len(items[0]["prosody"])
    batch = {"tokens": torch.zeros(B, n_tok, dtype=torch.long), "stress": torch.zeros(B, n_tok, dtype=torch.long),
             "durations": torch.zeros(B, n_tok, dtype=torch.long), "prosody": torch.zeros(B, n_fr, 3),
             "pitch_trusted": torch.zeros(B, n_fr), "mel": torch.zeros(B, n_fr, items[0]["mel"].shape[1]),
             "wave": torch.zeros(B, n_fr * hop)}
    for b, it in enumerate(items):
        t, f = len(it["tokens"]), len(it["prosody"])
        batch["tokens"][b, :t], batch["stress"][b, :t], batch["durations"][b, :t] = it["tokens"], it["stress"], it["durations"]
        batch["prosody"][b, :f], batch["pitch_trusted"][b, :f], batch["mel"][b, :f] = it["prosody"], it["pitch_trusted"], it["mel"]
        batch["wave"][b, : f * hop] = it["wave"]
    batch["token_lengths"] = torch.tensor([len(it["tokens"]) for it in items])
    batch["frame_lengths"] = torch.tensor([len(it["prosody"]) for it in items])
    batch["speakers"] = torch.full((B,), speaker, dtype=torch.long)
    return batch
