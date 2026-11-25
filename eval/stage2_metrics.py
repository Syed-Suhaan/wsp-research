"""Stage 2 metrics for a TinyVoca3 state dict (or model) on one split and one noise seed.

Per sentence of the split:
    speaker_sim     cosine similarity between the WavLM x-vector of the output
                    (microsoft/wavlm-base-plus-sv, 16 kHz) and the target-voice centroid: the
                    normalised mean x-vector of the 50 ENROLLMENT target utterances. Primary
                    metric; higher is better.
    mcd_db          mel-cepstral distortion with DTW against the target audio of the same
                    sentence (eval.stage2_audio.compare). Lower is better.
    f0_rmse_cents   F0 RMSE in cents over DTW-aligned voiced frame pairs. Lower is better.
    duration_ratio  output length / target length.
    wer             word error rate of Whisper small.en (greedy) against the sentence text,
                    after lowercasing and removing punctuation. Guardrail.

`preserved_voices` checks that F2 and M2 do not change: for voices 0 and 1 it compares the edited
model's output with the base model's output on the same sentences and seed.

Usage (library):
    ev = Evaluator()
    rows = ev.evaluate(model, split="test", seed=3, voice=2, controls={})
"""

from __future__ import annotations

import json
import re
import statistics
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SR = 24000
SV_MODEL = ("microsoft/wavlm-base-plus-sv", "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e")
ASR_MODEL = ("openai/whisper-small.en", "e8727524f962ee844a7319d92be39ac1bd25655a")


def stage2_config() -> dict:
    return json.loads((ROOT / "data" / "stage2_config.json").read_text())


def sentences() -> dict:
    return json.loads((ROOT / "data" / "stage2_sentences.json").read_text())


def target_path(split: str, i: int) -> Path:
    cfg = stage2_config()
    return ROOT / cfg["target_dir"] / split / f"{split}_{i:02d}_{cfg['target_file_tag']}.wav"


def to16k(wave: np.ndarray) -> np.ndarray:
    from scipy.signal import resample_poly
    return resample_poly(np.asarray(wave, dtype=np.float64), 2, 3)


def normalise_text(s: str) -> list[str]:
    return re.sub(r"[^a-z' ]+", " ", s.lower().replace("-", " ")).split()


def wer(ref: str, hyp: str) -> float:
    r, h = normalise_text(ref), normalise_text(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(len(r), 1)


def synthesize(model, text: str, voice: int, seed: int, controls: dict | None = None) -> np.ndarray:
    from tiny_voca.infer import phonemize
    c = {"speed": 1.0, "pitch_shift": 0.0, "pitch_range": 1.0, **(controls or {})}
    p = phonemize(text)
    with torch.inference_mode():
        audio, lengths, _ = model.synthesize(torch.tensor([p.ids]), torch.tensor([p.stress]),
                                             torch.tensor([len(p)]), torch.tensor([voice]),
                                             speed=c["speed"], pitch_shift=c["pitch_shift"],
                                             seed=seed, pitch_range=c["pitch_range"])
    return audio[0, : int(lengths[0])].double().numpy()


class Evaluator:
    def __init__(self, asr: bool = True):
        from transformers import AutoFeatureExtractor, WavLMForXVector
        torch.set_num_threads(4)
        self.fe = AutoFeatureExtractor.from_pretrained(SV_MODEL[0], revision=SV_MODEL[1])
        self.sv = WavLMForXVector.from_pretrained(SV_MODEL[0], revision=SV_MODEL[1]).eval()
        self.asr = None
        if asr:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            self.proc = WhisperProcessor.from_pretrained(ASR_MODEL[0], revision=ASR_MODEL[1])
            self.asr = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL[0], revision=ASR_MODEL[1]).eval()
        self.centroid = self._centroid()
        self._ref_feats: dict = {}

    def embed(self, wave24: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            e = self.sv(**self.fe(to16k(wave24), sampling_rate=16000, return_tensors="pt")).embeddings[0]
        return torch.nn.functional.normalize(e, dim=-1)

    def _centroid(self) -> torch.Tensor:
        from eval.stage2_audio import load_wav
        cache = ROOT / "data" / "cache" / "stage2_target_centroid.pt"
        tag = stage2_config()["target_dir"]
        if cache.exists():
            c = torch.load(cache, weights_only=True)
            if c.get("tag") == tag:
                return c["centroid"]
        embs = torch.stack([self.embed(load_wav(target_path("enroll", i))) for i in range(len(sentences()["enroll"]))])
        cen = torch.nn.functional.normalize(embs.mean(0), dim=-1)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"tag": tag, "centroid": cen}, cache)
        return cen

    def transcribe(self, wave24: np.ndarray) -> str:
        with torch.no_grad():
            feats = self.proc(to16k(wave24), sampling_rate=16000, return_tensors="pt").input_features
            ids = self.asr.generate(feats)
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def ref(self, split: str, i: int):
        from eval.stage2_audio import load_wav, world_features
        key = (split, i)
        if key not in self._ref_feats:
            w = load_wav(target_path(split, i))
            self._ref_feats[key] = (world_features(w), len(w))
        return self._ref_feats[key]

    def score_wave(self, wave: np.ndarray, split: str, i: int, text: str, with_asr: bool = True) -> dict:
        from eval.stage2_audio import compare
        m = compare(wave, *self.ref(split, i))
        row = {"sentence": i, "speaker_sim": float(self.embed(wave) @ self.centroid), **m}
        if with_asr and self.asr is not None:
            hyp = self.transcribe(wave)
            row.update({"asr": hyp, "wer": wer(text, hyp)})
        return row

    def evaluate(self, model, split: str, seed: int, voice: int = 2, controls: dict | None = None,
                 with_asr: bool = True, count: int | None = None) -> list[dict]:
        texts = sentences()[split][:count] if count else sentences()[split]
        return [self.score_wave(synthesize(model, t, voice, seed, controls), split, i, t, with_asr)
                for i, t in enumerate(texts)]


def summarize(rows: list[dict]) -> dict:
    keys = ["speaker_sim", "mcd_db", "f0_rmse_cents", "duration_ratio", "wer"]
    return {k: statistics.mean(r[k] for r in rows) for k in keys if rows and k in rows[0]}


def preserved_voices(model, base, split: str = "test", seed: int = 0, count: int = 5) -> dict:
    """F2 and M2 must not change: compare edited vs base output for voices 0 and 1."""
    from eval.stage2_audio import compare, world_features
    out = {}
    for v, name in ((0, "F2"), (1, "M2")):
        identical, mcds = True, []
        for i, t in enumerate(sentences()[split][:count]):
            a, b = synthesize(model, t, v, seed), synthesize(base, t, v, seed)
            if len(a) != len(b) or not np.array_equal(a, b):
                identical = False
                mcds.append(compare(a, world_features(b), len(b))["mcd_db"])
        out[name] = {"identical": identical, "mcd_vs_base_db": statistics.mean(mcds) if mcds else 0.0,
                     "sentences_checked": count}
    return out
