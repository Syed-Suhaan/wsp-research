"""Stage 2 gradient baselines: fine-tune TinyVoca3 toward the target voice (trainer revision 2).

    B1  trainable: row 2 (the new voice) of `voices.weight` and `frame_voices.weight`
    B2  trainable: every parameter of the model
    H1  B1's training applied to a WSP (W1) checkpoint instead of the base

The objective is tiny-voca's own training objective (`tiny_voca.train.TrainingSystem.objective`):
durations and pitch/voicing/energy are teacher-forced from annotations of the recordings
(train/stage2_annotate.py: the author's forced aligner, pyworld pitch); the acoustic decoder is
conditioned on the model's own detached prosody (the author's final teacher probability, 0.0);
the oscillator follows the recorded pitch with a random phase; the waveform is compared with
the recording by multi-resolution STFT and log-mel losses. Loss weights are the author's
(`tiny_voca.train.WEIGHTS`), except the auxiliary mel head, which is training-only and not
released (weight 0). The adversarial terms are left out (the discriminators are not
released). Optimizer: Adam. Batch: 4 utterances drawn with a generator seeded by the seed.

Training data: the 50 ENROLLMENT sentences and their target audio (or, for the pipeline
sanity check only, another set of waveforms for the same sentences).

Revision 1 of this trainer (free-running synthesis with a DTW-aligned log-mel loss) failed a
pipeline sanity check and was replaced; see docs/decisions_stage2.md (D11, D12).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]

from eval.stage2_audio import load_wav  # noqa: E402
from eval.stage2_metrics import sentences, target_path  # noqa: E402
from models.tiny_voca3 import NEW_VOICE  # noqa: E402
from train.stage2_annotate import annotate, collate  # noqa: E402

BATCH = 4
TEACHER_PROBABILITY = 0.0


def target_annotations() -> list[dict]:
    texts = sentences()["enroll"]
    waves = [load_wav(target_path("enroll", i)) for i in range(len(texts))]
    return annotate("target_enroll", texts, waves)


def trainable_params(model, which: str) -> list[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad_(which == "all")
    if which == "voice":
        model.voices.weight.requires_grad_(True)
        model.frame_voices.weight.requires_grad_(True)
        return [model.voices.weight, model.frame_voices.weight]
    return [p for p in model.parameters() if p.requires_grad]


def _restrict_voice_grad(model) -> None:
    """For B1 and H1 only row 2 may change: zero the gradient of rows 0 and 1."""
    for w in (model.voices.weight, model.frame_voices.weight):
        if w.grad is not None:
            w.grad[:NEW_VOICE] = 0


def train(model, which: str, lr: float, steps: int, seed: int, snapshot_steps: list[int], log=None,
          annotations: list[dict] | None = None, loss_log: list | None = None) -> dict:
    """Fine-tune `model` in place; returns {step: state_dict} for the snapshot steps.
    annotations: annotated utterances (default: the target's enrollment audio)."""
    from tiny_voca.train import WEIGHTS, TrainingSystem
    torch.manual_seed(seed)
    items = annotations if annotations is not None else target_annotations()
    system = TrainingSystem(model.config)
    system.model = model
    weights = dict(WEIGHTS, mel_head=0.0)
    params = trainable_params(model, which)
    opt = torch.optim.Adam(params, lr=lr)
    rng = random.Random(seed)
    snaps = {}
    model.train(False)   # dropout off (fine-tuning a released model)
    for step in range(1, steps + 1):
        batch = collate([items[i] for i in rng.sample(range(len(items)), BATCH)], NEW_VOICE)
        opt.zero_grad()
        loss, terms, _ = system.objective(batch, TEACHER_PROBABILITY, weights=weights, random_phase=True)
        loss.backward()
        if which == "voice":
            _restrict_voice_grad(model)
        opt.step()
        if loss_log is not None:
            loss_log.append(float(loss.detach()))
        if log and (step % 25 == 0 or step == 1):
            log(f"step {step} loss {float(loss.detach()):.4f} pitch_err {float(terms['pitch_error_semitones']):.2f} st")
        if step in snapshot_steps:
            snaps[step] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for p in model.parameters():
        p.requires_grad_(False)
    return snaps
