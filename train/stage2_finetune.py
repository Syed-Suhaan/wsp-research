"""Stage 2 gradient baselines: fine-tune TinyVoca3 toward the target voice.

    B1  trainable: row 2 (the new voice) of `voices.weight` and `frame_voices.weight` (384 values)
    B2  trainable: every parameter of the model
    H1  B1's training applied to a WSP (W1) checkpoint instead of the base

Training data: the 50 ENROLLMENT sentences and their target audio (no other target audio).
There are no token durations for the target audio, so the loss is alignment-free:
    1. synthesize the sentence free-running with the new voice (predicted durations, noise seed
       = training seed * 100000 + step), with gradients;
    2. compute log-mel spectrograms (80 mels, n_fft 1024, hop 256) of output and target;
    3. find the DTW path between them on the detached features (Euclidean frame distance);
    4. loss = mean L1 between output and target log-mel frames along that path.
Durations are not learned (rounding is not differentiable). Optimizer: Adam. Each step uses a
batch of 4 enrollment sentences drawn with a generator seeded by the training seed.

Usage (library): train(model, trainable, lr, steps, seed, snapshot_steps) -> {step: state_dict}
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]

from eval.stage2_audio import dtw_path, load_wav  # noqa: E402
from eval.stage2_metrics import sentences, target_path  # noqa: E402
from models.tiny_voca3 import NEW_VOICE  # noqa: E402

N_FFT, HOP, N_MELS, SR = 1024, 256, 80, 24000
BATCH = 4


def _mel_basis() -> torch.Tensor:
    from tiny_voca.losses import mel_filterbank
    return mel_filterbank(SR, N_FFT, N_MELS)


def log_mel(wave: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    spec = torch.stft(wave, N_FFT, HOP, window=torch.hann_window(N_FFT), return_complex=True).abs()
    mel = basis.to(spec.dtype) @ spec if basis.shape[1] == spec.shape[0] else spec.T @ basis.T
    mel = mel if mel.shape[0] == N_MELS else mel.T
    return torch.log(mel.clamp_min(1e-5)).T            # [frames, mels]


def synth_with_grad(model, tokens, stress, voice: int, seed: int) -> torch.Tensor:
    """TinyVoca.synthesize without inference_mode (same steps, default controls)."""
    lengths = torch.tensor([tokens.shape[1]])
    speakers = torch.tensor([voice])
    hidden, mask = model.encode(tokens, stress, lengths, speakers)
    durations = model.predict_durations(hidden, mask, tokens, 1.0)
    frames, frame_mask, frame_lengths = model.expand(hidden, durations, mask, speakers)
    prosody_hidden, prosody = model.predict_prosody(frames, frame_mask)
    ends = durations.cumsum(1)
    frame = torch.arange(frames.shape[1])[None].expand(1, -1).contiguous()
    owner = torch.searchsorted(ends, frame, right=True).clamp(max=tokens.shape[1] - 1)
    vowel = model.vowel_tokens[tokens].gather(1, owner) & frame_mask
    prosody = torch.cat((prosody[..., :1], torch.where(vowel, prosody[..., 1].clamp_min(2.0), prosody[..., 1])[..., None],
                         prosody[..., 2:]), dim=-1)
    acoustic = model.decode(prosody_hidden, model.hard_prosody(prosody), frame_mask)
    n = int(frame_lengths[0]) * model.config.hop
    noise = torch.randn(n, generator=torch.Generator().manual_seed(seed))[None]
    wave = model.vocoder(acoustic, model.frequency(prosody[..., 0]), frame_lengths, noise=noise)
    return wave[0, :n]


def trainable_params(model, which: str) -> list[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad_(which == "all")
    if which == "voice":
        model.voices.weight.requires_grad_(True)
        model.frame_voices.weight.requires_grad_(True)
        return [model.voices.weight, model.frame_voices.weight]
    return [p for p in model.parameters() if p.requires_grad]


def _restrict_voice_grad(model) -> None:
    """For B1 only row 2 may change: zero the gradient of rows 0 and 1."""
    for w in (model.voices.weight, model.frame_voices.weight):
        if w.grad is not None:
            w.grad[:NEW_VOICE] = 0


def train(model, which: str, lr: float, steps: int, seed: int, snapshot_steps: list[int], log=None,
          targets: list | None = None, loss_log: list | None = None) -> dict:
    """targets: optional list of enrollment-sentence waveforms to use instead of the target audio
    (used only by the pipeline sanity check). loss_log: optional list that receives the loss of
    every step."""
    from tiny_voca.infer import phonemize
    torch.manual_seed(seed)
    texts = sentences()["enroll"]
    phones = [phonemize(t) for t in texts]
    if targets is None:
        targets = [torch.from_numpy(load_wav(target_path("enroll", i))).float() for i in range(len(texts))]
    basis = _mel_basis()
    tgt_mels = [log_mel(t, basis) for t in targets]
    params = trainable_params(model, which)
    opt = torch.optim.Adam(params, lr=lr)
    rng = random.Random(seed)
    snaps = {}
    model.train(False)  # dropout off; the model has no batch-norm
    for step in range(1, steps + 1):
        idx = rng.sample(range(len(texts)), BATCH)
        opt.zero_grad()
        total = 0.0
        for k, i in enumerate(idx):
            p = phones[i]
            wave = synth_with_grad(model, torch.tensor([p.ids]), torch.tensor([p.stress]), NEW_VOICE,
                                   seed * 100000 + step * BATCH + k)
            gm = log_mel(wave, basis)
            pi, pj = dtw_path(gm.detach().numpy(), tgt_mels[i].numpy())
            loss = (gm[torch.from_numpy(pi)] - tgt_mels[i][torch.from_numpy(pj)]).abs().mean() / BATCH
            loss.backward()
            total += float(loss.detach())
        if which == "voice":
            _restrict_voice_grad(model)
        opt.step()
        if loss_log is not None:
            loss_log.append(total)
        if log and (step % 25 == 0 or step == 1):
            log(f"step {step} loss {total:.4f}")
        if step in snapshot_steps:
            snaps[step] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for p in model.parameters():
        p.requires_grad_(False)
    return snaps
