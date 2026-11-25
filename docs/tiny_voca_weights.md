# tiny-voca weights: parts relevant to Stage 2

**Status: verified against the model code.** The code is `tiny_voca/model.py` at commit
`4bf796d29ea775edd5a7cc5ee3a6806b3c3b91f6` of github.com/n4ze3m/tiny-voca. The full tensor list
is in `docs/tiny_voca_weight_list.md` (`eval/inspect_tiny_voca.py`).

Model: 4,537,291 parameters, 184 tensors, 24 kHz output, 192 channels, hop 256 samples per
acoustic frame. Voices: index 0 = F2, index 1 = M2.

Data flow (from the model docstring and `TinyVoca.synthesize`):

    phoneme + stress + voice embedding -> encoder (4 blocks) -> duration predictor
      -> repeat token states by duration, add frame-position features and frame voice embedding
      -> prosody stack (2 blocks) -> prosody head: log F0, voicing, energy
      -> acoustic stack (5 blocks), conditioned on the clamped prosody values
      -> source-filter synthesizer: pulse train at F0 + noise, each shaped by a
         per-frame filter predicted from the acoustic stack -> waveform

| Part | Tensors | Shape | Role (verified) |
|---|---|---|---|
| Voice embedding, phoneme level | `voices.weight` | [2, 192] | `nn.Embedding`; the row of the chosen voice is added to every token embedding (with the phoneme and stress embeddings) before the encoder (`TinyVoca.encode`). |
| Voice embedding, frame level | `frame_voices.weight` | [2, 192] | `nn.Embedding`; the voice row is added to every frame after duration expansion, before the prosody stack (`TinyVoca.expand`). |
| Duration path | `duration.conv1`, `duration.conv2`, `duration.norm1/2`, `duration.out` [2, 128] | | `Predictor` with 2 outputs per token: (0) log(1 + frames); (1) for pause-capable tokens, a logit for whether the pause has any frames. Frames = round(expm1(output 0) / speed) (`predict_durations`). |
| Frame position | `frame_position` | [192, 3] | Linear map of 3 features per frame: relative position inside its token, 1 − that, and log(1 + token length) / 4. |
| Pitch / voicing / energy path | `prosody_stack.*` (2 blocks), `prosody_head` | head [3, 192] | Outputs per frame: (0) log(F0 / 200 Hz), (1) voicing logit (> 0 = voiced), (2) normalised log RMS energy. F0 in Hz = 200 · exp(output 0), clamped to 50-600 Hz (`frequency`). |
| Prosody conditioning | `prosody_projection` | [192, 3] | Maps the clamped prosody values (pitch clamped to [−2, 1.5], voicing as 0/1, energy clamped to [−4.5, 3]) into the acoustic stack input (`decode`). |
| Acoustic stack | `acoustic_stack.*` (5 blocks) | | Frame-level transformer whose output drives both synthesizer heads. |
| Vocoder, harmonic branch | `vocoder.harmonic` | [98, 192] | 98 = 80 causal + 16 anticausal complex-cepstrum coefficients (the spectral envelope, i.e. timbre / formants) + 2 per-sub-frame gains (hop 256 / synth hop 128). Shapes a band-limited pulse train at the predicted F0. |
| Vocoder, noise branch | `vocoder.noise` | [36, 192] | 36 = 32 noise-cepstrum coefficients + 4 per-sub-frame gains (hop 256 / noise hop 64). Shapes a noise source. |

Built-in inference controls (`TinyVoca.synthesize`; they change no weights):
- `speed`: durations are divided by it (0.25-4).
- `pitch_shift`: added to log F0 for the oscillator only. ln 2 / 12 ≈ 0.058 is one semitone.
  The acoustic stack is conditioned on the unshifted prosody, so the spectral envelope
  (formants) does not move.
- `pitch_range`: scales log-F0 movement around each utterance's voiced mean (0-3).

Other tensors: `tokens.weight` [76, 192] (phoneme embedding), `stress.weight` [3, 192],
`encoder.*` (4 blocks).
