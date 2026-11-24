# tiny-voca weights: parts relevant to Stage 2

**Status: provisional.** The mapping below comes from tensor names and shapes
(`eval/inspect_tiny_voca.py`, full list in `docs/tiny_voca_weight_list.md`), from the model's
`config.json`, and from its README. It has not yet been checked against the model code
(`github.com/n4ze3m/tiny-voca`), which is not installed in this environment.

Model: 4,537,291 parameters, 184 tensors, 24 kHz output, hop 256, 192 channels. Voices are
indexed 0 = F2 and 1 = M2 (`config.json` `voices`; the ONNX interface uses the same order).

| Part | Tensors | Shape | Reading |
|---|---|---|---|
| Voice embedding (phoneme level) | `voices.weight` | [2, 192] | One row per voice (F2, M2), added to or combined with the phoneme encoder input. |
| Voice embedding (frame level) | `frame_voices.weight` | [2, 192] | One row per voice, used by the frame-level stacks after duration expansion. |
| Duration path | `duration.conv1`, `duration.conv2`, `duration.norm1/2`, `duration.out` | out: [2, 128] | Convolutional duration predictor on the encoder output; 2 outputs per phoneme. |
| Pitch / voicing / energy path | `prosody_stack.*` (2 blocks), `prosody_head` | head: [3, 192] | Frame-level transformer, then 3 outputs per frame. These are probably log F0, voicing, and energy; the config has `f0_min`, `f0_max`, `pitch_reference_hz`, `energy_mean`, `energy_scale`. |
| Prosody feedback | `prosody_projection` | [192, 3] | Maps the 3 prosody values back into the 192-channel frame stream. |
| Acoustic stack | `acoustic_stack.*` (5 blocks) | | Frame-level transformer that predicts the synthesizer parameters. |
| Source-filter vocoder: harmonic branch | `vocoder.harmonic` | [98, 192] | Filter for the pulse-train source. 98 outputs; the config has `cep_causal` 80 and `cep_anti` 16 (96), so likely cepstral filter coefficients plus 2 gain or shape terms. |
| Source-filter vocoder: noise branch | `vocoder.noise` | [36, 192] | Filter for the noise source. 36 outputs; the config has `noise_cep` 32, so likely 32 cepstral coefficients plus 4 other terms. |

Other tensors: `tokens.weight` [76, 192] (phoneme embedding), `stress.weight` [3, 192],
`encoder.*` (4 blocks), `frame_position` (frame position features).
