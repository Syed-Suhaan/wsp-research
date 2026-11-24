# Stage 2 spec (orchestrator only; WSP writers must not read notes/)

Stage 2 asks whether WSP can add a new voice style to tiny-voca. Do Stage 2 only when the
user says so. The original Stage 2 section of the project instructions is reproduced below
verbatim.

Export: the public README must credit tiny-voca (`n4ze3m/tiny-voca`, Apache-2.0).

## Original Stage 2 section (verbatim)

## Stage 2 — Add a new voice style to tiny-voca

### Task 2.0 — Access check
- Try to download `n4ze3m/tiny-voca` from Hugging Face.
- If it fails, write in `BLOCKED.md` that the user must commit `tiny-voca.pt`, `config.json`, and the `phonemizer/` folder to `vendor/tiny-voca/`, with the Apache-2.0 license file. Stop.

### Task 2.1 — Inspect
- Write `eval/inspect_tiny_voca.py`. List every weight name, shape, and parameter count.
- In `docs/tiny_voca_weights.md`, identify: the voice embedding (F2, M2), the pitch path, the duration path, and the source-filter vocoder parts.

### Task 2.2 — Synthetic target
- Use 40 fixed English sentences: 20 for validation, 20 for test.
- Make the target style "deep narrator" with DSP on F2 output: pitch −4 semitones, formants −8%, rate 0.9x. Use pyworld or an equivalent.
- The target durations are the F2 durations divided by 0.9. They are known exactly, so fine-tuning needs no aligner.

### Task 2.3 — Conditions
| ID | Condition | Uses gradients |
|----|-----------|----------------|
| B1 | Fine-tune a new voice vector only | Yes |
| B2 | Fine-tune the whole model on target audio | Yes |
| W1 | WSP: add a third voice with no gradients | No |
| H1 | Hybrid: start from the W1 checkpoint, then B1 | Yes |

### Task 2.4 — Metrics (TEST sentences)
- F0 RMSE in cents on voiced frames.
- Mel-cepstral distortion with DTW.
- Duration ratio against the target.
- F2 and M2 must not change: report their mel-cepstral distortion against the original model output.
- Time per condition and percentage of weights changed.

### Task 2.5 — Samples
- Save 3 test sentences per condition as 24 kHz WAV in `samples/stage2/`.
- Save one before/after pair: original F2 and W1 on the same sentence.

### Task 2.6 — Export
Do the Export task (below) for Stage 2. Then stop and wait for the user.
