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

### Stage 2 additions (from the user; these override the original text where they conflict)

Process:
- Every job that can take more than 30 minutes saves after each unit (setting, seed, or run)
  and skips saved units on restart. Build this in from the start.
- Tuned baselines from the start. Before any grid runs, commit a selection rule: choose one
  global setting per condition on VALIDATION using seeds 0-2, then test the chosen setting on
  seeds 0-7. H1 uses the chosen B1 setting.
- Stop points:
  1. Listening check: commit 3 validation sentences as 24 kHz WAV, original F2 and the
     synthetic target, then stop and wait for the user to listen. The user placed this
     "after Task 2.3", but the audio it needs exists at the end of Task 2.2 (no condition
     needed), and Task 2.3 includes W1, which has its own earlier stop. So the
     orchestrator stops at the end of Task 2.2, before any condition runs, unless the
     user says otherwise.
  2. Before the first W1 run: show the user the W1 instruction file and wait for an OK.
- Model code: `tiny-voca` @ 4bf796d29ea775edd5a7cc5ee3a6806b3c3b91f6 and `dizione` @
  8cd889e4de7be6bd1dd2e6891344a0729fa14dd4, installed by the environment setup script,
  pinned in `requirements.txt`. Read-only source copies are in `vendor/src/` (not installed
  from there).

New baseline:
- **B3 (no weight edit):** F2 with only the model's built-in inference controls
  (`pitch_shift`, `pitch_range`, `speed`). Choose their values on VALIDATION with a fixed
  grid, committed before the grid runs.

Metrics:
- Mel-cepstral distortion (timbre) is the **primary** metric for Stage 2. F0 RMSE and the
  duration ratio are secondary.
- Report W1 − B3 with 95% bootstrap CIs. The Stage 2 claim needs W1 to beat B3 on
  mel-cepstral distortion.

Writer information:
- WSP writers must not know the target values. The W1 instruction must not mention
  `pitch_shift`, `speed`, or the DSP settings.

### Stage 2 target revision (from the user, 2026-11-25; overrides the DSP target above)

The DSP "deep narrator" target (Task 2.2 above) was rejected after a listening check: it
sounded like slowed F2 with reverb, with the same accent and delivery, not a new voice. It
was rejected before any test evaluation or WSP run (`docs/decision_stage2_target.md`). Its
artifacts are kept for the record under `results/stage2_dsp_rejected/` and
`samples/stage2_dsp_rejected/`.

Target:
- A real, different voice: Kokoro-82M (hexgrad/Kokoro-82M, Apache-2.0), voice **am_michael**
  (American English, male), chosen by the user from five candidates
  (`samples/stage2/candidates/`).
- Generated in the separate environment `.venv-target` (`requirements-target.txt`); the
  target WAVs are committed as data in `data/stage2_target/am_michael/`, so no condition needs
  Kokoro.
- Credit Kokoro-82M (Apache-2.0) in the public README, next to tiny-voca.

Sentences (`data/stage2_sentences.json`, all fixed and committed):
- 50 enrollment, 20 validation, 20 test.
- Every condition (B1, B2, H1, W1) may use only the enrollment and validation audio of the
  target voice. Test audio is for evaluation only.

Metrics:
- Primary: speaker similarity to the target voice: cosine similarity of Resemblyzer speaker
  embeddings.
- Secondary: mel-cepstral distortion with DTW, and F0 RMSE.
- Guardrail: word error rate with Whisper small.en (from Hugging Face). A condition that
  breaks intelligibility must be visible in the results.
- F2 and M2 must not change: keep that check (their output before vs after the edit).
- Report W1 − B3 with 95% bootstrap CIs. The Stage 2 claim needs W1 to beat B3 on the
  primary metric (speaker similarity).
- (Resemblyzer and Whisper are not installed yet; ask the user before installing them.)

Baselines:
- **B3 (built-in controls only, no weight edit):** the grid includes the source voice (F2 or
  M2), `pitch_shift`, and `pitch_range`; `speed` is fixed at 1.0. Commit the grid and the
  selection rule before B3 runs. Select one global setting by speaker similarity on
  VALIDATION (seeds 0-2), then test it on seeds 0-7.
- **Reference row "nearest built-in voice":** M2 and F2, unchanged, scored against the target
  on TEST, so readers see the starting distance.

Writer information (unchanged): WSP writers must not know the target values. The W1
instruction must not mention `pitch_shift`, `speed`, or any target-generation settings.

The earlier additions that made MCD the primary metric and defined B3 for the DSP target are
superseded by this section.

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
