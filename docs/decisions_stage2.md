# Stage 2 decisions log

Each entry: what was decided, why, when, and which results had been seen at that point.
"No results seen" means that no Stage 2 condition (B1, B2, B3, H1, W1) had been evaluated on any
split with the Kokoro target.

## D1. Speaker-similarity model: WavLM x-vector instead of Resemblyzer
- **What:** the primary metric uses cosine similarity of `microsoft/wavlm-base-plus-sv`
  embeddings (WavLMForXVector, revision feb593a6c23c1cc3d9510425c29b0a14d2b07b1e), through
  `transformers` 5.24.1, on 16 kHz audio.
- **Why:** `resemblyzer` could not be installed wheels-only, because its dependency
  `webrtcvad` exists only as a source distribution. The user pre-approved an equivalent
  speaker model from Hugging Face for a technical failure. WavLM-base-plus-SV is a standard
  speaker-verification model, and it needs no package beyond `transformers`, which Whisper
  needs anyway.
- **When:** 2026-11-25, before any condition ran. No results seen. Check: two different
  sentences of the target voice score 0.937 cosine.

## D2. ASR guardrail: Whisper small.en via transformers
- **What:** WER uses `openai/whisper-small.en` (revision e8727524f962ee844a7319d92be39ac1bd25655a),
  greedy decoding, on audio resampled to 16 kHz. Text is lowercased and punctuation is
  removed before WER.
- **Why:** pre-approved. Check: the first validation target sentence is transcribed exactly.
- **When:** 2026-11-25, before any condition ran. No results seen.

## D3. Fixed model class and base: a third voice slot initialised as M2
- **What:** `models/tiny_voca3.py` (TinyVoca3) is tiny-voca with voices [F2, M2, NEW]. The
  only shape change is `voices.weight` and `frame_voices.weight`, [3, 192]. In the base
  checkpoint (`checkpoints/stage2_base.pt`), row 2 is a copy of M2. Every condition is
  evaluated through this class with strict=True.
- **Why:** WSP rule 4 needs one fixed class, and the new voice needs its own rows so that F2
  and M2 can stay unchanged. Among the built-in voices, M2 is nearest to the target
  (candidate distances: MCD 11.09 vs 14.01 dB), so starting from it is the natural
  initialisation for B1, B2, and W1 alike.
- **When:** 2026-11-25, before any condition ran. No results seen.

## D4. Gradient baselines train with a DTW-aligned log-mel loss
- **What:** B1, B2, and H1 synthesize free-running, then DTW-align output and target
  log-mels (on detached features) and minimise L1 along the path. Durations are not trained.
  Batch: 4 enrollment sentences, Adam.
- **Why:** the target audio has no token durations. tiny-voca's own aligner needs an extra
  phoneme recogniser (wav2vec2), which is not approved. A DTW-aligned spectral loss is a
  standard alignment-free choice and uses only enrollment audio.
- **When:** 2026-11-25, before any condition ran. No results seen. A 4-step check showed
  that B1 changes only row 2 of the two voice tables.

## D5. Speaker similarity is measured against an enrollment centroid
- **What:** speaker similarity = cosine(x-vector of the output, normalised mean x-vector of
  the 50 enrollment target utterances).
- **Why:** this compares against the voice, not against a same-text recording, and it uses
  only enrollment audio. Check: a validation target utterance scores 0.986 against the
  centroid.
- **When:** 2026-11-25, before any condition ran. On 3 validation sentences of the base model
  (not a condition result): M2 0.621, F2 0.476.

## D6. Pairing and seeds
- **What:** run i of W1 and H1 is paired with seed i of every baseline. On TEST, each run is
  scored on the 20 test sentences with noise seed i. Selection uses VALIDATION seeds 0-2.
- **Why:** this matches Stage 1 v2 and gives 8 pairs for bootstrap CIs.
- **When:** 2026-11-25, before any condition ran. No results seen.

## D7. Selection rule (all tuned Stage 2 conditions)
- **What:** `results/stage2/selection_rule.json`. Choose the highest mean validation speaker
  similarity (ties within 0.005 broken by lower MCD), after excluding settings with mean
  validation WER above 0.10.
- **Why:** speaker similarity is the primary metric. The WER cap stops a baseline from being
  chosen for an unintelligible output, the same risk the guardrail exists for. 0.10 is well
  above the base voices' WER on the check sentences (0.0).
- **When:** 2026-11-25, committed before any grid ran. No results seen.

## D8. Grids
- **What:** B3: voice {F2, M2} x pitch shift {-12 ... 0 semitones, step 2} x pitch range
  {0.6, 0.8, 1.0, 1.2}, speed 1.0 (56 settings). B1: lr {1e-3, 3e-3, 1e-2, 3e-2} x steps
  {100, 300}. B2: lr {1e-5, 3e-5, 1e-4} x steps {100, 300}. H1 uses B1's chosen setting.
- **Why:** the pitch-shift range covers the distance from M2 (median about 163 Hz) to the
  target (about 111 Hz), about 7 semitones, with margin. The B1 and B2 learning rates span
  typical Adam values for embedding-only and full fine-tuning. Snapshots at 100 and 300 steps
  bound the cost to about 5 minutes per training run.
- **When:** 2026-11-25, committed before any grid ran. No results seen.

## D9. W1 instruction check (prompts/stage2_W1.md), before the first W1 run
- **Checked against the user's list:**
  - target voice name, "Kokoro", "am_michael": none;
  - slider names (`pitch_shift`, `pitch_range`, `speed`): none;
  - metric thresholds: none. The only regex hit is "F0 ... 24 kHz", in the model
    description. The numbers in the file are counts, sizes, and limits (voice indices 0-2,
    50/20 sentences, 4/2/5 transformer blocks, [3, 192] tables, 24 kHz, at most 3 validation
    evaluations, five-minute edit runtime, python3.13 in a path);
  - test sentences: none.
- **Also:** the writer-readable code (`eval/wsp_tools.py`, `models/tiny_voca3.py`,
  `eval/common.py`, `tools/check_wsp_edit.py`) contains none of these names; the target
  location is read at run time from `data/stage2_config.json`, which writers may not read.
  The `tiny_voca` package source, which writers may read to understand the model, names the
  model's built-in controls (it is the published code). It contains no target values, and the
  evaluation always uses the default settings.
- **When:** 2026-11-25, before the first W1 run. While it was written, the reference rows and
  the B3 grid were running; no baseline result had been seen.

## D10. Stage 2 export reproduction check
- **What:** the export's reproduction check gains a Stage 2 part: download the tiny-voca files,
  rebuild the Stage 2 base, and score REF_M2 and the chosen B3 setting on TEST with seed 0.
  Tolerance: 0.005 speaker similarity (0.5 points) and 0.05 dB MCD.
- **Why:** CLAUDE.md asks for the reference baseline named in the stage spec; Stage 2 has no
  base training (the model is pretrained). REF_M2 and B3 exercise the full metric pipeline
  (synthesis, WavLM, Whisper, MCD) without training, so the check stays short.
- **When:** 2026-11-26, after all W1 runs and the reference rows were on disk, while the B3
  grid was running; no B3, B1, or B2 test result had been seen.

## D11. B1 pipeline bug found by the sanity check; B1/B2/H1 halted
- **What:** the B1 validation grid gave no gain over unedited M2 at lr 1e-3 and 3e-3
  (similarity 0.586-0.600, F0 RMSE about 700 cents, like M2). Per the user's rule (c), a sanity
  check was run (`eval/b1_sanity_stage2.py`, `results/stage2/b1_sanity.json`). Trained toward F2's
  own audio (a reachable target), B1's loss did not fall (0.957 -> 0.957), MCD to F2 did not
  change, and F0 moved the wrong way (161 -> 140 Hz; F2 is 251 Hz). The loss itself scores the
  answer far lower (0.23 vs 0.99). So the optimisation path is broken, most likely the gradient
  through the oscillator phase. B1, B2, and H1 share this trainer.
- **Action:** the background baseline job was stopped (its saved units are kept), and the
  orchestrator stopped to report to the user, as rule (c) requires. No B1, B2, or H1 test result
  exists.
- **When:** 2026-11-26, after the B3 test results and all W1 test results had been seen, and
  after 4 of the 12 B1 grid units.

## D12. Trainer revision 2 follows the author's recipe
- **What:** B1, B2, and H1 now use tiny-voca's own training objective
  (`tiny_voca.train.TrainingSystem.objective`):
  - durations are forced-aligned with the author's aligner (facebook/wav2vec2-lv-60-espeak-cv-ft,
    revision ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4, pre-approved by the user);
  - pitch, voicing, and energy targets come from the recording, and the oscillator follows the
    recorded pitch with a random phase;
  - the acoustic decoder is conditioned on the model's own detached prosody (the author's final
    teacher probability, 0.0);
  - losses: duration, pause, pitch, pitch spread, voicing, energy, and multi-resolution
    STFT + log-mel, with the author's weights.
  Left out: the adversarial terms (discriminators not released) and the auxiliary mel head
  (training-only, not released; weight 0). Deviation: pitch and voicing come from pyworld
  Harvest instead of librosa.pyin, because librosa is not installed.
- **Why:** the user asked to base the fine-tune on the author's training code when it exists.
  It does (`vendor/src/tiny-voca/src/tiny_voca/train.py`). Teacher-forcing pitch also removes
  the gradient path through the oscillator phase that broke revision 1.
- **When:** 2026-11-26. The trainer was fixed **after the W1 test results were known** (and
  after the B3 and reference-row test results). The fix is validated only on the F2 target,
  by the sanity criterion committed before the sanity run (`results/stage2/sanity_criterion.json`).
  All grids and selection rules for B1, B2, and H1 are committed before they run.

## D13. Sanity check passed; selection rule revision 2; target-ceiling row
- **Sanity result** (`results/stage2/sanity_v2.json`, criterion committed before the run):
  trained toward F2's own audio, B1 closes 99.3% of the M2-to-F2 gap in speaker similarity
  (0.334 -> 0.981; F2 itself 0.985) and 84.2% in MCD (10.00 -> 4.38 dB; F2's noise floor
  3.32 dB). Pass (threshold 70% in both).
- **Selection rule revision 2** (`results/stage2/selection_rule.json`): same grids, seeds, and
  choice; only the trainer changed. The 4 B1 grid units from trainer revision 1 are moved to
  `results/stage2_trainer_v1_rejected/` and are not used. Committed before any revision-2 grid
  ran.
- **Target ceiling** (user request): reference row REF_TARGET scores the target's own test
  recordings like a condition. Every condition also reports "gap closed" in speaker
  similarity = (condition - B3) / (ceiling - B3).
- **When:** 2026-11-26, after the W1, B3, and reference-row test results were known.

## D14. Housekeeping: a finished grid is never re-run
- An earlier restart of the job queue re-entered the B3 grid, because `run_grid` did not skip
  a finished grid. It recomputed 4 units before the job was stopped. Those units were deleted;
  the committed `results/stage2/B3/grid.json`, which was never overwritten, is the one used.
  `run_grid` now skips a condition whose `grid.json` exists.

## D15. H1 below W1 and B1: checked, not a bug
- **Observation:** H1 (W1 run i, then B1's setting) has a mean test speaker similarity below
  both W1 and B1.
- **Check** (`results/stage2/h1_check.json`): with the same setting and seed, the training
  loss falls just as much from a W1 start as from the M2 start (27.8 -> 18.2 vs 28.6 -> 17.9).
  The trainer works on H1's starting point. The training objective (spectral and prosody
  losses) is not speaker similarity. From W1's starting points, which lie outside the region
  the voice table was trained on, it reaches a similar loss at a lower speaker similarity.
  Reported as measured.
- **When:** 2026-11-26, after all test results were known.

## D16. requirements.txt: setuptools pinned
- **What:** added `setuptools==84.2.1` (the version in the working environment).
- **Why:** the first export reproduction check failed in a fresh Python 3.13 environment
  because pyworld 0.3.5 imports `pkg_resources` (part of setuptools), which a fresh venv does
  not include. The Stage 1 part of that check had passed.
- **When:** 2026-11-26, during the Stage 2 export, after all results were known.
