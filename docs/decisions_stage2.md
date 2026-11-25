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
