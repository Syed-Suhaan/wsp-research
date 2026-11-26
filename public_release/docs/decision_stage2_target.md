# Decision record: Stage 2 target voice

## The first target was rejected

The first Stage 2 target was a synthetic "deep narrator" style. It was made by DSP on the
output of tiny-voca's F2 voice: pitch lowered, formants shifted down, and speech slowed,
using the WORLD vocoder (pyworld).

After a listening check, this target was **rejected**. It sounded like slowed F2 with reverb,
with the same accent and delivery as F2. It was not a new voice, so it could not test
whether a weight edit adds a new voice.

The rejection happened **before any test evaluation and before any WSP run**. The only results
that existed were:
- the target audio itself;
- a validation-only grid for baseline B3 (built-in controls only), which was never evaluated on
  test.

These files are kept for the record in `results/stage2_dsp_rejected/` and
`samples/stage2_dsp_rejected/`. One observation from that pilot: on mel-cepstral distortion, B3
preferred settings near the default voice, because the DSP target kept F2's spectral frames.

## The new target

A real, different voice: Kokoro-82M (`hexgrad/Kokoro-82M`, Apache-2.0, model revision
`f3ff3571791e39611d31c381e3a41a3af07b4987`), voice **am_michael** (American English, male).
The user chose it after listening to five candidates (`samples/stage2/candidates/`). The
descriptive distances of the candidates to F2 and M2 are in
`samples/stage2/candidates/distance_to_F2_M2.json`.

- The target audio for 50 enrollment, 20 validation, and 20 test sentences
  (`data/stage2_sentences.json`) is committed as data in `data/stage2_target/am_michael/`. No
  condition needs Kokoro to run.
- Conditions may use only the enrollment and validation audio. Test audio is for evaluation
  only.
- The audio was generated in a separate environment (`.venv-target`, Python 3.12;
  `requirements-target.txt`) with `data/make_stage2_kokoro.py`. The noise seed is fixed per
  sentence, and two runs produce byte-identical files.
- Primary metric: speaker similarity (Resemblyzer, cosine). Secondary: MCD with DTW, and F0
  RMSE. Guardrail: word error rate (Whisper small.en).
