"""[Rejected pilot, kept for the record: the rejected DSP target. See docs/decision_stage2_target.md.]

Stage 2, Task 2.2: synthesize F2 for the 40 fixed sentences and build the synthetic target.

For every sentence (data/stage2_sentences.json): F2 with the model's default controls and
noise seed 0, then the target style by DSP (eval.stage2_audio.make_target: pitch, formant,
and rate change with pyworld). The target durations are the F2 durations divided by the rate.

Output:
    data/cache/stage2/{split}_{i:02d}_F2.wav, ..._target.wav     (regenerable, not committed)
    results/stage2_dsp_rejected/target/manifest.json                          lengths, F2 token durations,
                                                                 F2-vs-target metrics
    samples/stage2_dsp_rejected/listening/val_{i:02d}_F2.wav, ..._target.wav  first 3 validation sentences

Usage:
    python data/make_stage2_target.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.stage2_audio import (SR, TARGET_STYLE, compare, load_tts, make_target, save_wav,  # noqa: E402
                               sentences, world_features)

CACHE = ROOT / "data" / "cache" / "stage2"
LISTEN = ROOT / "samples" / "stage2_dsp_rejected" / "listening"
N_LISTEN = 3


def main():
    from tiny_voca.infer import phonemize, _voice
    model = load_tts()
    items = []
    for split in ("val", "test"):
        for i, text in enumerate(sentences()[split]):
            phones = phonemize(text)
            wave, details = _voice(model, phones, "F2", 1.0, 0.0, 0, 1.0)
            f2 = wave.double().numpy()
            target = make_target(f2, TARGET_STYLE)
            stem = f"{split}_{i:02d}"
            save_wav(CACHE / f"{stem}_F2.wav", f2)
            save_wav(CACHE / f"{stem}_target.wav", target)
            if split == "val" and i < N_LISTEN:
                save_wav(LISTEN / f"{stem}_F2.wav", f2)
                save_wav(LISTEN / f"{stem}_target.wav", target)
            m = compare(f2, world_features(target), len(target))
            durations = details["durations"][0].tolist()
            items.append({"split": split, "index": i, "text": text,
                          "f2_samples": len(f2), "target_samples": len(target),
                          "target_over_f2_length": len(target) / len(f2),
                          "f2_token_durations_frames": durations,
                          "target_token_durations_frames": [d / TARGET_STYLE["rate"] for d in durations],
                          "f2_vs_target": m})
            print(f"{stem}: F2 {len(f2) / SR:.2f}s target {len(target) / SR:.2f}s  "
                  f"MCD {m['mcd_db']:.2f}  F0 {m['f0_rmse_cents']:.0f}c", flush=True)
    out = {"target_style": TARGET_STYLE, "f2_controls": {"speed": 1.0, "pitch_shift": 0.0, "pitch_range": 1.0},
           "f2_noise_seed": 0, "sample_rate": SR,
           "note": "Target durations are the F2 token durations divided by the rate (exact, in frames "
                   "of 256 samples). The target waveform length can differ by a few milliseconds "
                   "because WORLD analysis and synthesis use 5 ms frames.",
           "items": items}
    p = ROOT / "results" / "stage2_dsp_rejected" / "target" / "manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
