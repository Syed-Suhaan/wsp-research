"""Before/after listening files for Stage 2: one TEST sentence, every clip with noise seed 0.

    F2, M2          base model, voices 0 and 1
    target          the target recording of the sentence
    B3              base model with B3's chosen built-in controls
    B1              B1 trained with its chosen setting and seed 0 (identical to its run 0)
    W1_best, W1_worst  the W1 runs with the highest and lowest test speaker similarity

Output: samples/stage2/final/test_<i>_<name>.wav and samples/stage2/final/manifest.json
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.stage2_audio import save_wav  # noqa: E402
from eval.stage2_metrics import sentences, synthesize, target_path  # noqa: E402
from models.tiny_voca3 import NEW_VOICE, load_state  # noqa: E402
from train.stage2_baselines import b3_controls, model_from, train_condition  # noqa: E402

SENTENCE = 0
OUT = ROOT / "samples" / "stage2" / "final"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    text = sentences()["test"][SENTENCE]
    base = load_state()
    res = ROOT / "results" / "stage2"
    w1 = sorted((json.loads(p.read_text()) for p in (res / "W1").glob("run_*.json")), key=lambda r: r["speaker_sim"])
    worst, best = w1[0], w1[-1]
    b3 = json.loads((res / "B3" / "grid.json").read_text())["chosen_setting"]
    b1 = json.loads((res / "B1" / "grid.json").read_text())["chosen_setting"]
    b1_model = model_from(train_condition("B1", b1["lr"], [b1["steps"]], 0)[b1["steps"]])
    clips = {
        "F2": synthesize(base, text, 0, 0),
        "M2": synthesize(base, text, 1, 0),
        "B3": synthesize(base, text, b3["voice"], 0, b3_controls(b3)),
        "B1": synthesize(b1_model, text, NEW_VOICE, 0),
        "W1_best": synthesize(load_state(ROOT / "checkpoints" / f"stage2_W1_run{best['run']}.pt"), text, NEW_VOICE, 0),
        "W1_worst": synthesize(load_state(ROOT / "checkpoints" / f"stage2_W1_run{worst['run']}.pt"), text, NEW_VOICE, 0),
    }
    stem = f"test_{SENTENCE:02d}"
    for name, wave in clips.items():
        save_wav(OUT / f"{stem}_{name}.wav", wave)
    shutil.copy2(target_path("test", SENTENCE), OUT / f"{stem}_target.wav")
    manifest = {"sentence": text, "test_index": SENTENCE, "noise_seed": 0,
                "W1_best_run": best["run"], "W1_best_test_speaker_sim": best["speaker_sim"],
                "W1_worst_run": worst["run"], "W1_worst_test_speaker_sim": worst["speaker_sim"],
                "B3_setting": b3, "B1_setting": b1, "files": sorted(p.name for p in OUT.glob("*.wav"))}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
