"""[Rejected pilot, kept for the record: B3 for the rejected DSP target. See docs/decision_stage2_target.md.]

Stage 2 baseline B3: F2 with only the model's built-in inference controls (no weight edit).

Phases:
    grid     every setting of the B3 grid in results/stage2_dsp_rejected/selection_rule.json, on the 20
             VALIDATION sentences x seeds 0-2. Each finished setting is appended to
             results/stage2_dsp_rejected/B3/grid.partial.jsonl; a restart skips saved settings. When all
             settings are done, the selection rule is applied and results/stage2_dsp_rejected/B3/grid.json
             is written (the partial file is then removed).
    listen   write the first 3 validation sentences with the chosen setting (seed 0) to
             samples/stage2_dsp_rejected/listening/val_XX_B3.wav.
    (test    seeds 0-7 on the TEST sentences with the chosen setting; not run yet.)

Usage:
    python train/stage2_b3.py --phase grid
    python train/stage2_b3.py --phase listen
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from eval.stage2_audio import compare, load_tts, load_wav, save_wav, sentences, synth, world_features  # noqa: E402

RULE = json.loads((ROOT / "results" / "stage2_dsp_rejected" / "selection_rule.json").read_text())
OUT = ROOT / "results" / "stage2_dsp_rejected" / "B3"
VAL_SEEDS = [0, 1, 2]
CACHE = ROOT / "data" / "cache" / "stage2"


def grid() -> list[dict]:
    g = RULE["conditions"]["B3"]["grid"]
    keys = list(g)
    return [dict(zip(keys, v)) for v in itertools.product(*(g[k] for k in keys))]


def controls(setting: dict) -> dict:
    return {"pitch_shift": setting["pitch_shift_semitones"] * math.log(2) / 12,
            "pitch_range": setting["pitch_range"], "speed": setting["speed"]}


def select(settings: list[dict]) -> dict:
    best_mcd = min(s["mcd_db"] for s in settings)
    a = [s for s in settings if s["mcd_db"] <= best_mcd + 0.05]
    best_f0 = min(s["f0_rmse_cents"] for s in a)
    b = [s for s in a if s["f0_rmse_cents"] <= best_f0 + 5.0]
    return sorted(b, key=lambda s: (s["abs_log_duration_ratio"], s["index"]))[0]


def run_grid() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    partial = OUT / "grid.partial.jsonl"
    done = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            r = json.loads(line)
            done[r["index"]] = r
    model = load_tts()
    texts = sentences()["val"]
    refs = []
    for i in range(len(texts)):
        tgt = load_wav(CACHE / f"val_{i:02d}_target.wav")
        refs.append((world_features(tgt), len(tgt)))
    settings = []
    for idx, setting in enumerate(grid()):
        if idx in done:
            settings.append(done[idx])
            continue
        t0 = time.time()
        rows = []
        for seed in VAL_SEEDS:
            for i, text in enumerate(texts):
                wave = synth(model, text, "F2", seed=seed, **controls(setting))
                m = compare(wave, *refs[i])
                rows.append({"seed": seed, "sentence": i, **m})
        rec = {"index": idx, "setting": setting, "controls": controls(setting),
               "mcd_db": statistics.mean(r["mcd_db"] for r in rows),
               "f0_rmse_cents": statistics.mean(r["f0_rmse_cents"] for r in rows),
               "duration_ratio": statistics.mean(r["duration_ratio"] for r in rows),
               "abs_log_duration_ratio": statistics.mean(abs(math.log(r["duration_ratio"])) for r in rows),
               "wall_clock_s": time.time() - t0, "per_item": rows}
        with partial.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        settings.append(rec)
        print(f"B3 {idx:3d} {setting}: MCD {rec['mcd_db']:.3f} F0 {rec['f0_rmse_cents']:.1f}c "
              f"dur {rec['duration_ratio']:.3f} ({rec['wall_clock_s']:.0f}s)", flush=True)
    chosen = select(settings)
    out = {"condition": "B3", "rule": RULE["choice"], "grid": RULE["conditions"]["B3"]["grid"],
           "seeds": VAL_SEEDS, "split": "val", "num_settings": len(settings),
           "tuning_wall_clock_s": sum(s["wall_clock_s"] for s in settings),
           "chosen_index": chosen["index"], "chosen_setting": chosen["setting"],
           "chosen_controls": chosen["controls"],
           "chosen_val": {k: chosen[k] for k in ("mcd_db", "f0_rmse_cents", "duration_ratio")},
           "settings": settings}
    (OUT / "grid.json").write_text(json.dumps(out, indent=2) + "\n")
    partial.unlink()
    print("chosen B3:", chosen["setting"], out["chosen_val"])


def run_listen(n: int = 3) -> None:
    g = json.loads((OUT / "grid.json").read_text())
    model = load_tts()
    for i, text in enumerate(sentences()["val"][:n]):
        save_wav(ROOT / "samples" / "stage2_dsp_rejected" / "listening" / f"val_{i:02d}_B3.wav",
                 synth(model, text, "F2", seed=0, **g["chosen_controls"]))
    print("wrote", n, "B3 listening samples with", g["chosen_setting"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["grid", "listen"])
    a = ap.parse_args()
    run_grid() if a.phase == "grid" else run_listen()
