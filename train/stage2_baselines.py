"""Stage 2 baselines and reference rows: validation grids and TEST evaluation.

Conditions (grids and rule: results/stage2/selection_rule.json):
    B3       built-in controls only (voice F2/M2, pitch shift, pitch range; speed 1.0)
    B1       fine-tune the new voice's rows only            (train/stage2_finetune.py)
    B2       fine-tune every parameter                      (train/stage2_finetune.py)
    REF_F2   reference "nearest built-in voice": F2 unchanged (TEST only)
    REF_M2   reference "nearest built-in voice": M2 unchanged (TEST only)
    REF_TARGET  reference "target ceiling": the target's own TEST recordings scored like a
             condition (speaker similarity of the target to its enrollment centroid)

Phases:
    grid   VALIDATION, seeds 0-2. One unit = (B3: setting, seed) or (B1/B2: lr, seed; both step
           snapshots are scored). Units are appended to results/stage2/<cond>/grid.partial.jsonl
           and skipped on restart. Then the rule is applied -> results/stage2/<cond>/grid.json.
    test   the chosen setting on TEST, seeds 0-7 (run i = seed i), with all metrics, the F2/M2
           check, and edit statistics -> results/stage2/<cond>/run_<i>.json (skips saved runs).

Usage:
    python train/stage2_baselines.py --condition B3 --phase grid
    python train/stage2_baselines.py --condition B3 --phase test
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

import torch  # noqa: E402

from eval.stage2_metrics import Evaluator, preserved_voices, summarize  # noqa: E402
from models.tiny_voca3 import NEW_VOICE, TinyVoca3, load_state  # noqa: E402

RULE = json.loads((ROOT / "results" / "stage2" / "selection_rule.json").read_text())
RES = ROOT / "results" / "stage2"
VAL_SEEDS, TEST_SEEDS = [0, 1, 2], list(range(8))
MAX_WER = 0.10


def grid(cond: str) -> list[dict]:
    g = RULE["conditions"][cond]["grid"]
    keys = list(g)
    return [dict(zip(keys, v)) for v in itertools.product(*(g[k] for k in keys))]


def b3_controls(s: dict) -> dict:
    return {"pitch_shift": s["pitch_shift_semitones"] * math.log(2) / 12, "pitch_range": s["pitch_range"], "speed": 1.0}


def model_from(sd: dict) -> TinyVoca3:
    m = TinyVoca3()
    m.load_state_dict(sd, strict=True)
    return m.eval()


def train_condition(cond: str, lr: float, steps: list[int], seed: int, start_sd: dict | None = None, log=None) -> dict:
    from train.stage2_finetune import train
    m = load_state() if start_sd is None else model_from(start_sd)
    return train(m, "voice" if cond in ("B1", "H1") else "all", lr, max(steps), seed, steps, log=log)


def choose(settings: list[dict]) -> tuple[dict, str]:
    ok = [s for s in settings if s["val_wer"] <= MAX_WER]
    note = "settings with mean validation WER <= 0.10"
    if not ok:
        low = min(s["val_wer"] for s in settings)
        ok = [s for s in settings if s["val_wer"] == low]
        note = "NO setting had WER <= 0.10; kept the lowest-WER setting(s)"
    best = max(s["val_speaker_sim"] for s in ok)
    near = [s for s in ok if s["val_speaker_sim"] >= best - 0.005]
    return sorted(near, key=lambda s: (s["val_mcd_db"], s["index"]))[0], note


def run_grid(cond: str) -> None:
    out = RES / cond
    out.mkdir(parents=True, exist_ok=True)
    partial = out / "grid.partial.jsonl"
    done = [json.loads(l) for l in partial.read_text().splitlines()] if partial.exists() else []
    keys = {(json.dumps(u["unit"], sort_keys=True), u["seed"]) for u in done}
    ev = Evaluator()
    if cond == "B3":
        base = load_state()
        units = [(s, seed) for s in grid(cond) for seed in VAL_SEEDS]
        for s, seed in units:
            if (json.dumps(s, sort_keys=True), seed) in keys:
                continue
            t0 = time.time()
            rows = ev.evaluate(base, "val", seed, voice=s["voice"], controls=b3_controls(s))
            rec = {"unit": s, "seed": seed, "scores": {json.dumps(s, sort_keys=True): summarize(rows)}, "wall_clock_s": time.time() - t0}
            with partial.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            done.append(rec)
            print(f"B3 {s} seed {seed}: {summarize(rows)} ({rec['wall_clock_s']:.0f}s)", flush=True)
    else:
        g = RULE["conditions"][cond]["grid"]
        for lr in g["lr"]:
            for seed in VAL_SEEDS:
                unit = {"lr": lr}
                if (json.dumps(unit, sort_keys=True), seed) in keys:
                    continue
                t0 = time.time()
                snaps = train_condition(cond, lr, g["steps"], seed)
                scores = {}
                for st, sd in snaps.items():
                    rows = ev.evaluate(model_from(sd), "val", seed, voice=NEW_VOICE)
                    scores[json.dumps({"lr": lr, "steps": st}, sort_keys=True)] = summarize(rows)
                rec = {"unit": unit, "seed": seed, "scores": scores, "wall_clock_s": time.time() - t0}
                with partial.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                done.append(rec)
                print(f"{cond} lr {lr} seed {seed}: " + "; ".join(f"{k}: sim {v['speaker_sim']:.3f} wer {v['wer']:.3f}" for k, v in scores.items())
                      + f" ({rec['wall_clock_s']:.0f}s)", flush=True)
    per = {}
    for rec in done:
        for k, v in rec["scores"].items():
            per.setdefault(k, []).append(v)
    settings = []
    for idx, s in enumerate(grid(cond)):
        k = json.dumps(s, sort_keys=True)
        vals = per[k]
        assert len(vals) == len(VAL_SEEDS), (k, len(vals))
        settings.append({"index": idx, "setting": s, **{f"val_{m}": statistics.mean(v[m] for v in vals)
                                                         for m in ("speaker_sim", "mcd_db", "f0_rmse_cents", "duration_ratio", "wer")}})
    chosen, note = choose(settings)
    res = {"condition": cond, "rule": RULE["choice"], "grid": RULE["conditions"][cond]["grid"], "seeds": VAL_SEEDS,
           "num_settings": len(settings), "tuning_wall_clock_s": sum(r["wall_clock_s"] for r in done),
           "chosen_index": chosen["index"], "chosen_setting": chosen["setting"], "choice_note": note,
           "chosen_val": {k: v for k, v in chosen.items() if k.startswith("val_")}, "settings": settings, "units": done}
    (out / "grid.json").write_text(json.dumps(res, indent=2) + "\n")
    partial.unlink()
    print(f"chosen {cond}: {chosen['setting']} ({note})", flush=True)


def edit_stats(base_sd: dict, sd: dict) -> dict:
    total = sum(v.numel() for v in base_sd.values())
    changed = {k: int((sd[k] != base_sd[k]).sum()) for k in base_sd}
    return {"pct_weights_changed": 100.0 * sum(changed.values()) / total, "num_weights_changed": sum(changed.values()),
            "changed_tensors": [k for k, n in changed.items() if n]}


def score_run(ev, model, base, seed: int, voice: int, controls: dict | None = None) -> dict:
    rows = ev.evaluate(model, "test", seed, voice=voice, controls=controls)
    return {**summarize(rows), "per_sentence": rows, "preserved_voices": preserved_voices(model, base, "test", seed)}


def run_test(cond: str, seeds: list[int] | None = None) -> None:
    out = RES / cond
    out.mkdir(parents=True, exist_ok=True)
    ev = Evaluator()
    base = load_state()
    base_sd = base.state_dict()
    chosen = None if cond.startswith("REF") else json.loads((out / "grid.json").read_text())["chosen_setting"]
    if cond == "REF_TARGET":
        from eval.stage2_audio import load_wav
        from eval.stage2_metrics import sentences, target_path
        path = out / "run_0.json"
        if not path.exists():
            texts = sentences()["test"]
            waves = [load_wav(target_path("test", i)) for i in range(len(texts))]
            rows = [ev.score_wave(w, "test", i, t, with_asr=False) for i, (w, t) in enumerate(zip(waves, texts))]
            for r, t, hyp in zip(rows, texts, ev.transcribe_batch(waves)):
                from eval.stage2_metrics import wer
                r.update({"asr": hyp, "wer": wer(t, hyp)})
            res = {"stage": "stage2", "condition": cond, "run": 0, "valid": True, "uses_gradients": False,
                   "note": "Target recordings themselves (no model, no seed); the ceiling for speaker similarity.",
                   **summarize(rows), "per_sentence": rows}
            path.write_text(json.dumps(res, indent=2) + "\n")
            print(f"REF_TARGET: sim {res['speaker_sim']:.3f} wer {res['wer']:.3f}", flush=True)
        return
    for seed in (seeds if seeds is not None else TEST_SEEDS):
        path = out / f"run_{seed}.json"
        if path.exists():
            continue
        t0 = time.time()
        if cond in ("REF_F2", "REF_M2", "B3"):
            voice = {"REF_F2": 0, "REF_M2": 1}.get(cond, chosen["voice"] if chosen else None)
            ctrl = b3_controls(chosen) if cond == "B3" else None
            model, train_s = base, 0.0
        else:
            snaps = train_condition(cond, chosen["lr"], [chosen["steps"]], seed)
            train_s = time.time() - t0
            model, voice, ctrl = model_from(snaps[chosen["steps"]]), NEW_VOICE, None
        r = score_run(ev, model, base, seed, voice, ctrl)
        result = {"stage": "stage2", "condition": cond, "run": seed, "seed": seed, "valid": True,
                  "uses_gradients": cond in ("B1", "B2"), "chosen_setting": chosen, "voice_index": voice,
                  "wall_clock_s": train_s, "eval_wall_clock_s": time.time() - t0 - train_s,
                  **edit_stats(base_sd, model.state_dict()), **r}
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"{cond} seed {seed}: sim {r['speaker_sim']:.3f} mcd {r['mcd_db']:.2f} f0 {r['f0_rmse_cents']:.0f}c "
              f"wer {r['wer']:.3f} F2/M2 identical {r['preserved_voices']['F2']['identical']}/{r['preserved_voices']['M2']['identical']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["B3", "B1", "B2", "REF_F2", "REF_M2", "REF_TARGET"])
    ap.add_argument("--phase", required=True, choices=["grid", "test"])
    ap.add_argument("--seeds", type=int, nargs="+", help="test phase only: subset of seeds 0-7")
    a = ap.parse_args()
    run_grid(a.condition) if a.phase == "grid" else run_test(a.condition, a.seeds)
