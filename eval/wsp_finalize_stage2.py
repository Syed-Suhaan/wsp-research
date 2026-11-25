"""Finalize one Stage 2 WSP run (W1) or hybrid run (H1); run by the orchestrator.

W1 steps:
    1. Delete the run's output state dict and re-run edit.py in a fresh process (TMPDIR = the
       run's scratch folder).
    2. Validity: tools/check_wsp_edit.py with the Stage 2 fixed class
       (models.tiny_voca3:TinyVoca3), including the frozen-code check against --git-rev.
    3. Score voice 2 on the 20 TEST sentences with noise seed = run number (all metrics), check
       that F2 and M2 are unchanged, and compute edit statistics.
    4. Write results/stage2/W1/run_<n>.json.
H1 (--condition H1): start from W1 run n's state dict, apply the chosen B1 setting
(results/stage2/B1/grid.json) with training seed n, then score as above ->
results/stage2/H1/run_<n>.json.

Usage:
    python eval/wsp_finalize_stage2.py --condition W1 --run 0 --git-rev <commit> \
        [--orchestrator-start S --orchestrator-end E]
    python eval/wsp_finalize_stage2.py --condition H1 --run 0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import torch  # noqa: E402

from check_wsp_edit import check_run  # noqa: E402
from eval.common import environment_info  # noqa: E402
from eval.stage2_metrics import Evaluator  # noqa: E402
from models.tiny_voca3 import BASE_PATH, NEW_VOICE, load_state  # noqa: E402
from train.stage2_baselines import edit_stats, model_from, score_run, train_condition  # noqa: E402

FIXED = "models.tiny_voca3:TinyVoca3"
WRITER_MODEL_ID = "claude-opus-5-5"


def files_read(rationale: str) -> list[str]:
    m = re.search(r"^#+\s*Files read\s*$(.*?)(?=^#+\s|\Z)", rationale, flags=re.M | re.S | re.I)
    return [l.strip().lstrip("-*0123456789. ").strip() for l in m.group(1).splitlines() if l.strip()] if m else []


def edit_type(stats: dict) -> str:
    t = set(stats["changed_tensors"])
    if not t:
        return "none"
    if t <= {"voices.weight", "frame_voices.weight"}:
        return "new-voice rows"
    if t & {"voices.weight", "frame_voices.weight"}:
        return "mixed (voice rows + shared weights)"
    return "shared weights only"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["W1", "H1"], default="W1")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--git-rev")
    ap.add_argument("--orchestrator-start", type=float)
    ap.add_argument("--orchestrator-end", type=float)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    base = load_state()
    base_sd = base.state_dict()
    out_dir = ROOT / "results" / "stage2" / a.condition
    out_dir.mkdir(parents=True, exist_ok=True)
    w1_ckpt = ROOT / "checkpoints" / f"stage2_W1_run{a.run}.pt"
    ev = Evaluator()

    if a.condition == "H1":
        w1 = json.loads((ROOT / "results" / "stage2" / "W1" / f"run_{a.run}.json").read_text())
        chosen = json.loads((ROOT / "results" / "stage2" / "B1" / "grid.json").read_text())["chosen_setting"]
        start = torch.load(w1_ckpt, map_location="cpu", weights_only=True)
        t0 = time.time()
        snaps = train_condition("H1", chosen["lr"], [chosen["steps"]], a.run, start_sd=start)
        train_s = time.time() - t0
        model = model_from(snaps[chosen["steps"]])
        torch.save(model.state_dict(), ROOT / "checkpoints" / f"stage2_H1_run{a.run}.pt")
        stats = edit_stats(base_sd, model.state_dict())
        r = score_run(ev, model, base, a.run, NEW_VOICE)
        result = {"stage": "stage2", "condition": "H1", "run": a.run, "seed": a.run, "valid": bool(w1["valid"]),
                  "uses_gradients": True, "start": f"W1 run {a.run}", "chosen_setting": chosen,
                  "wall_clock_s": train_s, **stats, "edit_type": edit_type(stats), **r}
    else:
        run_dir = ROOT / "wsp" / "stage2" / "W1" / f"run_{a.run}"
        if w1_ckpt.exists():
            w1_ckpt.unlink()
        (run_dir / "scratch").mkdir(exist_ok=True)
        t0 = time.time()
        rerun = subprocess.run([sys.executable, str(run_dir / "edit.py")], cwd=ROOT,
                               env=dict(os.environ, TMPDIR=str(run_dir / "scratch")), capture_output=True, text=True)
        edit_runtime = time.time() - t0
        check = (check_run(run_dir, w1_ckpt, BASE_PATH, a.git_rev, fixed_class=FIXED) if w1_ckpt.exists()
                 else {"valid": False, "errors": [f"edit.py did not produce {w1_ckpt.name}: {rerun.stderr[-500:]}"]})
        result = {"stage": "stage2", "condition": "W1", "run": a.run, "seed": a.run, "valid": bool(check["valid"]),
                  "check_errors": check["errors"], "uses_gradients": False, "writer_model_id": WRITER_MODEL_ID,
                  "effort_level": "unverified",
                  "effort_observations": json.loads((ROOT / "results/stage1_v2/effort_probe.json").read_text())["observations"],
                  "git_rev_at_start": a.git_rev, "edit_script_runtime_s": edit_runtime}
        if w1_ckpt.exists():
            try:
                sd = torch.load(w1_ckpt, map_location="cpu", weights_only=True)
                model = model_from(sd)
                stats = edit_stats(base_sd, sd)
                slot = {k: int((sd[k][NEW_VOICE] != base_sd[k][NEW_VOICE]).sum()) for k in ("voices.weight", "frame_voices.weight")}
                result.update({**stats, "edit_type": edit_type(stats), "new_voice_row_values_changed": slot,
                               **score_run(ev, model, base, a.run, NEW_VOICE)})
            except Exception as e:  # noqa: BLE001
                result.update({"evaluation_error": str(e)[:500], "edit_type": "invalid"})
        timing = json.loads((run_dir / "timing.json").read_text())
        rationale = (run_dir / "rationale.md").read_text() if (run_dir / "rationale.md").exists() else ""
        log = run_dir / "val_log.jsonl"
        result.update({"wall_clock_s": timing["end"] - timing["start"],
                       "orchestrator_wall_clock_s": (a.orchestrator_end - a.orchestrator_start
                                                     if a.orchestrator_start and a.orchestrator_end else None),
                       "num_val_evaluations": len(log.read_text().splitlines()) if log.exists() else 0,
                       "files_read": files_read(rationale), "process_notes": a.note})
    result["environment"] = environment_info()
    (out_dir / f"run_{a.run}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{a.condition} run {a.run}: valid {result['valid']} sim {result.get('speaker_sim', float('nan')):.3f} "
          f"mcd {result.get('mcd_db', float('nan')):.2f} wer {result.get('wer', float('nan')):.3f} "
          f"type {result.get('edit_type')} changed {result.get('pct_weights_changed', float('nan')):.4f}% "
          f"F2/M2 identical {result.get('preserved_voices', {}).get('F2', {}).get('identical')}/"
          f"{result.get('preserved_voices', {}).get('M2', {}).get('identical')}")
    for e in result.get("check_errors", []):
        print("  " + e)


if __name__ == "__main__":
    main()
