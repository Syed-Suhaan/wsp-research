"""Finalize one Stage 1 v2 WSP run (run by the orchestrator after the writer finishes).

Steps:
    1. Delete the run's output state dict and re-run edit.py in a fresh process (TMPDIR set to
       the run's scratch folder), so the recorded result comes from the committed script.
    2. Validity: tools/check_wsp_edit.py (rules 3, 4, and 5; rule 5 uses --git-rev, the
       commit at the start of the run).
    3. Evaluate on the TEST split (eval/evaluate_v2.py) and compute edit statistics.
    4. Write results/stage1_v2/<condition>/run_<n>.json.

Usage:
    python eval/wsp_finalize_v2.py --condition W1 --run 0 --git-rev <commit> \
        --orchestrator-start <unix s> --orchestrator-end <unix s>
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
from data.shapes_v2 import k_shot_indices, load_config  # noqa: E402
from eval.common import environment_info, set_determinism, write_json  # noqa: E402
from eval.evaluate_v2 import edit_stats, evaluate_test  # noqa: E402
from models.shapes_v2 import load_state  # noqa: E402

BASE = ROOT / "checkpoints" / "stage1_v2_base.pt"
WRITER_MODEL_ID = "claude-opus-5-5"


def files_read(rationale: str) -> list[str]:
    """Bullet or plain lines of the 'Files read' section of rationale.md."""
    m = re.search(r"^#+\s*Files read\s*$(.*?)(?=^#+\s|\Z)", rationale, flags=re.M | re.S | re.I)
    if not m:
        return []
    items = []
    for line in m.group(1).splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        if line:
            items.append(line)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["W1", "W2"])
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--git-rev", required=True)
    ap.add_argument("--orchestrator-start", type=float)
    ap.add_argument("--orchestrator-end", type=float)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    cfg = load_config()
    run_dir = ROOT / "wsp" / "stage1_v2" / args.condition / f"run_{args.run}"
    out = ROOT / "checkpoints" / f"stage1_v2_{args.condition}_run{args.run}.pt"

    if out.exists():
        out.unlink()
    env = dict(os.environ, TMPDIR=str(run_dir / "scratch"))
    (run_dir / "scratch").mkdir(exist_ok=True)
    t0 = time.time()
    rerun = subprocess.run([sys.executable, str(run_dir / "edit.py")], cwd=ROOT, env=env,
                           capture_output=True, text=True)
    edit_runtime = time.time() - t0

    check = check_run(run_dir, out, BASE, args.git_rev) if out.exists() else {
        "valid": False, "errors": [f"edit.py did not produce {out.name} "
                                   f"(exit {rerun.returncode}): {rerun.stderr[-500:]}"]}
    result = {"stage": "stage1_v2", "condition": args.condition, "run": args.run,
              "valid": bool(check["valid"]), "check_errors": check["errors"],
              "uses_gradients": False, "writer_model_id": WRITER_MODEL_ID,
              "effort_level": "unverified",
              "effort_observations": json.loads(
                  (ROOT / "results" / "stage1_v2" / "effort_probe.json").read_text())["observations"],
              "k": cfg["k"],
              "k_shot_val_indices_class9": (k_shot_indices(args.run, cfg["k"]).tolist()
                                            if args.condition == "W1" else None),
              "git_rev_at_start": args.git_rev}
    if out.exists():
        set_determinism(0)
        base = load_state(BASE)
        try:
            model = load_state(out)
            test, base_test = evaluate_test(model, cfg), evaluate_test(base, cfg)
            result.update(test)
            result["old_drop"] = base_test["old_acc"] - test["old_acc"]
            result["base_old_acc"] = base_test["old_acc"]
            result.update(edit_stats(base.state_dict(),
                                     torch.load(out, map_location="cpu", weights_only=True)))
        except Exception as e:  # noqa: BLE001  (an invalid output cannot be evaluated)
            result["evaluation_error"] = str(e)[:500]
            result.update({"new_acc": 0.0, "precision9": 0.0, "edit_type": "invalid"})
    timing = json.loads((run_dir / "timing.json").read_text())
    rationale = (run_dir / "rationale.md").read_text() if (run_dir / "rationale.md").exists() else ""
    val_log = run_dir / "val_log.jsonl"
    result.update({
        "wall_clock_s": timing["end"] - timing["start"],
        "orchestrator_wall_clock_s": (args.orchestrator_end - args.orchestrator_start
                                      if args.orchestrator_start and args.orchestrator_end else None),
        "edit_script_runtime_s": edit_runtime,
        "num_val_evaluations": len(val_log.read_text().splitlines()) if val_log.exists() else 0,
        "files_read": files_read(rationale),
        "process_notes": args.note,
        "environment": environment_info(),
    })
    write_json(ROOT / "results" / "stage1_v2" / args.condition / f"run_{args.run}.json", result)
    print(f"{args.condition} run {args.run}: valid {result['valid']}  "
          f"new {result.get('new_acc', 0):.1f}  prec {result.get('precision9', 0):.1f}  "
          f"fpr near {result.get('heldout_fpr_near', float('nan')):.1f} "
          f"far {result.get('heldout_fpr_far', float('nan')):.1f}  "
          f"old {result.get('old_acc', float('nan')):.2f} drop {result.get('old_drop', float('nan')):.2f}  "
          f"{result.get('edit_type')}  changed {result.get('pct_weights_changed', float('nan')):.3f}%  "
          f"slot {result.get('slot_params_used')}  evals {result['num_val_evaluations']}  "
          f"wall {result['wall_clock_s']:.0f}s")
    for e in result["check_errors"]:
        print("  " + e)


if __name__ == "__main__":
    main()
