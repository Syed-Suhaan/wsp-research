"""Finalize one WSP run: gradient check, re-run the edit, evaluate on TEST, write the run JSON.

This is run by the experiment orchestrator after the writer has finished, never by the writer.

Steps:
    1. Run tools/check_no_gradients.py on every .py file in the run folder. Failure marks the
       run invalid (it is still evaluated and recorded, but excluded from means).
    2. Delete the run's output checkpoint and re-run edit.py in a fresh process, so the
       recorded result comes from the committed script.
    3. Evaluate on the TEST split; compare parameters with the base checkpoint.
    4. Write results/<stage>/<condition>/run_<n>.json.

Usage:
    python eval/wsp_finalize.py --condition W1 --run 0 --writer-model MODEL_ID --effort LEVEL
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.common import environment_info, k_shot_indices, set_determinism, write_json  # noqa: E402
from eval.common import REPO_ROOT, weight_change_stats  # noqa: E402
from eval.evaluate import evaluate_model  # noqa: E402
from models.cnn import load_model  # noqa: E402

BASE_CKPT = REPO_ROOT / "checkpoints" / "stage1_base.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="stage1")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--writer-model", required=True)
    ap.add_argument("--effort", required=True)
    ap.add_argument("--note", default="", help="process notes recorded in the run JSON")
    args = ap.parse_args()

    run_dir = REPO_ROOT / "wsp" / args.stage / args.condition / f"run_{args.run}"
    ckpt = REPO_ROOT / "checkpoints" / f"{args.stage}_{args.condition}_run{args.run}.pt"
    py_files = sorted(run_dir.glob("*.py"))
    check = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "check_no_gradients.py"),
                            *map(str, py_files)], capture_output=True, text=True)
    valid = check.returncode == 0 and (run_dir / "edit.py").exists()

    if ckpt.exists():
        ckpt.unlink()
    t0 = time.time()
    rerun = subprocess.run([sys.executable, str(run_dir / "edit.py")], cwd=REPO_ROOT,
                           capture_output=True, text=True)
    edit_runtime = time.time() - t0
    if rerun.returncode != 0 or not ckpt.exists():
        raise SystemExit(f"edit.py failed:\n{rerun.stdout}\n{rerun.stderr}")

    set_determinism(0)
    base, model = load_model(BASE_CKPT), load_model(ckpt)
    test, base_test = evaluate_model(model, "test"), evaluate_model(base, "test")
    timing = json.loads((run_dir / "timing.json").read_text())
    val_log = run_dir / "val_log.jsonl"
    result = {
        "stage": args.stage, "condition": args.condition, "run": args.run,
        "valid": valid, "gradient_check_output": check.stdout.strip(),
        "uses_gradients": False,
        "writer_model_id": args.writer_model, "effort_level": args.effort,
        "k_shot_val_indices_class9": (k_shot_indices(args.run).tolist()
                                      if args.condition == "W1" else None),
        "new_acc": test["new_acc"], "old_acc": test["old_acc"],
        "old_drop": base_test["old_acc"] - test["old_acc"], "base_old_acc": base_test["old_acc"],
        "per_class": test["per_class"], "new_pred_hist": test["new_pred_hist"],
        "old_pred_as_new": test["old_pred_as_new"],
        "wall_clock_s": timing["end"] - timing["start"],
        "edit_script_runtime_s": edit_runtime,
        "num_val_evaluations": (len(val_log.read_text().splitlines()) if val_log.exists() else 0),
        **weight_change_stats(base, model),
        "process_notes": args.note,
        "environment": environment_info(),
    }
    write_json(REPO_ROOT / "results" / args.stage / args.condition / f"run_{args.run}.json", result)
    print(f"{args.condition} run {args.run}: valid {valid}  new {test['new_acc']:.1f}  "
          f"old {test['old_acc']:.2f}  drop {result['old_drop']:.2f}  "
          f"changed {result['pct_weights_changed']:.3f}%  new params {result['num_new_params']}  "
          f"wall {result['wall_clock_s']:.0f}s")


if __name__ == "__main__":
    main()
