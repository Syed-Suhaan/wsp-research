"""Aggregate results/<stage>/<condition>/run_*.json into RESULTS.md.

For every condition the table reports mean, sample standard deviation (sd), and range
(min-max) over runs. Invalid WSP runs (failed gradient check) are listed but excluded from
the means.

Usage:
    python eval/make_results.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.common import REPO_ROOT  # noqa: E402

STAGE1_ORDER = ["base", "B0", "B0s", "B1", "B2", "W1", "W2"]
STAGE1_DESC = {
    "base": "Base model (class-9 row zero)",
    "B0": "Imprinting, unit-norm row, zero bias",
    "B0s": "Imprinting, norm-matched row and mean bias (supplementary)",
    "B1": "Fine-tune class-9 row only (gradients)",
    "B2": "Fine-tune whole model with replay (gradients)",
    "W1": "WSP with k = 5 examples",
    "W2": "WSP from text description only",
}


def load_runs(stage: str, cond: str) -> list[dict]:
    d = REPO_ROOT / "results" / stage / cond
    return [json.loads(p.read_text()) for p in sorted(d.glob("run_*.json"))] if d.exists() else []


def fmt(values: list[float], digits: int = 2) -> str:
    if not values:
        return "—"
    m = statistics.mean(values)
    if len(values) == 1:
        return f"{m:.{digits}f}"
    sd = statistics.stdev(values)
    return f"{m:.{digits}f} ± {sd:.{digits}f} [{min(values):.{digits}f}–{max(values):.{digits}f}]"


def summarize(stage: str, cond: str) -> dict | None:
    runs = load_runs(stage, cond)
    if not runs:
        return None
    valid = [r for r in runs if r.get("valid", True)]
    if cond == "base":
        return {"n": f"{len(runs)}", "new": fmt([r["test_new_acc"] for r in runs]),
                "old": fmt([r["test_old_acc"] for r in runs]), "drop": "0",
                "time": fmt([r["wall_clock_s"] for r in runs], 2), "pct": "—", "newp": "—",
                "valid": valid, "runs": runs}
    return {
        "n": f"{len(valid)}" + (f" (+{len(runs) - len(valid)} invalid)" if len(valid) < len(runs) else ""),
        "new": fmt([r["new_acc"] for r in valid]),
        "old": fmt([r["old_acc"] for r in valid]),
        "drop": fmt([r["old_drop"] for r in valid]),
        "time": fmt([r["wall_clock_s"] for r in valid], 2),
        "pct": fmt([r["pct_weights_changed"] for r in valid], 3),
        "newp": fmt([float(r["num_new_params"]) for r in valid], 0),
        "valid": valid, "runs": runs,
    }


def mean_of(s: dict, key: str) -> float:
    return statistics.mean(r[key] for r in s["valid"])


def gate_text(s1: dict) -> list[str]:
    b0, w1, w2 = s1.get("B0"), s1.get("W1"), s1.get("W2")
    lines = ["## Stage 1 gate", ""]
    if not (b0 and w1 and w1["valid"]):
        return lines + ["Gate not yet evaluated (W1 results pending).", ""]
    w1_new, b0_new, w1_drop = mean_of(w1, "new_acc"), mean_of(b0, "new_acc"), mean_of(w1, "old_drop")
    c1, c2 = w1_new >= b0_new, w1_drop <= 2.0
    verdict = "PASS" if (c1 and c2) else "FAIL"
    lines += [f"**{verdict}**", "",
              f"- W1 mean new-class accuracy {w1_new:.2f}% vs B0 mean {b0_new:.2f}%: "
              f"{'met' if c1 else 'not met'} (W1 ≥ B0).",
              f"- W1 mean old-class drop {w1_drop:.2f} points: {'met' if c2 else 'not met'} (≤ 2 points)."]
    if w2 and w2["valid"]:
        w2_new = mean_of(w2, "new_acc")
        lines.append(f"- W2 (text only) mean new-class accuracy {w2_new:.2f}%: "
                     f"{'better' if w2_new > 10 else 'not better'} than chance (10%).")
    else:
        lines.append("- W2 pending.")
    return lines + [""]


def main():
    s1 = {c: summarize("stage1", c) for c in STAGE1_ORDER}
    s1 = {c: v for c, v in s1.items() if v}
    out = ["# Results", ""]
    out += gate_text(s1)
    out += ["## Stage 1 — add class 9 (heart) to a 9-class shape CNN", "",
            "Test split: 500 images per class. Accuracies in %, argmax over all 10 outputs. "
            "Cells show mean ± sample sd [min–max] over runs (seeds 0–4 for baselines, "
            "runs 0–4 for WSP). Old-class drop = base old-class accuracy − condition old-class "
            "accuracy (points). Time is the wall-clock time of the condition itself "
            "(for WSP: the writer's full session, reading to saved checkpoint).", "",
            "| ID | Condition | Runs | New-class acc | Old-class acc | Old-class drop | "
            "Wall-clock (s) | Weights changed (%) | New params |",
            "|---|---|---|---|---|---|---|---|---|"]
    for c, s in s1.items():
        out.append(f"| {c} | {STAGE1_DESC[c]} | {s['n']} | {s['new']} | {s['old']} | {s['drop']} | "
                   f"{s['time']} | {s['pct']} | {s['newp']} |")
    invalid = [(c, r) for c, s in s1.items() for r in s["runs"] if not r.get("valid", True)]
    if invalid:
        out += ["", "Invalid WSP runs (failed `tools/check_no_gradients.py`, excluded from means):"]
        out += [f"- {c} run {r['run']}" for c, r in invalid]
    writers = sorted({(r["writer_model_id"], r["effort_level"]) for c in ("W1", "W2") if c in s1
                      for r in s1[c]["runs"]})
    if writers:
        out += ["", "WSP writer: " + "; ".join(f"`{m}` (effort: {e})" for m, e in writers)]
    (REPO_ROOT / "RESULTS.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
