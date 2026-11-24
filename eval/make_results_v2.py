"""Build RESULTS.md for Stage 1 v2 (with the frozen Stage 1 v1 pilot table appended).

Reads results/stage1_v2/<condition>/run_*.json, results/stage1_v2/calibration.json, and
results/stage1_v1/RESULTS_v1.md + recheck.json.

Gate (W1 run i paired with B0 seed i, i = 0-7; 10,000 bootstrap resamples of the 8 pairs,
95% percentile intervals; invalid WSP runs count as 0% new-class accuracy and 0% precision):
    PASS if  CI lower bound of mean(W1 - B0) new-class accuracy >= -2 points
        and  CI lower bound of mean(W1 - B0) class-9 precision >= -2 points
        and  mean W1 old-class drop <= 2 points.
The same numbers are reported for W2 vs B0, W2 new-class accuracy vs chance (10%), and
(without affecting the gate) paired CIs for W1 - B1 and W2 - B1.

Usage:
    python eval/make_results_v2.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

RES = ROOT / "results" / "stage1_v2"
ORDER = ["base", "B0", "B0s", "B0b", "B1", "B1t", "B2", "B2t", "W1", "W2"]
DESC = {
    "base": "Base model (class-9 row, bias, adapter zero)",
    "B0": "Imprinting: unit-norm mean feature, bias 0",
    "B0s": "Imprinting, norm-matched row and mean bias (supplementary)",
    "B1": "Fine-tune class-9 row and bias (gradients)",
    "B2": "Fine-tune whole model with replay (gradients)",
    "B0b": "B0 with class-9 bias tuned on validation (v2.1)",
    "B1t": "B1 tuned on validation: lr, steps, negatives (v2.1)",
    "B2t": "B2 tuned on validation: lr, steps, positive weight (v2.1)",
    "W1": "WSP with the k examples",
    "W2": "WSP from a text description only",
}
N_PAIRS = 8
N_BOOT = 10_000
METRICS = [("new_acc", "New-class acc"), ("precision9", "Class-9 precision"),
           ("heldout_fpr_near", "Held-out FPR near"), ("heldout_fpr_far", "Held-out FPR far"),
           ("heldout_fpr", "Held-out FPR all"), ("old_acc", "Old-class acc"),
           ("old_drop", "Old-class drop"), ("wall_clock_s", "Wall-clock (s)"),
           ("pct_weights_changed", "Weights changed (%)"), ("slot_params_used", "Slot params used")]


def runs(cond: str) -> list[dict]:
    d = RES / cond
    return [json.loads(p.read_text()) for p in sorted(d.glob("run_*.json"),
                                                      key=lambda p: int(p.stem.split("_")[1]))] \
        if d.exists() else []


def fmt(vals, digits=2) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    m = statistics.mean(vals)
    if len(vals) == 1:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} ± {statistics.stdev(vals):.{digits}f} [{min(vals):.{digits}f}–{max(vals):.{digits}f}]"


def gate_value(r: dict, key: str):
    """Per-run value used in gate and CI computations (invalid runs: 0 for accuracy/precision)."""
    if not r.get("valid", True) and key in ("new_acc", "precision9"):
        return 0.0
    return r.get(key)


def paired_ci(a: list[dict], b: list[dict], key: str, seed: int = 0) -> tuple[float, float, float] | None:
    pa = {r["run"]: r for r in a}
    pb = {r["run"]: r for r in b}
    common = sorted(set(pa) & set(pb))
    if len(common) < N_PAIRS:
        return None
    d = np.array([gate_value(pa[i], key) - gate_value(pb[i], key) for i in common], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def one_sample_ci(a: list[dict], key: str, seed: int = 0):
    v = np.array([gate_value(r, key) for r in a], dtype=float)
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def ci_str(ci) -> str:
    return "—" if ci is None else f"{ci[0]:+.2f} [{ci[1]:+.2f}, {ci[2]:+.2f}]"


def base_row() -> dict | None:
    cache = RES / "base" / "test_eval.json"
    if not (RES / "base" / "run_0.json").exists():
        return None
    if not cache.exists():
        from eval.evaluate_v2 import evaluate_test
        from models.shapes_v2 import load_state
        t = evaluate_test(load_state(ROOT / "checkpoints" / "stage1_v2_base.pt"))
        cache.write_text(json.dumps(t, indent=2) + "\n")
    t = json.loads(cache.read_text())
    b = json.loads((RES / "base" / "run_0.json").read_text())
    return {**t, "old_drop": 0.0, "wall_clock_s": b["wall_clock_s"], "pct_weights_changed": None,
            "slot_params_used": 0, "valid": True, "run": 0}


def tuning_lines() -> list[str]:
    rule_path = ROOT / "results" / "stage1_v2_1" / "selection_rule.json"
    if not rule_path.exists():
        return []
    lines = ["### Tuned baselines (Stage 1 v2.1)", "",
             "Tuning budget: each tuned baseline used its full grid on the VALIDATION split "
             "(every setting x seeds 0-7, with the k examples excluded from the class-9 validation "
             "images). Each WSP writer had at most 3 validation evaluations of complete candidate "
             "edits. One global setting per condition was chosen by the mean of new-class accuracy "
             "and class-9 precision, subject to an old-class drop of at most 2 points. The rule "
             "(`results/stage1_v2_1/selection_rule.json`) was committed before any grid result "
             "existed. Only the chosen setting was evaluated on TEST.", "",
             "| Condition | Grid | Settings | Chosen setting | Val new acc | Val precision | Val old drop | Val score |",
             "|---|---|---|---|---|---|---|---|"]
    for c in ("B0b", "B1t", "B2t"):
        p = ROOT / "results" / "stage1_v2_1" / f"tuning_{c}.json"
        if not p.exists():
            lines.append(f"| {c} | pending | | | | | | |")
            continue
        t = json.loads(p.read_text())
        g = "; ".join(f"{k} {v}" for k, v in t["grid"].items())
        cv = t["chosen_val"]
        lines.append(f"| {c} | {g} | {t['num_settings']} | "
                     f"{', '.join(f'{k}={v}' for k, v in t['chosen_params'].items())} | "
                     f"{cv['val_new_acc']:.2f} | {cv['val_precision9']:.2f} | {cv['val_old_drop']:.2f} | "
                     f"{cv['score']:.2f} |")
        if "NO setting" in t["choice_note"]:
            lines.append(f"| | {t['choice_note']} | | | | | | |")
    return lines + [""]


def calibration_lines() -> list[str]:
    cal = json.loads((RES / "calibration.json").read_text())
    att = cal["attempts"]
    chosen = [a for a in att if a["in_target_range"]]
    lowest = min(att, key=lambda a: a["b0_val_new_acc_mean"])
    lines = ["## Calibration disclosure", "",
             f"The Stage 1 v2 setup was chosen by a difficulty calibration over {len(att)} "
             f"attempts (`results/stage1_v2/calibration.json`). The goal was a B0 mean new-class "
             "accuracy between 60% and 80% on the VALIDATION split (seeds 0-7). The selected setup "
             f"(attempt {chosen[0]['attempt']}) has the lowest B0 validation accuracy of all "
             f"attempts ({lowest['b0_val_new_acc_mean']:.1f}%). All other attempts scored "
             f"{min(a['b0_val_new_acc_mean'] for a in att if a is not lowest):.1f}-"
             f"{max(a['b0_val_new_acc_mean'] for a in att if a is not lowest):.1f}%.",
             "",
             "Selection rule: apply the levers in order and stop at the first setup in range. "
             "(1) k = 2 instead of 5; (2) a new class closer to an old class (rounded square, then "
             "hexagon); (3) more noise, blur, and distortion, with a retrained base model. Lever 3 "
             "was tried at increasing strength. At each strength the new classes were tried in "
             "the order octagon, rounded square, hexagon, with k = 2, and the first setup in range "
             "was taken. The base model had to keep old-class test accuracy of at least 95%. "
             "This rule was fixed before the lever-3 results were seen.", "",
             "| Attempt | Lever | New class | k | Noise | Blur | Distortion | Base old test acc | "
             "B0 val new acc | In range |", "|---|---|---|---|---|---|---|---|---|---|"]
    for a in att:
        c = a["config"]
        lines.append(f"| {a['attempt']} | {a['lever']} | {c['new_class']} | {c['k']} | "
                     f"{c['noise_sigma'][0]}–{c['noise_sigma'][1]} | {c['blur_sigma'][0]}–{c['blur_sigma'][1]} | "
                     f"{c['distortion']} | {a['base_test_old_acc']:.2f} | "
                     f"{a['b0_val_new_acc_mean']:.1f} ± {a['b0_val_new_acc_sd']:.1f} | "
                     f"{'yes' if a['in_target_range'] else 'no'} |")
    return lines + [""]


def main():
    data = {c: runs(c) for c in ORDER if c != "base"}
    b = base_row()
    if b:
        data["base"] = [b]
    b0, b1, w1, w2 = data.get("B0", []), data.get("B1", []), data.get("W1", []), data.get("W2", [])
    out = ["# Results", "", "## Stage 1 v2 gate", ""]

    def gate_block(w, name):
        lines = []
        ci_new = paired_ci(w, b0, "new_acc")
        ci_prec = paired_ci(w, b0, "precision9")
        drop = statistics.mean(r["old_drop"] for r in w) if len(w) == N_PAIRS else None
        lines += [f"- {name} − B0 new-class accuracy: mean {ci_str(ci_new)} (95% CI); "
                  f"criterion lower bound ≥ −2: {'met' if ci_new and ci_new[1] >= -2 else 'not met'}.",
                  f"- {name} − B0 class-9 precision: mean {ci_str(ci_prec)} (95% CI); "
                  f"criterion lower bound ≥ −2: {'met' if ci_prec and ci_prec[1] >= -2 else 'not met'}.",
                  f"- {name} mean old-class drop: {drop:.2f} points; criterion ≤ 2: "
                  f"{'met' if drop is not None and drop <= 2 else 'not met'}." if drop is not None
                  else f"- {name} mean old-class drop: pending."]
        ok = bool(ci_new and ci_prec and drop is not None and ci_new[1] >= -2 and ci_prec[1] >= -2 and drop <= 2)
        return lines, ok

    if len(w1) == N_PAIRS and len(b0) == N_PAIRS:
        lines, ok = gate_block(w1, "W1")
        out += [f"**{'PASS' if ok else 'FAIL'}** (W1 vs B0, {N_PAIRS} paired runs, "
                f"{N_BOOT:,} bootstrap resamples)", ""] + lines
    else:
        out += ["Gate pending (needs 8 W1 runs and 8 B0 seeds)."]
    out += [""]
    if len(w2) == N_PAIRS and len(b0) == N_PAIRS:
        lines, ok = gate_block(w2, "W2")
        ch = one_sample_ci(w2, "new_acc")
        out += ["Same criteria for W2 vs B0 (reported, not part of the gate): "
                f"{'all met' if ok else 'not all met'}.", ""] + lines + [
                f"- W2 new-class accuracy vs chance (10%): mean {ch[0]:.2f}% "
                f"[{ch[1]:.2f}, {ch[2]:.2f}] (95% CI); "
                f"{'above' if ch[1] > 10 else 'not clearly above'} chance.", ""]
    invalid = [(c, r["run"]) for c in ("W1", "W2") for r in data.get(c, []) if not r.get("valid", True)]
    out += ["Invalid WSP runs (counted as 0% new-class accuracy and 0% precision in the gate "
            "and in all CIs): " + (", ".join(f"{c} run {i}" for c, i in invalid) if invalid else "none") + ".", ""]

    out += ["### Paired comparisons with other baselines (not part of the gate)", "",
            "Mean difference with 95% bootstrap CI (10,000 resamples) over the 8 pairs (run i with "
            "seed i). A positive old-class-drop difference means the WSP condition loses more "
            "old-class accuracy. B1t, B2t, and B0b are the v2.1 tuned baselines.", "",
            "| Comparison | New-class acc | Class-9 precision | Old-class drop |", "|---|---|---|---|"]
    comparisons = [(w1, "W1", "B1"), (w2, "W2", "B1"), (w1, "W1", "B1t"), (w1, "W1", "B2t"),
                   (w1, "W1", "B0b"), (w2, "W2", "B1t")]
    for w, wname, bname in comparisons:
        bb = data.get(bname, [])
        out.append(f"| {wname} − {bname} | {ci_str(paired_ci(w, bb, 'new_acc'))} | "
                   f"{ci_str(paired_ci(w, bb, 'precision9'))} | {ci_str(paired_ci(w, bb, 'old_drop'))} |")
    out += [""]
    out += tuning_lines()

    out += calibration_lines()

    out += ["## Stage 1 v2 — main table (TEST split)", "",
            "Mean ± sample sd [min–max] over runs (baselines: seeds 0-7; WSP: runs 0-7; run i uses "
            "the same k examples as seed i). Accuracies and rates in %. Held-out FPR is the "
            "fraction of held-out images predicted as class 9; 'near' is the held-out shape close "
            "to the new class, 'far' the other one. The means include invalid WSP runs at their "
            "measured values; the gate counts them as 0.", "",
            "| ID | Condition | Runs | " + " | ".join(m[1] for m in METRICS) + " |",
            "|---|---|---|" + "---|" * len(METRICS)]
    for c in ORDER:
        rs = data.get(c)
        if not rs:
            continue
        cells = []
        for key, _ in METRICS:
            digits = 3 if key == "pct_weights_changed" else (0 if key == "slot_params_used" else 2)
            cells.append(fmt([r.get(key) for r in rs], digits))
        n_inv = sum(not r.get("valid", True) for r in rs)
        out.append(f"| {c} | {DESC[c]} | {len(rs)}{f' ({n_inv} invalid)' if n_inv else ''} | " + " | ".join(cells) + " |")
    out += [""]

    for c in ("W1", "W2"):
        rs = data.get(c, [])
        if not rs:
            continue
        out += [f"### {c} per run", "",
                "| Run | Edit type | New acc | Precision | FPR near | FPR far | Old acc | Old drop | "
                "Weights changed (%) | Slot params | Val evals | Wall-clock (s) | Valid |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rs:
            out.append(f"| {r['run']} | {r['edit_type']} | {r['new_acc']:.1f} | {r['precision9']:.1f} | "
                       f"{r['heldout_fpr_near']:.1f} | {r['heldout_fpr_far']:.1f} | {r['old_acc']:.2f} | "
                       f"{r['old_drop']:.2f} | {r['pct_weights_changed']:.3f} | {r['slot_params_used']} | "
                       f"{r['num_val_evaluations']} | {r['wall_clock_s']:.0f} | "
                       f"{'yes' if r['valid'] else 'NO'} |")
        out += [""]
    if w1 or w2:
        out += ["### WSP edit types", "", "| Condition | Edit type | Runs | Mean new-class acc | Mean class-9 precision |",
                "|---|---|---|---|---|"]
        for c in ("W1", "W2"):
            types = sorted({r["edit_type"] for r in data.get(c, [])})
            for t in types:
                rs = [r for r in data[c] if r["edit_type"] == t]
                out.append(f"| {c} | {t} | {len(rs)} | {statistics.mean(r['new_acc'] for r in rs):.2f} | "
                           f"{statistics.mean(r['precision9'] for r in rs):.2f} |")
        out += [""]
        writers = sorted({(r["writer_model_id"], r["effort_level"]) for c in ("W1", "W2") for r in data.get(c, [])})
        probe = json.loads((RES / "effort_probe.json").read_text())
        out += ["WSP writer: " + "; ".join(f"`{m}` (effort level: {e})" for m, e in writers) +
                ". The effort level could not be confirmed from a record outside the model "
                "(`results/stage1_v2/effort_probe.json`). Recorded observations:", ""]
        out += [f"- {o['source']}: {o['value']}" for o in probe["observations"]] + [""]

    sanity = ROOT / "results" / "stage1_v2_1" / "b2_sanity.json"
    extra = []
    if sanity.exists():
        b = json.loads(sanity.read_text())
        extra = [f"- **B2 pipeline check (diagnostic, not a result).** With the B2t setting but "
                 f"200 class-9 training images as positives (seed 0), B2 reaches "
                 f"{b['val_new_acc']:.1f}% class-9 validation accuracy "
                 f"(`results/stage1_v2_1/b2_sanity.json`). The pipeline learns the class when it "
                 f"has enough positives, so B2's near-zero accuracy at k = 2 is a few-shot "
                 f"failure, not a pipeline bug."]
    out += ["### Protocol notes (Stage 1 v2)", "", *NOTES, *extra, ""]

    v1 = (ROOT / "results" / "stage1_v1" / "RESULTS_v1.md").read_text().splitlines()
    out += ["---", "", "# Stage 1 v1 (pilot, frozen)", "",
            "The v1 pilot table below is unchanged. A later re-check with the v2 rules "
            "(`results/stage1_v1/recheck.json`) marks v1 W2 run 4 invalid (pixel rule outside "
            "the network, no tensor changed). Six other v1 WSP runs used a new layer or module, "
            "which the v1 instruction allowed and the v2 rules do not. The v1 table still "
            "includes run 4 as it was recorded.", ""]
    out += [("#" + ln) if ln.startswith("#") else ln for ln in v1[1:]]
    (ROOT / "RESULTS.md").write_text("\n".join(out).rstrip() + "\n")
    print("\n".join(out[:40]))


NOTES = [
    "- **Fixed model.** Every condition is evaluated by loading a state dict into "
    "`models/shapes_v2.ShapesV2Net` with strict=True. Edits can change behaviour only through "
    "tensor values (head, a zero-initialised adapter slot A2(ReLU(A1(f))), or the backbone).",
    "- **B1 negatives.** Only the class-9 head row and bias are trained. Each step uses the k "
    "examples as positives and 59 replayed class 0-8 training images as negatives.",
    "- **B2 underfits the new class.** B2 uses the hyperparameters carried over unchanged "
    "from v1 (200 Adam steps, lr 3e-4, 2 positives and 59 replayed images per step). With "
    "k = 2 this barely moves the class-9 logit. They were not tuned after the v2 results "
    "were seen. B2 is not part of the gate.",
    "- **W1 class name.** The W1 instruction calls the new class 'class 9' and gives only the k "
    "example images, not the shape's name. W2 gets only a text description of the shape, with "
    "no images.",
    "- **WSP isolation.** WSP runs ran one at a time, each as a writer subagent with a fresh "
    "context. Each writer had its own scratch folder (`wsp/stage1_v2/<condition>/run_<n>/scratch/`, "
    "also its TMPDIR) and a fixed instruction (`prompts/stage1_v2_<condition>.md`). Writers read "
    "data only through `eval/wsp_tools.py` (base checkpoint, class 0-8 validation images, and "
    "for W1 the run's k class-9 validation images). Each writer lists the files it read in "
    "`rationale.md`. The project instruction file may be loaded into a writer's context "
    "automatically by the agent harness. It does not describe the held-out classes or the "
    "test set.",
    "- **Instruction revision and aborted run.** The agent harness loads the project "
    "instruction file into every writer's context automatically. That file describes the v1 "
    "pilot, where class 9 was a 'heart'. The first W1 writer (instruction revision 1) "
    "therefore called class 9 'hearts'. Before its test results were seen, both W1 and W2 "
    "instructions were revised (revision 2) to tell writers not to use that file. That run "
    "was kept as aborted (`wsp/stage1_v2_aborted/`, `results/stage1_v2_aborted/`, excluded "
    "from all tables and the gate), and W1 restarted from run 0. All 16 counted runs used "
    "revision 2.",
    "- **Residual pilot-name priming.** Even under revision 2, five W1 writers (runs 0, 2, 3, "
    "6, 7) used the word 'heart' for the new class in their edit script, rationale, or scratch "
    "scripts (`pilot_class_name_mentions` in each run JSON). No W2 writer did. The pilot "
    "name is wrong (the new class is a hexagon) and gives no correct information. All W1 "
    "edits were derived from the k examples and class 0-8 statistics. W1's 'no class name' "
    "condition is still not clean.",
    "- **Validity.** Each run is checked by `tools/check_wsp_edit.py`: no gradients, fitting, "
    "or candidate-scoring loop; nothing but tensor values changed; strict load into the fixed "
    "class; at least one changed tensor; no change under models/, eval/, data/ since the run "
    "started (git).",
    "- **Held-out classes.** Two shape classes appear only in the test split (500 images each): "
    "one visually near the new class, one far from it. They were never shown to any model, "
    "baseline, or writer. One of them was replaced before any v2 condition ran "
    "(`results/stage1_v2/heldout_note.json`).",
]


if __name__ == "__main__":
    main()
