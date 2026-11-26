"""Stage 2 section of RESULTS.md (used by eval/make_results_v2.py).

Reads results/stage2/<condition>/run_<n>.json (TEST, run n = seed n), results/stage2/<cond>/grid.json
(VALIDATION selection), and the rejected DSP pilot in results/stage2_dsp_rejected/.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "stage2"
PILOT = ROOT / "results" / "stage2_dsp_rejected"
ORDER = ["REF_TARGET", "REF_F2", "REF_M2", "B3", "B1", "B2", "W1", "H1"]
DESC = {
    "REF_TARGET": "Target ceiling: the target's own test recordings (reference)",
    "REF_F2": "Nearest built-in voice: F2 unchanged (reference)",
    "REF_M2": "Nearest built-in voice: M2 unchanged (reference)",
    "B3": "Built-in controls only (voice, pitch shift, pitch range), tuned on validation",
    "B1": "Fine-tune the new voice's embedding rows (gradients), tuned on validation",
    "B2": "Fine-tune the whole model (gradients), tuned on validation",
    "W1": "WSP: new voice written without gradients",
    "H1": "Hybrid: W1 run i, then B1's setting",
}
METRICS = [("speaker_sim", "Speaker sim", 3), ("mcd_db", "MCD (dB)", 2), ("f0_rmse_cents", "F0 RMSE (cents)", 0),
           ("duration_ratio", "Duration ratio", 3), ("wer", "WER", 3)]
N_BOOT = 10_000


def runs(cond: str) -> list[dict]:
    d = RES / cond
    return sorted((json.loads(p.read_text()) for p in d.glob("run_*.json")), key=lambda r: r["run"]) if d.exists() else []


def fmt(vals, digits) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    m = statistics.mean(vals)
    if len(vals) == 1:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} ± {statistics.stdev(vals):.{digits}f} [{min(vals):.{digits}f}–{max(vals):.{digits}f}]"


def value(r: dict, key: str):
    """Invalid WSP runs count as similarity 0 and WER 1 in paired comparisons."""
    if not r.get("valid", True):
        return {"speaker_sim": 0.0, "wer": 1.0}.get(key, r.get(key))
    return r.get(key)


def paired_ci(a: list[dict], b: list[dict], key: str, seed: int = 0):
    pa, pb = {r["run"]: r for r in a}, {r["run"]: r for r in b}
    common = sorted(set(pa) & set(pb))
    if len(common) < 8:
        return None
    d = np.array([value(pa[i], key) - value(pb[i], key) for i in common], dtype=float)
    boots = d[np.random.default_rng(seed).integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def ci_str(ci, digits=3) -> str:
    return "—" if ci is None else f"{ci[0]:+.{digits}f} [{ci[1]:+.{digits}f}, {ci[2]:+.{digits}f}]"


def preserved(rs: list[dict]) -> str:
    if not rs or "preserved_voices" not in rs[0]:
        return "—"
    f2 = sum(r["preserved_voices"]["F2"]["identical"] for r in rs)
    m2 = sum(r["preserved_voices"]["M2"]["identical"] for r in rs)
    return f"{f2}/{len(rs)} · {m2}/{len(rs)}"


def pilot_lines() -> list[str]:
    man = PILOT / "target" / "manifest.json"
    if not man.exists():
        return []
    items = json.loads(man.read_text())["items"]
    lines = ["### Rejected pilot: DSP target", "",
             "The first Stage 2 target, a DSP-processed copy of F2, was rejected after a listening check, "
             "before any test evaluation or WSP run (`docs/decision_stage2_target.md`). Its files are in "
             "`results/stage2_dsp_rejected/`. Pilot metrics (superseded): MCD with DTW was the primary "
             "metric. F2 vs that target over 40 sentences: "
             f"MCD {statistics.mean(i['f2_vs_target']['mcd_db'] for i in items):.2f} dB, "
             f"F0 RMSE {statistics.mean(i['f2_vs_target']['f0_rmse_cents'] for i in items):.0f} cents."]
    g = PILOT / "B3" / "grid.json"
    if g.exists():
        b = json.loads(g.read_text())
        lines.append(f"The pilot B3 was chosen on validation ({b['chosen_setting']}) and never evaluated on test.")
    return lines + [""]


def stage2_lines() -> list[str]:
    data = {c: runs(c) for c in ORDER}
    if not any(data.values()):
        return pilot_lines()
    w1, b3 = data["W1"], data["B3"]
    out = ["## Stage 2 — add a new voice to tiny-voca", "",
           "Target: the Kokoro-82M voice am_michael (`docs/decision_stage2_target.md`). The fixed model "
           "`models/tiny_voca3.py` has voices F2, M2, and a new voice 2, which starts as a copy of M2. "
           "Every condition may use only the enrollment (50) and validation (20) audio of the target. "
           "All numbers below are on the 20 TEST sentences, run i = seed i (seeds 0-7). Primary "
           "metric: speaker similarity (WavLM x-vector cosine to the target-voice centroid, higher is "
           "better). Secondary: MCD with DTW and F0 RMSE (lower is better). Guardrail: Whisper "
           "small.en WER. Design decisions: `docs/decisions_stage2.md`.", ""]
    ci = paired_ci(w1, b3, "speaker_sim")
    if ci:
        met = ci[1] > 0
        out += [f"**Stage 2 claim (W1 beats B3 on speaker similarity): {'MET' if met else 'NOT MET'}.** "
                f"W1 − B3 speaker similarity: {ci_str(ci)} (mean, 95% bootstrap CI, 8 paired runs, "
                f"{N_BOOT:,} resamples); the claim requires the CI to lie above 0.", ""]
    else:
        out += ["Stage 2 claim (W1 beats B3 on speaker similarity): pending (needs 8 W1 runs and 8 B3 seeds).", ""]
    out += ["### Stage 2 main table (TEST)", "",
            "Mean ± sample sd [min–max] over runs. 'F2 · M2 unchanged' counts runs whose F2 and M2 "
            "output is bit-identical to the base model (5 test sentences checked per run).", "",
            "'Gap closed' = (condition − B3) / (ceiling − B3) in mean speaker similarity, where the "
            "ceiling is the target's own test recordings (REF_TARGET).", "",
            "| ID | Condition | Runs | " + " | ".join(m[1] for m in METRICS) + " | Gap closed | F2 · M2 unchanged | Weights changed (%) |",
            "|---|---|---|" + "---|" * (len(METRICS) + 3)]
    ceiling = statistics.mean(r["speaker_sim"] for r in data["REF_TARGET"]) if data["REF_TARGET"] else None
    b3_sim = statistics.mean(r["speaker_sim"] for r in b3) if b3 else None
    for c in ORDER:
        rs = data[c]
        if not rs:
            continue
        n_inv = sum(not r.get("valid", True) for r in rs)
        cells = [fmt([r.get(k) for r in rs], d) for k, _, d in METRICS]
        gap = "—"
        if ceiling is not None and b3_sim is not None and ceiling != b3_sim:
            gap = f"{(statistics.mean(value(r, 'speaker_sim') for r in rs) - b3_sim) / (ceiling - b3_sim):+.1%}"
        out.append(f"| {c} | {DESC[c]} | {len(rs)}{f' ({n_inv} invalid)' if n_inv else ''} | " + " | ".join(cells)
                   + f" | {gap} | {preserved(rs)} | {fmt([r.get('pct_weights_changed') for r in rs], 4)} |")
    out += ["", "### Paired comparisons (TEST, 95% bootstrap CI, 8 pairs)", "",
            "| Comparison | Speaker sim | MCD (dB) | F0 RMSE (cents) | WER |", "|---|---|---|---|---|"]
    for a, b in (("W1", "B3"), ("W1", "REF_M2"), ("W1", "B1"), ("W1", "B2"), ("H1", "B1"), ("H1", "W1")):
        if data[a] and data[b]:
            out.append(f"| {a} − {b} | {ci_str(paired_ci(data[a], data[b], 'speaker_sim'))} | "
                       f"{ci_str(paired_ci(data[a], data[b], 'mcd_db'), 2)} | "
                       f"{ci_str(paired_ci(data[a], data[b], 'f0_rmse_cents'), 0)} | "
                       f"{ci_str(paired_ci(data[a], data[b], 'wer'))} |")
    if w1:
        out += ["", "### W1 per run (TEST, seed = run)", "",
                "| Run | Edit type | Tensors changed | Speaker sim | MCD | F0 RMSE | WER | Duration ratio | Val evals | Wall-clock (s) | Valid |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in w1:
            out.append(f"| {r['run']} | {r.get('edit_type')} | {', '.join(r.get('changed_tensors', []))} | "
                       f"{r.get('speaker_sim', 0):.3f} | {r.get('mcd_db', 0):.2f} | {r.get('f0_rmse_cents', 0):.0f} | "
                       f"{r.get('wer', 0):.3f} | {r.get('duration_ratio', 0):.3f} | {r.get('num_val_evaluations')} | "
                       f"{r.get('wall_clock_s', 0):.0f} | {'yes' if r['valid'] else 'NO'} |")
        writers = sorted({(r["writer_model_id"], r["effort_level"]) for r in w1})
        out += ["", "WSP writer: " + "; ".join(f"`{m}` (effort level: {e})" for m, e in writers) +
                ". Session setting: high; probe subagent self-report: reasoning effort 15 on a 0-100 scale "
                "(`results/stage1_v2/effort_probe.json`).", ""]
    man = ROOT / "samples" / "stage2" / "final" / "manifest.json"
    if man.exists():
        mf = json.loads(man.read_text())
        out += ["Listening files (`samples/stage2/final/`): test sentence "
                f"{mf['test_index']} (\"{mf['sentence']}\"), noise seed 0, for F2, M2, the target, B3, B1, "
                f"and the best and worst W1 runs by test speaker similarity (runs {mf['W1_best_run']} and "
                f"{mf['W1_worst_run']}).", ""]
    tune = []
    for c in ("B3", "B1", "B2"):
        g = RES / c / "grid.json"
        if g.exists():
            t = json.loads(g.read_text())
            v = t["chosen_val"]
            tune.append(f"| {c} | {t['num_settings']} | {t['chosen_setting']} | {v['val_speaker_sim']:.3f} | "
                        f"{v['val_mcd_db']:.2f} | {v['val_wer']:.3f} | {t['choice_note']} |")
    if tune:
        out += ["### Tuned baselines (VALIDATION, seeds 0-2)", "",
                "Tuning budget: each baseline used its full grid on validation; each WSP writer had at "
                "most 3 validation evaluations. Rule: `results/stage2/selection_rule.json` (committed before "
                "any grid ran).", "",
                "| Condition | Settings | Chosen | Val speaker sim | Val MCD | Val WER | Note |", "|---|---|---|---|---|---|---|"] + tune + [""]
    return out + pilot_lines()
