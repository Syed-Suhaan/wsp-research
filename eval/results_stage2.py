"""Stage 2 section of RESULTS.md (used by eval/make_results_v2.py)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "stage2_dsp_rejected"


def stage2_lines() -> list[str]:
    man = RES / "target" / "manifest.json"
    if not man.exists():
        return []
    m = json.loads(man.read_text())
    items = m["items"]
    lines = ["## Stage 2 — add a new voice to tiny-voca (in progress)", "",
             "The first Stage 2 target, a DSP-processed copy of F2, was rejected after a listening "
             "check, before any test evaluation or WSP run (`docs/decision_stage2_target.md`). The new "
             "target is the Kokoro-82M voice am_michael. The rejected pilot is summarised below for the "
             "record; its files are in `results/stage2_dsp_rejected/`.", "",
             "### Rejected pilot: DSP target", "",
             "Pilot metrics (superseded): mel-cepstral distortion with DTW (MCD, dB) as primary, F0 "
             "RMSE in cents and the duration ratio as secondary. The current Stage 2 uses speaker "
             "similarity as its primary metric.", "",
             "Synthetic target (Task 2.2): F2 output with a fixed DSP style (`results/stage2_dsp_rejected/target/manifest.json`; "
             "the style values are in the manifest). F2 vs target over the 40 sentences: "
             f"MCD {statistics.mean(i['f2_vs_target']['mcd_db'] for i in items):.2f} dB, "
             f"F0 RMSE {statistics.mean(i['f2_vs_target']['f0_rmse_cents'] for i in items):.0f} cents, "
             f"duration ratio {statistics.mean(i['f2_vs_target']['duration_ratio'] for i in items):.3f}.", ""]
    g = RES / "B3" / "grid.json"
    if g.exists():
        b = json.loads(g.read_text())
        v = b["chosen_val"]
        lines += ["B3 (built-in controls only, no weight edit): chosen on VALIDATION (seeds 0-2) under "
                  "`results/stage2_dsp_rejected/selection_rule.json`, which was committed before the grid ran: "
                  f"{b['chosen_setting']} (MCD {v['mcd_db']:.3f} dB, F0 RMSE {v['f0_rmse_cents']:.1f} cents, "
                  f"duration ratio {v['duration_ratio']:.3f}). Over the {b['num_settings']}-setting grid "
                  f"the mean MCD ranges from {min(s['mcd_db'] for s in b['settings']):.3f} to "
                  f"{max(s['mcd_db'] for s in b['settings']):.3f} dB. MCD rises as speed falls toward the "
                  "target's rate, because the target keeps F2's normal-speed spectra and the model changes "
                  "its spectra when it speaks slower. MCD-first selection therefore picks a near-default "
                  "setting. This B3 was never evaluated on test.", ""]
    return lines
