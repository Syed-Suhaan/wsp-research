"""Stage 2 section of RESULTS.md (used by eval/make_results_v2.py)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "stage2"


def stage2_lines() -> list[str]:
    man = RES / "target" / "manifest.json"
    if not man.exists():
        return []
    m = json.loads(man.read_text())
    items = m["items"]
    lines = ["## Stage 2 — add a new voice style to tiny-voca (in progress)", "",
             "Primary metric: mel-cepstral distortion with DTW (MCD, dB, lower is better). "
             "Secondary: F0 RMSE in cents on voiced aligned frames, and the duration ratio "
             "candidate/target. The Stage 2 claim needs W1 to beat B3 on MCD (95% bootstrap CI).", "",
             "Synthetic target (Task 2.2): F2 output with a fixed DSP style (`results/stage2/target/manifest.json`; "
             "the style values are in the manifest). F2 vs target over the 40 sentences: "
             f"MCD {statistics.mean(i['f2_vs_target']['mcd_db'] for i in items):.2f} dB, "
             f"F0 RMSE {statistics.mean(i['f2_vs_target']['f0_rmse_cents'] for i in items):.0f} cents, "
             f"duration ratio {statistics.mean(i['f2_vs_target']['duration_ratio'] for i in items):.3f}.", ""]
    g = RES / "B3" / "grid.json"
    if g.exists():
        b = json.loads(g.read_text())
        v = b["chosen_val"]
        lines += ["B3 (built-in controls only, no weight edit): chosen on VALIDATION (seeds 0-2) under "
                  "`results/stage2/selection_rule.json`, which was committed before the grid ran: "
                  f"{b['chosen_setting']} (MCD {v['mcd_db']:.3f} dB, F0 RMSE {v['f0_rmse_cents']:.1f} cents, "
                  f"duration ratio {v['duration_ratio']:.3f}). Over the {b['num_settings']}-setting grid "
                  f"the mean MCD ranges from {min(s['mcd_db'] for s in b['settings']):.3f} to "
                  f"{max(s['mcd_db'] for s in b['settings']):.3f} dB. MCD rises as speed falls toward the "
                  "target's rate, because the target keeps F2's normal-speed spectra and the model changes "
                  "its spectra when it speaks slower. MCD-first selection therefore picks a near-default "
                  "setting. Test results: pending.", ""]
    return lines
