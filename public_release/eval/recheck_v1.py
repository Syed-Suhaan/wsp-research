"""Re-check the Stage 1 v1 (pilot) WSP runs against the v2 validity rules.

For every v1 run (W1 and W2, runs 0-4) this applies tools/check_wsp_edit.py:
  * rule 3 and the static part of rule 4 on edit.py, and the "no other code file" check;
  * rule 4 on the output: the v1 checkpoint's state dict is loaded with strict=True into the
    v1 fixed class (models.cnn.ShapeCNN) and compared with the v1 base checkpoint;
  * rule 5 (frozen code) cannot be checked after the fact: v1 did not record a revision per
    run. It is reported as not checked.

The v1 results table is not changed. Output: results/stage1_v1/recheck.json

Usage:
    python eval/recheck_v1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import torch  # noqa: E402

from check_wsp_edit import check_source  # noqa: E402
from models.cnn import ShapeCNN  # noqa: E402

V1_BASE = ROOT / "checkpoints" / "stage1_base.pt"
# Decision recorded at the user's request: v1 W2 run 4 decides class 9 with a pixel rule
# outside the network and changes no tensor, so it is marked invalid.
FORCED_INVALID = {("W2", 4): "pixel rule outside the network; no tensor changed"}


def main():
    base_sd = torch.load(V1_BASE, map_location="cpu", weights_only=False)["state_dict"]
    runs = []
    for cond in ("W1", "W2"):
        for r in range(5):
            run_dir = ROOT / "wsp" / "stage1_v1" / cond / f"run_{r}"
            v1 = json.loads((ROOT / "results" / "stage1_v1" / cond / f"run_{r}.json").read_text())
            errors = [f"edit.py line {n}: {why}" for n, why in check_source((run_dir / "edit.py").read_text())]
            extra = sorted(p.name for p in run_dir.glob("*.py") if p.name != "edit.py")
            if extra:
                errors.append("rule 4: run folder contains code besides edit.py: " + ", ".join(extra))
            ckpt = torch.load(ROOT / "checkpoints" / f"stage1_{cond}_run{r}.pt",
                              map_location="cpu", weights_only=False)
            sd = ckpt["state_dict"]
            if ckpt.get("arch") != "ShapeCNN":
                errors.append(f"rule 4: checkpoint uses a custom model class "
                              f"({ckpt.get('class_name')}), not the fixed class")
            changed = []
            try:
                ShapeCNN(10).load_state_dict(sd, strict=True)
                changed = [k for k, v in sd.items() if not torch.equal(v, base_sd[k])]
                if not changed:
                    errors.append("rule 4: no tensor value differs from the base checkpoint")
            except RuntimeError as e:
                errors.append("rule 4: strict load into the fixed class failed: "
                              + str(e).replace("\n", " ")[:300])
            passes_v2 = not errors
            if (cond, r) in FORCED_INVALID:
                status = "invalid"
                reason = FORCED_INVALID[(cond, r)]
            elif passes_v2:
                status, reason = "valid", "passes the v2 rules (rule 5 not checkable)"
            else:
                status = "valid under v1 rules"
                reason = "the v1 instruction permitted a new layer or module; fails v2 rule 4"
            runs.append({
                "condition": cond, "run": r, "v1_recorded_valid": v1["valid"],
                "passes_v2_rules": passes_v2, "status_after_recheck": status, "reason": reason,
                "errors": errors, "changed_tensors": changed,
                "rule5_frozen_code": "not checked (no per-run revision recorded in v1)",
            })
    out = {"description": "Stage 1 v1 WSP runs re-checked with tools/check_wsp_edit.py "
                          "(v2 rules). The v1 results table is unchanged.",
           "fixed_class_v1": "models.cnn.ShapeCNN", "runs": runs}
    path = ROOT / "results" / "stage1_v1" / "recheck.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    for x in runs:
        print(f"{x['condition']} run {x['run']}: passes v2 {x['passes_v2_rules']!s:5}  "
              f"{x['status_after_recheck']:22}  {'; '.join(x['errors'])[:110]}")


if __name__ == "__main__":
    main()
