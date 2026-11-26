"""Diagnostic (not a result): can the Stage 2 B1 pipeline move the new voice at all?

Uses a target the model can certainly reach: F2's own synthesized audio for the 50 enrollment
sentences (noise seed 0). Starting from the base (new voice = copy of M2), B1 is trained with
lr 1e-2 for 100 steps (seed 0). If the pipeline works, the loss falls and voice 2 moves toward
F2: its median F0 rises and its MCD to F2's audio falls. Measured on 5 validation sentences.

Output: results/stage2/b1_sanity.json
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from eval.stage2_audio import compare, world_features  # noqa: E402
from eval.stage2_metrics import sentences, synthesize  # noqa: E402
from models.tiny_voca3 import load_state  # noqa: E402
from train.stage2_finetune import train  # noqa: E402


def median_f0(w):
    import pyworld as pw
    f, _ = pw.harvest(np.ascontiguousarray(w), 24000)
    return float(np.median(f[f > 0]))


def describe(model, refs, texts):
    outs = [synthesize(model, t, 2, 0) for t in texts]
    return {"median_f0_hz": statistics.mean(median_f0(o) for o in outs),
            "mcd_to_F2_db": statistics.mean(compare(o, world_features(r), len(r))["mcd_db"] for o, r in zip(outs, refs))}


def main():
    torch.set_num_threads(4)
    base = load_state()
    enroll = sentences()["enroll"]
    f2_targets = [torch.from_numpy(synthesize(base, t, 0, 0)).float() for t in enroll]
    val = sentences()["val"][:5]
    refs = [synthesize(base, t, 0, 0) for t in val]
    before = describe(base, refs, val)
    model = load_state()
    losses = []
    snaps = train(model, "voice", 1e-2, 100, 0, [100], targets=f2_targets, loss_log=losses)
    model.load_state_dict(snaps[100])
    after = describe(model, refs, val)
    f2 = {"median_f0_hz": statistics.mean(median_f0(r) for r in refs)}
    out = {"purpose": "Diagnostic only: B1 pipeline trained toward F2's own audio (a reachable target).",
           "setting": {"lr": 0.01, "steps": 100, "seed": 0}, "loss_first_10_mean": float(np.mean(losses[:10])),
           "loss_last_10_mean": float(np.mean(losses[-10:])), "voice2_before": before, "voice2_after": after,
           "F2_reference": f2}
    (ROOT / "results" / "stage2" / "b1_sanity.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "purpose"}, indent=1))


if __name__ == "__main__":
    main()
