"""Diagnostic (not a result): does the B2 pipeline learn the new class when given many positives?

Runs train.baselines_v2.finetune_all with the B2t setting chosen in Stage 1 v2.1
(lr 3e-4, 500 steps, positive weight 1, 59 replayed class 0-8 training images per step), but
with 200 class-9 TRAINING images as positives instead of the k validation examples. Seed 0.
Reports validation accuracy on all 200 class-9 validation images (disjoint from the 200
training positives) and old-class validation accuracy.

Output: results/stage1_v2_1/b2_sanity.json

Usage:
    python eval/b2_sanity.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from data.shapes_v2 import NEW_CLASS, load_config, load_split  # noqa: E402
from eval.common import predict_logits, set_determinism, write_json  # noqa: E402
from models.shapes_v2 import load_state  # noqa: E402
from train.baselines_v2 import finetune_all  # noqa: E402

N_POS = 200
SEED = 0


def main():
    cfg = load_config()
    tune = json.loads((ROOT / "results/stage1_v2_1/tuning_B2t.json").read_text())
    params = dict(tune["chosen_params"], replay_per_step=59)
    set_determinism(SEED)
    xt, yt = load_split("train", cfg, classes=[NEW_CLASS])
    rng = np.random.default_rng(SEED)
    pos = xt[np.sort(rng.choice(len(xt), N_POS, replace=False))]
    base = load_state(ROOT / "checkpoints/stage1_v2_base.pt")
    model = load_state(ROOT / "checkpoints/stage1_v2_base.pt")
    t0 = time.time()
    finetune_all(model, pos, SEED, cfg, hp=params)
    wall = time.time() - t0
    xv, yv = load_split("val", cfg)
    pred = predict_logits(model, xv).argmax(1).numpy()
    bpred = predict_logits(base, xv).argmax(1).numpy()
    new, old = yv == NEW_CLASS, yv != NEW_CLASS
    p9 = pred == NEW_CLASS
    train_pred = predict_logits(model, pos).argmax(1).numpy()
    out = {
        "purpose": "Diagnostic only, not a result: check that the B2 pipeline can learn class 9 "
                   "when given many positives.",
        "setting": params, "positives": f"{N_POS} class-9 TRAINING images (seed {SEED})", "seed": SEED,
        "val_new_acc": float(p9[new].mean() * 100),
        "val_precision9": float((p9 & new).sum() / p9.sum() * 100) if p9.sum() else 0.0,
        "val_old_acc": float((pred[old] == yv[old]).mean() * 100),
        "base_val_old_acc": float((bpred[old] == yv[old]).mean() * 100),
        "train_positives_predicted_9_pct": float((train_pred == NEW_CLASS).mean() * 100),
        "new_pred_hist": np.bincount(pred[new], minlength=10).tolist(),
        "wall_clock_s": wall,
    }
    write_json(ROOT / "results/stage1_v2_1/b2_sanity.json", out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("purpose",)}, indent=1))


if __name__ == "__main__":
    main()
