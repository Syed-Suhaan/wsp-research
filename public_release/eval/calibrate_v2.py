"""Stage 1 v2 difficulty calibration (run before any WSP run).

Goal: B0 mean new-class accuracy between 60% and 80% on the VALIDATION split (seeds 0-7).
Each call evaluates one candidate setup and appends it to results/stage1_v2/calibration.json:
  1. train (or reuse) a base model for the setup's image configuration
     (checkpoints/calib/base_<tag>.pt; class-9 row, bias, and adapter zero);
  2. apply B0 for seeds 0-7 with the setup's k;
  3. report class-9 VALIDATION accuracy (all 200 images, and excluding the k examples),
     old-class validation accuracy, and the drop from the base model.

Usage:
    python eval/calibrate_v2.py --lever 0 --new-class octagon --k 5 [--noise 0.03 0.12]
        [--blur 0 0] [--distortion 0] [--note TEXT]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.shapes_v2 import DEFAULT_CONFIG, NEW_CLASS, config_tag, k_shot_indices, load_split  # noqa: E402
from eval.common import environment_info, predict_logits, set_determinism  # noqa: E402
from models.shapes_v2 import load_state  # noqa: E402
from train.baselines_v2 import imprint  # noqa: E402
from train.train_base_v2 import old_class_accuracy, train_base  # noqa: E402

OUT = ROOT / "results" / "stage1_v2" / "calibration.json"
SEEDS = range(8)
TARGET = (60.0, 80.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lever", type=int, required=True, help="0 = default setup, 1-3 = lever")
    ap.add_argument("--new-class", default=DEFAULT_CONFIG["new_class"])
    ap.add_argument("--k", type=int, default=DEFAULT_CONFIG["k"])
    ap.add_argument("--noise", type=float, nargs=2, default=DEFAULT_CONFIG["noise_sigma"])
    ap.add_argument("--blur", type=float, nargs=2, default=DEFAULT_CONFIG["blur_sigma"])
    ap.add_argument("--distortion", type=float, default=DEFAULT_CONFIG["distortion"])
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    cfg = dict(DEFAULT_CONFIG, new_class=args.new_class, k=args.k, noise_sigma=list(args.noise),
               blur_sigma=list(args.blur), distortion=args.distortion)
    tag = config_tag(cfg)
    # Old-class images do not depend on the new class, so the base model is keyed on the
    # image configuration without it.
    base_tag = config_tag(dict(cfg, new_class="*"))
    base_path = ROOT / "checkpoints" / "calib" / f"base_{base_tag}.pt"
    base_info_path = base_path.with_suffix(".json")
    if not base_path.exists():
        model, wall = train_base(cfg, log=lambda s: print(s, flush=True))
        base_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), base_path)
        base_info_path.write_text(json.dumps({
            "val_old_acc": old_class_accuracy(model, "val", cfg),
            "test_old_acc": old_class_accuracy(model, "test", cfg),
            "train_wall_clock_s": wall}, indent=2) + "\n")
    base_info = json.loads(base_info_path.read_text())

    set_determinism(0)
    xv, yv = load_split("val", cfg)
    old, new = yv != NEW_CLASS, yv == NEW_CLASS
    new_idx_all = np.where(new)[0]
    base = load_state(base_path)
    base_old = float((predict_logits(base, xv[old]).argmax(1).numpy() == yv[old]).mean() * 100)
    per_seed = []
    t0 = time.time()
    for seed in SEEDS:
        model = load_state(base_path)
        kidx = k_shot_indices(seed, cfg["k"])
        imprint(model, xv[new_idx_all[kidx]])
        pred = predict_logits(model, xv).argmax(1).numpy()
        mask_rest = np.ones(new.sum(), bool)
        mask_rest[kidx] = False
        pred_new = pred[new]
        per_seed.append({
            "seed": seed, "k_indices": kidx.tolist(),
            "val_new_acc": float((pred_new == NEW_CLASS).mean() * 100),
            "val_new_acc_excl_k": float((pred_new[mask_rest] == NEW_CLASS).mean() * 100),
            "val_old_acc": float((pred[old] == yv[old]).mean() * 100),
            "new_pred_hist": np.bincount(pred_new, minlength=10).tolist(),
        })
    mean_new = statistics.mean(s["val_new_acc"] for s in per_seed)
    attempt = {
        "lever": args.lever, "note": args.note, "config": cfg, "image_config_tag": tag, "base_config_tag": base_tag,
        "base_checkpoint": str(base_path.relative_to(ROOT)),
        "base_val_old_acc": base_info["val_old_acc"], "base_test_old_acc": base_info["test_old_acc"],
        "base_train_wall_clock_s": base_info["train_wall_clock_s"],
        "b0_val_new_acc_mean": mean_new,
        "b0_val_new_acc_sd": statistics.stdev(s["val_new_acc"] for s in per_seed),
        "b0_val_new_acc_excl_k_mean": statistics.mean(s["val_new_acc_excl_k"] for s in per_seed),
        "b0_val_old_acc_mean": statistics.mean(s["val_old_acc"] for s in per_seed),
        "b0_val_old_drop_mean": base_old - statistics.mean(s["val_old_acc"] for s in per_seed),
        "in_target_range": TARGET[0] <= mean_new <= TARGET[1],
        "per_seed": per_seed, "b0_wall_clock_s": time.time() - t0,
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "environment": environment_info(),
    }
    log = json.loads(OUT.read_text()) if OUT.exists() else {
        "goal": "B0 mean new-class accuracy in [60, 80] % on the VALIDATION split, seeds 0-7",
        "attempts": []}
    attempt["attempt"] = len(log["attempts"])
    log["attempts"].append(attempt)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(log, indent=2) + "\n")
    print(f"attempt {attempt['attempt']} lever {args.lever} {cfg['new_class']} k={cfg['k']} "
          f"noise={cfg['noise_sigma']} blur={cfg['blur_sigma']} dist={cfg['distortion']}: "
          f"base test old {base_info['test_old_acc']:.2f}  B0 val new {mean_new:.1f} "
          f"(sd {attempt['b0_val_new_acc_sd']:.1f}, excl k {attempt['b0_val_new_acc_excl_k_mean']:.1f})  "
          f"old drop {attempt['b0_val_old_drop_mean']:.2f}  in range {attempt['in_target_range']}")


if __name__ == "__main__":
    main()
