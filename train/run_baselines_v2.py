"""Run one Stage 1 v2 baseline condition for one seed and evaluate it on the TEST split.

Conditions (see train/baselines_v2.py): B0, B0s, B1, B2. The k class-9 VALIDATION examples
for seed s are data.shapes_v2.k_shot_examples(s); k comes from data/stage1_v2_config.json.

Output:
    checkpoints/stage1_v2_<cond>_seed<s>.pt     plain state dict
    results/stage1_v2/<cond>/run_<s>.json

Usage:
    python train/run_baselines_v2.py --condition B0 --seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from data.shapes_v2 import k_shot_examples, k_shot_indices, load_config  # noqa: E402
from eval.common import REPO_ROOT, environment_info, set_determinism, write_json  # noqa: E402
from eval.evaluate_v2 import edit_stats, evaluate_test  # noqa: E402
from models.shapes_v2 import load_state  # noqa: E402
from train.baselines_v2 import HPARAMS, finetune_all, finetune_row, imprint  # noqa: E402

BASE = REPO_ROOT / "checkpoints" / "stage1_v2_base.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["B0", "B0s", "B1", "B2"])
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    cfg = load_config()
    set_determinism(args.seed)
    base = load_state(BASE)
    model = load_state(BASE)
    xk = k_shot_examples(args.seed, cfg)
    t0 = time.time()
    if args.condition in ("B0", "B0s"):
        imprint(model, xk, rescale=args.condition == "B0s")
    elif args.condition == "B1":
        finetune_row(model, xk, args.seed, cfg)
    else:
        finetune_all(model, xk, args.seed, cfg)
    wall = time.time() - t0
    model.eval()
    ckpt = REPO_ROOT / "checkpoints" / f"stage1_v2_{args.condition}_seed{args.seed}.pt"
    torch.save(model.state_dict(), ckpt)
    test, base_test = evaluate_test(model, cfg), evaluate_test(base, cfg)
    result = {
        "stage": "stage1_v2", "condition": args.condition, "seed": args.seed, "run": args.seed,
        "valid": True, "uses_gradients": args.condition in ("B1", "B2"),
        "k": cfg["k"], "k_shot_val_indices_class9": k_shot_indices(args.seed, cfg["k"]).tolist(),
        **test, "old_drop": base_test["old_acc"] - test["old_acc"],
        "base_old_acc": base_test["old_acc"], "wall_clock_s": wall,
        **edit_stats(base.state_dict(), model.state_dict()),
        "hyperparameters": HPARAMS.get(args.condition, {"formula": "imprinting, see train/baselines_v2.py"}),
        "environment": environment_info(),
    }
    write_json(REPO_ROOT / "results" / "stage1_v2" / args.condition / f"run_{args.seed}.json", result)
    print(f"{args.condition} seed {args.seed}: new {test['new_acc']:.1f}  prec {test['precision9']:.1f}  "
          f"fpr near {test['heldout_fpr_near']:.1f} far {test['heldout_fpr_far']:.1f}  "
          f"old {test['old_acc']:.2f} drop {result['old_drop']:.2f}  {result['edit_type']}  {wall:.1f}s")


if __name__ == "__main__":
    main()
