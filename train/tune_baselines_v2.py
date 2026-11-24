"""Stage 1 v2.1: tune baselines B1t, B2t, and B0b on the VALIDATION split, then evaluate the
chosen setting once on TEST.

The grids and the selection rule are fixed in results/stage1_v2_1/selection_rule.json (it was
committed before any grid result existed). Summary:
    score(setting)   = mean over seeds 0-7 of (new-class acc + class-9 precision) / 2 on VALIDATION,
                       with each seed's k examples excluded from the class-9 validation images
    constraint       = mean old-class drop on VALIDATION <= 2 points
    choice           = highest score meeting the constraint (ties: smaller drop, then grid order);
                       if none meets it, the smallest mean drop (reported)

Usage:
    python train/tune_baselines_v2.py --condition B1t --phase grid   # validation grid
    python train/tune_baselines_v2.py --condition B1t --phase test   # chosen setting on TEST

Output:
    results/stage1_v2_1/tuning_<cond>.json     every setting x seed on validation, and the choice
    results/stage1_v2/<cond>/run_<seed>.json   TEST results of the chosen setting
    checkpoints/stage1_v2_<cond>_seed<seed>.pt
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.shapes_v2 import NEW_CLASS, k_shot_indices, load_config, load_split  # noqa: E402
from eval.common import environment_info, predict_logits, set_determinism, write_json  # noqa: E402
from eval.evaluate_v2 import edit_stats, evaluate_test  # noqa: E402
from models.shapes_v2 import load_state  # noqa: E402
from train.baselines_v2 import finetune_all, finetune_row, imprint, old_train_features  # noqa: E402

BASE = ROOT / "checkpoints" / "stage1_v2_base.pt"
RULE = json.loads((ROOT / "results" / "stage1_v2_1" / "selection_rule.json").read_text())
SEEDS = range(8)
MAX_DROP = 2.0


def grid(cond: str) -> list[dict]:
    g = RULE["conditions"][cond]["grid"]
    keys = list(g)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(g[k] for k in keys))]


def apply(cond: str, model, xk, seed: int, cfg: dict, params: dict, cache: dict) -> None:
    if cond == "B0b":
        imprint(model, xk, bias=float(params["bias"]))
    elif cond == "B1t":
        if "old_feats" not in cache:
            cache["old_feats"] = old_train_features(model, cfg)
        finetune_row(model, xk, seed, cfg, hp=params, old_feats=cache["old_feats"])
    elif cond == "B2t":
        finetune_all(model, xk, seed, cfg, hp=dict(params, replay_per_step=59))
    else:
        raise ValueError(cond)


def val_metrics(model, xv, yv, kidx_global, base_old_acc) -> dict:
    keep = np.ones(len(yv), bool)
    keep[kidx_global] = False
    x, y = xv[keep], yv[keep]
    pred = predict_logits(model, x).argmax(1).numpy()
    old, new = y != NEW_CLASS, y == NEW_CLASS
    p9 = pred == NEW_CLASS
    old_acc = float((pred[old] == y[old]).mean() * 100)
    return {"new_acc": float(p9[new].mean() * 100),
            "precision9": float((p9 & new).sum() / p9.sum() * 100) if p9.sum() else 0.0,
            "old_acc": old_acc, "old_drop": base_old_acc - old_acc}


def run_grid(cond: str, cfg: dict) -> None:
    xv, yv = load_split("val", cfg)
    new_idx = np.where(yv == NEW_CLASS)[0]
    base = load_state(BASE)
    base_old = float((predict_logits(base, xv[yv != NEW_CLASS]).argmax(1).numpy()
                      == yv[yv != NEW_CLASS]).mean() * 100)
    cache: dict = {}
    settings = []
    t0 = time.time()
    for i, params in enumerate(grid(cond)):
        per_seed = []
        for seed in SEEDS:
            set_determinism(seed)
            kidx = k_shot_indices(seed, cfg["k"])
            model = load_state(BASE)
            apply(cond, model, xv[new_idx[kidx]], seed, cfg, params, cache)
            model.eval()
            per_seed.append({"seed": seed, **val_metrics(model, xv, yv, new_idx[kidx], base_old)})
        m = {k: statistics.mean(s[k] for s in per_seed) for k in ("new_acc", "precision9", "old_acc", "old_drop")}
        score = (m["new_acc"] + m["precision9"]) / 2
        settings.append({"index": i, "params": params, **{f"val_{k}": v for k, v in m.items()},
                         "score": score, "meets_constraint": m["old_drop"] <= MAX_DROP, "per_seed": per_seed})
        print(f"{cond} {i:2d} {params}: new {m['new_acc']:.1f} prec {m['precision9']:.1f} "
              f"drop {m['old_drop']:.2f} score {score:.2f}", flush=True)
    ok = [s for s in settings if s["meets_constraint"]]
    if ok:
        chosen = sorted(ok, key=lambda s: (-s["score"], s["val_old_drop"], s["index"]))[0]
        note = "highest score among settings meeting the constraint"
    else:
        chosen = sorted(settings, key=lambda s: (s["val_old_drop"], s["index"]))[0]
        note = "NO setting met the constraint; chose the smallest mean old-class drop"
    out = {"condition": cond, "rule": RULE["selection"], "grid": RULE["conditions"][cond]["grid"],
           "num_settings": len(settings), "num_runs": len(settings) * len(SEEDS),
           "base_val_old_acc": base_old, "tuning_wall_clock_s": time.time() - t0,
           "chosen_index": chosen["index"], "chosen_params": chosen["params"], "choice_note": note,
           "chosen_val": {k: chosen[k] for k in ("score", "val_new_acc", "val_precision9", "val_old_drop")},
           "settings": settings}
    write_json(ROOT / "results" / "stage1_v2_1" / f"tuning_{cond}.json", out)
    print(f"chosen {cond}: {chosen['params']} ({note})")


def run_test(cond: str, cfg: dict) -> None:
    tune = json.loads((ROOT / "results" / "stage1_v2_1" / f"tuning_{cond}.json").read_text())
    params = tune["chosen_params"]
    base = load_state(BASE)
    base_test = evaluate_test(base, cfg)
    cache: dict = {}
    for seed in SEEDS:
        set_determinism(seed)
        model = load_state(BASE)
        xv, yv = load_split("val", cfg)
        kidx = k_shot_indices(seed, cfg["k"])
        xk = xv[np.where(yv == NEW_CLASS)[0][kidx]]
        t0 = time.time()
        apply(cond, model, xk, seed, cfg, params, cache)
        wall = time.time() - t0
        model.eval()
        torch.save(model.state_dict(), ROOT / "checkpoints" / f"stage1_v2_{cond}_seed{seed}.pt")
        test = evaluate_test(model, cfg)
        result = {"stage": "stage1_v2", "condition": cond, "seed": seed, "run": seed, "valid": True,
                  "uses_gradients": cond in ("B1t", "B2t"), "k": cfg["k"],
                  "k_shot_val_indices_class9": kidx.tolist(), "tuned": True,
                  "chosen_params": params, "tuning_file": f"results/stage1_v2_1/tuning_{cond}.json",
                  "tuning_budget": f"{tune['num_settings']} settings x 8 seeds on VALIDATION",
                  **test, "old_drop": base_test["old_acc"] - test["old_acc"],
                  "base_old_acc": base_test["old_acc"], "wall_clock_s": wall,
                  "tuning_wall_clock_s": tune["tuning_wall_clock_s"],
                  **edit_stats(base.state_dict(), model.state_dict()), "environment": environment_info()}
        write_json(ROOT / "results" / "stage1_v2" / cond / f"run_{seed}.json", result)
        print(f"{cond} seed {seed}: new {test['new_acc']:.1f} prec {test['precision9']:.1f} "
              f"old drop {result['old_drop']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["B1t", "B2t", "B0b"])
    ap.add_argument("--phase", required=True, choices=["grid", "test"])
    args = ap.parse_args()
    cfg = load_config()
    (run_grid if args.phase == "grid" else run_test)(args.condition, cfg)


if __name__ == "__main__":
    main()
