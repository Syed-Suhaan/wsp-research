"""Stage 1 baselines for adding class 9 (heart) to the base CNN.

All conditions use the k = 5 class-9 VALIDATION examples chosen by `k_shot_indices(seed)`.

B0  Imprinting (no gradients). The class-9 weight row is the L2-normalised (unit-norm) mean
    penultimate feature of the k examples. The class-9 bias keeps its initial value, zero.
B0s Supplementary imprinting variant (no gradients). As B0, but the row is rescaled to the
    mean L2 norm of the nine trained rows and the bias is set to the mean of the nine trained
    biases. Reported for reference; the gate uses B0.
B1  Fine-tune the class-9 row (weight and bias) only, by gradient descent. Starts from zero.
    Each step uses the k examples plus 59 replayed class 0-8 training images as negatives;
    all other parameters are frozen, so old-class logits cannot change.
B2  Fine-tune the whole model by gradient descent. Each step uses the k examples plus 59
    replayed class 0-8 training images. BatchNorm statistics stay frozen (eval mode).

Output:
    checkpoints/stage1_<cond>_seed<seed>.pt
    results/stage1/<cond>/run_<seed>.json

Usage:
    python train/baselines.py --condition B0 --seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from data.shapes import NEW_CLASS, load_split, to_tensor  # noqa: E402
from eval.common import (REPO_ROOT, environment_info, k_shot_examples, k_shot_indices,  # noqa: E402
                         predict_features, set_determinism, weight_change_stats, write_json)
from eval.evaluate import evaluate_model  # noqa: E402
from models.cnn import load_model, save_checkpoint  # noqa: E402

BASE_CKPT = REPO_ROOT / "checkpoints" / "stage1_base.pt"
HPARAMS = {
    "B1": {"steps": 300, "lr": 1e-2, "replay_per_step": 59, "optimizer": "Adam"},
    "B2": {"steps": 200, "lr": 3e-4, "replay_per_step": 59, "optimizer": "Adam",
           "batchnorm": "frozen statistics"},
}


def imprint(model, xk, rescale: bool) -> None:
    with torch.no_grad():
        mu = predict_features(model, xk).mean(0)
        w = model.fc.weight
        w[NEW_CLASS] = mu / mu.norm()
        if rescale:
            w[NEW_CLASS] *= w[:NEW_CLASS].norm(dim=1).mean()
            model.fc.bias[NEW_CLASS] = model.fc.bias[:NEW_CLASS].mean()


def finetune_row(model, xk, seed, hp) -> None:
    x_old, y_old = load_split("train", classes=range(NEW_CLASS))
    f_old = predict_features(model, x_old)
    f_new = predict_features(model, xk)
    w_old = model.fc.weight[:NEW_CLASS].detach().clone()
    b_old = model.fc.bias[:NEW_CLASS].detach().clone()
    w9 = torch.zeros(128, requires_grad=True)
    b9 = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w9, b9], lr=hp["lr"])
    g = torch.Generator().manual_seed(seed)
    y_new = torch.full((len(xk),), NEW_CLASS)
    y_old_t = torch.from_numpy(y_old)
    for _ in range(hp["steps"]):
        idx = torch.randint(len(f_old), (hp["replay_per_step"],), generator=g)
        f = torch.cat([f_new, f_old[idx]])
        y = torch.cat([y_new, y_old_t[idx]])
        logits = torch.cat([f @ w_old.T + b_old, (f @ w9 + b9).unsqueeze(1)], 1)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        model.fc.weight[NEW_CLASS] = w9.detach()
        model.fc.bias[NEW_CLASS] = b9.detach()[0]


def finetune_all(model, xk, seed, hp) -> None:
    x_old, y_old = load_split("train", classes=range(NEW_CLASS))
    xo, yo = to_tensor(x_old), torch.from_numpy(y_old)
    xn, yn = to_tensor(xk), torch.full((len(xk),), NEW_CLASS)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    g = torch.Generator().manual_seed(seed)
    model.eval()  # BatchNorm uses frozen running statistics
    for _ in range(hp["steps"]):
        idx = torch.randint(len(xo), (hp["replay_per_step"],), generator=g)
        logits = model(torch.cat([xn, xo[idx]]))
        loss = F.cross_entropy(logits, torch.cat([yn, yo[idx]]))
        opt.zero_grad()
        loss.backward()
        opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["B0", "B0s", "B1", "B2"])
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    set_determinism(args.seed)

    base = load_model(BASE_CKPT)
    model = load_model(BASE_CKPT)
    xk = k_shot_examples(args.seed)
    t0 = time.time()
    if args.condition in ("B0", "B0s"):
        imprint(model, xk, rescale=args.condition == "B0s")
    elif args.condition == "B1":
        finetune_row(model, xk, args.seed, HPARAMS["B1"])
    else:
        finetune_all(model, xk, args.seed, HPARAMS["B2"])
    wall = time.time() - t0
    model.eval()

    ckpt = REPO_ROOT / "checkpoints" / f"stage1_{args.condition}_seed{args.seed}.pt"
    save_checkpoint(model, ckpt, meta={"condition": args.condition, "seed": args.seed})
    test = evaluate_model(model, "test")
    base_test = evaluate_model(base, "test")
    result = {
        "stage": "stage1", "condition": args.condition, "seed": args.seed,
        "uses_gradients": args.condition in ("B1", "B2"),
        "k_shot_val_indices_class9": k_shot_indices(args.seed).tolist(),
        "new_acc": test["new_acc"], "old_acc": test["old_acc"],
        "old_drop": base_test["old_acc"] - test["old_acc"],
        "base_old_acc": base_test["old_acc"],
        "per_class": test["per_class"], "new_pred_hist": test["new_pred_hist"],
        "old_pred_as_new": test["old_pred_as_new"],
        "wall_clock_s": wall,
        **weight_change_stats(base, model),
        "hyperparameters": HPARAMS.get(args.condition, {"formula": "imprinting, see module docstring"}),
        "environment": environment_info(),
    }
    write_json(REPO_ROOT / "results" / "stage1" / args.condition / f"run_{args.seed}.json", result)
    print(f"{args.condition} seed {args.seed}: new {test['new_acc']:.1f}  old {test['old_acc']:.2f}  "
          f"drop {result['old_drop']:.2f}  time {wall:.1f}s")


if __name__ == "__main__":
    main()
