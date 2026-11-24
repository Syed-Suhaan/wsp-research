"""Train the Stage 1 base CNN on classes 0-8, then add a zero-initialised output row for class 9.

Output:
    checkpoints/stage1_base.pt          10-output ShapeCNN; fc row 9 and bias 9 are zero
    results/stage1/base/run_0.json      test/val accuracy, parameter count, training time

Usage:
    python train/train_base.py [--seed 0] [--epochs 12]
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
from eval.common import REPO_ROOT, environment_info, set_determinism, write_json  # noqa: E402
from eval.evaluate import evaluate_model  # noqa: E402
from models.cnn import ShapeCNN, count_params, save_checkpoint  # noqa: E402

OLD_CLASSES = list(range(NEW_CLASS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default=str(REPO_ROOT / "checkpoints" / "stage1_base.pt"))
    args = ap.parse_args()
    set_determinism(args.seed)
    t0 = time.time()

    x, y = load_split("train", classes=OLD_CLASSES)
    xt, yt = to_tensor(x), torch.from_numpy(y)
    model9 = ShapeCNN(num_classes=len(OLD_CLASSES))
    opt = torch.optim.Adam(model9.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(xt) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    g = torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs):
        model9.train()
        perm = torch.randperm(len(xt), generator=g)
        total, correct, loss_sum = 0, 0, 0.0
        for i in range(0, len(xt), args.batch):
            idx = perm[i:i + args.batch]
            logits = model9(xt[idx])
            loss = F.cross_entropy(logits, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(idx)
            correct += (logits.argmax(1) == yt[idx]).sum().item()
            total += len(idx)
        print(f"epoch {epoch + 1}/{args.epochs} loss {loss_sum / total:.4f} "
              f"train acc {100 * correct / total:.2f} ({time.time() - t0:.0f}s)", flush=True)

    # Expand to 10 outputs: copy the 9 trained rows, class-9 row and bias are zero.
    model = ShapeCNN(num_classes=10)
    sd = model9.state_dict()
    w = torch.zeros(10, 128)
    b = torch.zeros(10)
    w[:9] = sd["fc.weight"]
    b[:9] = sd["fc.bias"]
    sd["fc.weight"], sd["fc.bias"] = w, b
    model.load_state_dict(sd)
    model.eval()
    train_time = time.time() - t0

    save_checkpoint(model, args.out, meta={"seed": args.seed, "epochs": args.epochs,
                                           "trained_classes": OLD_CLASSES})
    test = evaluate_model(model, "test")
    val = evaluate_model(model, "val")
    result = {"stage": "stage1", "condition": "base", "seed": args.seed,
              "test_old_acc": test["old_acc"], "test_new_acc": test["new_acc"],
              "val_old_acc": val["old_acc"], "test_per_class": test["per_class"],
              "num_params": count_params(model), "wall_clock_s": train_time,
              "hyperparameters": vars(args) | {"out": "checkpoints/stage1_base.pt"},
              "environment": environment_info()}
    write_json(REPO_ROOT / "results" / "stage1" / "base" / f"run_{args.seed}.json", result)
    print(f"test old-class acc {test['old_acc']:.2f}  params {count_params(model)}  "
          f"time {train_time:.0f}s")


if __name__ == "__main__":
    main()
