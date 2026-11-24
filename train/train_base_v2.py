"""Train the Stage 1 v2 base model (models/shapes_v2.ShapesV2Net) on classes 0-8.

Training uses only logits 0-8, and the adapter (A1, A2) is frozen at zero. After training,
the class-9 head row, the class-9 bias, and every adapter tensor are set to exactly zero,
so every condition starts from the same 10-output model.

Output: a plain state dict (default checkpoints/stage1_v2_base.pt) and a JSON report
(default results/stage1_v2/base/run_0.json) with validation and test accuracy.

Usage:
    python train/train_base_v2.py [--config data/stage1_v2_config.json] [--out PATH] [--report PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from data.shapes_v2 import CONFIG_PATH, NEW_CLASS, load_config, load_split, to_tensor  # noqa: E402
from eval.common import REPO_ROOT, environment_info, predict_logits, set_determinism, write_json  # noqa: E402
from models.shapes_v2 import ADAPTER_PREFIXES, ShapesV2Net, count_params  # noqa: E402

OLD = list(range(NEW_CLASS))


def old_class_accuracy(model, split, cfg) -> float:
    x, y = load_split(split, cfg, classes=OLD)
    pred = predict_logits(model, x).argmax(1).numpy()
    return float((pred == y).mean() * 100)


def train_base(cfg: dict, epochs: int = 12, batch: int = 128, lr: float = 2e-3,
               seed: int = 0, log=print) -> tuple[ShapesV2Net, float]:
    set_determinism(seed)
    t0 = time.time()
    x, y = load_split("train", cfg, classes=OLD)
    xt, yt = to_tensor(x), torch.from_numpy(y)
    model = ShapesV2Net()
    for n, p in model.named_parameters():
        if n.startswith(ADAPTER_PREFIXES):
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    steps = epochs * ((len(xt) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    g = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=g)
        correct = total = 0
        loss_sum = 0.0
        for i in range(0, len(xt), batch):
            idx = perm[i:i + batch]
            logits = model(xt[idx])[:, :NEW_CLASS]          # classes 0-8 only
            loss = F.cross_entropy(logits, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(idx)
            correct += (logits.argmax(1) == yt[idx]).sum().item()
            total += len(idx)
        log(f"epoch {epoch + 1}/{epochs} loss {loss_sum / total:.4f} train acc "
            f"{100 * correct / total:.2f} ({time.time() - t0:.0f}s)")
    with torch.no_grad():
        model.head.weight[NEW_CLASS].zero_()
        model.head.bias[NEW_CLASS].zero_()
        for n, p in model.named_parameters():
            if n.startswith(ADAPTER_PREFIXES):
                p.zero_()
    model.eval()
    return model, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--out", default=str(REPO_ROOT / "checkpoints" / "stage1_v2_base.pt"))
    ap.add_argument("--report", default=str(REPO_ROOT / "results" / "stage1_v2" / "base" / "run_0.json"))
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    cfg = load_config(args.config)
    model, wall = train_base(cfg, epochs=args.epochs, log=lambda s: print(s, flush=True))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    report = {"stage": "stage1_v2", "condition": "base", "config": cfg,
              "val_old_acc": old_class_accuracy(model, "val", cfg),
              "test_old_acc": old_class_accuracy(model, "test", cfg),
              "num_params": count_params(model), "wall_clock_s": wall, "epochs": args.epochs,
              "environment": environment_info()}
    write_json(args.report, report)
    print(json.dumps({k: report[k] for k in ("val_old_acc", "test_old_acc", "wall_clock_s")}))


if __name__ == "__main__":
    main()
