"""Evaluate a Stage 1 checkpoint on a split.

Metrics:
    new_acc   accuracy on class 9 (heart) images, argmax over all 10 outputs
    old_acc   accuracy on classes 0-8 images, argmax over all 10 outputs
    per_class accuracy for each class
    new_pred_hist   how the class-9 images were classified (counts per predicted class)

Usage:
    python eval/evaluate.py CHECKPOINT [--split test]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from data.shapes import NEW_CLASS, NUM_CLASSES, load_split  # noqa: E402
from eval.common import predict_logits, set_determinism  # noqa: E402
from models.cnn import load_model  # noqa: E402


def evaluate_model(model, split: str = "test") -> dict:
    x, y = load_split(split)
    pred = predict_logits(model, x).argmax(1).numpy()
    old = y != NEW_CLASS
    new = y == NEW_CLASS
    return {
        "split": split,
        "new_acc": float((pred[new] == y[new]).mean() * 100),
        "old_acc": float((pred[old] == y[old]).mean() * 100),
        "per_class": [float((pred[y == c] == c).mean() * 100) for c in range(NUM_CLASSES)],
        "new_pred_hist": np.bincount(pred[new], minlength=NUM_CLASSES).tolist(),
        "old_pred_as_new": int((pred[old] == NEW_CLASS).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()
    set_determinism(0)
    print(json.dumps(evaluate_model(load_model(args.checkpoint), args.split), indent=2))


if __name__ == "__main__":
    main()
