"""[Stage 1 v1 pilot, frozen copy. Stage 1 v2 uses eval/wsp_tools.py.]

Data and evaluation access for a WSP writer during a Stage 1 run.

This module exposes only what a WSP run is allowed to see: the base checkpoint, the
VALIDATION images of classes 0-8, and (for condition W1) the k = 5 class-9 VALIDATION
examples assigned to the run. It never loads the test split.

Functions:
    load_base()                     base model (10 outputs, class-9 row zero), eval mode
    load_val_old()                  validation images and labels of classes 0-8
    k_shot_examples(run)            the 5 class-9 validation images for run `run` (W1 only)
    to_tensor(x)                    uint8 (N, 32, 32) -> float tensor (N, 1, 32, 32)
    evaluate_val(ckpt, run_dir, run=None)
        Validation report for a candidate checkpoint: old-class accuracy (classes 0-8),
        how many old-class images are predicted as class 9, and, if `run` is given, the
        predictions and logits for that run's 5 class-9 examples. Every call is appended to
        <run_dir>/val_log.jsonl so the number of validation evaluations is auditable.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from data.shapes import NEW_CLASS, to_tensor  # noqa: E402,F401
from data.shapes import load_split as _load_split  # noqa: E402
from eval.common import REPO_ROOT, k_shot_examples, predict_logits  # noqa: E402,F401
from models.cnn import load_model  # noqa: E402

BASE_CKPT = REPO_ROOT / "checkpoints" / "stage1_base.pt"


def load_base():
    return load_model(BASE_CKPT)


def load_val_old():
    return _load_split("val", classes=range(NEW_CLASS))


def evaluate_val(ckpt: str | Path, run_dir: str | Path, run: int | None = None) -> dict:
    model = load_model(ckpt)
    x, y = load_val_old()
    pred = predict_logits(model, x).argmax(1).numpy()
    report = {
        "checkpoint": str(ckpt),
        "val_old_acc": float((pred == y).mean() * 100),
        "val_old_per_class": [float((pred[y == c] == c).mean() * 100) for c in range(NEW_CLASS)],
        "val_old_predicted_as_9": int((pred == NEW_CLASS).sum()),
    }
    if run is not None:
        logits = predict_logits(model, k_shot_examples(run))
        report["k_shot_pred"] = logits.argmax(1).tolist()
        report["k_shot_logits"] = [[round(v, 3) for v in row] for row in logits.tolist()]
    log = Path(run_dir) / "val_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(json.dumps({"time": time.time(), **report}) + "\n")
    return report
