"""Data and evaluation access for a WSP writer during a Stage 1 v2 run.

This module is the only way a writer reads data. It exposes the base checkpoint, the
VALIDATION images of classes 0-8, and (condition W1 only) the run's k class-9 VALIDATION
examples. It does not load any test data.

Functions:
    BASE_PATH                      path of the frozen base checkpoint (a plain state dict)
    load_base()                    ShapesV2Net with the base weights, eval mode
    base_state_dict()              the base state dict (a fresh copy each call)
    load_edited(path)              ShapesV2Net with an edited state dict (strict=True), eval mode
    load_val_old()                 validation images and labels of classes 0-8:
                                   x uint8 (1800, 32, 32), y int64 (1800,)
    k_shot_examples(run)           W1 only: the run's k class-9 validation images,
                                   uint8 (k, 32, 32)
    to_tensor(x)                   uint8 (N, 32, 32) -> float tensor (N, 1, 32, 32) in [0, 1]
    evaluate_val(path, run_dir, run=None)
        Validation report for a complete candidate edit (a saved state dict): old-class
        accuracy overall and per class (argmax over all 10 outputs), and how many old-class
        images are predicted as class 9. With `run` (W1 only), also the predictions and
        logits for that run's k class-9 examples. Every call is appended to
        <run_dir>/val_log.jsonl; at most 3 calls are allowed per run.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.shapes_v2 import NEW_CLASS, to_tensor  # noqa: E402,F401
from data.shapes_v2 import k_shot_examples as _k_shot_examples  # noqa: E402
from data.shapes_v2 import load_config as _load_config  # noqa: E402
from data.shapes_v2 import load_split as _load_split  # noqa: E402
from eval.common import REPO_ROOT, predict_logits  # noqa: E402
from models.shapes_v2 import ShapesV2Net, load_state  # noqa: E402

BASE_PATH = REPO_ROOT / "checkpoints" / "stage1_v2_base.pt"


def load_base() -> ShapesV2Net:
    return load_state(BASE_PATH)


def base_state_dict() -> dict:
    import torch
    return torch.load(BASE_PATH, map_location="cpu", weights_only=True)


def load_edited(path) -> ShapesV2Net:
    return load_state(path)


def load_val_old():
    return _load_split("val", _load_config(), classes=range(NEW_CLASS))


def k_shot_examples(run: int):
    """W1 only. The run's k class-9 validation images (k is fixed for the experiment)."""
    return _k_shot_examples(run, _load_config())


def evaluate_val(path, run_dir, run: int | None = None) -> dict:
    model = load_state(path)
    x, y = load_val_old()
    pred = predict_logits(model, x).argmax(1).numpy()
    report = {
        "checkpoint": str(path),
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
