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


# ============================================================================ Stage 2
# Data access for a WSP writer in Stage 2 (text-to-speech). Exposes the base checkpoint of the
# fixed model class (models.tiny_voca3.TinyVoca3; voices 0 = F2, 1 = M2, 2 = the new voice), the
# ENROLLMENT and VALIDATION sentences, and the target speaker's recordings of those sentences.
# No test data is exposed.
#
#   STAGE2_BASE_PATH                  path of the frozen base checkpoint (plain state dict)
#   tts_load_base()                   TinyVoca3 with the base weights, eval mode
#   tts_base_state_dict()             the base state dict (a fresh copy each call)
#   tts_load_edited(path)             TinyVoca3 with an edited state dict (strict=True)
#   tts_texts(split)                  list of sentences; split is "enroll" (50) or "val" (20)
#   tts_target_audio(split, i)        the target speaker's recording of sentence i:
#                                     float64 waveform, 24 kHz, mono
#   tts_synthesize(model, text, voice, seed=0)
#                                     waveform (24 kHz) of the model's standard synthesis with
#                                     its default settings; voice is 0, 1, or 2
#   tts_evaluate_val(path, run_dir, seed=0)
#       Validation report for a complete candidate edit (a saved state dict): for voice 2 on
#       the 20 validation sentences, mean speaker similarity to the target speaker (cosine of
#       speaker embeddings, higher is better), mel-cepstral distortion to the target
#       recordings (lower is better), F0 RMSE in cents (lower is better), and word error rate
#       of a speech recogniser (lower is better); plus whether voices 0 and 1 are unchanged.
#       Every call is appended to <run_dir>/val_log.jsonl; at most 3 calls are allowed per run.

STAGE2_BASE_PATH = REPO_ROOT / "checkpoints" / "stage2_base.pt"
_STAGE2_SPLITS = ("enroll", "val")


def tts_load_base():
    from models.tiny_voca3 import load_state as _load
    return _load(STAGE2_BASE_PATH)


def tts_base_state_dict() -> dict:
    import torch
    return torch.load(STAGE2_BASE_PATH, map_location="cpu", weights_only=True)


def tts_load_edited(path):
    from models.tiny_voca3 import load_state as _load
    return _load(path)


def tts_texts(split: str) -> list[str]:
    if split not in _STAGE2_SPLITS:
        raise ValueError("only 'enroll' and 'val' are available")
    from eval.stage2_metrics import sentences
    return list(sentences()[split])


def tts_target_audio(split: str, i: int):
    if split not in _STAGE2_SPLITS:
        raise ValueError("only 'enroll' and 'val' are available")
    from eval.stage2_audio import load_wav
    from eval.stage2_metrics import target_path
    return load_wav(target_path(split, i))


def tts_synthesize(model, text: str, voice: int, seed: int = 0):
    from eval.stage2_metrics import synthesize
    return synthesize(model, text, voice, seed)


def tts_evaluate_val(path, run_dir, seed: int = 0) -> dict:
    from eval.stage2_metrics import Evaluator, preserved_voices, summarize
    model, base = tts_load_edited(path), tts_load_base()
    rows = Evaluator().evaluate(model, "val", seed, voice=2)
    report = {"checkpoint": str(path), "seed": seed, **summarize(rows),
              "voices_0_1_unchanged": preserved_voices(model, base, "val", seed, count=3),
              "per_sentence": [{k: r[k] for k in ("sentence", "speaker_sim", "mcd_db", "wer")} for r in rows]}
    log = Path(run_dir) / "val_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(json.dumps({"time": time.time(), **report}) + "\n")
    return report
