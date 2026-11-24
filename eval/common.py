"""Shared helpers: determinism, batched inference, k-shot example selection, JSON output."""

from __future__ import annotations

import json
import platform
import random
from pathlib import Path

import numpy as np
import torch

from data.shapes import NEW_CLASS, load_split, to_tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
K_SHOT = 5


def set_determinism(seed: int) -> None:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_logits(model: torch.nn.Module, x_uint8: np.ndarray, batch: int = 1000) -> torch.Tensor:
    model.eval()
    out = [model(to_tensor(x_uint8[i:i + batch])) for i in range(0, len(x_uint8), batch)]
    return torch.cat(out)


@torch.no_grad()
def predict_features(model: torch.nn.Module, x_uint8: np.ndarray, batch: int = 1000) -> torch.Tensor:
    model.eval()
    out = [model.features(to_tensor(x_uint8[i:i + batch])) for i in range(0, len(x_uint8), batch)]
    return torch.cat(out)


def k_shot_indices(seed: int, k: int = K_SHOT) -> np.ndarray:
    """Indices (within the class-9 VALIDATION images) of the k examples used for `seed`."""
    rng = np.random.default_rng(1000 + seed)
    return np.sort(rng.choice(200, size=k, replace=False))


def k_shot_examples(seed: int, k: int = K_SHOT) -> np.ndarray:
    """The k class-9 VALIDATION images for `seed`, uint8 (k, 32, 32)."""
    x, _ = load_split("val", classes=[NEW_CLASS])
    return x[k_shot_indices(seed, k)]


def environment_info() -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "threads": torch.get_num_threads(),
            "cpu": platform.processor() or platform.machine()}


def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def weight_change_stats(base_model: torch.nn.Module, new_model: torch.nn.Module) -> dict:
    """Compare learnable parameters of two models.

    pct_weights_changed: elements of the base model's parameters whose value differs in the
        new model, as a percentage of all base parameters.
    new_params: parameter elements in the new model that have no counterpart (by name and
        shape) in the base model.
    """
    base = dict(base_model.named_parameters())
    new = dict(new_model.named_parameters())
    total = sum(p.numel() for p in base.values())
    changed = 0
    new_params = 0
    for name, p in new.items():
        if name in base and base[name].shape == p.shape:
            changed += int((base[name].detach() != p.detach()).sum())
        else:
            new_params += p.numel()
    missing = [n for n in base if n not in new]
    return {"pct_weights_changed": 100.0 * changed / total, "num_weights_changed": changed,
            "num_new_params": new_params, "base_params": total, "missing_base_params": missing}
