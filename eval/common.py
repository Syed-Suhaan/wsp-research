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
