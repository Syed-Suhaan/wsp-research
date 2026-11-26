"""Fixed model class for Stage 2: tiny-voca with a third voice slot.

`TinyVoca3` is the tiny-voca model (package `tiny_voca`, pinned in requirements.txt) built from
the published checkpoint's configuration, with one more speaker: voices are
["F2", "M2", "NEW"], index 2 is the new voice. The only shape change from the published model
is two embedding tables: `voices.weight` and `frame_voices.weight` are [3, 192] instead of
[2, 192].

The Stage 2 base checkpoint (checkpoints/stage2_base.pt, written by
train/make_stage2_base.py) is the published weights with row 2 of both tables set to a copy of
row 1 (M2), so the new voice starts as M2. Every Stage 2 condition is evaluated by loading a
plain state dict into this class with strict=True and calling its standard `synthesize`.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "tiny-voca"
os.environ.setdefault("DIZIONE_DATA", str(VENDOR / "phonemizer"))

from tiny_voca.config import Config  # noqa: E402
from tiny_voca.model import TinyVoca  # noqa: E402

VOICES = ["F2", "M2", "NEW"]
NEW_VOICE = 2
BASE_PATH = ROOT / "checkpoints" / "stage2_base.pt"


def published_config() -> dict:
    return torch.load(VENDOR / "tiny-voca.pt", map_location="cpu", weights_only=True)["config"]


class TinyVoca3(TinyVoca):
    def __init__(self):
        cfg = dict(published_config())
        cfg["speakers"] = list(VOICES)
        super().__init__(Config.from_dict(cfg))


def use_checkpoint_frontend() -> None:
    """Select the phonemizer front end the published checkpoint was trained with (dizione)."""
    from tiny_voca.infer import match_frontend
    match_frontend(str(VENDOR / "tiny-voca.pt"))


def load_state(path: str | Path = BASE_PATH) -> TinyVoca3:
    use_checkpoint_frontend()
    sd = torch.load(path, map_location="cpu", weights_only=True)
    model = TinyVoca3()
    model.load_state_dict(sd, strict=True)
    return model.eval()
