"""Write the Stage 2 base checkpoint (plain state dict for models.tiny_voca3.TinyVoca3).

The published tiny-voca weights, with the voice tables extended to 3 rows; row 2 (the new
voice) is a copy of row 1 (M2) in both `voices.weight` and `frame_voices.weight`.

Usage:
    python train/make_stage2_base.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from models.tiny_voca3 import BASE_PATH, NEW_VOICE, VENDOR, TinyVoca3  # noqa: E402

sd = dict(torch.load(VENDOR / "tiny-voca.pt", map_location="cpu", weights_only=True)["model"])
for name in ("voices.weight", "frame_voices.weight"):
    sd[name] = torch.cat([sd[name], sd[name][1:2].clone()], dim=0)
TinyVoca3().load_state_dict(sd, strict=True)
BASE_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(sd, BASE_PATH)
assert torch.equal(sd["voices.weight"][NEW_VOICE], sd["voices.weight"][1])
print("wrote", BASE_PATH)
