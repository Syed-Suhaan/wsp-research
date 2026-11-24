"""WSP edit for Stage 1, condition W2, run 2 (text description only, no class-9 examples).

Loads the frozen base checkpoint, wraps it in ShapeCNNHeartRule (model_def.py in this
folder), writes the two new head constants, and saves checkpoints/stage1_W2_run2.pt.
No existing weight of the base model is changed. See rationale.md for the reasoning.
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)

from models.cnn import load_model, save_checkpoint  # noqa: E402

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))
from model_def import ShapeCNNHeartRule  # noqa: E402

BASE = REPO / "checkpoints" / "stage1_base.pt"
OUT = REPO / "checkpoints" / "stage1_W2_run2.pt"

# Class indices of the old classes used as evidence (fixed by the shape description):
#   "pointed" group: 2 triangle, 1 square -- outlines with straight sides and corners.
#     A heart has a sharp bottom point, and the classic heart is a 45-degree-rotated
#     square with two half-circles on its upper sides.
#   "curved" group: 5 ring, 6 ellipse -- curved outlines. The two heart lobes are arcs.
#   Pentagon (7) is left out: its corners are blunt, and it is confused with square in the
#   base model, so including it would raise the AND score for real squares/pentagons.
POINTED = (2, 1)
CURVED = (5, 6)

# offset = 4.5. On the old-class validation images, max_old_logit - min(pointed, curved)
# is at least 5.6 (worst case: an ellipse image), so with offset 4.5 no old validation
# image is moved to class 9 by the AND term, with a safety margin of about 1.1 logits.
# An image that is about equally triangle/square-like and ring/ellipse-like (both
# logits moderate) gets logit_9 = that level + 4.5 and wins.
OFFSET = 4.5

# floor = 5.0. Novelty threshold: old-class validation images have max old logit >= 4.8
# (1st percentile 6.2); a novel shape that no old class claims with logit >= 5 is
# assigned to class 9.
FLOOR = 5.0

base = load_model(BASE)
model = ShapeCNNHeartRule(num_classes=10, pointed=POINTED, curved=CURVED)
model.base.load_state_dict(base.state_dict())  # all base weights copied unchanged
with torch.no_grad():
    model.heart_offset.fill_(OFFSET)
    model.heart_floor.fill_(FLOOR)
model.eval()

save_checkpoint(
    model, OUT,
    meta={"condition": "W2", "run": 2, "edit": "rule-based class-9 head on old logits",
          "pointed": list(POINTED), "curved": list(CURVED), "offset": OFFSET, "floor": FLOOR},
    arch="custom",
    module_file="wsp/stage1/W2/run_2/model_def.py",
    class_name="ShapeCNNHeartRule",
    init_kwargs={"num_classes": 10, "pointed": list(POINTED), "curved": list(CURVED)},
    num_classes=10,
)
print("saved", OUT)
