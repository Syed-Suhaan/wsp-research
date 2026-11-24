"""WSP edit, Stage 1, condition W2 (text description only), run 4.

Loads the frozen base checkpoint, keeps every ShapeCNN weight unchanged, and wraps it in
HeartRuleCNN (wsp/stage1/W2/run_4/model_def.py). HeartRuleCNN computes the class-9 logit
from a fixed, parameter-free contour rule: an outline shape with exactly one concave corner
(the cusp between the two lobes of a heart). No value is learned or fitted; the rule
constants below are explicit and justified in rationale.md.

Run from the repository root:
    .venv/bin/python wsp/stage1/W2/run_4/edit.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from models.cnn import load_model, save_checkpoint  # noqa: E402

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)

RUN_DIR = Path("wsp/stage1/W2/run_4")
BASE = ROOT / "checkpoints" / "stage1_base.pt"
OUT = ROOT / "checkpoints" / "stage1_W2_run4.pt"

# Load the custom class from this run's folder.
sys.path.insert(0, str(ROOT / RUN_DIR))
from model_def import HeartRuleCNN  # noqa: E402

# Rule constants (see rationale.md for the reasoning behind each value).
INIT_KWARGS = {
    "num_classes": 10,
    "n_angles": 64,     # rays cast from the shape centroid (5.6 degrees apart)
    "k": 4,             # convexity stencil half-width: 4 rays = 22.5 degrees
    "floor": 0.25,      # raster-noise floor subtracted from -kappa (typical convex-outline noise)
    "fill_thr": 0.88,   # outline vs filled: filled shapes have fill ~0.9-1.0, outlines <= 0.8
    "c_thr": 0.5,       # minimum cusp depth (kappa < -0.75); expected heart cusp kappa ~ -1 to -3
    "q_thr": 0.6,       # one dominant concave corner (star 0.25, cross 0.3, single cusp ~1)
    "gain": 10.0,       # logit scale of the rule margin; only its sign affects the argmax
}

base = load_model(BASE)
model = HeartRuleCNN(**INIT_KWARGS)
# Copy every base weight unchanged (identical parameter names and shapes). fc row 9 stays
# zero; its output is replaced by the rule in HeartRuleCNN.forward.
model.load_state_dict(base.state_dict())
model.eval()

save_checkpoint(
    model, OUT,
    meta={"stage": 1, "condition": "W2", "run": 4, "base": "checkpoints/stage1_base.pt",
          "edit": "parameter-free contour rule for class 9; base weights unchanged"},
    arch="custom",
    module_file=str(RUN_DIR / "model_def.py"),
    class_name="HeartRuleCNN",
    init_kwargs=INIT_KWARGS,
)
print(f"saved {OUT}")
