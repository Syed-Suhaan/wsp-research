"""W2 run 0: add class 9 ("heart") to the base ShapeCNN without examples and without gradients.

No image of class 9 is available, so class 9 is added as the network's "none of the old
classes" output. This is justified because the evaluation set contains only the nine old
classes and class 9: an image that the frozen base model does not recognise as a typical
member of any old class is, in that set, most likely a heart. The description of the heart
(a closed outline with rounded lobes, a concave cusp and one sharp point) combines traits
of the outline classes (ring, ellipse, triangle, pentagon, square) with the concave corners
seen only in star, cross and arrow, so its penultimate feature is expected to lie between
old-class prototypes: lower cosine to its nearest prototype and a smaller logit margin
than old-class images.

Edit (see model_def.py for the forward rule):
  * All base weights (conv, BN, fc) are copied unchanged. fc row 9 stays zero and is
    unused, because logit 9 is replaced by the novelty rule.
  * class_dirs[c]   = unit-normalised mean penultimate feature of validation class c
                      (the same formula as imprinting, applied to the old classes).
  * cos_thresh[c]   = 2nd smallest cosine(feature, class_dirs[c]) among the 200
                      validation images of class c (so at most 1 of 200 is flagged).
  * margin_thresh[c]= 2nd smallest (top1 - top2) logit margin among the 200 validation
                      images of class c (at most 1 of 200 flagged).
  * cos_scale = 0.01 (a cosine shortfall of 0.01 counts as 1 logit), margin_scale = 1.
These are order statistics and means: explicit closed-form formulas, no fitting.
Only VALIDATION images of classes 0-8 are passed through the network.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

from eval.wsp_tools import load_base, load_val_old  # noqa: E402
from eval.common import predict_features  # noqa: E402
from models.cnn import save_checkpoint  # noqa: E402

RUN_DIR = "wsp/stage1/W2/run_0"
MODULE_FILE = f"{RUN_DIR}/model_def.py"
OUT = ROOT / "checkpoints" / "stage1_W2_run0.pt"
NUM_OLD = 9

sys.path.insert(0, str(ROOT / RUN_DIR))
from model_def import ShapeCNNNoveltyHead  # noqa: E402

# 1. Frozen base model and its penultimate features on the old-class validation images.
base = load_base()
x_val, y_val = load_val_old()
y = torch.as_tensor(y_val, dtype=torch.long)
with torch.no_grad():
    feats = predict_features(base, x_val)                    # (1800, 128)
    old_logits = base.fc(feats)[:, :NUM_OLD]                 # (1800, 9)

# 2. Class prototypes: unit-normalised mean feature of each old class (closed form).
onehot = F.one_hot(y, NUM_OLD).float()                       # (1800, 9)
class_means = onehot.t() @ feats / onehot.sum(0)[:, None]    # (9, 128)
class_dirs = F.normalize(class_means, dim=1)

# 3. Per-image novelty statistics with respect to the image's own (true = predicted) class.
#    The base model classifies every validation image correctly, so own class = argmax.
cos_own = (F.normalize(feats, dim=1) * class_dirs[y]).sum(1)
top2 = old_logits.topk(2, dim=1).values
margin = top2[:, 0] - top2[:, 1]

# 4. Thresholds: the 2nd smallest value per class (order statistic). Images strictly below
#    it are flagged, so at most 1 of the 200 validation images per class is flagged per rule.
cos_thresh = torch.stack([cos_own[y == c].sort().values[1] for c in range(NUM_OLD)])
margin_thresh = torch.stack([margin[y == c].sort().values[1] for c in range(NUM_OLD)])

# 5. Build the extended model: copy every base weight unchanged, then set the new values.
model = ShapeCNNNoveltyHead(num_classes=10, num_old=NUM_OLD, feat_dim=128)
missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
assert not unexpected and set(missing) == {
    "class_dirs", "cos_thresh", "margin_thresh", "cos_scale", "margin_scale"}
with torch.no_grad():
    model.class_dirs.copy_(class_dirs)
    model.cos_thresh.copy_(cos_thresh)
    model.margin_thresh.copy_(margin_thresh)
    model.cos_scale.fill_(0.01)
    model.margin_scale.fill_(1.0)
model.eval()

save_checkpoint(
    model, OUT,
    meta={"condition": "W2", "run": 0, "method": "open-set novelty head on penultimate features"},
    arch="custom", module_file=MODULE_FILE, class_name="ShapeCNNNoveltyHead",
    init_kwargs={"num_classes": 10, "num_old": NUM_OLD, "feat_dim": 128},
)
print("saved", OUT)
print("cos_thresh", [round(v, 4) for v in cos_thresh.tolist()])
print("margin_thresh", [round(v, 3) for v in margin_thresh.tolist()])
