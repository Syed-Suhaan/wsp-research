"""Stage 1 v2 evaluation on the TEST split, and edit statistics.

Test images: classes 0-9 (500 each) plus two held-out classes (500 each, labels 10 and 11,
from data/heldout.py; one far from the new class and one near it). Predictions are the argmax
over the 10 outputs of models.shapes_v2.ShapesV2Net.

Metrics (accuracies and rates in %):
    new_acc             class-9 images predicted as 9
    precision9          correct class-9 predictions / all class-9 predictions, over old, new,
                        and held-out test images. If nothing is predicted as 9 it is set to 0
                        and precision9_defined is false.
    heldout_fpr         held-out images predicted as 9 (both held-out classes)
    heldout_fpr_near    the held-out class near the new class
    heldout_fpr_far     the held-out class far from the new class
    old_acc             class 0-8 images predicted correctly
    old_drop            base old_acc - old_acc (points)

Edit statistics (edited model vs base state dict):
    pct_weights_changed     changed parameter elements / all parameter elements of the base
    num_weights_changed     changed parameter elements
    slot_params_used        nonzero elements of the adapter (A1, A2) after the edit
    changed_groups          which of {row9, head_other, adapter, backbone} changed
    edit_type               row-only | row-composition | adapter | backbone | mixed
        row-only            only head row 9 / bias 9 changed, row 9 not in the span of rows 0-8
        row-composition     only head row 9 / bias 9 changed, row 9 is a linear combination of
                            the trained rows 0-8 (relative residual < 1e-3)
        adapter             only adapter tensors changed
        backbone            only backbone (conv / BatchNorm) tensors changed
        mixed               more than one group changed, or head rows 0-8 changed
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.heldout import HELDOUT_LABELS, HELDOUT_ROLE, load_heldout_test  # noqa: E402
from data.shapes_v2 import NEW_CLASS, load_config, load_split  # noqa: E402
from eval.common import predict_logits  # noqa: E402
from models.shapes_v2 import ADAPTER_PREFIXES, BACKBONE_PREFIXES, ShapesV2Net  # noqa: E402

LABEL_NEAR = next(lbl for lbl in HELDOUT_LABELS if HELDOUT_ROLE[lbl] == "near")
LABEL_FAR = next(lbl for lbl in HELDOUT_LABELS if HELDOUT_ROLE[lbl] == "far")


def test_set(cfg: dict | None = None):
    cfg = cfg or load_config()
    x, y = load_split("test", cfg)
    xh, yh = load_heldout_test(cfg)
    return np.concatenate([x, xh]), np.concatenate([y, yh])


def evaluate_test(model: ShapesV2Net, cfg: dict | None = None) -> dict:
    x, y = test_set(cfg)
    pred = predict_logits(model, x).argmax(1).numpy()
    old = y < NEW_CLASS
    new = y == NEW_CLASS
    held = y > NEW_CLASS
    pred9 = pred == NEW_CLASS
    n_pred9 = int(pred9.sum())
    tp = int((pred9 & new).sum())
    return {
        "new_acc": float(pred9[new].mean() * 100),
        "precision9": float(tp / n_pred9 * 100) if n_pred9 else 0.0,
        "precision9_defined": bool(n_pred9),
        "num_pred9": n_pred9,
        "pred9_breakdown": {"new": tp, "old": int((pred9 & old).sum()),
                            "heldout_near": int((pred9 & (y == LABEL_NEAR)).sum()),
                            "heldout_far": int((pred9 & (y == LABEL_FAR)).sum())},
        "heldout_fpr": float(pred9[held].mean() * 100),
        "heldout_fpr_near": float(pred9[y == LABEL_NEAR].mean() * 100),
        "heldout_fpr_far": float(pred9[y == LABEL_FAR].mean() * 100),
        "old_acc": float((pred[old] == y[old]).mean() * 100),
        "per_class_acc": [float((pred[y == c] == c).mean() * 100) for c in range(NEW_CLASS + 1)],
        "new_pred_hist": np.bincount(pred[new], minlength=10).tolist(),
    }


def edit_stats(base_sd: dict, sd: dict) -> dict:
    params = dict(ShapesV2Net().named_parameters())
    total = sum(p.numel() for p in params.values())
    changed = 0
    groups = set()
    changed_tensors = []
    for name in params:
        diff = (sd[name] != base_sd[name])
        n = int(diff.sum())
        if not n:
            continue
        changed += n
        changed_tensors.append(name)
        if name.startswith(ADAPTER_PREFIXES):
            groups.add("adapter")
        elif name.startswith(BACKBONE_PREFIXES):
            groups.add("backbone")
        elif name == "head.weight":
            rows = diff.any(dim=1).nonzero().flatten().tolist()
            groups.update("row9" if r == NEW_CLASS else "head_other" for r in rows)
        elif name == "head.bias":
            idx = diff.nonzero().flatten().tolist()
            groups.update("row9" if i == NEW_CLASS else "head_other" for i in idx)
    buffers_changed = [k for k in sd if k not in params and not torch.equal(sd[k], base_sd[k])]
    if groups == {"row9"}:
        w = sd["head.weight"]
        old_rows, row9 = w[:NEW_CLASS].T.double(), w[NEW_CLASS].double()
        coef = torch.linalg.lstsq(old_rows, row9.unsqueeze(1)).solution
        resid = (old_rows @ coef).squeeze(1) - row9
        rel = float(resid.norm() / row9.norm()) if row9.norm() > 0 else 0.0
        edit_type = "row-composition" if rel < 1e-3 else "row-only"
    elif groups == {"adapter"}:
        edit_type, rel = "adapter", None
    elif groups == {"backbone"}:
        edit_type, rel = "backbone", None
    elif not groups:
        edit_type, rel = "none", None
    else:
        edit_type, rel = "mixed", None
    slot = sum(int((sd[n] != 0).sum()) for n in params if n.startswith(ADAPTER_PREFIXES))
    return {"pct_weights_changed": 100.0 * changed / total, "num_weights_changed": changed,
            "total_params": total, "slot_params_used": slot, "changed_groups": sorted(groups),
            "changed_tensors": changed_tensors, "buffers_changed": buffers_changed,
            "edit_type": edit_type, "row9_span_residual": rel}
