"""Stage 1 v2 baselines for adding class 9 to the base model.

Every condition uses the k class-9 VALIDATION examples chosen by
data.shapes_v2.k_shot_indices(seed, k); k is set in data/stage1_v2_config.json.

B0  Imprinting (no gradients): class-9 head row = unit-norm mean penultimate feature of the
    k examples; class-9 bias = 0.
B0s Supplementary imprinting (no gradients): as B0, with the row scaled to the mean L2 norm
    of the nine trained head rows and the bias set to the mean of the nine trained biases.
B1  Fine-tune the class-9 head row and bias only (gradients). Positives: the k examples.
    Negatives: replayed class 0-8 training images.
B2  Fine-tune the whole model (gradients), k examples plus class 0-8 replay. BatchNorm
    statistics stay frozen; the adapter stays at zero (its tensors are not trained).

Each function edits a ShapesV2Net in place.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from data.shapes_v2 import NEW_CLASS, load_split, to_tensor
from eval.common import predict_features
from models.shapes_v2 import ADAPTER_PREFIXES

HPARAMS = {
    "B1": {"steps": 300, "lr": 1e-2, "replay_per_step": 59, "optimizer": "Adam"},
    "B2": {"steps": 200, "lr": 3e-4, "replay_per_step": 59, "optimizer": "Adam",
           "batchnorm": "frozen statistics", "adapter": "frozen at zero"},
}


def imprint(model, xk, rescale: bool = False) -> None:
    with torch.no_grad():
        mu = predict_features(model, xk).mean(0)
        w = model.head.weight
        w[NEW_CLASS] = mu / mu.norm()
        model.head.bias[NEW_CLASS] = 0.0
        if rescale:
            w[NEW_CLASS] *= w[:NEW_CLASS].norm(dim=1).mean()
            model.head.bias[NEW_CLASS] = model.head.bias[:NEW_CLASS].mean()


def finetune_row(model, xk, seed: int, cfg: dict, hp=HPARAMS["B1"]) -> None:
    x_old, y_old = load_split("train", cfg, classes=range(NEW_CLASS))
    f_old = predict_features(model, x_old)
    f_new = predict_features(model, xk)
    w_old = model.head.weight[:NEW_CLASS].detach().clone()
    b_old = model.head.bias[:NEW_CLASS].detach().clone()
    w9 = torch.zeros(w_old.shape[1], requires_grad=True)
    b9 = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w9, b9], lr=hp["lr"])
    g = torch.Generator().manual_seed(seed)
    y_new = torch.full((len(xk),), NEW_CLASS)
    y_old_t = torch.from_numpy(y_old)
    for _ in range(hp["steps"]):
        idx = torch.randint(len(f_old), (hp["replay_per_step"],), generator=g)
        f = torch.cat([f_new, f_old[idx]])
        y = torch.cat([y_new, y_old_t[idx]])
        # The adapter is zero in the base model, so head logits are the full logits.
        logits = torch.cat([f @ w_old.T + b_old, (f @ w9 + b9).unsqueeze(1)], 1)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        model.head.weight[NEW_CLASS] = w9.detach()
        model.head.bias[NEW_CLASS] = b9.detach()[0]


def finetune_all(model, xk, seed: int, cfg: dict, hp=HPARAMS["B2"]) -> None:
    x_old, y_old = load_split("train", cfg, classes=range(NEW_CLASS))
    xo, yo = to_tensor(x_old), torch.from_numpy(y_old)
    xn, yn = to_tensor(xk), torch.full((len(xk),), NEW_CLASS)
    params = [p for n, p in model.named_parameters() if not n.startswith(ADAPTER_PREFIXES)]
    opt = torch.optim.Adam(params, lr=hp["lr"])
    g = torch.Generator().manual_seed(seed)
    model.eval()  # BatchNorm uses frozen running statistics
    for _ in range(hp["steps"]):
        idx = torch.randint(len(xo), (hp["replay_per_step"],), generator=g)
        logits = model(torch.cat([xn, xo[idx]]))
        loss = F.cross_entropy(logits, torch.cat([yn, yo[idx]]))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():  # the adapter receives no gradient step, but make zero explicit
        for n, p in model.named_parameters():
            if n.startswith(ADAPTER_PREFIXES):
                p.zero_()
