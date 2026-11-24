"""Tests for tools/check_no_gradients.py: bad examples must fail, good examples must pass."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from check_no_gradients import check_source  # noqa: E402

GOOD = '''
import torch
ckpt = torch.load("checkpoints/stage1_base.pt")
sd = ckpt["state_dict"]
with torch.no_grad():
    feats = []
    for x in examples:                      # accumulate features, no selection
        feats.append(model.features(x))
    mean = torch.stack(feats).mean(0)
    sd["fc.weight"][9] = mean / mean.norm() * 3.0   # explicit formula
    sd["fc.bias"][9] = -1.5                         # explicit value
changed = 0
for name, p in model.named_parameters():    # bookkeeping loop with a comparison
    if name.startswith("fc"):
        changed += p.numel()
w = torch.zeros(4, requires_grad=False)
torch.save(ckpt, "checkpoints/edited.pt")
'''

BAD_EXAMPLES = {
    "backward": "loss = model(x).sum()\nloss.backward()\n",
    "autograd": "import torch\ng = torch.autograd.grad(y, w)\n",
    "optim": "import torch\nopt = torch.optim.SGD(params, lr=0.1)\n",
    "requires_grad_": "w.requires_grad_(True)\n",
    "requires_grad_kw": "w = torch.zeros(3, requires_grad=True)\n",
    "lstsq": "import torch\nsol = torch.linalg.lstsq(A, B).solution\n",
    "kmeans": "from sklearn.cluster import KMeans\nkm = KMeans(3)\n",
    "search_loop": (
        "best = None\n"
        "for b in [-3, -2, -1, 0]:\n"
        "    sd['fc.bias'][9] = b\n"
        "    acc = evaluate(model)\n"
        "    if best is None or acc > best:\n"
        "        best = acc\n"
    ),
    "random_search": (
        "import torch\n"
        "for _ in range(100):\n"
        "    w = torch.randn(128)\n"
        "    s = model(x).max()\n"
    ),
}


def test_good_example_passes():
    assert check_source(GOOD) == []


def test_each_bad_example_fails():
    for name, src in BAD_EXAMPLES.items():
        assert check_source(src), f"bad example '{name}' was not detected"


def test_comment_is_ignored():
    assert check_source("x = 1  # do not call .backward() here\n") == []


def test_cli_exit_codes(tmp_path):
    good = tmp_path / "good.py"
    bad = tmp_path / "bad.py"
    good.write_text(GOOD)
    bad.write_text(BAD_EXAMPLES["backward"])
    tool = ROOT / "tools" / "check_no_gradients.py"
    assert subprocess.run([sys.executable, str(tool), str(good)]).returncode == 0
    assert subprocess.run([sys.executable, str(tool), str(bad)]).returncode == 1
