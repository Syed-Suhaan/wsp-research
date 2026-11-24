"""Tests for tools/check_wsp_edit.py.

Required behaviour: FAIL on (a) .backward(), (b) a forward function or pixel rule, (c) no
tensor changed, (d) a new tensor name; PASS on a small valid row edit. Also covers the
frozen-code (git diff) check and the candidate-scoring loop check.
"""

import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from check_wsp_edit import check_frozen_code, check_run, check_source  # noqa: E402
from models.shapes_v2 import ShapesV2Net  # noqa: E402

VALID_EDIT = '''
import torch
sd = torch.load("checkpoints/stage1_v2_base.pt")
with torch.no_grad():
    feats = []
    for x in examples:                       # accumulate features, no selection
        feats.append(model.features(x))
    mu = torch.stack(feats).mean(0)
    sd["head.weight"][9] = mu / mu.norm()     # explicit formula
    sd["head.bias"][9] = -1.0                 # explicit value
torch.save(sd, "checkpoints/stage1_v2_W1_run0.pt")
'''

BACKWARD_EDIT = VALID_EDIT + "loss = model(x).sum()\nloss.backward()\n"

PIXEL_RULE_EDIT = '''
import torch
from models.shapes_v2 import ShapesV2Net
class RuleNet(ShapesV2Net):
    def forward(self, x):
        logits = super().forward(x)
        logits[:, 9] = (x > 0.5).float().mean(dim=(1, 2, 3)) * 100   # pixel rule
        return logits
'''

SEARCH_LOOP_EDIT = '''
best = None
for b in [-3, -2, -1, 0]:
    sd["head.bias"][9] = b
    acc = evaluate(model)
    if best is None or acc > best:
        best = acc
'''


def _setup(tmp_path, edit_source, mutate):
    torch.manual_seed(0)
    base_sd = ShapesV2Net().state_dict()
    base = tmp_path / "base.pt"
    torch.save(base_sd, base)
    run_dir = tmp_path / "run_0"
    run_dir.mkdir()
    (run_dir / "edit.py").write_text(edit_source)
    out_sd = {k: v.clone() for k, v in base_sd.items()}
    out_sd = mutate(out_sd)
    out = tmp_path / "out.pt"
    torch.save(out_sd, out)
    return run_dir, out, base


def _row_edit(sd):
    sd["head.weight"][9] = torch.linspace(-1, 1, 128)
    sd["head.bias"][9] = -1.0
    return sd


def test_valid_row_edit_passes(tmp_path):
    run_dir, out, base = _setup(tmp_path, VALID_EDIT, _row_edit)
    report = check_run(run_dir, out, base, git_rev=None)
    assert report["valid"], report["errors"]
    assert report["changed_tensors"] == ["head.weight", "head.bias"]


def test_a_backward_fails(tmp_path):
    run_dir, out, base = _setup(tmp_path, BACKWARD_EDIT, _row_edit)
    report = check_run(run_dir, out, base, git_rev=None)
    assert not report["valid"]
    assert any(".backward" in e for e in report["errors"])


def test_b_forward_function_or_pixel_rule_fails(tmp_path):
    run_dir, out, base = _setup(tmp_path, PIXEL_RULE_EDIT, _row_edit)
    report = check_run(run_dir, out, base, git_rev=None)
    assert not report["valid"]
    assert any("forward" in e or "class" in e for e in report["errors"])


def test_b_model_definition_file_fails(tmp_path):
    run_dir, out, base = _setup(tmp_path, VALID_EDIT, _row_edit)
    (run_dir / "model_def.py").write_text(PIXEL_RULE_EDIT)
    report = check_run(run_dir, out, base, git_rev=None)
    assert not report["valid"]
    assert any("model_def.py" in e for e in report["errors"])


def test_b_custom_checkpoint_format_fails(tmp_path):
    run_dir, out, base = _setup(tmp_path, VALID_EDIT, _row_edit)
    torch.save({"arch": "custom", "state_dict": torch.load(out)}, out)
    report = check_run(run_dir, out, base, git_rev=None)
    assert not report["valid"]


def test_c_no_tensor_changed_fails(tmp_path):
    run_dir, out, base = _setup(tmp_path, VALID_EDIT, lambda sd: sd)
    report = check_run(run_dir, out, base, git_rev=None)
    assert not report["valid"]
    assert any("no tensor value differs" in e for e in report["errors"])


def test_d_new_tensor_name_fails(tmp_path):
    def add_tensor(sd):
        sd = _row_edit(sd)
        sd["heart_gate.weight"] = torch.ones(1, 128)
        return sd
    run_dir, out, base = _setup(tmp_path, VALID_EDIT, add_tensor)
    report = check_run(run_dir, out, base, git_rev=None)
    assert not report["valid"]
    assert any("strict load" in e for e in report["errors"])


def test_search_loop_fails():
    assert any("scoring candidate" in r for _, r in check_source(SEARCH_LOOP_EDIT))


def test_comment_is_ignored():
    assert check_source("x = 1  # do not call .backward() here\n") == []


def test_frozen_code_check(tmp_path):
    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "m.py").write_text("x = 1\n")
    git("add", "-A")
    git("commit", "-qm", "init")
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True,
                         text=True).stdout.strip()
    assert check_frozen_code(rev, repo=tmp_path) == []
    (tmp_path / "models" / "m.py").write_text("x = 2\n")
    assert check_frozen_code(rev, repo=tmp_path)
    git("checkout", "--", "models/m.py")
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "new.py").write_text("y = 1\n")
    assert check_frozen_code(rev, repo=tmp_path)


def test_cli_exit_codes(tmp_path):
    good = tmp_path / "good.py"
    bad = tmp_path / "bad.py"
    good.write_text(VALID_EDIT)
    bad.write_text(BACKWARD_EDIT)
    tool = ROOT / "tools" / "check_wsp_edit.py"
    assert subprocess.run([sys.executable, str(tool), "--static", str(good)]).returncode == 0
    assert subprocess.run([sys.executable, str(tool), "--static", str(bad)]).returncode == 1
