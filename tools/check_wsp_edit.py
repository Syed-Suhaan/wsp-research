"""Validity check for a weight-space-programming (WSP) edit.

A WSP edit is a script `edit.py` that loads a frozen checkpoint, writes values into its
state dict with explicit numbers or explicit formulas, and saves a new state dict. This tool
checks four rules:

  Rule 3, no gradients or fitting (static check of edit.py). Fails on `.backward`,
      `autograd`, `torch.func` differentiation, `functorch`, `enable_grad`,
      `requires_grad_(True)`, `requires_grad=True`, `.grad`, `torch.optim` / optimizer
      classes, `scipy.optimize`, `sklearn`, `lstsq`, `pinv`, k-means, `polyfit`,
      `curve_fit`, `minimize`, and any loop that both calls a model and selects among the
      results (a loop that scores candidate weights).
  Rule 4, the edit lives inside the network (static + output checks).
      Static: edit.py must not define a class or a `forward` function, register hooks,
      replace methods, or script/trace modules, and the run folder must not contain any
      other top-level .py file (for example a model definition).
      Output: the saved file must be a plain state dict (str -> Tensor) that loads into the
      fixed model class with strict=True (no new or missing tensor names, same shapes),
      and at least one tensor value must differ from the base checkpoint.
  Rule 5, frozen code: `git diff` against a given revision shows no change, and
      `git status` shows no untracked or modified file, under models/, eval/, and data/.

Usage:
    # full check of one run (the orchestrator runs this after each WSP run)
    python tools/check_wsp_edit.py --run-dir wsp/stage1_v2/W1/run_0 \
        --output checkpoints/stage1_v2_W1_run0.pt --base checkpoints/stage1_v2_base.pt \
        --git-rev <commit at run start> [--json report.json]

    # static check only (rule 3 and the static part of rule 4) of one or more scripts
    python tools/check_wsp_edit.py --static FILE [FILE ...]

Exit code 0 if every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXED_CLASS = "models.shapes_v2:ShapesV2Net"
FROZEN_DIRS = ("models", "eval", "data")

# ---------------------------------------------------------------- rule 3: no gradients/fitting
GRADIENT_PATTERNS = [
    (r"\.backward\b", "calls .backward"),
    (r"\bautograd\b", "uses torch.autograd"),
    (r"\btorch\.func\.(grad|vjp|jvp|jacrev|jacfwd)\b", "uses torch.func differentiation"),
    (r"\bfunctorch\b", "uses functorch"),
    (r"\benable_grad\b", "enables gradient tracking"),
    (r"requires_grad_\(\s*(True)?\s*\)", "calls requires_grad_(True)"),
    (r"requires_grad\s*=\s*True", "sets requires_grad=True"),
    (r"\.grad\b", "reads a .grad attribute"),
    (r"\btorch\.optim\b", "uses torch.optim"),
    (r"\bfrom\s+torch\s+import\s+[^\n]*\boptim\b", "imports torch.optim"),
    (r"\boptim\.", "uses an optimizer module"),
    (r"\w*Optimizer\b", "uses an optimizer class"),
    (r"\bscipy\.optimize\b|\bfrom\s+scipy\s+import\s+[^\n]*\boptimize\b", "uses scipy.optimize"),
    (r"\bsklearn\b", "uses scikit-learn"),
    (r"\blstsq\b", "uses least squares (lstsq)"),
    (r"\bpinv\b", "uses a pseudo-inverse fit (pinv)"),
    (r"\b[Kk][Mm]eans\b|\bKMeans\b", "uses k-means"),
    (r"\bpolyfit\b", "uses polyfit"),
    (r"\bcurve_fit\b", "uses curve_fit"),
    (r"\bminimize\b", "uses a minimize routine"),
]

# ------------------------------------------------------- rule 4 (static): no logic outside weights
OUTSIDE_NETWORK_PATTERNS = [
    (r"\bregister_\w*hook\b", "registers a hook"),
    (r"\.forward\s*=", "replaces a forward method"),
    (r"\bMethodType\b", "binds a new method"),
    (r"\btorch\.jit\b|\btorch\.export\b|\btorch\.fx\b", "scripts, traces, or exports a module"),
    (r"\bsetattr\s*\(", "sets attributes dynamically"),
    (r"(?<![.\w])(exec|eval)\s*\(", "executes dynamic code"),
    (r"\bmodule_file\b|\bclass_name\b", "saves a custom model reference"),
]

MODEL_CALL_RE = re.compile(
    r"(model|net|cnn|forward|logits|features|evaluate|predict|accuracy|score)", re.IGNORECASE)
SELECTION_NAME_RE = re.compile(r"(best|score|loss|candidate)", re.IGNORECASE)
SELECTION_FUNCS = {"max", "min", "argmax", "argmin", "sorted", "sort", "topk"}


def _code_lines(source: str) -> list[tuple[int, str]]:
    """(line number, line without trailing comment). A '#' inside a string literal is rare in
    edit scripts; cutting there only makes the check stricter."""
    return [(i, line.split("#", 1)[0]) for i, line in enumerate(source.splitlines(), start=1)]


def _callee_name(node: ast.Call) -> str:
    f = node.func
    parts = []
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
    return ".".join(reversed(parts))


def _loop_violations(tree: ast.AST) -> list[tuple[int, str]]:
    found = []
    for loop in ast.walk(tree):
        if not isinstance(loop, (ast.For, ast.While, ast.AsyncFor)):
            continue
        calls_model = selects = False
        for node in ast.walk(loop):
            if isinstance(node, ast.Call):
                last = _callee_name(node).split(".")[-1]
                if last and MODEL_CALL_RE.search(last):
                    calls_model = True
                if last in SELECTION_FUNCS:
                    selects = True
            elif isinstance(node, ast.Compare):
                selects = True
            elif isinstance(node, ast.Name) and SELECTION_NAME_RE.search(node.id):
                selects = True
            elif isinstance(node, ast.Attribute) and SELECTION_NAME_RE.search(node.attr):
                selects = True
        if calls_model and selects:
            found.append((loop.lineno, "rule 3: loop calls a model and selects among results "
                                       "(looks like scoring candidate weights)"))
    return found


def _definition_violations(tree: ast.AST) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            found.append((node.lineno, "rule 4: defines a class (new module or model)"))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and \
                getattr(node, "name", "") == "forward":
            found.append((node.lineno, "rule 4: defines a forward function"))
    return found


def check_source(source: str, outside_network: bool = True) -> list[tuple[int, str]]:
    """Static violations of rule 3 (and of rule 4 if outside_network) in a source string."""
    violations = []
    patterns = [(p, "rule 3: " + r) for p, r in GRADIENT_PATTERNS]
    if outside_network:
        patterns += [(p, "rule 4: " + r) for p, r in OUTSIDE_NETWORK_PATTERNS]
    for lineno, code in _code_lines(source):
        for pattern, reason in patterns:
            if re.search(pattern, code):
                violations.append((lineno, reason))
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return violations + [(e.lineno or 0, f"syntax error: {e.msg}")]
    violations += _loop_violations(tree)
    if outside_network:
        violations += _definition_violations(tree)
    return sorted(set(violations))


# ------------------------------------------------------------- rule 4 (output): state dict only
def _fixed_class(spec: str):
    module, name = spec.split(":")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return getattr(importlib.import_module(module), name)


def check_output(output: str | Path, base: str | Path, fixed_class: str = FIXED_CLASS) -> dict:
    """Check that `output` is a plain state dict that loads strictly into the fixed class and
    differs from `base` in at least one tensor value."""
    import torch

    report = {"errors": [], "changed_tensors": [], "num_values_changed": 0}
    try:
        sd = torch.load(output, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001
        report["errors"].append(f"rule 4: output is not a loadable plain state dict ({e})")
        return report
    if not isinstance(sd, dict) or not all(isinstance(v, torch.Tensor) for v in sd.values()):
        report["errors"].append("rule 4: output is not a plain state dict of tensors")
        return report
    model = _fixed_class(fixed_class)()
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        report["errors"].append("rule 4: strict load into the fixed class failed: "
                                + str(e).replace("\n", " ")[:500])
        return report
    base_sd = torch.load(base, map_location="cpu", weights_only=True)
    for name, t in sd.items():
        diff = int((t != base_sd[name]).sum()) if t.shape == base_sd[name].shape else t.numel()
        if diff:
            report["changed_tensors"].append(name)
            report["num_values_changed"] += diff
    if not report["changed_tensors"]:
        report["errors"].append("rule 4: no tensor value differs from the base checkpoint")
    return report


# ------------------------------------------------------------------------ rule 5: frozen code
def check_frozen_code(git_rev: str, repo: str | Path = ROOT, dirs=FROZEN_DIRS) -> list[str]:
    errors = []
    diff = subprocess.run(["git", "diff", "--name-only", git_rev, "--", *dirs], cwd=repo,
                          capture_output=True, text=True)
    if diff.returncode != 0:
        return [f"rule 5: git diff failed: {diff.stderr.strip()}"]
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", *dirs],
                            cwd=repo, capture_output=True, text=True)
    changed = sorted(set(diff.stdout.split()) |
                     {ln[3:] for ln in status.stdout.splitlines() if ln.strip()})
    # Python byte-code caches are not code changes.
    changed = [c for c in changed if "__pycache__" not in c and not c.endswith(".pyc")]
    if changed:
        errors.append("rule 5: frozen code changed: " + ", ".join(changed))
    return errors


# ------------------------------------------------------------------------------- full check
def check_run(run_dir: str | Path, output: str | Path, base: str | Path,
              git_rev: str | None, fixed_class: str = FIXED_CLASS) -> dict:
    run_dir = Path(run_dir)
    report = {"run_dir": str(run_dir), "static": [], "errors": []}
    edit = run_dir / "edit.py"
    if not edit.exists():
        report["errors"].append("rule 2: edit.py is missing")
    else:
        report["static"] = [f"line {n}: {r}" for n, r in check_source(edit.read_text())]
        report["errors"] += report["static"]
    extra = sorted(p.name for p in run_dir.glob("*.py") if p.name != "edit.py")
    if extra:
        report["errors"].append("rule 4: run folder contains code besides edit.py: "
                                + ", ".join(extra))
    out = check_output(output, base, fixed_class)
    report["errors"] += out["errors"]
    report["changed_tensors"] = out["changed_tensors"]
    report["num_values_changed"] = out["num_values_changed"]
    if git_rev is not None:
        report["errors"] += check_frozen_code(git_rev)
    else:
        report["frozen_code"] = "not checked (no --git-rev)"
    report["valid"] = not report["errors"]
    return report


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--static", nargs="+", metavar="FILE", help="static check only")
    ap.add_argument("--run-dir")
    ap.add_argument("--output")
    ap.add_argument("--base")
    ap.add_argument("--git-rev")
    ap.add_argument("--fixed-class", default=FIXED_CLASS)
    ap.add_argument("--json", help="write the report to this file")
    args = ap.parse_args(argv)

    if args.static:
        failed = False
        for path in args.static:
            v = check_source(Path(path).read_text())
            print(("FAIL " if v else "PASS ") + path)
            for n, r in v:
                print(f"  line {n}: {r}")
            failed |= bool(v)
        return 1 if failed else 0

    if not (args.run_dir and args.output and args.base):
        ap.error("--run-dir, --output, and --base are required (or use --static)")
    report = check_run(args.run_dir, args.output, args.base, args.git_rev, args.fixed_class)
    print(("PASS " if report["valid"] else "FAIL ") + args.run_dir)
    for e in report["errors"]:
        print("  " + e)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
