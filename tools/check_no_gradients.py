"""Static check that a weight-edit script uses no gradients, optimizers, or fitting.

Usage:
    python tools/check_no_gradients.py FILE [FILE ...]

Exit code 0 if every file passes, 1 if any file fails. Each violation is printed
with its line number.

A file fails if it contains any of the following:
  * gradient machinery: `.backward`, `autograd`, `torch.func.grad`, `functorch`,
    `.grad` attribute access, `requires_grad_(True)`, `requires_grad=True`,
    `enable_grad`
  * optimizers: `torch.optim`, `scipy.optimize`, any name ending in `Optimizer`
  * fitting routines: `lstsq`, `pinv`, `KMeans` / `kmeans`, `sklearn`, `polyfit`,
    `curve_fit`, `minimize`
  * a loop that scores candidate weights: a `for` / `while` loop whose body both
    calls a model (a call whose final callee name contains `model`, `net`, `cnn`,
    `forward`, `logits`, `features`, `evaluate`, `predict`, `accuracy`, or `score`)
    and selects among results (a comparison
    operator, a call to `max` / `min` / `argmax` / `argmin` / `sorted`, or a
    name containing `best`, `score`, `loss`, or `candidate`).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

# Text patterns that are forbidden anywhere in the source (comments excluded).
FORBIDDEN_PATTERNS = [
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

MODEL_CALL_RE = re.compile(
    r"(model|net|cnn|forward|logits|features|evaluate|predict|accuracy|score)", re.IGNORECASE)
SELECTION_NAME_RE = re.compile(r"(best|score|loss|candidate)", re.IGNORECASE)
SELECTION_FUNCS = {"max", "min", "argmax", "argmin", "sorted", "sort", "topk"}


def _strip_comments(source: str) -> list[tuple[int, str]]:
    """Return (line_number, code_without_comment) pairs. Docstrings are not removed."""
    out = []
    for i, line in enumerate(source.splitlines(), start=1):
        # Remove a trailing comment. A '#' inside a string literal is rare in edit scripts;
        # a false positive only makes the check stricter.
        code = line.split("#", 1)[0]
        out.append((i, code))
    return out


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
        calls_model = False
        selects = False
        for node in ast.walk(loop):
            if isinstance(node, ast.Call):
                name = _callee_name(node)
                last = name.split(".")[-1] if name else ""
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
            found.append((loop.lineno, "loop calls a model and selects among results "
                                       "(looks like scoring candidate weights)"))
    return found


def check_source(source: str) -> list[tuple[int, str]]:
    """Return a list of (line, reason) violations for a source string."""
    violations = []
    for lineno, code in _strip_comments(source):
        for pattern, reason in FORBIDDEN_PATTERNS:
            if re.search(pattern, code):
                violations.append((lineno, reason))
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        violations.append((e.lineno or 0, f"syntax error: {e.msg}"))
        return violations
    violations.extend(_loop_violations(tree))
    return sorted(set(violations))


def check_file(path: str | Path) -> list[tuple[int, str]]:
    return check_source(Path(path).read_text())


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    failed = False
    for path in argv:
        violations = check_file(path)
        if violations:
            failed = True
            print(f"FAIL {path}")
            for lineno, reason in violations:
                print(f"  line {lineno}: {reason}")
        else:
            print(f"PASS {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
