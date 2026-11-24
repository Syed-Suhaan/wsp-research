"""Rebuild public_release/ from zero, and optionally test that it reproduces.

Usage:
    python tools/export_public.py            # rebuild public_release/
    python tools/export_public.py --check    # rebuild, then run the reproduction check

The reproduction check copies public_release/ into a temporary folder, creates a fresh
virtual environment there, installs public_release/requirements.txt, runs the Stage 1 base
training and the B0 evaluation with the README commands, and compares the new numbers with
the exported results (tolerance 0.5 points). The outcome is written to
public_release/REPRODUCE_CHECK.md.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public_release"

COPY_DIRS = ["data", "models", "train", "eval", "wsp", "prompts", "results", "samples", "docs"]
COPY_FILES = ["tools/check_no_gradients.py", "RESULTS.md", "requirements.txt", "LICENSE"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "cache", ".pytest_cache", "*.pt")
SCRUB_WORDS = ["CLAUDE" + ".md", "BLOCK" + "ED"]   # split so this file does not scrub itself
TOLERANCE = 0.5

REPRO_COMMANDS_BASE = [
    "python data/shapes.py",
    "python train/train_base.py",
]
REPRO_COMMANDS_B0 = [
    "for s in 0 1 2 3 4; do python train/baselines.py --condition B0 --seed $s; done",
]


def copy_tree() -> list[str]:
    missing = []
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    for d in COPY_DIRS:
        src = ROOT / d
        if src.exists():
            shutil.copytree(src, OUT / d, ignore=IGNORE)
        else:
            missing.append(d + "/")
    for f in COPY_FILES:
        src = ROOT / f
        if src.exists():
            (OUT / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, OUT / f)
        else:
            missing.append(f)
    # tools/ is a package-less folder; nothing else from tools/ is exported.
    return missing


def writer_models() -> list[str]:
    found = set()
    for p in (ROOT / "results").glob("*/W*/run_*.json"):
        r = json.loads(p.read_text())
        found.add((r.get("writer_model_id"), r.get("effort_level")))
    return [f"`{m}` (effort level: {e})" for m, e in sorted(found) if m]


def results_body() -> str:
    text = (ROOT / "RESULTS.md").read_text()
    # Drop the top-level "# Results" heading; keep everything else, one heading level down.
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(("#" + ln) if ln.startswith("## ") else ln for ln in lines).strip()


def finalize_commands() -> list[str]:
    cmds = []
    for p in sorted((ROOT / "results").glob("stage1/W*/run_*.json")):
        r = json.loads(p.read_text())
        cmds.append(f"python eval/wsp_finalize.py --condition {r['condition']} --run {r['run']} "
                    f"--writer-model {r['writer_model_id']} --effort \"{r['effort_level']}\"")
    return cmds


def write_readme() -> None:
    writers = writer_models()
    wsp_cmds = finalize_commands()
    readme = f"""# Weight-space programming: can a model add a capability to a trained network without gradients?

## Research question

Can weight-space programming (WSP) add a new capability to a trained neural network?

**WSP** means: a writer model reads a trained network's weights (and may run forward passes
and read activations on validation data), then writes new weights by reasoning. It does not
use gradients, optimizers, or automated fitting. Each edit is a short Python script
(`wsp/<stage>/<condition>/run_<n>/edit.py`) that sets weights to explicit values or explicit
closed-form formulas, with a written rationale (`rationale.md`) next to it.

- **Stage 1 (gate):** add a new class (heart) to a small CNN that was trained on 9 other
  synthetic shape classes.
- **Stage 2:** add a new voice style to the tiny-voca text-to-speech model (not yet run).

## Stage 1 method

**Data** (`data/shapes.py`): 32x32 grayscale images of 10 shape classes (circle, square,
triangle, star, cross, ring, ellipse, pentagon, arrow, heart) with random size, position,
rotation, line width, brightness, and pixel noise. Per class: 2000 train, 200 validation,
500 test images; seed 0. Class 9 (heart) is the new class. The base model never sees it.

**Base model** (`models/cnn.py`, `train/train_base.py`): a 5-layer CNN (140,778 parameters)
trained on classes 0-8, then given a zero-initialised 10th output row, so that every
condition starts from the same 10-output model.

**Conditions.** Every condition that uses examples uses the same k = 5 class-9 images from
the validation split, chosen per seed (baselines) or per run number (WSP) by
`eval/common.py:k_shot_indices`.

| ID | Condition | Uses gradients |
|----|-----------|----------------|
| B0 | Imprinting: class-9 row = unit-norm mean penultimate feature of the k examples (bias 0) | No (fixed formula) |
| B0s | Supplementary imprinting variant: row rescaled to the mean old-row norm, bias = mean old bias | No (fixed formula) |
| B1 | Fine-tune the class-9 output row only, k examples (old-class training images as negatives) | Yes |
| B2 | Fine-tune the whole model, k examples plus replay of old classes | Yes |
| W1 | WSP with the k examples. A new layer is permitted. | No |
| W2 | WSP from a text description of a heart only. No examples. | No |

Baselines run once per seed (0-4). Each WSP condition has 5 independent runs, each written
by a separate writer session with a fresh context. All runs of a condition receive the same
fixed instruction (`prompts/stage1_<condition>.md`), with only the run number changed.
During a WSP run the writer may read the base weights and run forward passes on validation
data only (old classes 0-8, plus the 5 examples for W1). It never reads the test split.
Every `edit.py` is checked by `tools/check_no_gradients.py`. A run that fails the check is
recorded as invalid and excluded from means.

**Metrics** (test split): new-class accuracy (class 9), old-class accuracy (classes 0-8) and
its drop from the base model (argmax over all 10 outputs), wall-clock time, percentage of
weights changed, and number of new parameters.

**Gate:** PASS if the W1 mean new-class accuracy is at least the B0 mean AND the W1 mean
old-class drop is at most 2 points. The table also reports whether W2 beats chance (10%).

## WSP writer

{chr(10).join("- " + w for w in writers) if writers else "- (no WSP runs recorded yet)"}

The writer model ID and effort level are recorded in every WSP run JSON
(`results/<stage>/<condition>/run_<n>.json`).

## Hardware

4 vCPU, 15 GiB RAM, no GPU. CPU PyTorch, 4 threads, deterministic algorithms, fixed seeds.

## Results

{results_body()}

## Reproduce from zero

All commands run from the root of this folder.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# Stage 1 data and base model (writes checkpoints/stage1_base.pt)
{chr(10).join(REPRO_COMMANDS_BASE)}

# Stage 1 baselines
{chr(10).join(REPRO_COMMANDS_B0)}
for c in B0s B1 B2; do for s in 0 1 2 3 4; do python train/baselines.py --condition $c --seed $s; done; done

# Stage 1 WSP: re-run every committed edit.py, check it for gradients, evaluate on test
{chr(10).join(wsp_cmds) if wsp_cmds else "# (no WSP runs recorded yet)"}

# Rebuild the results table
python eval/make_results.py

# Gradient-check a single edit script
python tools/check_no_gradients.py wsp/stage1/W1/run_0/edit.py
```

`requirements.txt` pins `torch` from PyPI. On Linux this is the CUDA-enabled build; it runs
on CPU when no GPU is present. A CPU-only build of the same version can be installed with
`pip install torch==<version> --index-url https://download.pytorch.org/whl/cpu`.

The wall-clock times of WSP runs cover the writer's whole session (reading, analysis,
writing the edit, and validation checks), so they cannot be reproduced by these commands.
Re-running `wsp_finalize.py` reports the script runtime separately (`edit_script_runtime_s`).

## Credits

Stage 2 uses tiny-voca (`n4ze3m/tiny-voca`, Apache-2.0).
"""
    (OUT / "README.md").write_text(readme)


def write_requirements() -> None:
    reqs = (ROOT / "requirements.txt").read_text().splitlines()
    pinned = [r for r in reqs if r.strip() and not r.startswith("#")]
    bad = [r for r in pinned if "==" not in r]
    if bad:
        raise SystemExit(f"unpinned requirements: {bad}")
    (OUT / "requirements.txt").write_text("\n".join(pinned) + "\n")


def scrub() -> list[str]:
    removed = []
    for p in OUT.rglob("*"):
        if not p.is_file() or p.suffix in {".pt", ".npz", ".png", ".wav"}:
            continue
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue
        lines = text.splitlines(keepends=True)
        keep = [ln for ln in lines if not any(w in ln for w in SCRUB_WORDS)]
        if len(keep) != len(lines):
            removed.append(f"{p.relative_to(OUT)}: {len(lines) - len(keep)} line(s)")
            p.write_text("".join(keep))
    return removed


def reproduce_check() -> str:
    exported = {
        "base_old": json.loads((OUT / "results/stage1/base/run_0.json").read_text())["test_old_acc"],
        "B0": [json.loads((OUT / f"results/stage1/B0/run_{s}.json").read_text()) for s in range(5)],
    }
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="wsp_repro_") as tmp:
        work = Path(tmp) / "public_release"
        shutil.copytree(OUT, work)
        shutil.rmtree(work / "results")               # results must be produced afresh
        env_bin = Path(tmp) / "venv" / "bin"
        log = []

        def sh(cmd: str) -> None:
            full = f". {env_bin}/activate && {cmd}"
            r = subprocess.run(["bash", "-c", full], cwd=work, capture_output=True, text=True)
            log.append(f"$ {cmd}\n(exit {r.returncode})")
            if r.returncode != 0:
                raise SystemExit(f"command failed: {cmd}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")

        subprocess.run([sys.executable, "-m", "venv", str(Path(tmp) / "venv")], check=True)
        sh("pip install -q --upgrade pip")
        sh("pip install -q -r requirements.txt")
        sh("pip freeze > installed.txt")
        installed = (work / "installed.txt").read_text()
        for cmd in REPRO_COMMANDS_BASE + REPRO_COMMANDS_B0:
            sh(cmd)
        new_base = json.loads((work / "results/stage1/base/run_0.json").read_text())["test_old_acc"]
        new_b0 = [json.loads((work / f"results/stage1/B0/run_{s}.json").read_text()) for s in range(5)]
    minutes = (time.time() - t0) / 60

    rows = [("Base old-class acc (test)", exported["base_old"], new_base)]
    for key, label in [("new_acc", "B0 new-class acc, mean of seeds 0-4"),
                       ("old_acc", "B0 old-class acc, mean of seeds 0-4"),
                       ("old_drop", "B0 old-class drop, mean of seeds 0-4")]:
        rows.append((label, statistics.mean(r[key] for r in exported["B0"]),
                     statistics.mean(r[key] for r in new_b0)))
    ok = all(abs(a - b) <= TOLERANCE for _, a, b in rows)
    table = "\n".join(f"| {n} | {a:.2f} | {b:.2f} | {abs(a - b):.2f} |" for n, a, b in rows)
    report = f"""# Reproduction check

Result: **{"PASS" if ok else "FAIL"}** (tolerance {TOLERANCE} points)

Procedure: copied this folder to a temporary directory, deleted `results/`, created a fresh
virtual environment, ran `pip install -r requirements.txt`, then ran the README commands:

```
{chr(10).join(REPRO_COMMANDS_BASE + REPRO_COMMANDS_B0)}
```

| Metric | Exported | Reproduced | Abs. difference |
|---|---|---|---|
{table}

Total time (install + data + training + B0): {minutes:.1f} min on 4 vCPU, no GPU.
Date: {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}

Installed packages in the fresh environment:

```
{installed.strip()}
```
"""
    (OUT / "REPRODUCE_CHECK.md").write_text(report)
    return "PASS" if ok else "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="run the reproduction check")
    args = ap.parse_args()
    missing = copy_tree()
    write_readme()
    write_requirements()
    if args.check:
        print("reproduction check:", reproduce_check())
    removed = scrub()
    if missing:
        print("not found, not copied:", ", ".join(missing))
    if removed:
        print("scrubbed lines:", "; ".join(removed))
    print(f"public_release/ rebuilt ({sum(1 for _ in OUT.rglob('*') if _.is_file())} files)")


if __name__ == "__main__":
    main()
