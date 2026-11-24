"""Rebuild public_release/ from zero, and optionally test that it reproduces.

Usage:
    python tools/export_public.py            # rebuild public_release/
    python tools/export_public.py --check    # rebuild, then run the reproduction check

Copied: data/, models/, train/, eval/, wsp/, prompts/, results/, samples/, docs/,
tools/check_wsp_edit.py, RESULTS.md, requirements.txt, LICENSE. Never copied: notes/,
checkpoints/, vendor/, public_release/, scratch/ folders, caches, and the project's private
instruction and status files. Files in prompts/ are copied byte for byte (checked).

The reproduction check copies public_release/ into a temporary folder, deletes results/,
creates a fresh virtual environment, installs public_release/requirements.txt, runs the
Stage 1 v2 base training and B0 (seeds 0-7) with the README commands, and compares the new
numbers with the exported results (tolerance 0.5 points). The outcome is written to
public_release/REPRODUCE_CHECK.md.
"""

from __future__ import annotations

import argparse
import json
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
COPY_FILES = ["tools/check_wsp_edit.py", "RESULTS.md", "requirements.txt", "LICENSE"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "cache", ".pytest_cache", "*.pt", "scratch")
SCRUB_WORDS = ["CLAUDE" + ".md", "BLOCK" + "ED"]   # split so this file does not scrub itself
TOLERANCE = 0.5
N_SEEDS = 8

REPRO_COMMANDS_BASE = [
    "python data/shapes_v2.py",
    "python train/train_base_v2.py",
]
REPRO_COMMANDS_B0 = [
    "for s in 0 1 2 3 4 5 6 7; do python train/run_baselines_v2.py --condition B0 --seed $s; done",
]
COMPARE_KEYS = [("new_acc", "B0 new-class acc"), ("precision9", "B0 class-9 precision"),
                ("heldout_fpr_near", "B0 held-out FPR near"), ("heldout_fpr_far", "B0 held-out FPR far"),
                ("old_acc", "B0 old-class acc"), ("old_drop", "B0 old-class drop")]


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
    return missing


def v2_runs(cond: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((ROOT / "results/stage1_v2" / cond).glob("run_*.json"))]


def writer_lines() -> list[str]:
    found = {}
    for c in ("W1", "W2"):
        for r in v2_runs(c):
            found[(r["writer_model_id"], r["effort_level"])] = r.get("effort_observations", [])
    lines = []
    for (m, e), obs in sorted(found.items()):
        lines.append(f"- Model ID `{m}`; effort level: **{e}**.")
        for o in obs:
            lines.append(f"  - {o['source']}: {o['value']}")
    return lines or ["- (no WSP runs recorded yet)"]


def results_body() -> str:
    lines = (ROOT / "RESULTS.md").read_text().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(("#" + ln) if ln.startswith("#") else ln for ln in lines).strip()


def finalize_commands() -> list[str]:
    cmds = []
    for c in ("W1", "W2"):
        for r in sorted(v2_runs(c), key=lambda r: r["run"]):
            cmds.append(f"python eval/wsp_finalize_v2.py --condition {c} --run {r['run']} "
                        f"--git-rev {r['git_rev_at_start']}")
    return cmds


def write_readme() -> None:
    cfg = json.loads((ROOT / "data/stage1_v2_config.json").read_text())
    readme = f"""# Weight-space programming: can a model add a capability to a trained network without gradients?

## Research question

Can weight-space programming (WSP) add a new capability to a trained neural network?

**WSP** means: a writer model reads a trained network's weights (and may run forward passes
and read activations on validation data), then writes new weights by reasoning. It does not
use gradients, optimizers, or automated fitting. Each edit is a short Python script
(`wsp/<stage>/<condition>/run_<n>/edit.py`) that loads the frozen checkpoint, writes tensor
values with explicit numbers or explicit closed-form formulas, and saves a new state dict.
A written rationale (`rationale.md`) sits next to it.

- **Stage 1 (gate):** add a new class to a small CNN trained on 9 synthetic shape classes.
  Stage 1 v2 (this release) is the main experiment. Stage 1 v1 was a pilot (see below).
- **Stage 2:** add a new voice style to the tiny-voca text-to-speech model (not yet run).

## Stage 1 v2 method

**Data** (`data/shapes_v2.py`, configuration frozen in `data/stage1_v2_config.json`):
32x32 grayscale images of 9 old classes (circle, square, triangle, star, cross, ring,
ellipse, pentagon, arrow) and a new class 9 ({cfg['new_class']}, outline). Every image has a
random size, position, rotation, line width, and brightness, plus pixel noise
(sigma {cfg['noise_sigma'][0]}-{cfg['noise_sigma'][1]}), blur (sigma
{cfg['blur_sigma'][0]}-{cfg['blur_sigma'][1]} px), and affine distortion with vertex jitter
(strength {cfg['distortion']}). Per class: 2000 train, 200 validation, 500 test; seed 0.
The test split also contains two **held-out shape classes** (500 images each, `data/heldout.py`).
One is visually near the new class and one is far from it. No model, baseline, or writer ever
sees them. They measure whether an edit calls unfamiliar shapes "class 9".

**Fixed model class** (`models/shapes_v2.py`): a 5-layer CNN backbone (139,488 parameters)
giving a 128-d feature f, and logits = head(f) + A2(ReLU(A1(f))). Here head = Linear(128, 10)
and the adapter slot is A1 = Linear(128, 64), A2 = Linear(64, 10), initialised to zero. The
base model (`train/train_base_v2.py`) is trained on classes 0-8; its class-9 head row, class-9
bias, and adapter are zero. Every condition is evaluated by loading a state dict into this
class with strict=True and running its standard forward. An edit can change behaviour only
through tensor values.

**Conditions.** k = {cfg['k']} class-9 validation examples per seed or run. Run i of every
condition uses the same k examples as seed i (`data.shapes_v2.k_shot_indices`).

| ID | Condition | Uses gradients |
|----|-----------|----------------|
| B0 | Imprinting: class-9 head row = unit-norm mean penultimate feature of the k examples; bias 0 | No (fixed formula) |
| B0s | Supplementary: B0 row scaled to the mean norm of the trained rows; bias = their mean | No (fixed formula) |
| B1 | Fine-tune the class-9 head row and bias only; k positives, replayed old-class negatives | Yes |
| B2 | Fine-tune the whole model, k examples plus old-class replay | Yes |
| B1t | B1 with lr, steps, and negatives per step tuned on validation (v2.1) | Yes |
| B2t | B2 with lr, steps, and positive-class weight tuned on validation (v2.1) | Yes |
| B0b | B0 with the class-9 bias tuned on validation (v2.1) | No (fixed formula) |
| W1 | WSP with the k examples (the class name is not given) | No |
| W2 | WSP from a text description of the new shape only, no examples | No |

Baselines run once per seed (0-7). Each WSP condition has 8 runs (0-7), run **one at a time**,
each by a separate writer session with a fresh context. All runs of a condition receive the
same fixed instruction (`prompts/stage1_v2_<condition>.md`), with only the run number changed.
Writers read data only through `eval/wsp_tools.py`: the base checkpoint, the class 0-8
validation images, and for W1 the run's k examples. They may make at most 3 validation
evaluations of complete candidate edits, and they list the files they read in `rationale.md`.
`tools/check_wsp_edit.py` checks every run: no gradients or fitting, only tensor values
changed, a strict load into the fixed class, at least one changed tensor, and no change to
models/, eval/, or data/ during the run (git). Invalid runs are kept and reported.

**Metrics** (test split): new-class accuracy; class-9 precision over old, new, and held-out
test images; held-out false-positive rate (near, far, and both); old-class accuracy and its
drop from the base model; wall-clock time; percentage of weights changed; adapter-slot
parameters used; and edit type (row-only, row-composition, adapter, backbone, mixed).

**Gate:** 10,000 bootstrap resamples over the 8 paired runs (W1 run i with B0 seed i).
PASS if the 95% CI lower bound of mean(W1 − B0) is at least −2 points for both new-class
accuracy and class-9 precision, and the W1 mean old-class drop is at most 2 points. Invalid
runs count as 0% new-class accuracy.

**Stage 1 v2.1 (baseline fairness).** B1 and B2 above use fixed hyperparameters carried
over from v1. v2.1 adds tuned versions (B1t, B2t) and a bias-tuned imprinting baseline (B0b).
Each tuned baseline used its full grid on the validation split (seeds 0-7, k examples
excluded). One global setting per condition was chosen by the mean of new-class accuracy and
class-9 precision, subject to an old-class drop of at most 2 points. That rule was committed
before any grid result existed (`results/stage1_v2_1/selection_rule.json`). The chosen setting
was then evaluated once on test. Each WSP writer had at most 3 validation evaluations. The
v2 gate is unchanged, and no WSP runs were added.

**Calibration disclosure.** The data setup was chosen by a difficulty calibration over 10
attempts (`results/stage1_v2/calibration.json`). The goal was a B0 mean new-class accuracy of
60-80% on the validation split. The selection rule applied the levers in order and stopped at
the first setup in range: (1) k = 2 instead of 5; (2) a new class closer to an old class; (3)
more noise, blur, and distortion, with a retrained base model. Lever 3 was tried at
increasing strength, with new classes in the fixed order octagon, rounded square, hexagon.
The selected setup has the lowest B0 validation accuracy of all attempts; the others were
94.8-100%.

## Stage 1 v1 pilot and what v2 changes

The v1 pilot (`results/stage1_v1/`, `wsp/stage1_v1/`) added a "heart" class with k = 5.
It showed four problems, and v2 addresses each one:

1. **Ceiling effect.** The base model reached 100% on old classes, and plain imprinting (B0)
   reached 100% on the new class, so no condition could beat B0.
   *v2:* harder data and k = 2, calibrated before any WSP run so that B0 scores 60-80% on
   validation, plus a paired bootstrap gate instead of a comparison of means.
2. **Open-set shortcut.** Several v1 writers made class 9 a "none of the old classes"
   detector. This works only because the v1 test set contained no other new shapes.
   *v2:* the test split has two held-out shape classes (one near the new class, one far);
   class-9 precision and held-out false-positive rates are reported, and precision is part
   of the gate.
3. **Edits outside the network.** v1 allowed new modules. One v1 run changed no weights and
   decided class 9 with a hand-written pixel rule.
   *v2:* every edit must be a state dict for one fixed model class (with a zero-initialised
   adapter slot for added capacity), loaded with strict=True. `tools/check_wsp_edit.py`
   rejects new modules, forward functions, hooks, and edits that change no tensor. A
   re-check of v1 with these rules is in `results/stage1_v1/recheck.json`; it marks the v1
   pixel-rule run invalid.
4. **Isolation.** The v1 writers ran in parallel and shared one scratch directory, and
   several saw fragments of other runs' files.
   *v2:* runs execute one at a time, each with its own scratch folder and TMPDIR. Writers may
   read only a fixed list of files and record the files they read.

## WSP writer

{chr(10).join(writer_lines())}

The writer model ID and effort level are recorded in every WSP run JSON
(`results/stage1_v2/<condition>/run_<n>.json`). The effort level is "unverified": it could
not be confirmed from a record outside the model (`results/stage1_v2/effort_probe.json`).

## Hardware

4 vCPU, 15 GiB RAM, no GPU. CPU PyTorch, 4 threads, deterministic algorithms, fixed seeds.

## Results

{results_body()}

## Reproduce from zero

All commands run from the root of this folder.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# Stage 1 v2 data and base model (writes checkpoints/stage1_v2_base.pt)
{chr(10).join(REPRO_COMMANDS_BASE)}

# Stage 1 v2 baselines
{chr(10).join(REPRO_COMMANDS_B0)}
for c in B0s B1 B2; do for s in 0 1 2 3 4 5 6 7; do python train/run_baselines_v2.py --condition $c --seed $s; done; done

# Stage 1 v2.1 tuned baselines: validation grid, then the chosen setting on test
for c in B0b B1t B2t; do python train/tune_baselines_v2.py --condition $c --phase grid; done
for c in B0b B1t B2t; do python train/tune_baselines_v2.py --condition $c --phase test; done

# Stage 1 v2 WSP: re-run each committed edit.py, check validity, evaluate on test.
# --git-rev is the commit at the start of the run, used for the frozen-code check; in a
# copy without git history, use the check from tools/check_wsp_edit.py without it.
{chr(10).join(finalize_commands()) if finalize_commands() else "# (no WSP runs recorded yet)"}

# Results table
python eval/make_results_v2.py

# Check a single edit
python tools/check_wsp_edit.py --run-dir wsp/stage1_v2/W1/run_0 \\
    --output checkpoints/stage1_v2_W1_run0.pt --base checkpoints/stage1_v2_base.pt

# Calibration attempts (each trains a base model for its data setup)
python eval/calibrate_v2.py --lever 0 --new-class octagon --k 5
```

`requirements.txt` pins `torch` from PyPI. On Linux this is the CUDA-enabled build, which
runs on CPU when no GPU is present. A CPU-only build of the same version can be installed
with `pip install torch==<version> --index-url https://download.pytorch.org/whl/cpu`.

WSP wall-clock times cover the writer's whole session, so these commands cannot reproduce
them. `wsp_finalize_v2.py` reports the edit script's own runtime separately.

The Stage 1 v1 pilot code is kept for reference (`data/shapes.py`, `models/cnn.py`,
`train/train_base.py`, `train/baselines.py`, `eval/make_results_v1.py`). The v1 WSP edit
scripts refer to their original folder (`wsp/stage1/...`) and to the v1 data-access module
(kept as `eval/wsp_tools_v1.py`), so they are a record and cannot be re-run as they are.

## License and credits

Apache-2.0 (see `LICENSE`). Stage 2 will use tiny-voca (`n4ze3m/tiny-voca`, Apache-2.0).
"""
    (OUT / "README.md").write_text(readme)


def write_requirements() -> None:
    reqs = [r for r in (ROOT / "requirements.txt").read_text().splitlines()
            if r.strip() and not r.startswith("#")]
    bad = [r for r in reqs if "==" not in r]
    if bad:
        raise SystemExit(f"unpinned requirements: {bad}")
    (OUT / "requirements.txt").write_text("\n".join(reqs) + "\n")


def scrub() -> list[str]:
    """Remove lines containing SCRUB_WORDS, except in prompts/: exported prompt files must be
    byte-identical to the files the writers used."""
    removed = []
    for p in OUT.rglob("*"):
        if not p.is_file() or p.suffix in {".pt", ".npz", ".png", ".wav"}:
            continue
        if p.relative_to(OUT).parts[0] == "prompts":
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
    def b0(root: Path) -> list[dict]:
        return [json.loads((root / f"results/stage1_v2/B0/run_{s}.json").read_text()) for s in range(N_SEEDS)]

    exp_base = json.loads((OUT / "results/stage1_v2/base/run_0.json").read_text())["test_old_acc"]
    exp_b0 = b0(OUT)
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="wsp_repro_") as tmp:
        work = Path(tmp) / "public_release"
        shutil.copytree(OUT, work)
        shutil.rmtree(work / "results")
        env_bin = Path(tmp) / "venv" / "bin"

        def sh(cmd: str) -> None:
            r = subprocess.run(["bash", "-c", f". {env_bin}/activate && {cmd}"], cwd=work,
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise SystemExit(f"command failed: {cmd}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")

        subprocess.run([sys.executable, "-m", "venv", str(Path(tmp) / "venv")], check=True)
        sh("pip install -q --upgrade pip")
        sh("pip install -q -r requirements.txt")
        sh("pip freeze > installed.txt")
        installed = (work / "installed.txt").read_text()
        for cmd in REPRO_COMMANDS_BASE + REPRO_COMMANDS_B0:
            sh(cmd)
        new_base = json.loads((work / "results/stage1_v2/base/run_0.json").read_text())["test_old_acc"]
        new_b0 = b0(work)
    minutes = (time.time() - t0) / 60
    rows = [("Base old-class acc (test)", exp_base, new_base)]
    for key, label in COMPARE_KEYS:
        rows.append((f"{label}, mean of seeds 0-7", statistics.mean(r[key] for r in exp_b0),
                     statistics.mean(r[key] for r in new_b0)))
    ok = all(abs(a - b) <= TOLERANCE for _, a, b in rows)
    table = "\n".join(f"| {n} | {a:.2f} | {b:.2f} | {abs(a - b):.2f} |" for n, a, b in rows)
    (OUT / "REPRODUCE_CHECK.md").write_text(f"""# Reproduction check

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
""")
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
    leftovers = [str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.name in ("scratch", "notes")]
    if leftovers:
        raise SystemExit(f"forbidden folders exported: {leftovers}")
    for p in (ROOT / "prompts").rglob("*"):
        if p.is_file() and p.read_bytes() != (OUT / "prompts" / p.relative_to(ROOT / "prompts")).read_bytes():
            raise SystemExit(f"exported prompt differs from the original: {p.name}")
    if missing:
        print("not found, not copied:", ", ".join(missing))
    if removed:
        print("scrubbed lines:", "; ".join(removed))
    print(f"public_release/ rebuilt ({sum(1 for p in OUT.rglob('*') if p.is_file())} files)")


if __name__ == "__main__":
    main()
