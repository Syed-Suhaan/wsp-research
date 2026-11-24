"""Held-out shape classes for the Stage 1 v2 TEST split.

These two classes appear only in the test split (500 images each). They are used to measure
how often an edited model calls an unfamiliar shape the new class (class-9 precision and the
held-out false-positive rate). They are never shown to the base model, the baselines, or
the WSP writers.

    held-out A (far from the new class):  crescent (filled: a disk minus an offset disk)
    held-out B (near the new class):      heptagon (outline: regular seven-sided polygon),
                                          visually close to the hexagon new class

The near/far assignment is also recorded in results/stage1_v2/heldout_note.json.

Images use the same random size, position, rotation, line width, brightness, noise, blur,
and distortion as data/shapes_v2.py under the same configuration.

Labels: 10 for held-out A (far), 11 for held-out B (near).

Usage:
    from data.heldout import load_heldout_test
    x, y = load_heldout_test(cfg)        # x uint8 (1000, 32, 32), y in {10, 11}
"""

from __future__ import annotations

import math

import numpy as np

from data.shapes_v2 import (CACHE_DIR, IMG, SS, Canvas, config_tag, ellipse_points, load_config,
                            regular_polygon)

HELDOUT_NAMES = ["crescent", "heptagon"]
HELDOUT_LABELS = [10, 11]
HELDOUT_ROLE = {10: "far", 11: "near"}
N_TEST = 500
SPLIT_ID = 3


def _draw(c: Canvas, name: str) -> None:
    if name == "crescent":
        c.fill(ellipse_points(1.0), jitter=False)
        # Cut an offset disk out of the filled disk (drawn in background colour).
        off = np.array([0.45, 0.0])
        inner = ellipse_points(1.0) * 0.85 + off
        c.draw.polygon(c.tf(inner, jitter=False), fill=0)
    elif name == "heptagon":
        c.outline(regular_polygon(7, -math.pi / 2))
    else:
        raise ValueError(name)


def generate_heldout_test(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for j, (name, label) in enumerate(zip(HELDOUT_NAMES, HELDOUT_LABELS)):
        rng = np.random.default_rng([cfg["seed"], SPLIT_ID, 100 + j])
        for _ in range(N_TEST):
            c = Canvas(rng, cfg)
            _draw(c, name)
            xs.append(np.round(c.finish() * 255).astype(np.uint8))
            ys.append(label)
    return np.stack(xs), np.array(ys, dtype=np.int64)


def load_heldout_test(cfg: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    cfg = cfg or load_config()
    path = CACHE_DIR / f"heldout_{'_'.join(HELDOUT_NAMES)}_{config_tag(cfg)}_test.npz"
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        x, y = generate_heldout_test(cfg)
        np.savez_compressed(path, x=x, y=y)
    d = np.load(path)
    return d["x"], d["y"]


assert IMG == 32 and SS == 4 and math.pi  # same rendering constants as data/shapes_v2.py
