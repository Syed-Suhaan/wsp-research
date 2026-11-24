"""Synthetic 32x32 grayscale shape dataset with 10 classes.

Classes (index: name, drawing style):
    0 circle    filled disk
    1 square    outline
    2 triangle  outline (equilateral)
    3 star      filled five-point star
    4 cross     plus sign drawn with two strokes
    5 ring      outline circle
    6 ellipse   outline ellipse, aspect ratio 0.45-0.65
    7 pentagon  outline (regular)
    8 arrow     filled arrow (shaft and head)
    9 heart     outline heart (classic parametric heart curve)

Each image has a random size, position, rotation, line width, foreground intensity, and
additive Gaussian pixel noise. Shapes are drawn at 4x resolution and downsampled with a
box filter, which anti-aliases the edges.

Splits per class: 2000 train, 200 val, 500 test. Generation is deterministic given the seed
(default 0): every (split, class) pair has its own random stream.

Usage:
    python data/shapes.py            # generate and cache all splits in data/cache/
    from data.shapes import load_split
    x, y = load_split("val", classes=range(9))   # x: uint8 (N, 32, 32), y: int64 (N,)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASS_NAMES = ["circle", "square", "triangle", "star", "cross",
               "ring", "ellipse", "pentagon", "arrow", "heart"]
NUM_CLASSES = 10
NEW_CLASS = 9                      # held out from base training
SPLIT_SIZES = {"train": 2000, "val": 200, "test": 500}
SPLIT_IDS = {"train": 0, "val": 1, "test": 2}
IMG = 32                           # output resolution
SS = 4                             # supersampling factor
CACHE_DIR = Path(__file__).resolve().parent / "cache"


def _regular_polygon(n: int, phase: float = 0.0) -> np.ndarray:
    a = phase + 2 * math.pi * np.arange(n) / n
    return np.stack([np.cos(a), np.sin(a)], 1)


def _star(points: int = 5, inner: float = 0.45) -> np.ndarray:
    a = -math.pi / 2 + math.pi * np.arange(2 * points) / points
    r = np.where(np.arange(2 * points) % 2 == 0, 1.0, inner)
    return np.stack([r * np.cos(a), r * np.sin(a)], 1)


def _ellipse(aspect: float, n: int = 72) -> np.ndarray:
    a = 2 * math.pi * np.arange(n) / n
    return np.stack([np.cos(a), aspect * np.sin(a)], 1)


def _arrow() -> np.ndarray:
    # Points right, centred on the origin, extent [-1, 1].
    return np.array([[-1.0, -0.18], [0.15, -0.18], [0.15, -0.55], [1.0, 0.0],
                     [0.15, 0.55], [0.15, 0.18], [-1.0, 0.18]])


def _heart(n: int = 120) -> np.ndarray:
    t = 2 * math.pi * np.arange(n) / n
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    pts = np.stack([x, -y], 1)                    # image y axis points down
    pts -= (pts.max(0) + pts.min(0)) / 2          # centre the bounding box
    return pts / np.abs(pts).max()


def _transform(pts: np.ndarray, cx: float, cy: float, r: float, theta: float) -> list:
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    p = (pts @ rot.T) * r + np.array([cx, cy])
    return [tuple(v) for v in (p * SS)]


def _outline(draw: ImageDraw.ImageDraw, pts: list, width: float, fill: int) -> None:
    w = max(1, int(round(width * SS)))
    draw.line(pts + pts[:2], fill=fill, width=w, joint="curve")


def render(cls: int, rng: np.random.Generator) -> np.ndarray:
    """Render one image of class `cls`. Returns float32 (32, 32) in [0, 1]."""
    r = rng.uniform(7.0, 12.0)                    # shape radius in output pixels
    cx = rng.uniform(r + 1.5, IMG - r - 1.5)
    cy = rng.uniform(r + 1.5, IMG - r - 1.5)
    theta = rng.uniform(0, 2 * math.pi)
    lw = rng.uniform(1.5, 3.0)                    # line width in output pixels
    fg = int(round(255 * rng.uniform(0.7, 1.0)))
    noise_sigma = rng.uniform(0.03, 0.12)

    canvas = Image.new("L", (IMG * SS, IMG * SS), 0)
    d = ImageDraw.Draw(canvas)
    name = CLASS_NAMES[cls]
    if name == "circle":
        d.ellipse([(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS], fill=fg)
    elif name == "square":
        _outline(d, _transform(_regular_polygon(4, math.pi / 4), cx, cy, r, theta), lw, fg)
    elif name == "triangle":
        _outline(d, _transform(_regular_polygon(3, -math.pi / 2), cx, cy, r, theta), lw, fg)
    elif name == "star":
        d.polygon(_transform(_star(), cx, cy, r, theta), fill=fg)
    elif name == "cross":
        for arm in (np.array([[-1.0, 0.0], [1.0, 0.0]]), np.array([[0.0, -1.0], [0.0, 1.0]])):
            d.line(_transform(arm, cx, cy, r, theta), fill=fg,
                   width=max(1, int(round(lw * 1.3 * SS))))
    elif name == "ring":
        _outline(d, _transform(_ellipse(1.0), cx, cy, r, theta), lw, fg)
    elif name == "ellipse":
        aspect = rng.uniform(0.45, 0.65)
        _outline(d, _transform(_ellipse(aspect), cx, cy, r, theta), lw, fg)
    elif name == "pentagon":
        _outline(d, _transform(_regular_polygon(5, -math.pi / 2), cx, cy, r, theta), lw, fg)
    elif name == "arrow":
        d.polygon(_transform(_arrow(), cx, cy, r, theta), fill=fg)
    elif name == "heart":
        _outline(d, _transform(_heart(), cx, cy, r, theta), lw, fg)
    else:
        raise ValueError(cls)

    img = np.asarray(canvas.resize((IMG, IMG), Image.Resampling.BOX), dtype=np.float32) / 255.0
    img = img + rng.normal(0.0, noise_sigma, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def generate_split(split: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    n = SPLIT_SIZES[split]
    xs, ys = [], []
    for cls in range(NUM_CLASSES):
        rng = np.random.default_rng([seed, SPLIT_IDS[split], cls])
        for _ in range(n):
            xs.append(np.round(render(cls, rng) * 255).astype(np.uint8))
            ys.append(cls)
    return np.stack(xs), np.array(ys, dtype=np.int64)


def load_split(split: str, classes=None, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Load a split from the cache (generating it if needed), optionally filtered by class."""
    path = CACHE_DIR / f"shapes_{split}_seed{seed}.npz"
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        x, y = generate_split(split, seed)
        np.savez_compressed(path, x=x, y=y)
    d = np.load(path)
    x, y = d["x"], d["y"]
    if classes is not None:
        keep = np.isin(y, np.array(list(classes)))
        x, y = x[keep], y[keep]
    return x, y


def to_tensor(x: np.ndarray):
    """uint8 (N, 32, 32) -> float32 torch tensor (N, 1, 32, 32) in [0, 1]."""
    import torch
    return torch.from_numpy(x.astype(np.float32) / 255.0).unsqueeze(1)


if __name__ == "__main__":
    for s in SPLIT_SIZES:
        x, y = load_split(s)
        print(f"{s}: {x.shape} images, classes {np.bincount(y).tolist()}")
