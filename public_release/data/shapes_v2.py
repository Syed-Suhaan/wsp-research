"""Synthetic 32x32 grayscale shape dataset for Stage 1 v2.

Classes:
    0 circle    filled disk
    1 square    outline
    2 triangle  outline (equilateral)
    3 star      filled five-point star
    4 cross     plus sign drawn with two strokes
    5 ring      outline circle
    6 ellipse   outline ellipse, aspect ratio 0.45-0.65
    7 pentagon  outline (regular)
    8 arrow     filled arrow (shaft and head)
    9 new class, set by "new_class" in the data configuration (default: octagon, outline)

Every image has a random size, position, rotation, line width, brightness, and additive
Gaussian pixel noise. The configuration can add Gaussian blur and a random affine distortion
(anisotropic stretch and shear) plus per-vertex jitter. Shapes are drawn at 4x resolution and
downsampled with a box filter.

Splits per class: 2000 train, 200 val, 500 test. Generation is deterministic given the
configuration; each (split, class) pair has its own random stream.

The configuration is a dict (see DEFAULT_CONFIG); the frozen Stage 1 v2 configuration is in
data/stage1_v2_config.json.

Usage:
    python data/shapes_v2.py [--config data/stage1_v2_config.json]
    from data.shapes_v2 import load_config, load_split
    cfg = load_config()
    x, y = load_split("val", cfg, classes=range(9))   # x uint8 (N, 32, 32), y int64 (N,)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OLD_CLASS_NAMES = ["circle", "square", "triangle", "star", "cross",
                   "ring", "ellipse", "pentagon", "arrow"]
NEW_CLASS = 9
NUM_CLASSES = 10
SPLIT_SIZES = {"train": 2000, "val": 200, "test": 500}
SPLIT_IDS = {"train": 0, "val": 1, "test": 2}
IMG = 32
SS = 4
DATA_DIR = Path(__file__).resolve().parent
CACHE_DIR = DATA_DIR / "cache"
CONFIG_PATH = DATA_DIR / "stage1_v2_config.json"

DEFAULT_CONFIG = {
    "new_class": "octagon",
    "k": 5,
    "seed": 0,
    "noise_sigma": [0.03, 0.12],   # per-image range of additive Gaussian noise (0-1 scale)
    "blur_sigma": [0.0, 0.0],      # per-image range of Gaussian blur sigma, output pixels
    "distortion": 0.0,             # max relative stretch / shear / vertex jitter
}
# Candidate new classes, in the order used for calibration lever 2.
NEW_CLASS_CANDIDATES = ["octagon", "hexagon", "rounded_square"]


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(json.loads(Path(path).read_text()))
    return cfg


def config_tag(cfg: dict) -> str:
    """Short hash of the fields that change the images (not k)."""
    keys = ["new_class", "seed", "noise_sigma", "blur_sigma", "distortion"]
    blob = json.dumps({k: cfg[k] for k in keys}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


# ----------------------------------------------------------------------------- geometry
def regular_polygon(n: int, phase: float = 0.0) -> np.ndarray:
    a = phase + 2 * math.pi * np.arange(n) / n
    return np.stack([np.cos(a), np.sin(a)], 1)


def star_points(points: int = 5, inner: float = 0.45) -> np.ndarray:
    a = -math.pi / 2 + math.pi * np.arange(2 * points) / points
    r = np.where(np.arange(2 * points) % 2 == 0, 1.0, inner)
    return np.stack([r * np.cos(a), r * np.sin(a)], 1)


def ellipse_points(aspect: float, n: int = 72) -> np.ndarray:
    a = 2 * math.pi * np.arange(n) / n
    return np.stack([np.cos(a), aspect * np.sin(a)], 1)


def arrow_points() -> np.ndarray:
    return np.array([[-1.0, -0.18], [0.15, -0.18], [0.15, -0.55], [1.0, 0.0],
                     [0.15, 0.55], [0.15, 0.18], [-1.0, 0.18]])


def rounded_square_points(n: int = 72, power: float = 4.0) -> np.ndarray:
    """Superellipse |x|^p + |y|^p = 1: a square with rounded corners."""
    a = 2 * math.pi * np.arange(n) / n
    c, s = np.cos(a), np.sin(a)
    x = np.sign(c) * np.abs(c) ** (2 / power)
    y = np.sign(s) * np.abs(s) ** (2 / power)
    return np.stack([x, y], 1) * 0.85


class Canvas:
    """Supersampled drawing surface with the per-image random transform applied to points."""

    def __init__(self, rng: np.random.Generator, cfg: dict, r_range=(7.0, 12.0)):
        self.rng = rng
        self.r = rng.uniform(*r_range)
        self.cx = rng.uniform(self.r + 1.5, IMG - self.r - 1.5)
        self.cy = rng.uniform(self.r + 1.5, IMG - self.r - 1.5)
        self.theta = rng.uniform(0, 2 * math.pi)
        self.lw = rng.uniform(1.5, 3.0)
        self.fg = int(round(255 * rng.uniform(0.7, 1.0)))
        self.noise_sigma = rng.uniform(*cfg["noise_sigma"])
        self.blur_sigma = rng.uniform(*cfg["blur_sigma"])
        d = cfg["distortion"]
        self.jitter = d * 0.3
        # Affine distortion: stretch along a random axis, then shear.
        stretch = 1.0 + rng.uniform(-d, d)
        shear = rng.uniform(-d, d)
        phi = rng.uniform(0, math.pi)
        rot_phi = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
        self.affine = rot_phi @ np.diag([stretch, 1.0 / max(stretch, 1e-3)]) @ rot_phi.T \
            @ np.array([[1.0, shear], [0.0, 1.0]])
        self.img = Image.new("L", (IMG * SS, IMG * SS), 0)
        self.draw = ImageDraw.Draw(self.img)

    def tf(self, pts: np.ndarray, jitter: bool = True) -> list:
        pts = pts @ self.affine.T
        if jitter and self.jitter > 0:
            pts = pts + self.rng.normal(0, self.jitter / 2, pts.shape).clip(-self.jitter, self.jitter)
        c, s = math.cos(self.theta), math.sin(self.theta)
        p = (pts @ np.array([[c, -s], [s, c]]).T) * self.r + np.array([self.cx, self.cy])
        return [tuple(v) for v in (p * SS)]

    def outline(self, pts: np.ndarray, width_scale: float = 1.0, jitter: bool = True) -> None:
        q = self.tf(pts, jitter)
        w = max(1, int(round(self.lw * width_scale * SS)))
        self.draw.line(q + q[:2], fill=self.fg, width=w, joint="curve")

    def fill(self, pts: np.ndarray, jitter: bool = True) -> None:
        self.draw.polygon(self.tf(pts, jitter), fill=self.fg)

    def stroke(self, pts: np.ndarray, width_scale: float = 1.0) -> None:
        w = max(1, int(round(self.lw * width_scale * SS)))
        self.draw.line(self.tf(pts, jitter=False), fill=self.fg, width=w)

    def finish(self) -> np.ndarray:
        img = self.img
        if self.blur_sigma > 0:
            img = img.filter(ImageFilter.GaussianBlur(self.blur_sigma * SS))
        out = np.asarray(img.resize((IMG, IMG), Image.Resampling.BOX), dtype=np.float32) / 255.0
        out = out + self.rng.normal(0.0, self.noise_sigma, out.shape).astype(np.float32)
        return np.clip(out, 0.0, 1.0)


def draw_named(c: Canvas, name: str) -> None:
    rng = c.rng
    if name == "circle":
        c.fill(ellipse_points(1.0), jitter=False)
    elif name == "square":
        c.outline(regular_polygon(4, math.pi / 4))
    elif name == "triangle":
        c.outline(regular_polygon(3, -math.pi / 2))
    elif name == "star":
        c.fill(star_points())
    elif name == "cross":
        c.stroke(np.array([[-1.0, 0.0], [1.0, 0.0]]), 1.3)
        c.stroke(np.array([[0.0, -1.0], [0.0, 1.0]]), 1.3)
    elif name == "ring":
        c.outline(ellipse_points(1.0), jitter=False)
    elif name == "ellipse":
        c.outline(ellipse_points(rng.uniform(0.45, 0.65)), jitter=False)
    elif name == "pentagon":
        c.outline(regular_polygon(5, -math.pi / 2))
    elif name == "arrow":
        c.fill(arrow_points())
    elif name == "octagon":
        c.outline(regular_polygon(8, math.pi / 8))
    elif name == "hexagon":
        c.outline(regular_polygon(6, 0.0))
    elif name == "rounded_square":
        c.outline(rounded_square_points(), jitter=False)
    else:
        raise ValueError(name)


def class_names(cfg: dict) -> list[str]:
    return OLD_CLASS_NAMES + [cfg["new_class"]]


def render(name: str, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    c = Canvas(rng, cfg)
    draw_named(c, name)
    return c.finish()


def generate_split(split: str, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    names = class_names(cfg)
    xs, ys = [], []
    for cls, name in enumerate(names):
        rng = np.random.default_rng([cfg["seed"], SPLIT_IDS[split], cls])
        for _ in range(SPLIT_SIZES[split]):
            xs.append(np.round(render(name, rng, cfg) * 255).astype(np.uint8))
            ys.append(cls)
    return np.stack(xs), np.array(ys, dtype=np.int64)


def load_split(split: str, cfg: dict | None = None, classes=None) -> tuple[np.ndarray, np.ndarray]:
    """Load a split (generating and caching it if needed), optionally filtered by class."""
    cfg = cfg or load_config()
    path = CACHE_DIR / f"shapes_v2_{config_tag(cfg)}_{split}.npz"
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        x, y = generate_split(split, cfg)
        np.savez_compressed(path, x=x, y=y)
    d = np.load(path)
    x, y = d["x"], d["y"]
    if classes is not None:
        keep = np.isin(y, np.array(list(classes)))
        x, y = x[keep], y[keep]
    return x, y


def k_shot_indices(seed: int, k: int) -> np.ndarray:
    """Indices (within the 200 class-9 VALIDATION images) of the k examples for `seed`."""
    rng = np.random.default_rng(1000 + seed)
    return np.sort(rng.choice(SPLIT_SIZES["val"], size=k, replace=False))


def k_shot_examples(seed: int, cfg: dict | None = None) -> np.ndarray:
    cfg = cfg or load_config()
    x, _ = load_split("val", cfg, classes=[NEW_CLASS])
    return x[k_shot_indices(seed, cfg["k"])]


def to_tensor(x: np.ndarray):
    """uint8 (N, 32, 32) -> float32 tensor (N, 1, 32, 32) in [0, 1]."""
    import torch
    return torch.from_numpy(x.astype(np.float32) / 255.0).unsqueeze(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    args = ap.parse_args()
    cfg = load_config(args.config) if Path(args.config).exists() else dict(DEFAULT_CONFIG)
    for s in SPLIT_SIZES:
        x, y = load_split(s, cfg)
        print(f"{s}: {x.shape} images, classes {np.bincount(y).tolist()}")
