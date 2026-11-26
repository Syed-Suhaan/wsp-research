"""List every weight of the tiny-voca checkpoint: name, shape, and parameter count.

Reads the checkpoint as plain tensors (torch.load with weights_only=True); no model code is
needed or executed.

Usage:
    python eval/inspect_tiny_voca.py [--checkpoint vendor/tiny-voca/tiny-voca.pt] [--markdown OUT]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "vendor" / "tiny-voca" / "tiny-voca.pt"))
    ap.add_argument("--markdown", help="also write the table to this Markdown file")
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    sd = ck["model"]
    rows = [(k, tuple(v.shape), v.numel()) for k, v in sd.items()]
    total = sum(n for _, _, n in rows)
    lines = [f"Checkpoint: {Path(args.checkpoint).name}  (training step {ck.get('step')}, "
             f"format version {ck.get('format_version')})",
             f"{len(rows)} tensors, {total:,} parameters", "",
             "| Name | Shape | Parameters |", "|---|---|---:|"]
    lines += [f"| `{k}` | {list(s)} | {n:,} |" for k, s, n in rows]
    text = "\n".join(lines)
    print(text)
    if args.markdown:
        Path(args.markdown).write_text(text + "\n")


if __name__ == "__main__":
    main()
