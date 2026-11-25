"""Generate Stage 2 target-voice audio with Kokoro-82M (hexgrad/Kokoro-82M, Apache-2.0).

Runs only in the target-generation environment (.venv-target, see requirements-target.txt).
The generated WAVs are committed as data, so no other step needs Kokoro.

Kokoro's American English pipeline (lang_code "a") synthesizes each sentence at 24 kHz with
default speed 1.0. Every chunk the pipeline yields for one sentence is concatenated.

Usage:
    .venv-target/bin/python data/make_stage2_kokoro.py --voices af_heart am_michael \
        --split val --count 3 --out samples/stage2/candidates
Writes <out>/<split>_<i>_<voice>.wav and <out>/generation.json (versions and the Hugging Face
revision of the model files).
"""

from __future__ import annotations

import argparse
import json
import platform
from importlib import metadata
from pathlib import Path

import wave as wavfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "hexgrad/Kokoro-82M"
SR = 24000


def write_wav(path: Path, x: np.ndarray) -> None:
    """16-bit PCM mono WAV with the standard library (no extra package needed)."""
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wavfile.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def hf_revision() -> str | None:
    """Commit hash of the cached Kokoro-82M snapshot that was used."""
    from huggingface_hub import scan_cache_dir
    for repo in scan_cache_dir().repos:
        if repo.repo_id == REPO_ID:
            revs = sorted(repo.revisions, key=lambda r: r.last_modified)
            return revs[-1].commit_hash if revs else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", nargs="+", required=True)
    ap.add_argument("--split", default="val", choices=["enroll", "val", "test"])
    ap.add_argument("--count", type=int, default=None, help="first N sentences of the split")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from kokoro import KPipeline
    torch.manual_seed(0)
    torch.set_num_threads(4)
    texts = json.loads((ROOT / "data" / "stage2_sentences.json").read_text())[args.split]
    texts = texts[: args.count] if args.count else texts
    pipeline = KPipeline(lang_code="a")  # model and voices from hexgrad/Kokoro-82M (KModel.REPO_ID)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for voice in args.voices:
        for i, text in enumerate(texts):
            chunks = [r.audio.numpy() for r in pipeline(text, voice=voice, speed=1.0) if r.audio is not None]
            wave = np.concatenate(chunks).astype(np.float32)
            path = out / f"{args.split}_{i:02d}_{voice}.wav"
            write_wav(path, wave)
            files.append({"file": path.name, "voice": voice, "split": args.split, "index": i, "text": text,
                          "seconds": round(len(wave) / SR, 3), "peak": float(np.abs(wave).max())})
            print(f"{path.name}: {len(wave) / SR:.2f}s", flush=True)
    info = {"model": REPO_ID, "hf_revision": hf_revision(), "lang_code": "a", "speed": 1.0, "sample_rate": SR,
            "python": platform.python_version(),
            "packages": {p: metadata.version(p) for p in ("kokoro", "misaki", "torch", "spacy", "en_core_web_sm", "numpy")},
            "files": files}
    (out / "generation.json").write_text(json.dumps(info, indent=2) + "\n")
    print("revision", info["hf_revision"])


if __name__ == "__main__":
    main()
