# Blocked downloads

## 2026-11-24 — PyTorch CPU wheel host

- `pip install --index-url https://download.pytorch.org/whl/cpu torch` failed:
  the index resolves, but the wheel host `download-r2.pytorch.org` is refused by the
  network proxy (`403 Forbidden: no rule or network setting allows this host`).
- Workaround used: `torch==2.15.0` from PyPI (default Linux build with CUDA libraries).
  No GPU is present, so it runs on CPU only (`torch.cuda.is_available() == False`).
  This does not block Stage 1. To get the CPU-only wheel, allow `download-r2.pytorch.org`
  in the environment's network policy.
