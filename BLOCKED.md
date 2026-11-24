# Blocked downloads

## 2026-11-24 — PyTorch CPU wheel host

- `pip install --index-url https://download.pytorch.org/whl/cpu torch` failed:
  the index resolves, but the wheel host `download-r2.pytorch.org` is refused by the
  network proxy (`403 Forbidden: no rule or network setting allows this host`).
- Workaround used: `torch==2.15.0` from PyPI (default Linux build with CUDA libraries).
  No GPU is present, so it runs on CPU only (`torch.cuda.is_available() == False`).
  This does not block Stage 1. To get the CPU-only wheel, allow `download-r2.pytorch.org`
  in the environment's network policy.

## 2026-11-24 — Stage 2: tiny-voca model code not installed

- Task 2.0 succeeded: the tiny-voca weights, config, license, and phonemizer data were
  downloaded from Hugging Face to `vendor/tiny-voca/` (checkpoint SHA256 matches the
  model card).
- Running the model needs its Python code: the `tiny_voca` package
  (github.com/n4ze3m/tiny-voca, commit 4bf796d) and the `dizione` phonemizer
  (github.com/n4ze3m/dizione, commit 8cd889e). Installing them with
  `pip install git+https://github.com/...` was refused by this session's permission
  classifier. The network itself can reach GitHub.
- Waiting for the user to decide how to proceed. Options: allow that install; or commit the
  two packages' source to `vendor/`; or let the orchestrator write its own model
  definition from the published code.
