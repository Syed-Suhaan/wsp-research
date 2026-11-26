# Reproduction check

Result: **PASS** (Stage 1: tolerance 0.5 points; Stage 2: 0.005 speaker similarity, 0.05 dB MCD)

Procedure: copied this folder to a temporary directory, deleted `results/`, created a fresh
virtual environment, ran `pip install -r requirements.txt`, then ran the README commands:

```
python data/shapes_v2.py
python train/train_base_v2.py
for s in 0 1 2 3 4 5 6 7; do python train/run_baselines_v2.py --condition B0 --seed $s; done
hf download n4ze3m/tiny-voca --local-dir vendor/tiny-voca
python train/make_stage2_base.py
python train/stage2_baselines.py --condition REF_M2 --phase test --seeds 0
python train/stage2_baselines.py --condition B3 --phase test --seeds 0
```

| Metric | Exported | Reproduced | Abs. difference | Tolerance |
|---|---|---|---|---|
| Base old-class acc (test) | 98.42 | 98.42 | 0.00 | 0.5 |
| B0 new-class acc, mean of seeds 0-7 | 78.25 | 78.25 | 0.00 | 0.5 |
| B0 class-9 precision, mean of seeds 0-7 | 25.73 | 25.73 | 0.00 | 0.5 |
| B0 held-out FPR near, mean of seeds 0-7 | 94.22 | 94.22 | 0.00 | 0.5 |
| B0 held-out FPR far, mean of seeds 0-7 | 41.95 | 41.95 | 0.00 | 0.5 |
| B0 old-class acc, mean of seeds 0-7 | 88.92 | 88.92 | 0.00 | 0.5 |
| B0 old-class drop, mean of seeds 0-7 | 9.50 | 9.50 | 0.00 | 0.5 |
| Stage 2 REF_M2 speaker_sim, test seed 0 | 0.6039 | 0.6039 | 0.0000 | 0.005 |
| Stage 2 REF_M2 mcd_db, test seed 0 | 11.0555 | 11.0555 | 0.0000 | 0.05 |
| Stage 2 B3 speaker_sim, test seed 0 | 0.6160 | 0.6160 | 0.0000 | 0.005 |
| Stage 2 B3 mcd_db, test seed 0 | 10.9725 | 10.9725 | 0.0000 | 0.05 |

Total time: 8.3 min on 4 vCPU, no GPU.
Date: 2026-11-26 10:12 UTC

Installed packages in the fresh environment:

```
annotated-doc==0.0.7
annotated-types==0.8.0
anyio==4.16.0
cffi==2.1.1
click==8.5.1
cuda-bindings==13.5.0
cuda-pathfinder==1.9.1
cuda-toolkit==13.2.2
dizione @ git+https://github.com/n4ze3m/dizione@8cd889e4de7be6bd1dd2e6891344a0729fa14dd4
filelock==4.3.1
fsspec==2026.9.0
h11==0.16.0
hf-xet==1.8.0
httpcore2==2.16.2
httpx2==2.16.2
huggingface_hub==2.4.0
idna==3.20
iniconfig==2.3.1
Jinja2==3.1.6
markdown-it-py==4.3.0
MarkupSafe==3.0.4
mdurl==0.1.2
mpmath==1.3.0
networkx==3.7
numpy==2.5.5
nvidia-cublas==13.4.1.3
nvidia-cuda-cccl==13.3.5.0.6
nvidia-cuda-cupti==13.2.86
nvidia-cuda-nvrtc==13.2.86
nvidia-cuda-runtime==13.2.86
nvidia-cudnn-cu13==9.26.0.51
nvidia-cufft==12.2.0.57
nvidia-cufile==1.17.1.22
nvidia-cusolver==12.2.0.11
nvidia-cusparse==12.7.10.12
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu13==2.30.7
nvidia-nvjitlink==13.5.37
nvidia-nvshmem-cu13==3.7.2
nvidia-nvtx==13.2.86
packaging==26.4
pillow==13.0.0
pluggy==1.7.0
pycparser==3.12
pydantic==2.14.4
pydantic_core==2.50.5
Pygments==2.21.0
pytest==9.2.0
pyworld==0.3.5
PyYAML==6.0.4
regex==2026.11.24
rich==15.0.0
safetensors==0.8.0
scipy==1.18.1
setuptools==80.9.0
shellingham==1.5.4
soundfile==0.14.0
sympy==1.14.0
tiny-voca @ git+https://github.com/n4ze3m/tiny-voca@4bf796d29ea775edd5a7cc5ee3a6806b3c3b91f6
tokenizers==0.23.3
torch==2.15.0
tqdm==4.70.1
transformers==5.24.1
triton==3.9.0
truststore==0.10.5
typer==0.29.3
typing-inspection==0.4.5
typing_extensions==4.16.0
```
