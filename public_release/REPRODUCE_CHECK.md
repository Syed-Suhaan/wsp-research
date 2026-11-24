# Reproduction check

Result: **PASS** (tolerance 0.5 points)

Procedure: copied this folder to a temporary directory, deleted `results/`, created a fresh
virtual environment, ran `pip install -r requirements.txt`, then ran the README commands:

```
python data/shapes.py
python train/train_base.py
for s in 0 1 2 3 4; do python train/baselines.py --condition B0 --seed $s; done
```

| Metric | Exported | Reproduced | Abs. difference |
|---|---|---|---|
| Base old-class acc (test) | 100.00 | 100.00 | 0.00 |
| B0 new-class acc, mean of seeds 0-4 | 100.00 | 100.00 | 0.00 |
| B0 old-class acc, mean of seeds 0-4 | 99.34 | 99.34 | 0.00 |
| B0 old-class drop, mean of seeds 0-4 | 0.66 | 0.66 | 0.00 |

Total time (install + data + training + B0): 6.4 min on 4 vCPU, no GPU.
Date: 2026-11-24 13:00 UTC

Installed packages in the fresh environment:

```
cuda-bindings==13.5.0
cuda-pathfinder==1.9.1
cuda-toolkit==13.2.2
filelock==4.3.1
fsspec==2026.9.0
iniconfig==2.3.1
Jinja2==3.1.6
MarkupSafe==3.0.4
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
Pygments==2.21.0
pytest==9.2.0
setuptools==84.2.1
sympy==1.14.0
torch==2.15.0
triton==3.9.0
typing_extensions==4.16.0
```
