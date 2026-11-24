"""WSP edit, Stage 1, condition W1, run 2: add class 9 (heart) to the base ShapeCNN.

Reads the frozen checkpoint checkpoints/stage1_base.pt and writes
checkpoints/stage1_W1_run2.pt. No gradients, optimizers, fitting routines or searches are
used. All new weights are closed-form statistics of penultimate features:
  * the 5 class-9 validation examples assigned to run 2, and
  * the validation images of classes 0-8.

Edit summary
  * All base weights (conv1-conv5, bn1-bn5, fc) are copied unchanged.
  * A new linear layer `heart_pair` (128 -> 9, 1,161 parameters) is added. Row c is a
    diagonal-covariance Gaussian discriminant between the heart class and old class c.
  * Output 9 = (largest old logit) + min_c heart_pair_c(f). See model_def.py.

Run from the repository root:
    .venv/bin/python wsp/stage1/W1/run_2/edit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)

from eval.common import predict_features  # noqa: E402
from eval.wsp_tools import k_shot_examples, load_val_old  # noqa: E402
from models.cnn import load_model, save_checkpoint  # noqa: E402

RUN = 2
RUN_DIR = Path("wsp/stage1/W1/run_2")
BASE_CKPT = REPO_ROOT / "checkpoints" / "stage1_base.pt"
OUT_CKPT = REPO_ROOT / "checkpoints" / "stage1_W1_run2.pt"
NUM_OLD = 9

# Variance floor added to every per-dimension class variance before dividing by it.
# Penultimate features are averages of ReLU maps; many dimensions are almost always zero for a
# given class, so their variance is near 0 and an unfloored 1/var weight would be unbounded.
# Typical per-dimension standard deviations are 0.06-0.2 (variance 0.004-0.04); a floor of
# 0.01 (= a standard deviation of 0.1) caps any single dimension's weight at 100 times its
# mean difference and so limits how much one noisy dimension can dominate.
VAR_FLOOR = 1e-2

sys.path.insert(0, str(REPO_ROOT / RUN_DIR))
from model_def import HeartPairwiseCNN  # noqa: E402


def main() -> None:
    base = load_model(BASE_CKPT)

    # ---- 1. Penultimate features (forward passes only) -------------------------------------
    x_old, y_old = load_val_old()
    y_old = torch.as_tensor(y_old)
    f_old = predict_features(base, x_old)                 # (1800, 128)
    f_heart = predict_features(base, k_shot_examples(RUN))  # (5, 128)

    # ---- 2. Per-class statistics (closed form) -------------------------------------------
    # one_hot (1800, 9) turns per-class means into a single matrix product, without a loop.
    one_hot = torch.nn.functional.one_hot(y_old, NUM_OLD).float()
    counts = one_hot.sum(0)                                             # (9,)
    mu = (one_hot.T @ f_old) / counts[:, None]                          # (9, 128) class means
    var = (one_hot.T @ f_old.pow(2) - counts[:, None] * mu.pow(2)) / (counts[:, None] - 1)
    mu_heart = f_heart.mean(0)                                          # (128,) heart mean

    # ---- 3. Pairwise discriminant directions ----------------------------------------------
    # Row c: (heart mean - class-c mean) / (class-c variance + floor). This is the linear
    # discriminant of two Gaussians with a shared diagonal covariance, where the covariance is
    # taken from class c (200 images) because 5 heart images cannot estimate 128 variances.
    # The base features show that hearts look like an ellipse outline plus corner evidence:
    # compared with ellipses, hearts have lower curvature-channel activations and higher
    # activations on channels that respond to squares, pentagons and triangles. The
    # heart-vs-ellipse direction therefore weights exactly those channels.
    direction = (mu_heart[None, :] - mu) / (var + VAR_FLOOR)            # (9, 128)

    # ---- 4. Projections and thresholds ---------------------------------------------------
    # proj_old[i, c] = projection of old image i on direction c.
    proj_old = f_old @ direction.T                                      # (1800, 9)
    proj_heart = f_heart @ direction.T                                  # (5, 9)
    # Mean and std of class c along its own direction c (diagonal of the per-class stats).
    m_c = ((one_hot * proj_old).sum(0)) / counts                        # (9,)
    s_c = (((one_hot * (proj_old - m_c[None, :]).pow(2)).sum(0)) / (counts - 1)).sqrt()
    m_h = proj_heart.mean(0)                                            # (9,)
    s_h = proj_heart.std(0)                                             # (9,) unbiased, n = 5
    # Threshold at the point that is the same number of standard deviations from both
    # means ("equal-z" point): (t - m_c) / s_c = (m_h - t) / s_h.
    threshold = (m_c * s_h + m_h * s_c) / (s_c + s_h)                   # (9,)

    # ---- 5. Write the new layer --------------------------------------------------------------
    # heart_pair_c(f) = (direction_c . f - threshold_c) / s_c: signed distance from the
    # heart/class-c boundary, in units of class-c standard deviations along that direction.
    model = HeartPairwiseCNN(num_classes=10)
    state = base.state_dict()                                           # all base weights
    state["heart_pair.weight"] = direction / s_c[:, None]
    state["heart_pair.bias"] = -threshold / s_c
    model.load_state_dict(state)
    model.eval()

    save_checkpoint(
        model, OUT_CKPT,
        meta={"condition": "W1", "run": RUN, "var_floor": VAR_FLOOR,
              "threshold_z": ((threshold - m_c) / s_c).tolist(),
              "heart_z_mean": ((m_h - m_c) / s_c).tolist()},
        arch="custom", module_file=str(RUN_DIR / "model_def.py"),
        class_name="HeartPairwiseCNN", init_kwargs={"num_classes": 10},
    )
    print("saved", OUT_CKPT)
    print("threshold (class-c sd above class-c mean):",
          [round(v, 2) for v in ((threshold - m_c) / s_c).tolist()])
    print("heart mean (class-c sd above class-c mean):",
          [round(v, 2) for v in ((m_h - m_c) / s_c).tolist()])


if __name__ == "__main__":
    main()
