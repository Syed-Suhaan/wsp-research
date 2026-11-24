# Results

## Stage 1 gate

Gate not yet evaluated (W1 results pending).

## Stage 1 — add class 9 (heart) to a 9-class shape CNN

Test split: 500 images per class. Accuracies in %, argmax over all 10 outputs. Cells show mean ± sample sd [min–max] over runs (seeds 0–4 for baselines, runs 0–4 for WSP). Old-class drop = base old-class accuracy − condition old-class accuracy (points). Time is the wall-clock time of the condition itself (for WSP: the writer's full session, reading to saved checkpoint).

| ID | Condition | Runs | New-class acc | Old-class acc | Old-class drop | Wall-clock (s) | Weights changed (%) | New params |
|---|---|---|---|---|---|---|---|---|
| base | Base model (class-9 row zero) | 1 | 0.00 | 100.00 | 0 | 232.5 | — | — |
| B0 | Imprinting, unit-norm row, zero bias | 5 | 100.00 ± 0.00 [100.00–100.00] | 99.34 ± 0.06 [99.24–99.40] | 0.66 ± 0.06 [0.60–0.76] | 0.0 ± 0.0 [0.0–0.0] | 0.091 ± 0.000 [0.091–0.091] | 0 ± 0 [0–0] |
| B0s | Imprinting, norm-matched row and mean bias (supplementary) | 5 | 100.00 ± 0.00 [100.00–100.00] | 70.12 ± 3.80 [66.04–74.47] | 29.88 ± 3.80 [25.53–33.96] | 0.0 ± 0.0 [0.0–0.0] | 0.092 ± 0.000 [0.092–0.092] | 0 ± 0 [0–0] |
