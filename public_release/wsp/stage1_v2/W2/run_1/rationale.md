# WSP edit — Stage 1 v2, condition W2, run 1

Writer: Claude Opus 5.5 (`claude-opus-5-5`). New class 9: regular hexagon outline (text description only).

## Tensors changed

- `head.weight[9]`: set to a copy of `head.weight[7]` (pentagon row).
- `head.bias[9]`: set to `head.bias[7] - 0.5`.
- `A1.weight[0]`, `A1.weight[1]`: both set to `w_dir / sigma` (the standardized hexagon-vs-pentagon discriminant direction, 128 values each).
- `A1.bias[0] = -mu/sigma - t0`, `A1.bias[1] = -mu/sigma - t0 - 0.5`.
- `A2.weight[9, 0] = +6.0`, `A2.weight[9, 1] = -6.0` (H/R = 3.0/0.5).
- Not changed: backbone, `head` rows 0-8, A1 units 2-63, A2 rows 0-8, A2 bias.

The resulting logit is `z9 = z7 - 0.5 + 3.0 * clamp((s(f) - t0) / 0.5, 0, 1)`, where `s(f)` is the score defined below.

## Why

1. **Which old class the hexagon resembles.** I checked the validation images: classes 1 (square), 2 (triangle), 5 (ring), 6 (ellipse), and 7 (pentagon) are outlines; 0, 3, and 8 are filled. A hexagon outline belongs to the family of outlined regular polygons: triangle (3), square (4), pentagon (5), hexagon (6), and the ring as the limit n -> infinity. Its nearest member is the pentagon. In feature space the base model's pentagon mean is closest to square (distance 6.5, cosine 0.81) and then to ring (cosine 0.59). Pentagon runner-up logits are mostly square (162/200) and ring (27/200).
2. **Prototype estimate (closed form).** I parameterised the family by the exterior turning angle phi = 360/n: square 90, pentagon 72, ring 0, hexagon 60. I estimated the hexagon feature mean by quadratic Lagrange interpolation through the three nearest nodes (square, pentagon, ring), evaluated at phi = 60: `m_hex = -4/9 m_sq + 25/18 m_pent + 1/18 m_ring`. Penultimate features are averages of ReLU outputs and so cannot be negative, so I clipped the result at 0 (28 of the 128 entries were negative before clipping). I also tried a purely linear pentagon-to-ring interpolation (12/72 of the way). It gives a much smaller shift. I chose the quadratic because it also uses the square-to-pentagon trend and does not depend on one segment of the family. The base head gives the estimated prototype a pentagon logit of 12.1 and ring 1.6, so the base model would most likely call hexagons "pentagon".
3. **Score s(f).** The score is a diagonal-covariance (per-feature variance) discriminant: `w_dir = (m_hex - m_pent) / (var_pent + 0.1*median(var_pent))`. The projection is standardized with the validation-pentagon mean and std, so pentagons have s ~ (0, 1). The estimated prototype sits at delta = 2.37. On this axis validation pentagons range from -2.6 to 2.0. Squares, triangles, stars, crosses, arrows, and circles are all below 0. Rings (about 4.4) and part of the ellipses are high, but their pentagon logit is far below their own logit. Because class 9 inherits z7, those classes stay protected.
4. **Threshold (closed form).** I used the equal-prior Bayes boundary between N(0,1) for pentagon and N(delta, k^2) for hexagon. I assumed k = 2, because an unseen class is likely to be spread more widely than a trained one. The boundary is the positive root of `(k^2-1)t^2 + 2*delta*t - delta^2 - 2k^2 ln k = 0`, which gives theta = 1.294.
5. **How it is placed in the weights.** Class 9 copies the pentagon row and is offset by -EPS (0.5), so with no bonus it always loses to pentagon. Two A1 ReLU units share the direction and have biases spaced 0.5 apart. Their difference, through A2 row 9, is a ramp that saturates at H = 3. The ramp start is t0 = theta - R*EPS/H = 1.210, so class 9 overtakes pentagon exactly at s = theta. The saturation limits the bonus over z7 to 2.5 logits for any input. This protects rings and ellipses whose pentagon logit is far below their own.

## Expected effect

- Hexagon images, which the base model probably predicts as pentagon, move to class 9 when s > 1.29. If hexagons are near the estimated prototype (delta = 2.37, std up to 2), roughly 70-85% of them would be captured. If the true hexagon features are closer to the pentagon than the quadratic estimate predicts (as the linear pentagon-to-ring interpolation suggests, delta about 0.7), capture would fall to about 30-40%.
- Cost: the pentagons with the highest s (about 10% of them, mostly clean pentagons with a low square logit) become class 9. That is an expected old-class drop of about 1-1.5 points. The other classes should be essentially unchanged.

## Validation evaluations

One candidate evaluation was used (1 of 3), with `evaluate_val` on the final edit:
- val old-class accuracy 97.5% (base 98.67% from my own forward pass of the unedited base model, so a drop of 1.17 points).
- Per class: [100.0, 99.5, 100.0, 100.0, 100.0, 96.5, 93.0, 89.0, 99.5]. The pentagon fell from 99.5 to 89.0; the other classes are unchanged.
- 22 old-class validation images were predicted as class 9 (21 pentagons and 1 other).

Other forward passes covered only the base model (features and logits of validation classes 0-8) for the analysis above. No candidate values were scored or searched.

## Files read

- `models/shapes_v2.py`
- `eval/wsp_tools.py`
- `eval/common.py`
- `tools/check_wsp_edit.py`
- Files in `wsp/stage1_v2/W2/run_1/` (my own edit.py, scratch scripts, and their outputs)
- `checkpoints/stage1_v2_base.pt` (via `torch.load`/`wsp_tools`) and my output `checkpoints/stage1_v2_W2_run1.pt`
- Data only through `eval/wsp_tools.load_val_old()` (validation images of classes 0-8)
