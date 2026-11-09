# Female validation conditioned on the right bank of the puddle

Held-out original clip frames: [481, 1252]. Exact epoch-75 centroid and the same selected centered model; no new inference.

The ROI follows the accepted right shoreline, extended vertically to image edges.
Top tip: [3290.0, 0.0]; bottom tip: [2590.0, 3170.0]. Thus the lower vertical boundary is x=2590 from y=3170 to y=4320; the upper tip already meets y=0.
Boundary/water points are excluded; both labels and predictions are filtered before matching.

## Primary comparison (20-pixel one-to-one matching radius)

| Selection | Labels | Predictions | TP | FP relative to labels | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full held-out frames | 184 | 142 | 117 | 25 | 67 | 82.39% | 63.59% | 71.78% |
| Right-bank ROI | 134 | 107 | 89 | 18 | 45 | 83.18% | 66.42% | 73.86% |

## Regional results by frame

| Frame | Labels | Predictions | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 481 | 65 | 54 | 42 | 12 | 23 | 77.78% | 64.62% | 70.59% |
| 1252 | 69 | 53 | 47 | 6 | 22 | 88.68% | 68.12% | 77.05% |

## Regional matching-radius checks

| Radius (px) | Precision | Recall | F1 | Median matched error (px) |
|---|---:|---:|---:|---:|
| 5 | 76.64% | 61.19% | 68.05% | 2.3386 |
| 10 | 83.18% | 66.42% | 73.86% | 2.4919 |
| 20 | 83.18% | 66.42% | 73.86% | 2.4919 |
| 50 | 83.18% | 66.42% | 73.86% | 2.4919 |

## Limits

- Only two held-out frames; same clip as training, possible temporal correlation.
- Validation frames also select best checkpoints/early stopping; not an independent final test set.
- Precision is relative to provided labels; unlabeled animals can be counted as false positives.
- FBody localization/detection only; no limb pose or identity accuracy measured.
- This is a post-hoc spatial subset requested after reviewing the overall validation; not a new independent test.
- ROI is a fixed coarse hand-drawn outline from the same camera; no new image-based registration or shoreline refinement.
- Point errors across the ROI boundary can change inclusion. No uncertainty band or special boundary tolerance is added.
- Regional filtering does not alter training, active video predictions, or canonical all-detection outputs.
