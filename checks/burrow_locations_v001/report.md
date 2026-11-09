# Burrow annotation counts and pixel spacing

Source: `data/labels/burrows/BurrowLabels3.slp`. 564 annotation instances over 2 labeled frames; do not assume the sum is unique physical burrows.

Same curved right-of-puddle region and vertical tip extensions as female conditional validation. Distances are within each frame, never across pooled repeated frames.

| Frame | Annotated points | Right of puddle | Full-frame mean nearest neighbor (px) | Right-region mean nearest neighbor (px) | Full-frame mean all-pairs distance (px) | Right-region mean all-pairs distance (px) |
|---|---:|---:|---:|---:|---:|---:|
| 563 | 282 | 220 | 113.56 | 108.47 | 1807.93 | 1531.68 |
| 948 | 282 | 214 | 119.77 | 117.60 | 1762.45 | 1463.26 |

Exact distinct coordinate pairs across frames: 564 overall, 434 in the right ROI. This is a coordinate audit, NOT a physical-burrow deduplication.

## Limits

- Annotations across different frames may repeat the same physical burrows; totals across frames are not certified unique-burrow counts.
- No cross-frame point merging or association threshold has been invented.
- Exact within-frame duplicate coordinates are removed ONLY for distance calculations and remain in annotation counts/audit CSV.
- Nearest neighbors are among annotated burrows, not necessarily all physical burrows; incomplete labeling can increase apparent spacing.
- No image-edge or ROI-edge statistical correction; right-anchor/full-frame-neighbor distances are also supplied for boundary context.
- Pixel spacing depends on perspective and is not a physical distance.
