# Uploaded burrow locations: counts and pixel spacing

[Documentation index](README.md) · [Right-bank region definition](FEMALE_V002.md#user-requested-right-bank-conditional-validation) · [Demo copy and fixed-reference GUI layer](demo/FEMALE_BURROWS.md)

All artifact paths are project-relative. This is read-only analysis of the
uploaded annotations, not burrow-model training or neural inference.

## Completed analysis

CPU Slurm job **17171189 COMPLETED**, exit **0:0**, in 8 seconds, with all
**7 known-answer tests** passing. Source:
`data/labels/burrows/BurrowLabels3.slp` (unchanged SHA-256
`1c8aa0bbda851459f470e49c2d1cc73cda93b4eb06e5bae083f5a848ef7f7903`).

One `Burrow` landmark per human instance. Three stored frame records: frames
**563 and 948** each have **282** human-annotated points; frame 949 has no human
instances. There are **564 annotation instances across the two labeled frames**,
not a certified count of 564 different physical burrows. No identities/tracks
are supplied. The coordinates are not exactly identical across frames (zero
exact overlap); their cross-frame nearest-position median is 4.60 pixels, so
many are close enough to plausibly be repeated annotations. No matching threshold
or physical-identity merge was invented. All annotated points are finite and
within the 7680×4320 image, with no exact within-frame duplicates.

## Counts and local spacing

Use the same curved right shoreline and vertical tip extensions accepted for
female conditional validation. Apply membership to `Burrow`, not `FBody`.
Boundary/water excluded; no extra uncertainty band.

| Original clip frame | Burrow annotations | Right of puddle | Mean nearest-neighbor distance, full frame (px) | Mean nearest-neighbor distance, right ROI (px) |
|---|---:|---:|---:|---:|
| 563 | 282 | 220 | 113.56 | 108.47 |
| 948 | 282 | 214 | 119.77 | 117.60 |

Nearest-neighbor spacing means: for each annotated point, find its closest other
annotated burrow in the **same frame**, then average those distances. The right-
ROI version restricts both the query and neighbor to that region. Frames are
never pooled, since repeated locations could create artificially tiny spacing.

For the alternative meaning “distance between a randomly chosen pair,” the mean
over **all unordered pairs** is:

| Frame | Full frame (px) | Right ROI (px) |
|---|---:|---:|
| 563 | 1807.93 | 1531.68 |
| 948 | 1762.45 | 1463.26 |

These are original-image Euclidean pixels, not physical lengths or perspective-
corrected distances. Missing annotations can inflate nearest-neighbor spacing;
there is no statistical boundary correction. The saved summary also provides
right-region anchors' nearest neighbors anywhere in the full frame for ROI-edge
context; here their averages equal the within-right-ROI averages.

## Evidence and reproducibility

- `checks/burrow_locations_v001/BURROW_STATS_COMPLETE.json`: counts, spacing
  distributions, track inventory, exact-coordinate/cross-frame diagnostics,
  source/code/ROI hashes, limitations and test receipt.
- `checks/burrow_locations_v001/annotations.csv`: every original point, frame,
  instance index and right-region membership.
- `checks/burrow_locations_v001/report.md`: run-specific numerical report.
- `scripts/burrow_location_stats.py`, `tests/test_burrow_location_stats.py`,
  `jobs/burrow_location_stats.sh`: finite reproducible analysis; output refuses
  overwriting an existing completed result.

No source label edits, cross-frame burrow merging, retraining, inference,
visualizations, production-prediction changes or recurring jobs were performed.
