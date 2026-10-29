# Male-only particle-system exploratory analysis

[Documentation index](../README.md) · [Dataset schema](../datasets/v007_best_body_first_dataframe_v001/README.md)

## Scope

Requested preliminary, low-compute exploratory analyses of the published frozen
v007 dataframe. Only male-targeted detections are available; no females or burrows
are inferred. Source dataset, model, tracking and prediction artifacts remain
unchanged. Output coordinates are image pixels, times nominal frame times.

Code and outputs: `analyses/male_particle_eda_v001/` (project-relative).

- `analysis.py`: counts, occupancy, pair structure, density correlations, nearest
  neighbors, local-spacing/speed relation, MSD, radial self van Hove distributions,
  non-Gaussian parameter, overlap, velocity correlations and track-support checks.
- `test_analysis.py`: 16 known-answer unit tests, run before analysis.
- `report.py`: technical PDF, Markdown narrative and page previews.
- `run.sh`: finite 2-CPU, 16-GB, 25-minute Slurm job; no requeue, retry or sweep.
- `finalize_report.py`, `finalize_report.sh`: saved-result readback validation and
  reviewed report generation only, without recomputing the real-data estimators.

## Fixed analysis choices

Videos are analyzed separately. Raw counts retain every source detection;
spatial/motion analyses use accepted tracking detections, with land-only as the
primary selection and all accepted as a sensitivity comparison. Land filtering
splits paths at water intervals; no interpolation, relinking or gap bridging.

Counts use all frames. Spatial measurements use every 30th frame (~1 s), motion
origins every 10th original frame. Lags: 1,3,10,30,60,150,300,600,1500 frames.
MSD uses surviving origin-endpoint paths; a second analysis holds origins fixed
to those surviving 300 frames. Endpoint score >=0.8 is a diagnostic, not a
calibrated confidence filter. MSD bands use 300 origin-time-block resamples
(60 s), not independent-animal uncertainty. Random seed 20261015.

Spatial diagnostics include rectangular translation-corrected pair correlation
for all accepted points, a same-time/time-mixed pair-distance ratio for land,
and uncorrected descriptive nearest-neighbor distances. The time-mixed reference
includes persistent self positions and is NOT a causal social-interaction null.
Coarse occupancy uses 24 x 14 image cells. Density ACF includes cell-centering
and a count-adjusted spatial-profile residual; count ACF includes detrending.

## Tests and validation

Known-answer controls: stationary/ballistic paths, Gaussian random walks,
localization jitter, gap/identity isolation, water-crossing splits, fixed origin
support, PDF normalization with zero displacement, Rayleigh CDF, exact pair/NN
geometry, uniform rectangle g(r) and independent time-mixed ratios, white noise,
AR(1), cell-mean invariance, constant-value bootstrap, rotation invariance, and
invalid inputs. Full input artifact checksums and per-frame count reconciliation
are required. Track displacement ceilings and PDF counts are asserted.

All controls are computational; identities remain experimentally assigned and
not ground-truth validated. No diffusion coefficient, social force, burrow
location, female interaction or true abundance is estimated.

## Completed outputs

**[Reviewed technical report PDF (17 pages)](../../analyses/male_particle_eda_v001/report_15600418/male_crab_preliminary_report.pdf)**

- [Matching Markdown narrative](../../analyses/male_particle_eda_v001/report_15600418/report.md)
- [Numerical results and plots](../../analyses/male_particle_eda_v001/run_15600144/)
- [Headline summary JSON](../../analyses/male_particle_eda_v001/run_15600144/summary.json)
- [Final validation receipt](../../analyses/male_particle_eda_v001/report_15600418/validation.json)
- [Frozen reviewed source](../../analyses/male_particle_eda_v001/report_15600418/source/)

Analysis job **15600144** completed, exit 0:0, in **1m27s**, 2 CPUs, peak RSS
1,815,948 KiB (~1.73 GiB). Report/readback job **15600418** completed, exit 0:0,
in **23s**, 1 CPU, peak RSS 497,660 KiB (~0.47 GiB). Both were finite CPU jobs;
no training/inference or real-data reanalysis was performed in the second job.
No follow-on job remains part of this workflow.

Both jobs passed all 16 unit tests. The first verified **84 pinned input
artifacts**, exact per-frame and global count reconciliation, coordinate/ID
validity, consecutive frames, and the single-frame displacement ceiling. The
second rehashed **78 saved analysis artifacts** and checked saved PDF histogram
normalization/counts, monotone support, fixed-origin accounting, nearest-neighbor
CDFs and velocity correlations. Completion manifests pin all saved outputs.

The original 16-page report is retained in `run_15600144/`. The **recommended
17-page report** in `report_15600418/` adds the reviewed-findings synthesis;
its own completion/provenance receipts and source snapshot distinguish it from
the first version. No original completed artifact was overwritten.

### Preliminary findings

| Quantity (accepted land detections) | Cam 2_2 | Cam 2_3 |
| --- | ---: | ---: |
| Mean detections/frame | 121.09 | 141.23 |
| Median nearest-neighbor distance (px) | 148.90 | 147.87 |
| Median displacement at 1.001 s (px) | 1.64 | 2.06 |
| RMS displacement at 1.001 s (px) | 7.05 | 10.59 |
| Non-Gaussian alpha2 at 1.001 s | 9.80 | 5.13 |
| Fraction moving <10 px at 1.001 s | 90.70% | 82.72% |
| Origin support at 10.01 s | 54.59% | 45.41% |
| Origin support at 50.05 s | 20.40% | 12.97% |

- Radial van Hove distributions have a strong near-zero component and broad
  tails relative to matched-MSD Gaussian references. A shared one-pixel-scale
  feature merits localization/discretization checks, not biological interpretation.
- Cam 2_3 has 2.26 times the 1-s land MSD of Cam 2_2. Fixed 10-s-surviving origins
  reduce 1-s MSD by 26.9% / 17.4%; the endpoint-score >=0.8 check changes it by
  only +1.1% / +1.2%. Selection is consequential even with many available rows.
- Full-rectangle g(r) is elevated, while land same-time/time-mixed pair ratios
  are close to one above roughly 200 px. Much of the uniform-null clustering
  follows the occupancy footprint. Short-distance deficits cannot be called
  social avoidance because time-mixed pairs include persistent self positions.
- Velocity correlations remain positive out to 10 s; coarse-cell density memory
  lasts tens of seconds. Cam 2_3 count memory is strongly sensitive to detrending.
  None is a validated collective relaxation time or social-interaction estimate.

### Visual/document review

All 11 standalone plot images were opened: counts, track support, both occupancy
maps, both longest-segment panels, motion, van Hove, spatial pairs, correlations,
and spacing-speed. Initial cover/motion page previews and final findings/test
page previews were also checked for readable layout. The selected existing
shoreline QC crop was inspected for context, not as a random identity audit.

PNG readback, page-count/layout checks and PDF header/EOF checks passed. System
`pdfinfo` and `pdftoppm` were unavailable, so **no independent PDF parse/render
is claimed**: the inspected page previews were exported from the same figures
used to write the PDF. This limitation is recorded in `validation.json`.

All identities remain experimental; all distances remain image pixels. The
report explicitly does not infer burrows, females, physical speeds, unique-male
abundance, social forces, or a biological diffusion coefficient.
