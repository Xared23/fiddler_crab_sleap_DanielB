# Bounded centroid-threshold prediction tests

[Documentation index](README.md) · [Original full-video run](PREDICTION_V007_BEST.md)

## User request and fixed scope

The user reports excellent visible male predictions but many missed males, and
requested ~200-frame tests at thresholds 0.1 and 0.05 against existing 0.2
predictions, with no retraining, no saved frame chunks, timestamped outputs under
`predictions/test runs/`, and tracking that permits varying animal counts.

This first comparison uses **Crab_Cam_2_2, original frames 0–199 inclusive**,
exactly 200 frames per condition (~6.67 s at 30000/1001 fps). Only two new
inferences are authorized/launched; the matching 0.2 baseline is extracted from
the existing raw SLP, not recomputed. This is not a full-video prediction run or
an expanded video/threshold sweep. The sample is a bounded first look, not a
representative recall evaluation or necessarily the user's observed missed-crab
interval.

## What changes and what remains fixed

- Same frozen validation-best v007 centroid and centered-instance checkpoints
  used for `v007_best_body_first_v001`, verified against baseline SHA-256 pins.
- **Centroid detection threshold** changes to 0.1 or 0.05.
- **Centered-instance/landmark threshold stays 0.2.** The old `peak_threshold=0.2`
  applied to both stages. Explicit stage-specific overrides isolate detection
  sensitivity without also changing which pose landmarks are reported.
- Native 7680 x 4320 coordinates, batch size 1, no ROI filtering, no instance cap,
  no count-target culling and no gap filling. Empty frames are retained.
- Package versions are checked against the original run manifest; input video
  size/mtime and the existing baseline SLP checksum are checked. Video bytes are
  not rehashed. This does not claim GPU bitwise reproducibility across hardware.

## Cross-frame tracking: no retraining or fixed count required

The existing `body_first_v001` algorithm already associates actual detections
across adjacent frames. This same algorithm is retained, rather than changing
tracking and threshold simultaneously. New tracks can appear and end at any
frame. There is no fixed animal count or track-count cap. Lowering the detection
threshold does NOT lower the tracker's independent Shell score >=0.5 rule or
change its duplicate, motion or ambiguity gates.

Every predicted instance is saved, including detections that fail association
eligibility; these simply have no ID. Thus a GUI that hides untracked instances
could hide some added predictions. Count comparisons distinguish all poses,
in-image Shell positions and observations assigned IDs.

Tracking does not create a pose where the detector missed the crab. Existing
IDs end at a missing/occluded frame; a crab emerging later from a burrow receives
a new ID. Longer-gap association/re-identification would require a separately
chosen tracking policy and scientific validation, but not necessarily neural
retraining. No such policy change is included in these tests.

The installed SLEAP API also supports built-in `TrackerConfig` without fixed
counts (`max_tracks=None`, no target-count pre/post culling). We are NOT enabling
that different tracker in this comparison. The existing body-first method
already satisfies dynamic births/deaths while preserving the previous method.

## Output layout and validation

Entry point: `jobs/predict_threshold_tests.sh` (one finite 20-minute GPU job,
no automatic requeue/retry). Config:
`configs/prediction/threshold_tests_200_v001.json`.
Implementation: `scripts/threshold_prediction_tests.py`.

Run directory: `predictions/test runs/threshold_YYMMDD_HHMM_<job-id>/` (UTC).
Each condition saves **one final SLP covering all 200 selected frames**, named
`Crab_Cam_2_2_YYMMDD_HHMM_c{200,100,050}.predictions.slp`. All pose measurements
are retained and IDs are optional metadata in that same file; there is no
separate chunk collection or duplicate raw/tracked SLP pair. A whole-selection
`.tmp.slp` is used solely for atomic save/readback and renamed on success.

Sidecars: exact config/model/code provenance; unit-test receipt; per-frame
raw/assigned/unassigned counts; counts plot; one-to-one 5-pixel Shell-position
matches to the existing baseline; three deliberately selected added-candidate
image comparisons; completion/hash manifests. Unmatched positions are candidate
extra detections, NOT confirmed recovered males. Selected crops do not measure
precision/recall. SLPs still reference the original full video and original
frame numbers, for GUI inspection without coordinate/time remapping.

Before real inference: the existing 20 tracker tests plus 12 bounded-workflow
controls exercise changing counts, births, empty frames/reappearance, rejected
row retention, actual SLP roundtrip, empty outputs, no overwrite, missing
coverage, corrupted points/scores, one-to-one spatial matching and dispatch of
thresholds/no cap/no file output. Actual loaded model-stage thresholds and
no-cap settings are checked before each inference. Final SLPs are fully read
back and compared with pre-save measurements (instance/tracking scores use
SLP's normal float32 representation; node measurements remain float64).
Existing-baseline frame counts/assigned counts must reproduce the published
export. No model, old predictions, dataset or pinned tracking code is modified.

## Status

Initial job **15640861** stopped at the synthetic test gate (22 seconds; no real
inference). Two SLP roundtrip tests exposed a fixture problem: `sio.Skeleton`
converts its input node-name list in place, mutating the shared expected names.
The fixture now passes a copy and asserts that shared names remain strings.
The failed test receipt is retained under
`predictions/test runs/threshold_261015_1957_15640861/`.

Corrected job **15641165 COMPLETED**, exit 0:0, in **5m19s**, on one GPU with
4 allocated CPUs. Peak host RSS was 4,047,792 KiB (~3.86 GiB; not GPU memory).
All **32 tests passed**, as did input validation, baseline count agreement,
effective stage-threshold/no-cap checks, and full final-SLP readback. Exactly
400 new frames were inferred (200 per requested threshold) and 200 existing
baseline frames were extracted. No other video, threshold or full-video run
was launched. No job remains running for this request.

**[Completed output directory](../predictions/test%20runs/threshold_261015_2000_15641165/)**
(timestamp `261015_2000` is UTC; each filename also includes it).

| Centroid threshold | Mean raw detections/frame | Mean assigned/frame | Final SLP |
| --- | ---: | ---: | --- |
| 0.2, existing baseline | 128.230 | 128.095 | [c200](../predictions/test%20runs/threshold_261015_2000_15641165/Crab_Cam_2_2_261015_2000_c200.predictions.slp) |
| 0.1, new inference | 135.065 | 134.770 | [c100](../predictions/test%20runs/threshold_261015_2000_15641165/Crab_Cam_2_2_261015_2000_c100.predictions.slp) |
| 0.05, new inference | 140.105 | 139.515 | [c050](../predictions/test%20runs/threshold_261015_2000_15641165/Crab_Cam_2_2_261015_2000_c050.predictions.slp) |

- [Count/spatial-match comparison plot](../predictions/test%20runs/threshold_261015_2000_15641165/threshold_comparison.png)
- [Summary JSON](../predictions/test%20runs/threshold_261015_2000_15641165/summary.json)
- [Frame 0 side-by-side](../predictions/test%20runs/threshold_261015_2000_15641165/review_frame_000000.png)
- [Frame 100 side-by-side](../predictions/test%20runs/threshold_261015_2000_15641165/review_frame_000100.png)
- [Frame 199 side-by-side](../predictions/test%20runs/threshold_261015_2000_15641165/review_frame_000199.png)

The raw count increases are approximately **5.3%** and **9.3%** relative to the
existing baseline. All baseline finite in-image Shell positions found a
one-to-one match within 5 pixels in BOTH new conditions; the mean unmatched
additional in-image positions were 6.815 and 11.830 per frame. This is a spatial
correspondence diagnostic, not evidence that every added detection is a male
or that baseline identities/poses are identical.

### Bounded visual review and recommended interpretation

The count plot and all three four-panel crop PNGs were opened after completion.
Each crop shows the same original image without overlays and at all three
thresholds. Markers are green for ID-assigned predictions, orange for unassigned
predictions, and magenta boxes for added/unmatched Shell positions.

- Frame 0 crop `(2080, 0, 1200, 900)`: an added pose is close to an existing
  pose near the shoreline. This merits a duplicate-versus-distinct-crab check;
  a >5-pixel matching distance is not proof of a new animal.
- Frame 100 crop `(3441, 1180, 1200, 900)`: additional overlays are visible in
  the central patch; inspect the original pixels and temporal sequence before
  judging whether these recover previously missed bodies.
- Frame 199 crop `(0, 0, 1200, 900)`: a 0.05-only pose appears at the upper-left
  image boundary. Its identity is visually ambiguous in the still, so it is
  not certified as a recovered male. This is an example worth checking for
  boundary/background false positives.

These are deliberately selected high-Shell-score added candidates, not a
random accuracy audit. Neither threshold is approved for a full-video run on
count gain alone. The final SLPs are the appropriate next interactive review
artifacts, paired with the original full `Crab_Cam_2_2.mp4`.

Provenance note: the completed `preview_selection.json` and its frozen source
use the field/selection label `c005` to refer to **0.05**; this is a metadata
naming typo, not a 0.005 inference. Actual config, effective-threshold checks,
summary and SLP names all correctly specify 0.05 / `c050`. The generator's
future-output label is corrected to `c050`; completed artifacts and checksums
were not rewritten, and no inference was rerun for this naming-only correction.
Exact executed code remains under the completed run's `source/` directory.

## Adjacent-frame count fluctuations

Requested follow-up completed in CPU job **15642396** (exit 0:0, 6 seconds).
The calculation uses every one of the **199 consecutive-frame pairs** in each
200-frame prediction, with no smoothing or time binning. Let
`delta = N(t+1)-N(t)` and `magnitude = abs(delta)`; the frame interval is
1001/30000 s (~33.37 ms). Primary `N` counts all pose predictions, matching the
raw-count comparison above. Variances below use the descriptive divisor 199
(`ddof=0`); they are not uncertainty estimates for the means.

| Centroid threshold | Mean absolute change | Variance of absolute change | Mean signed change | Variance of signed change |
| --- | ---: | ---: | ---: | ---: |
| 0.2 | 1.738693 | 2.524684 | 0.050251 | 5.545214 |
| 0.1 | 1.964824 | 2.737456 | 0.065327 | 6.593722 |
| 0.05 | 2.281407 | 3.197192 | 0.100503 | 8.391909 |

Mean changes have units of detection counts; variances have units of counts
squared. Mean absolute change divided by mean count is 1.356%, 1.455%, and
1.628%, respectively. Thus lower thresholds show larger net count fluctuations
in this sample, including after this simple normalization. This does not
establish that the lower thresholds are worse detectors: actual visibility,
entry/exit and false/missed detections can all contribute.

Seven known-answer tests passed: constants, linear increase, alternating counts,
hand-calculated signed/absolute variances with both denominator conventions,
count-offset invariance, gap rejection and invalid-count rejection. Source
artifact checksums, exact frames/times, count accounting, telescoping increments,
second-moment identities and saved-table readback passed.

- [Results and definitions](../analyses/threshold_count_fluctuations_v001/run_15642396/results.md)
- [All summary statistics](../analyses/threshold_count_fluctuations_v001/run_15642396/fluctuation_summary.csv)
- [Every adjacent-frame increment](../analyses/threshold_count_fluctuations_v001/run_15642396/adjacent_frame_changes.csv)

The CSV also includes sample variances (`ddof=1`), RMS fluctuations, unchanged
frame-pair fractions, finite-in-image counts and ID-assigned counts. These are
NET count fluctuations: simultaneous disappearance and appearance can cancel.
They do not measure coordinate jitter, individual detection turnover or ID
switches. No independent-increment or precision/recall inference is made.
