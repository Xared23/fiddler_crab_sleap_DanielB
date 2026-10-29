# 1,000-frame centroid-hysteresis / five-missed-frame tests

## Scope and status

User-authorized bounded test on **Crab_Cam_2_2, original frames 0–999**.
No retraining, full-video prediction, production replacement, or parameter sweep.
Submitted as Slurm job **15646126**. Submission is not completion; no test or
inference results are claimed here yet. Logs:
`logs/prediction_grace_tests/job_15646126.out` and `.err`.

| Condition | New-track centroid threshold | Continuation centroid threshold | Maximum intervening missed frames |
|---|---:|---:|---:|
| `b200_c050_g5` | 0.2 | 0.05 | 5 |
| `b100_c050_g5` | 0.1 | 0.05 | 5 |

The frozen v007 best model pair is used once at centroid threshold 0.05;
independent trackers then process the same 1,000-frame candidate pool. This
avoids redundant GPU inference while giving separate final prediction files.
The centered-instance **landmark threshold remains 0.2**. Actual centroid
confidence comes from `Outputs.pred_centroid_values`, checked against the
installed top-down model's `instance_scores`; it is **not Shell confidence**.

## Association and missing observations

- Existing Shell score >=0.5, optional Shoulder score >=0.2, image bounds,
  5-pixel duplicate rejection, and body/Shoulder geometry gates remain unchanged.
- A birth requires centroid confidence >= the condition's new-track threshold.
  An established track can use a pose down to centroid confidence 0.05, provided
  the same body-quality and association checks pass.
- A last observation at frame `f` can link again at `f+6`, with frames `f+1` to
  `f+5` missing. It cannot revive at `f+7` after six intervening misses.
- Track memory never produces a pose. Missing frames contain no fabricated,
  held, interpolated, or extrapolated measurements; gap CSVs enumerate them.
- Prediction uses bounded exponentially updated body velocity. For elapsed
  frames `dt`, displacement <= `12*dt` pixels and prediction residual <=18
  pixels are required. The residual bound is **not** expanded with the gap.
- Cost is `0.7 * displacement/dt + 0.3 * prediction_residual`, plus the existing
  optional capped Shoulder-vector penalty. Cost <=12, unmatched cost 7,
  Hungarian one-to-one matching, and minimum alternative margin 2 are retained.
  Both stale active tracks and current rejected body candidates participate in
  ambiguity checks where geometrically feasible.
- An unresolved match does not create a parallel/shadow ID for a competing
  high-confidence pose. That pose remains in the final SLP without an ID.
- No fixed animal count, track cap, water mask, limb association, or
  reidentification after grace expiry. A pose after expiry can start a new ID
  only if it meets the new-track threshold. Birth/death counts remain variable.

This is an experimental custom tracker, not a claim of validated animal identity.
The original adjacent-frame tracker and its published predictions are unchanged.

## Outputs

Run directory: `predictions/test runs/grace_YYMMDD_HHMM_<job-id>/` (UTC).

Exactly two final whole-selection prediction SLPs are written:

- `Crab_Cam_2_2_YYMMDD_HHMM_b200_c050_g5.predictions.slp`
- `Crab_Cam_2_2_YYMMDD_HHMM_b100_c050_g5.predictions.slp`

Each SLP retains every pose at/above its new-track threshold, even if rejected
for identity, plus lower-confidence poses **successfully linked** to established
tracks. Unmatched weak candidates do not become detections in these conditional
outputs. All original frame numbers, empty frames, and retained measurements are
preserved and verified by SLP readback. No frame-chunk SLPs are saved.

Raw predictions are independently preserved in the whole-selection
`raw_candidates.npz`, including candidates excluded from both final SLPs and
centroid slots whose pose is entirely missing. Keys:

- `frame`, `slot`: original frame number and model-output slot within that frame.
- `xy` (N,5,2), `landmark_scores` (N,5): original native-image pose measurements.
- `centroid_xy` (N,2), `centroid_scores`, `instance_scores`: detection-stage output.
- `selected_frames` (1000,), `node_names` (5,): frame coverage and node ordering.

Load with `numpy.load(path, allow_pickle=False)`. This is not an additional SLP
or a collection of saved inference chunks. Packaging excludes all-NaN poses
from SLPs, matching the SLEAP API; their slots remain in this raw archive.

Per condition: counts CSV, assignment Parquet (including rejected raw poses and
reasons), track CSV, explicit gaps CSV, and same-pool control CSVs. Controls
filter the candidate pool at 0.2 or 0.1 and use the unchanged adjacent-frame
tracker, without extra inference or extra SLP files. `comparison.png`,
`comparison.md`, and `summary.json` compare observed counts and track continuity.
Pending unobserved identities are reported separately, never counted as poses.

`source/`, `RUN_MANIFEST.json`, `TESTS.json`, `tests.txt`, per-condition completion
records, and a final `COMPLETE.json` provide provenance and validation receipts.
A queued/running job or partial condition marker is **not** a complete test run.

## Validation

The Slurm script runs the existing tracker/serialization tests and new controls
**before any real inference**, stopping on failure. New cases cover:

- Exact five-versus-six-missed-frame boundary, weak-score continuation, expired
  identities, and centroid-versus-landmark score separation.
- Independent 0.2/0.1 births, variable counts, entrances/exits, empty frames,
  absent Shoulder, shuffled instance order, and no fixed count cap.
- Ambiguous crossings, rejected nearby competitors, duplicate Shells, elapsed-
  time motion gates, and the fixed prediction-residual bound.
- No-grace adjacent-only control on isolated known tracks.
- Raw output field/slot mapping, missing poses, source immutability, independent
  conditions, no synthesized points, final SLP readback and complete coverage.

Runtime validations verify checkpoint/config/script hashes against the production
manifest, installed package versions, video size/mtime and dimensions, effective
thresholds and disabled caps/NMS, raw archive readback, one-to-one IDs, birth/link
thresholds, gap lengths, count conservation, and unchanged source measurements.
Hashes are rechecked before the final completion marker.

Longer tracks, more retained poses, and smoother net counts do **not** establish
better precision/recall or fewer ID switches. These need visual ground truth;
no manual accuracy claims or automatic production promotion are made.

## Implementation

- `configs/prediction/grace_tests_1000_v001.json`
- `scripts/grace_tracklets.py`
- `scripts/grace_prediction_tests.py`
- `tests/test_grace_prediction_tests.py`
- `jobs/predict_grace_tests.sh`: one finite GPU job, 4 CPUs, 64 GB, 30-minute cap,
  no requeue or continuation chain.
- Logs: `logs/prediction_grace_tests/job_<job-id>.{out,err}`.

SLPs reference `crab_vids/Crab_Cam_2_2.mp4` at 7680×4320. For local viewing,
relink to the matching full original video, not a resized or cropped excerpt.
See [viewing instructions](VISUALIZING_PREDICTIONS.md) and the preceding
[200-frame threshold experiment](prediction_threshold_tests.md).
