# Frozen-v007 prediction dataframe

[Documentation index](../../README.md) · [Prediction run](../../PREDICTION_V007_BEST.md) · [Tracker method](../../BODY_FIRST_TRACKLETS.md)

**Validated and published.** Build **15573035** and publication **15573203**
both completed successfully (exit 0). Publication took 5 seconds. No job remains
running for this export.
Initial test-only job 15572965 caught an empty-table string-type inference bug
before conversion; the dtype was made explicit before the successful build. Its
failed-test receipt serialization was also corrected. No dataset was published by
that attempt, and no validated output was overwritten.
Require `datasets/v007_best_body_first_dataframe_v001/DATASET_COMPLETE.json`
with `status="validated_complete"` before loading the organized dataset.
The published manifest SHA-256 is
`9dcec268c1de2d366cb57dc101a245c4ec009baa8588f4897bcc1fe121359756`.

All paths below are relative to the project root
`/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap`, unless marked dataset-relative.

## Source run and scope

Both full-video inference jobs and finalizers completed successfully (exit 0):

| Video | Inference job | Finalizer job | Source detections | Assigned observations | Unassigned observations | Tracklets |
|---|---:|---:|---:|---:|---:|---:|
| `Crab_Cam_2_2` | 15527343 | 15527453 | 3,673,641 | 3,661,979 | 11,662 | 106,326 |
| `Crab_Cam_2_3` | 15527355 | 15527462 | 4,312,189 | 4,295,087 | 17,102 | 154,059 |
| Total | | | **7,985,830** | **7,957,066** | **28,764** | **260,385** |

All counts above were independently reconciled by successful build **15573035**,
which also passed all **14 conversion tests** and full saved-table/source checks.
It completed in **3m30s**, with approximately **2.4 GiB** maximum RSS. These counts
describe a different model **and** tracking method from the legacy dataset, so fewer
tracklets or more continuity must not be interpreted alone as improved identity
accuracy. Each video has all 28,800 original frames (57,600 total).

The model pair is the frozen validation-best pair under
`models/v007_19f/pauses/step_00021212/{centroid_best,centered_best}/`.
This is **not** the historical `dan_*_native` model pair or its legacy dataset.
Training remains paused. No inference, tracking, model selection, or parameter
changes are performed by this dataframe export.

The existing finalizer tables under
`predictions/v007_best_body_first_v001/<video>/tracklets/` were already pandas
readable. This version adds explicit centroid coordinates, nominal time, exact
merged/chunk source references, dataset-scoped detection keys, nullable floating
measurements, frame summaries, and independent validation/provenance.

## Tables and retention policy

The following paths are relative to `datasets/v007_best_body_first_dataframe_v001/`:

| Path | Content |
|---|---|
| `tables/observations/<video>/part-*.parquet` | **Every raw detection**, including unassigned rows |
| `tables/unassigned_observations/<video>/part-*.parquet` | Convenience subset of the observations table; null tracklet IDs and explicit rejection reasons |
| `tables/tracklets.parquet` | Existing per-tracklet summaries, combined across the videos |
| `tables/frames.parquet` | Every original frame, source/assigned/unassigned counts, recorded status |
| `manifest.json` | Source/code/config/model/video SHA-256 pins, environment, artifact hashes, counts and limits |
| `config/` | Export settings, source inference settings, accepted static ROI |
| `audit/` | Source run manifest and conversion-test receipt |
| `qc/summary.json` | Counts, length quantiles, reasons, water counts, chunk-boundary links |
| `qc/previews.json`, `qc/*.png` | Six bounded adjacent-frame crop pairs; frames/crop coordinates recorded |
| `qc/review.json` | Bounded visual review with explicit limitations |
| `DATASET_COMPLETE.json` | Completion gate written only after structural validation and reviewed publication |

**Unlike the legacy dataset**, canonical `observations` includes *all* raw source
detections, not just tracking-accepted bodies. Select `accepted_for_tracking=True`
for the assigned-only view. Ignoring tracklet IDs does not remove any detections.
The unassigned table duplicates a subset for convenience: **do not concatenate it
back into observations**, or those rows will be counted twice.

No canonical rows are excluded based on water, Shoulder validity, or optional
identity usability. No coordinates are smoothed or interpolated. Serialized NaN
floating values become Arrow nulls; other measurements, including out-of-image
coordinates, low scores, and scores above 1, are not silently changed or clipped.
The exact original SLPs remain the pinned raw-byte store.

## Validated QC summary

These describe the existing experimental tracklets, not unique animals:

| Metric | `Crab_Cam_2_2` | `Crab_Cam_2_3` |
|---|---:|---:|
| Singleton tracklets | 40,181 | 56,121 |
| Median length (frames) | 2 | 2 |
| 95th-percentile length (frames) | 71 | 69 |
| 99th-percentile length (frames) | 757 | 589 |
| Maximum length (frames) | 17,051 | 9,777 |
| Observations in tracklets ≥30 frames | 3,261,679 | 3,692,744 |
| Verified links across chunk boundaries | 88,410 | 102,815 |
| Water / land / unknown, **all** observations | 174,716 / 3,497,590 / 1,335 | 227,813 / 4,081,891 / 2,485 |

All six paired previews (12 original frames) were opened and reviewed. Their
selected crop coordinates align with the underlying frames and fixed ROI, and
several IDs visibly persist across the selected chunk boundary. Some markers
appear or disappear between adjacent frames. Several water-region marks lie on
bright/reflection-like features whose biological identity is not resolved by these
stills; assigned status or track length does not establish that they are real crabs.
The crops preferentially sample water/shoreline and are not a representative
accuracy audit. See `qc/review.json` for the exact scope and limitations.

## Keys, coordinates and nullable schema

Use `dtype_backend="numpy_nullable"` with pandas. Strings, booleans, integers,
and floats are nullable (`string`, `boolean`, `Int64`/`Int32`, `Float64`).

| Columns | Meaning |
|---|---|
| `dataset_id`, `prediction_run_id` | Organized export version and the unchanged inference/tracking run |
| `video_id`, `frame` | Video key and **zero-based original-video frame**, never chunk-local time |
| `detection_id` | Export-scoped key: `v007_best_body_first_dataframe_v001/<video>/f000039/i0000` |
| `prediction_detection_id` | Original finalizer key: `<video>/f000039/i0000` |
| `instance_index`, `raw_instance_id` | Zero-based row within frame / row in the merged raw SLP |
| `tracklet_id`, `tracklet_local_id` | Existing full run-scoped ID / video-local number; null if unassigned |
| `accepted_for_tracking`, `rejection_reason`, `incoming_reason` | Existing assignment decision and diagnostic reasons, preserved unchanged |
| `tracking_algorithm`, `identity_status` | `body_first_v001`; `experimental_not_ground_truth_validated` |
| `body_x`, `body_y`, `body_score` | Centered model's **Shell** landmark coordinates and original node score |
| `centroid_x`, `centroid_y`, `centroid_source` | Same Shell coordinates, explicitly labeled **`shell_proxy`** |
| `<node>_x`, `<node>_y`, `<node>_score`, `<node>_visible` | Raw values for `shell`, `shoulder`, `elbow`, `claw`, `arm` |
| `instance_score`, `source_tracking_score` | Serialized SLEAP score fields; not calibrated probabilities or detector-centroid confidence |
| `water` | Nullable Shell-point membership in the accepted fixed polygon |
| `time_s` | Nominal `frame * 1001 / 30000`; not measured per-frame PTS |
| `source_path`, `source_instance_row`, `source_instance_index` | Project-relative **merged raw SLP**, zero-based merged instance row, within-frame row |
| `source_chunk_path`, `source_chunk_instance_row` | Project-relative original 40-frame chunk and zero-based instance row within that chunk |
| `source_observations_path` | Project-relative original finalizer Parquet table |

Positions are in **7680 × 4320 original-image pixels**, x right and y down.
Shell/body is a landmark, not size, segmentation, or a body outline. Separate
centroid-detector crop-center coordinates were not serialized in these SLPs;
the explicitly approved Shell proxy is not a recovered detector output.

Full tracklet IDs retain their original scope:
`v007_best_body_first_v001/<video>/t000000001`. They are deliberately **not
renumbered** for this export. Never combine videos by local ID alone.
The additional dataset prefix on detection IDs prevents collisions with legacy
predictions. There is no cross-model or cross-video animal correspondence.

## Tracking and water semantics

These IDs use the newer **body-first** method, not the legacy Shoulder-required
mutual-uniqueness method. Shell drives assignment; Shoulder is optional, short
motion history is bounded, and matching has unmatched options and ambiguity
checks. State crosses chunk boundaries, not video boundaries. No gap bridging,
re-identification, temporal interpolation, or limb/water association is added.
The exporter verifies the already-computed IDs; it does not rerun association.

IDs remain **experimental**, not validated physical-animal identities or counts.
Structural continuity and a few image crops do not establish tracking accuracy,
recall, absence of ID switches, or model superiority. False/missed detections,
duplicate predictions, and fragmentation may remain. Long tracks can also follow
persistent false detections; continuity alone is not evidence of a real crab.
Elbow/Claw/Arm values are preserved for inspection, not certified as reliable features.

Water uses the previously accepted fixed geometry, separately tied to each
video, with boundary included and **no 30-pixel uncertainty band**. The polygon
is an analysis ROI, not precise frame-by-frame water segmentation. As in the
source finalizer, membership requires finite in-image Shell coordinates; missing
or out-of-image coordinates are unknown, never automatic land. Membership does
**not** additionally gate on Shell score/visibility or tracking acceptance, so an
unassigned row can have a known water value. This differs from the stricter
legacy rejected-row water policy and must not be silently conflated with it.
Land-only filtering can create temporal gaps; retain original frame numbers.

`config/water_regions.json` is an unchanged copy of the effective legacy ROI.
Its embedded `acceptance_receipts` paths refer to the **original legacy dataset
root**, not this new export. The approvals remain at
[`user_decisions_001.json`](../../../datasets/legacy_native_tracklets_v001/config/user_decisions_001.json)
and [`user_decisions_002.json`](../../../datasets/legacy_native_tracklets_v001/config/user_decisions_002.json).
Historical uncertainty notes inside the copied ROI are superseded by those
receipts and the effective boundary/no-band policy; they are not new pending decisions.

## Pandas loading

Execute computation in an authorized Slurm job on Della. Start with selected
columns and a bounded video/frame range; loading all ~8 million rows and all
string/node columns at once can use substantial memory.

```python
import json
from pathlib import Path
import pandas as pd

root = Path('/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/datasets/v007_best_body_first_dataframe_v001')
complete = json.loads((root / 'DATASET_COMPLETE.json').read_text())
assert complete['status'] == 'validated_complete'

# All predictions in the selected frames, including null-ID detections.
df = pd.read_parquet(
    root / 'tables/observations/Crab_Cam_2_2',
    columns=['video_id', 'frame', 'time_s', 'detection_id', 'tracklet_id',
             'accepted_for_tracking', 'centroid_x', 'centroid_y',
             'centroid_source', 'body_score',
             'shoulder_x', 'shoulder_y', 'shoulder_score', 'water'],
    filters=[('frame', '>=', 0), ('frame', '<', 400)],
    dtype_backend='numpy_nullable',
)
tracked = df.loc[df['accepted_for_tracking'].fillna(False)].copy()
land = df.loc[df['water'].eq(False).fillna(False)].copy()  # NA is not land
tracklets = pd.read_parquet(root / 'tables/tracklets.parquet',
                           dtype_backend='numpy_nullable')
frames = pd.read_parquet(root / 'tables/frames.parquet',
                        dtype_backend='numpy_nullable')
```

For both videos, reading `root / 'tables/observations'` discovers the plain
video-named subdirectories; every row stores its own `video_id`. They are not
Hive `video_id=...` partitions. Keep column selection and frame predicates.
For assigned observations only, add `('accepted_for_tracking', '==', True)` to
the Parquet filters. Do **not** add that filter if you want all model outputs.

## Validation, publication and reproducibility

Implementation: `scripts/v007_dataframe.py`; tests:
`tests/test_v007_dataframe.py`; frozen export settings:
`configs/datasets/v007_best_body_first_dataframe_v001.json`.

`jobs/v007_dataframe.sh build` runs conversion tests first, then a finite,
bounded-memory CPU export. It independently hashes the saved model/run inputs,
all 1,440 chunks, merged raw files, tracked SLPs, original Parquet tables and
videos. It compares every chunk's frame/instance/point payload to the merged
source, every saved observation to its raw measurements, all track references
and catalog names to the tracked SLP, and all tracklet summaries to consecutive
frame counts. It recomputes static-ROI memberships without changing them.
Every new Parquet part is fully read back; saved-table continuity and source
ordinals are checked, and pandas nullable loading/filtering is exercised.

The finite tests cover original schema compatibility, unchanged raw node scores,
NaN-to-null conversion, complete row retention, source links and namespacing,
measurement/source-key corruption, tracked-SLP measurement corruption, chunk
and batch boundaries, prohibited gaps/duplicate track-frame keys, video isolation,
water boundary/unknown handling, nullable Parquet, filters, and empty tables.

Build stages under a hidden `.build_<job-id>` destination and does not mark the
dataset complete. After review of the six paired previews, a separate
`jobs/v007_dataframe.sh publish <build-job-id>` job checks all exported artifact
hashes, source sizes/mtimes, code/config hashes and the matching review receipt.
It writes the final completion marker last in staging and atomically renames the
new dataset into its final location. It refuses an existing destination.
This is publication of verified data, not certification of scientific identities.

No original predictions, legacy artifacts, model checkpoints, or pinned tracking
code/configuration are edited. No automatic retries, requeues, recurring services,
or additional experiments are installed. Future changes require a separately
reviewed dataset version/destination. Logs: `logs/v007_dataframe/job_<id>.{out,err}`.
