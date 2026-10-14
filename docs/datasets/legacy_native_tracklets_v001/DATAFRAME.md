# Conservative legacy tracklets and dataframe

[Dataset overview](README.md) · [Documentation index](../../README.md)

**Validated and published.** Build **15521792** and publication **15522072**
both completed with exit 0. Require the
[completion marker](../../../datasets/legacy_native_tracklets_v001/DATASET_COMPLETE.json)
before use. Artifact paths below are relative to
`datasets/legacy_native_tracklets_v001/` at the project root. No original videos,
SLPs, models or `v007_19f` files were changed. No inference was rerun.

## Published results

| Video | Accepted observations | Rejected observations | Tracklets | Singleton tracklets |
|---|---:|---:|---:|---:|
| `Crab_Cam_2_2` | 4,455,589 | 3,706 | 1,688,746 | 1,534,070 |
| `Crab_Cam_2_3` | 5,179,601 | 5,298 | 1,922,976 | 1,721,902 |
| Total | **9,635,190** | **9,004** | **3,611,722** | **3,255,972** |

All **9,644,194 source detections** and **57,600 original frame records**
reconcile. All 21 synthetic tests, exact per-part Parquet read-back, full
source-key/tracklet-continuity/summary/dtype checks, and publication hash checks
passed. Twelve selected adjacent-frame crop previews were visually reviewed.
The build used about 1 GiB maximum RSS and took 7m22s; publication took 7s.

**Important quality limitation:** most tracklets are singletons. Median length
is **1 frame** in each video; 95th-percentile lengths are 4/5 frames, and
99th-percentile lengths are 26/30 frames. Longest segments are 9,726/6,107 frames.
Missing Shoulder and ambiguous competitors deliberately fragment identities.
This is a reusable observation table and conservative contiguous-segment dataset,
not a table of long-lived animal identities or a validated count of unique crabs.
Residual model errors and duplicate detections beyond the rejection radius can
remain; their nearby competitors inhibit temporal linking rather than forcing IDs.

Accepted water/land rows are 254,017/4,201,572 for `Crab_Cam_2_2` and
283,728/4,895,873 for `Crab_Cam_2_3`. There are no unknown water values among the
accepted observations in this build, but the column retains nullable boolean
type and invalid/unknown cases are tested. See the
[QC summary](../../../datasets/legacy_native_tracklets_v001/qc/postprocessing/summary.json)
and [review receipt](../../../datasets/legacy_native_tracklets_v001/config/quality_review_v001.json).

## Meaning and association rules

A tracklet is a sequence of accepted, unambiguously associated detections on
**consecutive original-video frames**, not a persistent physical-animal ID.
There is no interpolation, gap bridging, re-identification, motion extrapolation
or later stitching. State crosses chunk boundaries, never video boundaries.
These rules cannot detect every occlusion or guarantee zero identity switches.

The single frozen parameter set is
[`config/tracklets.yaml`](../../../datasets/legacy_native_tracklets_v001/config/tracklets.yaml):

- Body needs finite, in-image, visible **Shell** with node score ≥0.5.
- Within-frame Shell pairs ≤5 pixels apart are duplicate candidates: reject both,
  not an arbitrary winner. Competitors include all finite in-image Shell points.
- Linking additionally requires Shoulder at both endpoints: finite, in-image,
  visible, node score ≥0.2, and Shell–Shoulder distance 10–65 pixels.
- Missing/unusable Shoulder **does not delete a reliable body**. It yields a
  singleton and breaks linking; no Shoulder or temporal continuity is invented.
- Adjacent-frame displacement: Shell ≤12 pixels; Shoulder ≤16 pixels;
  Shell-to-Shoulder vector change ≤12 pixels.
- Both endpoints must have exactly one opposite-frame candidate within **20
  pixels** (12-pixel motion gate + 8-pixel ambiguity guard). Even low-score or
  rejected finite in-image Shell competitors can veto a link. Shoulder gates
  cannot erase a competitor to manufacture uniqueness. No assignment fallback.
- Missing records and recorded empty frames break continuity. Real audited
  inputs are complete; these cases are tested synthetically.
- Instance score, Elbow/Claw/Arm and water do not control tracking/acceptance.

Thresholds were chosen once from the input audit, without a sweep. Shoulder
peaks are much lower than Shell peaks; a common 0.5 gate would incorrectly
remove essentially all Shoulder measurements. Scores are uncalibrated and not
clipped to [0, 1]. Strong fragmentation and many retained singletons are expected.

## Tables, keys and provenance

| Dataset-relative path | Content |
|---|---|
| `tables/observations/<video>/part-*.parquet` | One row per accepted detection, including singletons |
| `tables/rejected_observations/<video>/part-*.parquet` | Source-linked rejected rows; null tracklet ID and explicit reason |
| `tables/tracklets.parquet` | ID/video/start/end/length and start/termination reasons |
| `tables/frames.parquet` | Per-original-frame recorded status and source/accepted/rejected/link counts |
| `manifest.json` | SHA-256 sources/code/config/artifacts, package versions, job IDs, counts and limits |
| `config/water_regions_effective_v002.json` | Effective accepted fixed ROI, reference point and boundary policy |
| `qc/postprocessing/summary.json` | Length distributions, singleton/link/rejection/water counts, checks |
| `qc/postprocessing/review_examples.json` | Exact preview frames/crop bounds and observation-to-tracklet mapping |
| `DATASET_COMPLETE.json` | Final publication gate, hashes, counts and explicit water status |

Physical video partitions are plain video-named directories, **not** Hive
`video_id=...` directories; every row stores `video_id` itself. IDs are assigned
in chronological frame/source-instance order, deterministically:

- `legacy_native_tracklets_v001/<video>/t000000001`
- `legacy_native_tracklets_v001/<video>/f000039/i0000` (detection)

Use the full `tracklet_id`, not a local number, when combining videos. Rejected
observations do not consume a tracklet number.

The unchanged, SHA-256-pinned **original chunk SLPs** are the canonical raw-point
store. All five landmarks/scores/visibility flags remain there; Elbow/Claw/Arm
are untrusted for association. Each table row links back using `source_path`
(project-root-relative), `frame`, zero-based `source_instance_index` within the
frame, and `source_instance_row` in that chunk's HDF5 `instances` dataset. No
separate raw-points copy or derived tracked SLP is generated; the primary product
is the dataframe. Current model hashes do not prove historical checkpoint hashes.

## Observation schema and semantics

Use `pd.read_parquet(..., dtype_backend="numpy_nullable")`. Files use nullable
Arrow string/int64/float64/bool. Missing values are true nulls, not numeric zeros.

| Columns | Pandas dtype | Meaning |
|---|---|---|
| `tracklet_id`, `video_id`, `detection_id` | `string` | Dataset/video-scoped keys; rejected tracklet IDs are null |
| `tracklet_local_id`, `frame` | `Int64` | Video-local ID, zero-based original frame |
| `centroid_x`, `centroid_y`, `centroid_source` | `Float64`, `Float64`, `string` | Explicitly approved **Shell proxy**, labeled `shell_proxy` |
| `body_x`, `body_y`, `body_score` | `Float64` | Centered-instance Shell position and original node score |
| `shoulder_x`, `shoulder_y`, `shoulder_score` | `Float64` | Serialized Shoulder measurements, possibly missing/unusable |
| `shoulder_valid`, `association_eligible` | `boolean` | Shoulder validity and stricter full temporal-link eligibility |
| `body_visible`, `shoulder_visible` | `boolean` | Serialized visibility flags |
| `instance_score`, `source_tracking_score` | `Float64` | Original score semantics, not renamed centroid confidence |
| `source_track_index` | `Int64` | Serialized original index; -1 = no old track |
| `water` | nullable `boolean` | Shell point within the accepted fixed polygon: True / False / NA |
| `time_s` | `Float64` | Nominal `frame / container_fps`, not measured per-frame timestamps |
| `source_path`, `source_instance_index`, `source_instance_row` | `string`, `Int64`, `Int64` | Exact original-source references |
| `incoming_reason`, `rejection_reason` | `string` | Link/start classification and rejection reason (null if accepted) |
| `predecessor_detection_id` | `string` | Adjacent preceding detection only for a passed link; otherwise null |
| `body_displacement_px`, `shoulder_displacement_px` | `Float64` | Accepted-link displacements; null for starts/rejections |
| `duplicate_candidate` | `boolean` | ≤5-pixel within-frame Shell competitor flag |

Coordinates are **7680 × 4320 original-image pixels**, x rightward and y downward,
without export rescaling. Both videos have audited container FPS=30000/1001, so
nominal time is `frame × 1001 / 30000`. No exhaustive PTS scan was performed.

`body_x/y` and `centroid_x/y` deliberately duplicate the **same centered-model
Shell estimate**, with explicit user approval. Detector crop centers were not
saved; these are not recovered detector outputs. Body does not mean segmentation,
size, outline or guaranteed heading. Retained out-of-bounds/low-score Shoulder
measurements have `shoulder_valid=False`; non-null does not mean trustworthy.

## Fixed water annotation

The user confirmed the region is **the same in every frame**, accepted the
polygon for initial use, and approved the Shell centroid/body proxy. The polygon
is never redetected from frames. Both videos use the separately reviewed fixed
geometry. Membership uses that point, not any-part-of-crab overlap; the polygon
boundary is included. The earlier unconfirmed 30-pixel uncertainty-band proposal
is **not applied**. Precision of the shoreline/wet-mud distinction remains limited.

Water is per observation: a moving crab can cross the same fixed boundary during
a tracklet. It never excludes, weights or breaks canonical tracking. Missing or
unconfirmed ROI/invalid reference points yield NA, never automatic land. Invalid
Shell-score/visibility rejected rows have unknown membership; duplicate candidates
may still have a valid position. Filtering land-only rows can introduce gaps and
must not be interpreted as evidence of new continuity.

Approval receipts:
[`user_decisions_001.json`](../../../datasets/legacy_native_tracklets_v001/config/user_decisions_001.json)
and [`user_decisions_002.json`](../../../datasets/legacy_native_tracklets_v001/config/user_decisions_002.json).
Original audited proposal files/previews retain historical “unconfirmed” labels;
receipts and effective ROI supersede that status without rewriting hashed evidence.

## Pandas loading

Run computation in an authorized Slurm job on Della. Select columns/video/frame
ranges instead of loading all ~9.6 million source detections without enough RAM.

```python
import json
from pathlib import Path
import pandas as pd

root = Path('/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/datasets/legacy_native_tracklets_v001')
completion = json.loads((root / 'DATASET_COMPLETE.json').read_text())
assert completion['status'] == 'validated_complete'

df = pd.read_parquet(
    root / 'tables/observations/Crab_Cam_2_2',
    columns=['tracklet_id', 'detection_id', 'video_id', 'frame',
             'centroid_x', 'centroid_y', 'centroid_source', 'body_score',
             'shoulder_x', 'shoulder_y', 'shoulder_score', 'water'],
    filters=[('frame', '>=', 0), ('frame', '<', 400)],
    dtype_backend='numpy_nullable',
)
land = df.loc[df['water'].eq(False).fillna(False)].copy()
tracklets = pd.read_parquet(root / 'tables/tracklets.parquet',
                           dtype_backend='numpy_nullable')
```

## Validation, reproducibility and publication

`jobs/build_legacy_dataset.sh` runs the synthetic tests first and stops on failure.
It fully rehashes canonical chunks, videos and old-model artifacts against the
audit, streams chronologically and writes only `.build_job_<ID>/`. Each saved
observation/rejection part is read back and compared exactly. Full saved-table
checks reconcile **every source ordinal exactly once**, verify ID prefixes and
unique `(tracklet_id, frame)` keys, require consecutive frames, compare every
tracklet's summary, and check pandas nullable dtypes. Code/config/source mutation
is a hard error, not permission to combine revisions.

Bounded adjacent-frame crop previews are generated in Slurm and reviewed before
`jobs/publish_legacy_dataset.sh <validated-build-job-id>` is submitted. Both
stages use exclusive writer locks. Publication rechecks artifact hashes,
schemas/counts and input pins, renames staged tables/QC into place and writes
`DATASET_COMPLETE.json` **last**. Require that marker; intermediate directories
alone are not completion. No validated output is overwritten, no automatic retry
or follow-on job occurs, and no recurring process remains. Future changes need a
new reviewed version/destination. Scientific identity accuracy is not established
by structural tests or a small visual review.

The 21 build tests cover adjacency/chunk boundaries; explicit and implicit
missing spans; empty frames; ambiguous one-to-many/many-to-one matches; ambiguity
guards and low-confidence competitors; duplicate candidates; invalid coordinates;
missing Shoulder singleton preservation; displacement/vector gates; deterministic
video-isolated IDs; ROI inside/outside/boundary/missing behavior; unchanged
tracking when water membership changes; and nullable Parquet/end-to-end read-back.

Completed audit jobs: **15428610** (inputs), **15428784** (five audit tests),
**15429096** (ROI/provenance report). Completed build: **15521792**; completed
publication: **15522072**. Definitive counts and hashes are in
`DATASET_COMPLETE.json` and `manifest.json`. No job remains running for this task;
no new-model inference or follow-on experiment was submitted.
