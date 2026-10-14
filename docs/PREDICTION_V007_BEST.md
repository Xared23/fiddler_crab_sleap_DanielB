# Frozen v007 best models: raw predictions plus optional tracklets

[Documentation index](README.md) · [Tracking method and pilot](BODY_FIRST_TRACKLETS.md)

All artifact paths below are project-relative to
`/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap`.

## Authorization and scope

The user explicitly requested full inference on `Crab_Cam_2_2.mp4` and
`Crab_Cam_2_3.mp4` with the frozen validation-best centroid and centered models,
using the body-first tracklet method. They accepted that the identities might
later prove unsuitable, PROVIDED the raw pose predictions remain independently
usable. This supersedes the earlier requirement to approve image-based tracking
accuracy before starting full inference. It does not certify identity accuracy.

No training resume, parameter sweep, regional model treatment, gap bridging,
re-identification, or overwrite of existing data is included.

## Completed execution chain

Preflight **15527306** reported `PREFLIGHT_PASSED`, including 20 tracker tests and
6 SLP/Parquet integration tests. The initial check **15527136** safely refused a
previously unlisted `provenance_json` header dataset; the documented format was
supported and regression-tested before the successful verification rerun.

| Video | GPU inference job | Dependent CPU merge + tracklets job |
| --- | --- | --- |
| `Crab_Cam_2_2` | **15527343** | **15527453** |
| `Crab_Cam_2_3` | **15527355** | **15527462** |

Both GPU jobs have `afterok:15527306`; each CPU job depends only on its matching
GPU job. `jobs/finalize_v007_cam2_2.sh` and `jobs/finalize_v007_cam2_3.sh` pin these
dependencies and invoke the common finalizer. All four jobs were subsequently
checked and reported **COMPLETED**, exit **0:0**. Both videos have their raw,
tracklet and overall completion markers, covering all 28,800 frames each.
The finalizers report 3,673,641 / 4,312,189 source observations and 106,326 /
154,059 optional tracklets, respectively. These are not unique-animal counts.
The [submission receipt](../logs/prediction_v007/submission.json) remains the
historical submission record, not a live status record.

A separate [standardized dataframe export](datasets/v007_best_body_first_dataframe_v001/README.md)
is now **validated and published** under
`datasets/v007_best_body_first_dataframe_v001/` (build **15573035**, publication
**15573203**). It adds explicit centroid-proxy coordinates, source references,
nullable fields and independent validation. All **7,985,830** source detections
remain available, including **28,764** unassigned observations. Require its
completion marker before use. The original finalizer Parquet tables below also
remain unchanged. No inference or tracking was rerun by the export.

## Pinned models and inference settings

- Centroid: `models/v007_19f/pauses/step_00021212/centroid_best/`.
  Best validation loss `5.3800526075065136e-05`, epoch index 72 (human epoch 73).
- Centered: `models/v007_19f/pauses/step_00021212/centered_best/`.
  Best validation loss `0.000517778389621526`, epoch index 191 (human epoch 192).
- Those losses come from only two held-out frames used for model selection.
  They are not pixel-error, recall, or tracking-accuracy measurements.
- Exact configuration: `configs/prediction/v007_best_body_first_v001.json`.
- Native 7680x4320 inference; 28,800 frames per video; 40-frame chunks; batch 1;
  peak threshold 0.2; no cap on instances. No water mask or regional filtering.
- The existing `predict_dan_native.sh` and `scripts/gputest_predict.py` are reused.
  The batch entry point now selects the versioned frozen-best run explicitly;
  it no longer defaults to the historical model/output directories. The Python
  worker retains the legacy interface for existing callers, but these new jobs
  always pass `--run-config`.
- No built-in inference tracker or pose filter is enabled. Body-first tracking is
  a separate CPU postprocessing step after raw prediction publication.

## Outputs: identities are genuinely optional

New root: `predictions/v007_best_body_first_v001/`.
For each `<video>` (`Crab_Cam_2_2` or `Crab_Cam_2_3`):

```text
<video>/
  raw/
    <video>.frames_000000_000039.slp   # and 719 further raw chunks
    <video>.frames_000000_000039.json  # coverage/hash/model-run receipt
    CHUNKS_COMPLETE.json
    <video>.predictions.slp           # merged, untracked, all detections
    RAW_COMPLETE.json
  tracklets/
    <video>.tracked.slp               # same measurements + optional Track references
    observations.parquet             # EVERY raw detection, including unassigned ones
    tracklets.parquet                # start/end/length/reason for each assigned track
    TRACKLETS_COMPLETE.json
  hop_state.json
  PREDICT_COMPLETE.json
```

**To ignore identities, simply use `raw/<video>.predictions.slp`.** Raw chunks are
also independently readable and are never replaced by tracked chunks. If tracking
fails, the completed raw merged file and its completion receipt remain usable.
The finalizer publishes raw predictions before it starts tracking.

The derived SLP retains ALL instances: a body rejected for tracking has no Track,
not a deleted pose. The observations table likewise retains those rows with null
`tracklet_id` / `tracklet_local_id`, `accepted_for_tracking=false`, and a reason.
No coordinates or prediction scores are smoothed/replaced. Byte-level comparisons
verify that only track-reference fields changed in the derived SLP.

The table includes `video_id`, zero-based `frame`, within-frame `instance_index`,
`raw_instance_id` (index in the merged raw SLP), stable within-run `detection_id`,
all five nodes' x/y/score/visible fields, instance score, Shell body coordinates,
and optional per-observation water metadata. Water does not affect association.
`centroid_source="shell_proxy"` is explicit: the API's pose SLP does not retain
separate detector-centroid coordinates, and Shell is NOT advertised as those
coordinates. Concatenate tables with their `video_id` intact.

Example (execute analysis through Slurm on Della):

```python
import pandas as pd
observations = pd.read_parquet(
    "predictions/v007_best_body_first_v001/Crab_Cam_2_2/tracklets/observations.parquet"
)
# Ignoring tracklet_id does not remove any detections. Do NOT filter on
# accepted_for_tracking if you want all raw pose observations.
poses = observations.drop(columns=["tracklet_id", "tracklet_local_id"])
```

## Tracklet semantics and limitations

The pilot-tested `body_first_v001` method uses Shell first, optional Shoulder,
bounded short motion history, one-to-one matching with unmatched options, and
ambiguity checks. Tracking state persists across every 40-frame chunk boundary,
but never crosses videos. Missing/empty frames terminate identity. Parameters and
algorithm are unchanged from the pilot; only the track-name prefix is scoped to
the new prediction run. Elbow/Claw/Arm and water are not association features.

Pilot continuity improved greatly, but the old predictions showed missed bodies
and duplicate detections that can still split visible-crab trajectories or give
more than one tracklet for a crab. IDs are **experimental**, not unique-animal
counts, and have no ground-truth ID-switch guarantee. Keep raw predictions so
later tracker revisions require only CPU postprocessing, not neural inference.

## Execution and fail-safe behavior

1. Finite CPU preflight: synthetic tracker tests, actual SLP merge/tracking/Parquet
   integration tests, Bash syntax/API checks, frozen model hashes/configuration,
   video metadata/source pins, and agreement with the tested tracker version.
2. Two GPU jobs depend on successful preflight. Each uses 20-minute allocations,
   4 CPUs, 64 GB RAM and one GPU80 GPU. They process completed 40-frame chunks
   within a finite work budget and requeue the SAME Slurm job ID only on the
   worker's explicit exit 75. Limits: 1,000 allocations and 3 consecutive
   no-progress allocations per video. Fatal errors do not become retry loops.
3. Two finite CPU finalizers depend on the corresponding GPU job's success.
   Each has 4 CPUs, 32 GB RAM, and a 4-hour ceiling; neither requeues itself.
   They merge raw SLP tables with validated index offsets, then derive optional
   IDs and Parquet tables. This avoids a full-video list of millions of Python
   pose objects. Existing `merge_dan_native.sh` remains a historical-only entry
   point and is NOT submitted for this run.

No separate GPU benchmark is being launched. Actual model loading/inference and
raw serialization are checked by each production job's first chunk. A failure
stops that job and blocks its dependent finalizer rather than silently changing
models or parameters.

Code/model/configuration/environment hashes are pinned in the run manifest and
checked at each allocation/finalization. Per-video locks prevent concurrent
writers. Chunks use `.tmp.slp`, explicit SLP format, read-back, checksums, and
atomic publication. An unreceipted final file is retained for inspection rather
than overwritten. Unexpected schema or incomplete coverage is an error.

Preflight evidence: `checks/v007_best_body_first_v001/`.
Logs: `logs/prediction_v007/`.
Submission receipt: `logs/prediction_v007/submission.json` (written at submission).
The receipt records IDs/dependencies; it is NOT a completion claim or live status.
