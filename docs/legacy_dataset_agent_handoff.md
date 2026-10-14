# Agent handoff: organize the ORIGINAL-model predictions into a reusable dataset

## Mission and authorization

Project root: `/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap`.

The user wants the existing original-model predictions put into a clean,
manageable form BEFORE integrating the newly trained models. The resulting
schema, provenance, and validation should be a template for future datasets.
This is a separate data-organization task, not a model-training task.

User's requested sequence:

1. Evaluate whether the existing SLEAP predictions can support the conservative
   tracklets defined below. Evaluate first; do not immediately rerun inference.
2. If feasible, make tracklets and a pandas-compatible dataframe with:
   **tracklet ID; time (frame); centroid position; body info; shoulder info;
   Water (whether the crab is in the static water section).**
3. Work on the assumption that centroid, body, and shoulder positions are the
   reliable measurements. Other pose points may not be reliable.
4. Determine whether the water region can be established from the videos or
   whether the user must specify/confirm it. It must support excluding water
   observations for selected downstream analyses.

You may implement and run finite Slurm audits, tests, and the conditional
postprocessing requested above. If existing information is insufficient,
report exactly what is missing and propose a scoped remedy. **Do not launch
fresh neural-network predictions without explicit user approval after the audit.**
Do not invent additional training, performance experiments, or parameter sweeps.
This handoff does not claim that feasibility or tracklet quality has been tested.

## Project primer

This is top-down male fiddler-crab pose estimation: a centroid detector proposes
crops and a centered-instance model predicts five landmarks:

`Shell`, `Shoulder`, `Elbow`, `Claw`, `Arm`.

The centroid target and centered-pose crop are anchored on `Shell`. This does
NOT establish that a detector centroid is numerically identical to the final
pose model's Shell point, nor that both coordinates were retained in the SLP.

The old model pair is:

- `models/dan_centroid_native/`
- `models/dan_centered_native/`

The newer run is `v007_19f`, based solely on cumulative `danlabels.v007.slp`:
19 human-labeled frames, 17 train / 2 validation, seed 42; no special regional
handling. Those models are NOT the source of this dataset. At the prior training
handoff, the centered model had finished 200 epochs and the centroid was still
training under job `15256499`. That is historical status, not a fresh queue
check. A tested extension schedules centroid snapshots at 75/100 completed
epochs. Do not touch that run or its code/configuration/checkpoint integrity pins.

Read `docs/PROJECT.md` for general context and `docs/README.md` for the notes
index. Subsequent audit results and user decisions are recorded in
`docs/datasets/legacy_native_tracklets_v001/README.md`; this handoff is historical
task context, not a fresh status report. `docs/retraining_v007.md` includes useful
scientific decisions but also explicitly superseded historical sections.
`docs/milestone_checkpoints.md` concerns the separate new training run only.

## Inputs and evidence already inspected

All paths below are relative to the project root.

| Video ID | Raw video | Original predictions |
| --- | --- | --- |
| `Crab_Cam_2_2` | `crab_vids/Crab_Cam_2_2.mp4` | `predictions/Crab_Cam_2_2/` |
| `Crab_Cam_2_3` | `crab_vids/Crab_Cam_2_3.mp4` | `predictions/Crab_Cam_2_3/` |

Each prediction directory contains chunk files named
`<video>.frames_000000_000039.slp`, etc. Its `PREDICT_COMPLETE` text reports
28,800 frames and a merged path `<video>.predictions.slp` in the same directory.
Both `.predict_opts` files record `CHUNK_FRAMES=40` and `PEAK_THRESHOLD=0.2`.
These records are evidence of the historical pipeline's claims, NOT verification
of present completeness, correctness, or the merged SLP's readability.

Relevant historical code/logs:

- `scripts/gputest_predict.py`: explicitly selects the OLD model pair above;
  calls `sleap_nn.inference.run.predict` separately for each 40-frame chunk.
  Its invocation does NOT explicitly enable a tracker. Do not assume that the
  user's description "SLEAP-tracked predictions" means that usable identities
  actually exist in the saved instances. Inspect the data and applicable API.
- `predict_dan_native.sh`, `merge_dan_native.sh`: historical entry points; read
  for provenance only, do not launch them as an audit or replacement workflow.
- `predict_logs/`: inference/merge history.

A lack of existing track identities does not by itself require new inference:
valid per-frame positions can be associated retrospectively. A damaged/missing
merged SLP may also be repairable from intact chunks without inference.

File search tools respect `.gitignore` and may omit prediction artifacts. Do
not interpret an empty glob search as proof that ignored SLPs are absent.

The project uses `.venv/bin/python` and `sleap_io` / `sleap_nn`. Earlier inspection
reported Python 3.13, sleap-io 0.9.2, sleap-nn 0.3.3, Lightning 2.6.5, and Torch
2.13.0+cu130; record the versions actually used by this task rather than assuming
those observations are current. Do not modify the shared active environment.

## Phase 1: finite feasibility audit, before conversion or rerunning predictions

Use a dedicated CPU Slurm job and save a machine-readable inventory plus a
short assessment. Inspect source metadata, code, and actual saved instances.

1. Establish video identity, dimensions, frame counts, frame indexing and timing.
   Do not copy the training clip's resolution/frame rate onto these videos
   without checking. Verify SLP-to-local-video references without altering
   originals or reading external collaborator paths.
2. Inventory merged files and chunks; validate their readability, expected
   ranges, frame ordering, duplicate/overlapping frames, coverage, video and
   skeleton consistency, and model/inference provenance where recoverable.
   Distinguish a processed frame with zero detections from an unprocessed frame.
   Missing frame records must not be silently interpreted as successful inference.
3. Inspect whether instances have tracks, whether track labels restart across
   chunks, and whether old identities satisfy the new conservative definition.
   Treat old IDs as provenance, not authoritative identities.
4. Determine exactly which positions and scores were serialized: predicted
   nodes, detector centroids, node confidence, instance score, tracking score,
   and any crop transforms. Do not relabel an instance score as centroid
   confidence. Check coordinate units and full-image versus crop coordinates.
5. Quantify finite/missing trusted coordinates, suspicious outliers, duplicate
   detections and available confidence distributions. The user's reliability
   assumption guides the choice of features; it is not proof that every point
   or association is correct. Confidence values are not necessarily calibrated
   probabilities. Avoid rejecting otherwise reliable bodies solely because a
   mean whole-pose score is reduced by unreliable limbs.
6. Inspect a bounded, explicitly selected set of consecutive-frame examples,
   including dense interactions, water/land, and 40-frame chunk boundaries.
   Extract static annotated PNGs or a small contact sheet in Slurm; do not render
   full overlay movies. Record the reviewed frame indices and limitations.
7. Report a concrete disposition per video: usable as-is for post-hoc tracking;
   repairable from chunks; partially usable with explicit exclusions; or needing
   specified additional information/prediction. Do not infer perfect identity
   accuracy from serialization or structural checks alone.

Audit need not wait for water-ROI confirmation or the naming clarification below.
If usable, proceed to the requested bounded tracking/export work with documented
parameters and checks. If not, present evidence and a minimal remedy; do not
silently rerun full-video inference.

## Conservative tracklet definition (required)

- A tracklet is a sequence of reliable, unambiguously associated observations
  on consecutive original-video frames. It is NOT a persistent animal identity.
- Associate adjacent frames only, using the reliable centroid/body location and
  shoulder information where justified. Do not let Elbow/Claw/Arm drive matching.
- Require documented validity/confidence, displacement, one-to-one association,
  and ambiguity checks. Do not force a match just because an assignment
  algorithm returns one. Resolve neither ties nor near-equivalent competitors
  into a claimed identity without sufficient evidence.
- End continuity at missing/unreliable detections or ambiguous associations.
  Start a new ID on reappearance. No interpolation, gap bridging, re-ID,
  extrapolation across gaps, or later stitching. Prefer fragmentation to switches.
- Keep tracking state chronological across chunk boundaries, but never across
  videos. Missing/unprocessed spans also break continuity.
- Preserve reliable singleton tracklets rather than silently removing short
  tracks. Preserve rejected/unassigned detections in source data and an audit
  table with rejection reasons; do not make them disappear from accounting.
- Deterministic, versioned parameters and ID assignment are required. IDs must
  be unambiguous across the dataset, e.g. a dataset/video prefix plus a local ID.
- Explain that these operational rules cannot identify every physical occlusion
  or burrow entry or guarantee zero identity switches from detections alone.

## Dataframe contract: proposed explicit columns

Use one row per accepted crab observation per original-video frame. Prefer
numeric coordinate columns over opaque tuples/dicts in pandas object columns.
Store a pandas-readable Parquet dataset if an appropriate engine is already
available; document loading and dtypes. Report a missing dependency before
changing/installing anything in the active environment.

| User label | Proposed columns / meaning |
| --- | --- |
| A. Tracklet ID | `tracklet_id` (dataset/video-scoped unique string) |
| B. Time (frame) | `frame` (zero-based original-video integer frame index) |
| C. Centroid position | `centroid_x`, `centroid_y`, plus `centroid_source` |
| D. Body info | provisionally `body_x`, `body_y`, `body_score` for node `Shell` |
| E. Shoulder info | `shoulder_x`, `shoulder_y`, `shoulder_score` for `Shoulder` |
| F. Water | `water` (pandas nullable boolean: True / False / NA) |
| Required context | `video_id`, `detection_id` linking back to the source observation |

Coordinate convention: full-resolution original-video pixels, x rightward and
y downward; record dimensions and transformations. Keep missing values missing,
not zero. If useful, add `time_s`, but use verified video timestamps or a clearly
documented frame-rate conversion; frame remains the primary requested time key.
Retain source confidence fields with their original semantics. Preserve all five
raw nodes/confidences in source artifacts or a clearly linked raw-points table;
mark the non-body/shoulder measurements as untrusted for this analysis.

**Clarification to resolve before finalizing the schema:** does the user's
"body" mean the SLEAP `Shell` landmark, or something additional such as size,
outline, or orientation? A single landmark is a position, not a body outline.
Do not promise segmentation or heading information that the labels do not encode.

**Centroid distinction:** a centroid-model crop center and a pose-model Shell
point can differ. If detector centroids were not retained, report that fact.
A Shell-based position may be a sensible proxy, but requires explicit user
agreement and `centroid_source="shell_proxy"`; do not silently duplicate Shell
coordinates under the name "detector centroid". Never average unreliable limb
positions to manufacture a centroid. Missing detector centroids alone need not
force an expensive inference rerun if the user accepts a body-position proxy.

## Water annotation: static, explicit, and independent of inference

The user states that the water region is static. You may inspect representative
frames and propose a water polygon; the semantic boundary may still be visually
ambiguous. Obtain confirmation of a proposed polygon, or accept user-supplied
polygon vertices / an annotated screenshot, before calling the flags definitive.

- Define and record an ROI per video unless identical geometry is verified.
  Use original-resolution pixel coordinates, not display-scaled screenshot
  coordinates. Check representative early/middle/late frames for registration.
- Save the polygon(s), source frame/video, dimensions, confirmation status,
  coordinate convention and boundary-inclusion rule in a versioned ROI file.
- Propose point-in-polygon membership of the chosen centroid/body reference
  point. Confirm that this matches "crab in water"; it is not the same as any
  part of the crab overlapping water. Record the exact reference-point source.
- `water=True` means the selected position lies in the confirmed water ROI;
  `False` means it lies outside. Use `NA` for unknown/invalid positions,
  unconfirmed/missing ROI, or any explicitly defined uncertain-boundary zone.
  Missing ROI/coordinates must never turn into an automatic land label.
- Label each observation, not an entire tracklet: a crab can cross the boundary
  during a tracklet. Keep the canonical table intact; downstream land-only
  filtering removes rows and can create gaps, not new evidence of continuity.
- Do not discard water detections from canonical tracklets or raw data. This is
  analysis metadata, NOT water-specific model training, inference, weighting,
  or tracking exclusions. The original no-regional-treatment decision stands.

Provide a small annotated ROI preview for user confirmation. This can proceed
without waiting for any new neural-network predictions. Until confirmed, leave
water flags nullable/provisional and label the dataset's status accordingly.

## Suggested versioned organization and acceptance checks

Proposed new destination (check that it is unused before creating):

```text
datasets/legacy_native_tracklets_v001/
    README.md                     # short pointer to central docs/datasets/<id>/ notes
    manifest.json                 # sources/hashes, model provenance, versions
    config/tracklets.yaml          # frozen association/validity parameters
    config/water_regions.json     # polygons + confirmation/provenance
    audit/                        # input inventory, feasibility and review evidence
    tables/observations/           # pandas-readable Parquet, partitioned by video
    tables/tracklets.parquet       # ID/video/start/end/length and termination reason
    tables/rejected_observations/  # source references and reasons, if needed
    slp/                          # optional derived SLPs carrying the SAME new IDs
    qc/                           # summary metrics and bounded static previews
    DATASET_COMPLETE.json         # only after checks; explicit water status
```

This is a proposed layout, not a statement that outputs exist. Raw videos,
models and historical predictions stay in place; the manifest references them.
A derived tracked SLP is useful for inspection, but the user's requested primary
product here is the dataframe plus reproducible schema/provenance.

Acceptance checks must include:

- Source counts reconcile with accepted/rejected observations; no unexplained
  drops, duplication or invalid video/frame mappings.
- A `(tracklet_id, frame)` pair is unique, each tracklet belongs to one video,
  frame increments within each tracklet are exactly one, and IDs do not collide
  across videos. Test empty frames, missing spans, ambiguous collisions,
  invalid coordinates and chunk-boundary continuity on synthetic fixtures.
- Trusted coordinates, nullable scores and water flags have documented dtypes
  and meanings. Test ROI inside/outside/boundary/missing-position behavior.
- Tracklet-length distributions, unmatched/rejected counts and review examples
  support an honest quality assessment; structural checks alone do not prove
  scientific accuracy.
- Saved tables are read back and schema/counts/keys verified. If producing SLPs,
  validate their frame/node/track correspondence to the tables and video links.
- Artifacts are reproducible from pinned inputs/configs/code versions. Use locks
  and atomic publication; do not overwrite an existing validated dataset.
- Distinguish provisional ROI annotation, incomplete inputs, and validated
  completion clearly. Document known limits and remaining user decisions.

## Operational boundaries

- Work only inside `/scratch/gpfs/WBIALEK/db8682` and its project subdirectories.
  Inspect/edit using file tools. ALL Python, video decoding, binary SLP/HDF5
  inspection, tests, analyses and conversion run in finite Slurm jobs, never
  directly on vis1. Read scripts before submission and report job IDs/results.
- Keep scripts and logs project-local, preferably in task-specific directories.
  Do not add watchers, recurring services, auto-submitting loops or unbounded
  retries. Do not cancel training jobs or alter their Slurm state.
- Do not edit `configs/v007_19f/`, training scripts/helpers, pinned extension
  receipts, model checkpoints, original labels, original prediction files,
  `.predict_opts`, `.next_frame`, or historical completion markers.
- No moving, archiving, deleting or destructive relinking of originals without
  separate approval. Preserving source paths matters because SLPs reference
  videos rather than embedding them. Do not reuse legacy merge code that deletes
  an existing output before replacement validation.
- Stream/chunk large data where appropriate; carry only needed association state
  across boundaries rather than assuming both full videos fit in RAM.
- For derived SLPs, use `.tmp.slp` and explicit `format="slp"`, then read back
  before atomic publication. The earlier `.slp.tmp` extension caused a real save
  error. A save failure is not permission to start a requeue/final-save loop.
- Finish the requested work and become idle. Do not automatically transition to
  new-model inference, a prediction rerun, new training, or additional analyses.
