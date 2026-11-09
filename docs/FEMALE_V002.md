# Female FBody training and two-video prediction pipeline

[Documentation index](README.md) · [Male pipeline](PROJECT.md)

All code-formatted artifact paths are relative to the project root.

## Completion update and demo extension

Both full-video raw merges now have matching `PREDICT_COMPLETE.json` and
`raw/RAW_COMPLETE.json` receipts, each covering **28,800 frames**, node `FBody`,
with **no full-video track IDs**:

- `Crab_Cam_2_2`: **1,726,477 predictions**, merge job `16999056`, raw SHA-256
  `72b7184ea6a836efb005963e94c397eadd9e5a87a05415c1da5e0af32cdb3981`.
- `Crab_Cam_2_3`: **1,723,746 predictions**, merge job `16999177`, raw SHA-256
  `90e550df85999a9366aec8c89b6a9e456c1fab2f9997e647a995a4c55b7f6865`.

A later user-requested [demo extension](demo/FEMALE_BURROWS.md) adds female
predictions/experimental demo-only tracklets and copied static burrow references
to the existing 2,000-frame male demo. It does not modify the full-video outputs
or extend training. **Earlier launch/activation status statements below are
historical observations, superseded by this completion update and the completed
validation sections later in this document.**

## Updated user decision: stop centroid after exactly 75 completed epochs

The user subsequently requested **save and terminate the centroid at epoch 75**,
retain resumable progress, and validate/predict using **those exact epoch-75
weights**. This supersedes the original 200-epoch centroid launch target below.
The centered-instance training policy is unchanged; job **16998959** has since
completed successfully, early-stopping after **70 epochs** (14,000 steps), with
its validation-best weights from epoch index 27 (human epoch 28), step 5,600.

- Effective centroid stage: **minimum = maximum = 75 completed epochs**. The
  minimum defers ordinary early stopping until that requested target; its
  original patience/wait/best-score state and the LR scheduler remain recorded.
  All other training settings, labels, split and same-ID requeue limits stay
  unchanged. The original resolved YAMLs and run manifest still record the
  historical 200-epoch ceiling; they are immutable baseline provenance, not
  the active stop policy. The explicit override is
  `configs/female_v002/stage75.json`.
- Centroid job **16998958** keeps its ID. The tested launcher edit was applied
  with project file tools while it was **PENDING between normal allocations**;
  no cancellation, forced restart, fresh weights or replacement GPU job.
  Activation was subsequently confirmed in normal allocation hop 7: the log
  reports `female centroid75`, resume step **1,267**, epoch index **6**, and
  `effective_epochs=75`. `stage75_active.json` records the matching tested
  policy. No training progress was reset.
- At epoch 75, a full-state, read-back-validated immutable snapshot will be
  published at **`models/female_v002/centroid/stages/epoch_0075/`**, containing
  `best.ckpt`, `training_config.yaml`, baseline provenance and
  `STAGE_COMPLETE.json`. Here `best.ckpt` means **epoch-75 weights**, NOT the
  earlier validation-best weights. Canonical historical `centroid/best.ckpt`
  is preserved independently.
- The snapshot is saved after the training loop finishes all 75 epochs and its
  scheduler updates. Its serialized epoch counter is 75, unlike the male
  `on_train_epoch_end` snapshots' zero-based index 74; the receipt independently
  checks the loop's **75 completed epochs**. Adam state, LR scheduler,
  early-stopping callbacks and full loop progress are retained.
- Only successful snapshot validation writes the centroid stage-completion
  marker and allows job exit 0. This is completion of the **requested stage**,
  not a claim of 200 completed epochs. Save/corruption failures remain fatal.
- Existing validation **16998986**, predictions **16999003/16999004** and CPU
  merges **16999056/16999177** retain their dependencies and IDs. Validation
  now freezes the **exact epoch-75 centroid** with the completed centered
  model's validation-best weights. Both full videos use that same selected pair.
- No automatic training beyond 75 is scheduled. For future continuation, start
  from the immutable epoch-75 **full checkpoint**, not an earlier best or a
  weights-only initialization. Continuing requires a separate explicit resume
  action that preserves this stage marker/snapshot and does not let the current
  stage-completion guard return without training. Original early-stopping and
  scheduler state are preserved, not silently reset; if patience is already
  exhausted, the scientific stopping policy must be considered explicitly.

Finite CPU test job **17016724** passed all **18 tests** (12 existing female
controls and 6 new stage controls). Tests include real Lightning training to 75
and continuation to 76, exact optimizer-step counts, unchanged weights/Adam
moments/LR scheduler versus uninterrupted training, deferred early stopping,
full-state snapshots, fatal save handling and exact downstream model selection.
The original GPU preflight is retained unchanged; a narrowly scoped additive
receipt binds it to the two tested female-only entry-point changes and all new
code/policy hashes. Male shared code is unchanged. Evidence:
`checks/female_v002/stage75/{TESTS.json,APPROVED.json,INSTALLED.json}`.
Post-edit verification job **17016807 COMPLETED**, exit **0:0** in 18 seconds;
`INSTALLED.json` reports `passed=true`, unchanged original data/config/helper
hashes, the matching tested edits, readable full-state resume progress, and
unchanged completed centered model. Centroid job **16998958** subsequently
entered its next normal `gputest` allocation and activated the 75-epoch worker.
All downstream jobs remained **PENDING (Dependency)**. No 75-epoch checkpoint,
held-out accuracy report, or full-video prediction is claimed created yet.

## Scope and status

User-requested fresh female training, approximately 10% frame-level validation,
and predictions on both original full videos. This is a new isolated run,
`female_v002`; existing male models, paused training, predictions, demo and
burrow annotations are not changed. No burrow model, parameter sweep, water
filter, cross-frame identity algorithm, figure rendering, or environment change
is part of this run.

**Preparation and smoke tests completed successfully.** CPU preparation
**16997245** completed exit 0:0. CPU checks **16998377** passed all 20 tests.
GPU preflight **16998321** completed exit 0:0 in 6m17s, including interruption,
full-state resume, held-out smoke validation, and two-stage inference/SLP readback.

The full requested chain is submitted. At signoff both production training jobs
were **RUNNING in `gputest`**; downstream jobs were **PENDING (Dependency)**:

| Stage | Job ID | Successful predecessor(s) required |
|---|---:|---|
| Centroid training | **16998958** | Saved matching preflight/data/code gate |
| Centered training | **16998959** | Saved matching preflight/data/code gate |
| Held-out validation | **16998986** | 16998958 AND 16998959 |
| Cam 2_2 prediction | **16999003** | 16998986 |
| Cam 2_3 prediction | **16999004** | 16998986 |
| Cam 2_2 CPU merge | **16999056** | 16999003 |
| Cam 2_3 CPU merge | **16999177** | 16999004 |

The user requested signoff after queueing this dependency chain, rather than
waiting for training. **Production training, held-out accuracy evaluation, and
full-video predictions are not claimed complete.** Exact scripts/arguments,
dependencies and the historical queue observation are in
`logs/female_v002_submission.json`; use Slurm and completion markers for later
live status. The shared finalization script now shows the Cam 2_3 launch's
dependency; Slurm already holds the independently submitted Cam 2_2 script with
its own correct dependency. Do not resubmit historical launch scripts blindly.

## Input and fixed holdout

Read-only input audit **16997084** found:

- `data/labels/Female_data/female_labels.v002.slp`: 16 human-labeled frames,
  1,369 instances; exactly one landmark, **`FBody`**, no skeleton edges and no
  predicted instances. Females do not have the male dominant-claw annotation.
- `data/labels/burrows/BurrowLabels3.slp`: 2 human-labeled frames, 564 instances,
  node `Burrow`. Inventory only; no burrow training is launched.
- Both reference the collaborator's `clip_cam2_8k.mp4`, with recorded shape
  1,820 × 4,320 × 7,680 × 3. Original external paths are read as metadata only.

The female upload remains byte-identical (SHA-256
`be918ab7afc8760e380744bc02dc2c35327730a0676ca8c6995118d7418d062c`).
An unchanged backup is under `data/labels/female_v002/source/`. Only derived
splits reference the project-local `clip_cam2_8k.mp4`. Preparation verified the
same video basename/full shape, decoded every labeled image, checked every
FBody coordinate is finite/in-image, and compared annotation signatures after
SLP readback. This is metadata/mapping consistency, not an independent
pixel-hash comparison against an unavailable collaborator movie.

Seed **42**, whole-frame splitting:

| Split | Frames | Instances | Zero-based frame indices |
|---|---:|---:|---|
| Train | 14 | 1,185 | 350, 698, 713, 797, 847, 875, 892, 899, 967, 1001, 1046, 1470, 1600, 1760 |
| Validation | 2 | 184 | **481, 1252** |

Two frames are the closest whole-frame approximation to 10% of 16: **12.5%**
(one would be 6.25%). The same split is used by both models, without instance-
level leakage between train/validation. Frames come from the same short clip;
temporal correlation and the small holdout limit generalization claims. This
validation set also selects checkpoints/early stopping, so it is **not an
independent final test set**.

## Same core training logic as the latest male run

- Use the unchanged `scripts/train_versioned.py`, `pipeline_common.py`, and
  existing tested safe-stop/checkpoint helpers. The centroid's authorized
  `scripts/female_stage75.py` stage worker reuses these helpers and preserves
  its original run manifest; centered training uses the original worker.
  No edits to the pinned male implementation, preflight receipts, or configs.
- Inherit the current resolved `configs/v007_19f/{centroid,centered}.yaml`.
  Retain native grayscale resolution, UNet backbones, augmentations, batch 1,
  192-pixel centered crops, 200 train steps/epoch, Adam learning rate 1e-4,
  scheduler patience 20, early-stopping patience 40, original ceiling 200 epochs,
  seed 42. The later centroid-only stage override stops exactly at 75, as above.
- Both stages are anchored on **FBody**, and the centered stage has **one output
  landmark**. No male nodes, dummy claws, or fabricated body orientation.
  Although a one-point model could use a detector alone, the two-stage
  centroid → centered-instance structure is deliberately retained to match the
  requested male pipeline. Train fresh weights, not male-weight fine-tuning.
- Same 58-minute short GPU allocations, GPU80 constraint, 64-GB RAM, centroid
  1 CPU / centered 4 CPUs, three-minute USR1 warning, append logs, same-ID
  `scontrol requeue` **only for worker exit 75**. Max 400 hops/role and 3
  non-progress hops; fatal errors/save failures do not trigger replacement jobs.
- Scheduler state behavior and non-bit-identical mid-epoch dataloader replay
  are inherited from the male workflow; neither is silently repaired here.
- Della rejects explicit `--partition=gputest` directives. Removing that
  directive, as in the original male production scripts, lets the site route
  these short GPU jobs to `gputest`; preflight's assigned partition was checked.
  An obsolete dependency on already-completed preparation was also rejected
  before creating a job and removed; saved data-integrity gates still apply.
- Exclusive locks, data/config/code/environment manifests, progress-based full
  optimizer checkpoint selection, atomic save/readback, best-weight preservation
  and validated completion markers remain enforced. Only scheduling
  `#SBATCH --dependency=` lines are excluded from new shell-script hashes, so
  predecessor IDs can be added without changing executable behavior or resources.

## Preflight and tests

`jobs/preflight_female.sh` is one finite GPU job, not an autonomous retry loop:

1. Bash syntax checks; existing training regression tests; female-specific tests.
2. For each role, isolated two-epoch/two-step-per-epoch smoke training, interrupt
   after one optimizer step, verify exit 75, resume, validate best checkpoint.
3. One held-out-frame **API smoke inference** using only those smoke models,
   checking the one-node SLP format. This is not an accuracy experiment.
4. Only then write `checks/female_v002/PREFLIGHT_COMPLETE.json` with matching
   run/dataset/implementation/environment hashes. Production requires this gate.

`tests/test_female_workflow.py` covers exact/disjoint reproducible splitting,
unchanged uploaded annotations, inherited model settings and FBody heads,
maximum-cardinality one-to-one matching, empty inputs, localization metrics,
real one-node SLP merge/readback, all-empty chunks, empty-frame preservation,
wrong skeleton/coverage, corrupt offsets, no-overwrite, receipt/model signature
mismatches, and narrow dependency-only metadata changes. CPU checks can run via
`jobs/check_female_cpu.sh` without waiting for a GPU; they do not train/infer.

## Validation and prediction stages

After **both** production roles complete normally:

1. `jobs/validate_female.sh`: verify the centroid's completed 75-epoch stage and
   the centered model's training completion. Freeze the **exact epoch-75
   centroid / validation-best centered** pair at
   `models/female_v002/selected/{centroid,centered}/`.
   Infer only frames 481 and 1252 at native resolution, batch 1, threshold 0.2,
   no cap/filter/identities. Retain the held-out SLP and report counts, precision,
   recall, F1 and matched localization-error summaries at 5/10/**20**/50-pixel
   radii. Matching maximizes one-to-one match cardinality within each radius,
   then minimizes total distance. The primary descriptive radius is 20 pixels;
   no post-hoc threshold tuning is performed. Missing FBody cannot match but its
   instance remains in prediction counts. Apparent false positives are relative
   to provided labels, which may not be biologically exhaustive.
2. Two `jobs/predict_female.sh <video-key>` jobs depend on successful validation.
   Reuse the existing male `gputest_predict.predict_chunk` inference helper,
   with the selected female models explicitly passed. 40-frame chunks, native
   resolution, threshold 0.2, batch 1, no cap/masks/IDs. Same-ID short-allocation
   requeue, finite 1,000-hop and 3-no-progress limits. Completed chunks have
   range/model/code/source signatures, full SLP readback and SHA-256 receipts.
   Empty detection frames are processed frames, never silently missing coverage.
3. Separate finite CPU `jobs/finalize_female.sh <video-key>` jobs depend on their
   matching inference completion. Stream native SLP tables into one final SLP,
   validate every coordinate/score byte and index offset against chunks, check
   complete coverage, and publish atomically. No large merge in GPU requeue
   loops; existing/unreceipted finals are preserved for review rather than replaced.

The male finalizer assumes five landmarks. New `scripts/female_prediction_io.py`
adapts its validated streaming approach to the uploaded skeleton without
modifying the old finalizer. No motion smoothing, gap filling, tracklet IDs or
unique-animal accuracy claims are added. Raw female detections can support a
separately reviewed downstream association/dataframe pipeline later.

## Locations

| Artifact | Path |
|---|---|
| Run decisions | `configs/female_v002/run.yaml` |
| Resolved model configs | `configs/female_v002/{centroid,centered}.yaml` |
| Fixed split/source provenance | `data/labels/female_v002/{train.slp,val.slp,manifest.json,DATASET_COMPLETE.json}` |
| Production training | `models/female_v002/{centroid,centered}/` |
| Exact-75 resumable centroid stage | `models/female_v002/centroid/stages/epoch_0075/` |
| Explicit centroid stage override | `configs/female_v002/stage75.json` |
| Isolated software smoke models | `checks/female_v002/training_smoke/` — **not production** |
| Validation | `checks/female_v002/validation/{heldout.predictions.slp,VALIDATION_COMPLETE.json,report.md}` |
| Full-video raw predictions | `predictions/female_v002/<video>/raw/<video>.female.predictions.slp` |
| Full-video completion | `predictions/female_v002/<video>/PREDICT_COMPLETE.json` |
| Logs | `logs/female_{stage}_<job-id>.{out,err}` |

`<video>` is independently `Crab_Cam_2_2` or `Crab_Cam_2_3`. Each source movie
has previously verified 28,800 frames; this workflow reads actual video metadata
and records coverage/source pins rather than inferring lengths from chunk counts.
SLPs retain original full-video paths and zero-based frame indices. For local
SLEAP viewing, relink to the matching original movie, not a resized/cropped video.

Require the corresponding completion marker before treating models, validation,
or predictions as finished. Queue/accounting and durable progress are the live
status sources. A dependency failure blocks downstream execution; no background
watcher, recurring submitter, fallback chain, unapproved retraining or sweep is
installed. Job IDs are historical receipts, not a claim that jobs have completed.

## User-requested right-bank conditional validation

The user requested conditioning the existing held-out results on land to the
**right of the outlined puddle**, following the curved right bank, with vertical
extensions at the puddle's tips. This is a post-hoc validation-only subset, not
new training, threshold tuning, inference, or filtering of full-video outputs.

Use the accepted effective mask at
`datasets/v007_best_body_first_dataframe_v001/config/water_regions.json`, not
its older unconfirmed proposal. The source right-bank path runs from top tip
**(3290, 0)** to bottom tip **(2590, 3170)**. Extend the lower boundary vertically
at **x=2590 to y=4320**; the upper tip already meets the image top. Close the
region to the right image edge. Preserve actual bank indentations, including
folds where a horizontal scanline may meet the water more than once; do not
substitute a rightmost-x rectangle or a rowwise envelope.

Filter **both** human and predicted `FBody` positions to this same polygon before
one-to-one matching at the original 5/10/20/50-pixel radii. Exclude water, the
shared shoreline and its vertical extensions; retain valid image-top points to
the right of the boundary. No 30-pixel uncertainty band is added. Invalid or
out-of-image points are not assigned to this region. Before regional evaluation,
require exact reproduction of all original full-frame validation metrics.

Entry point: `jobs/validate_female_right_bank.sh`; finite CPU job
**17169644 COMPLETED**, exit **0:0** in **11 seconds**, with all **7 tests**
passing, exact original full-frame metric reproduction and source immutability
checks. No retraining/inference or production change was performed.

At the unchanged primary **20-pixel** matching radius:

| Selection | Labels | Predictions | TP | FP relative to labels | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full frames | 184 | 142 | 117 | 25 | 67 | 82.39% | 63.59% | 71.78% |
| Right-bank region | 134 | 107 | 89 | 18 | 45 | 83.18% | 66.42% | 73.86% |

The region excludes 50 human annotations and 35 predictions. Regional median
matched position error is **2.49 px**, mean **2.67 px**, p90 **4.43 px**.
Metrics are identical at 10, 20 and 50 pixels in this region. Thus regional
conditioning modestly improves recall/F1 but does not remove the missed-detection
problem. Full provenance and numerical results are under
`checks/female_v002/validation/right_bank_v001/`:

- `REGIONAL_VALIDATION_COMPLETE.json`: counts, pooled/per-frame metrics, source
  hashes/model provenance, limitations and test receipt.
- `report.md`: full-frame comparison and conditional results.
- `ROI.json`, `roi_vertices_px.csv`: exact geometry and boundary convention.
- `point_membership.csv`: every human/predicted point and its region membership.

Tests cover vertical extensions, strict boundary rules, excluded water/left
bank/invalid points, real curved-bank geometry and tips, reversed/closed polygon
ordering, land pockets at bank indentations, and symmetric pre-match filtering.
The two-frame/within-clip/model-selection limitations still apply. Regional
results cannot be advertised as an independent test or whole-scene performance.
