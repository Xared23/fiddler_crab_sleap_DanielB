# Fiddler crab SLEAP tracking

[Documentation index](README.md) · [Pause and resume instructions](RESUME_V007_TRAINING.md)

Unless a clickable link says otherwise, code-formatted paths below are relative
to the project root, not this documentation directory.

Top-down male crab pose estimation: a Shell-anchored centroid model followed by
Shell / Shoulder / Elbow / Claw / Arm pose estimation.

## Current run: `v007_19f`

**Centroid training is PAUSED at the user's request.** Job `15256499` was
cancelled while pending between allocations. The latest saved progress is epoch
index **104** (partway through epoch 105), optimizer step **21,212**. Both the
75-epoch and 100-epoch snapshots were saved. Centered-instance training completed
200 epochs. See the [resume guide](RESUME_V007_TRAINING.md) for frozen model
locations and the explicit future resume command. No resume has been submitted.
Slurm and validated records are status sources; submission/pause receipts are
historical records, not live job-status indicators.

The user selected **only the cumulative `danlabels.v007.slp`**: 19 human-labeled
frames, split into **17 training / 2 validation** frames using seed 42. Both
models use the same fixed split. Frames 84 and 1359 are held out (zero-based
indices). Validation is for model selection/early stopping, not an independent
final test set. **No special water/land handling** is used in training or planned
prediction: no region masks, weighting, filtering, or separate models.

### Where the new work lives

| Artifact | Path |
| --- | --- |
| Run decisions and limits | [`configs/v007_19f/run.yaml`](../configs/v007_19f/run.yaml) |
| Full centroid config | [`configs/v007_19f/centroid.yaml`](../configs/v007_19f/centroid.yaml) |
| Full pose config | [`configs/v007_19f/centered.yaml`](../configs/v007_19f/centered.yaml) |
| Fixed split, counts, checksums, source provenance | [`data/labels/v007_19f/manifest.json`](../data/labels/v007_19f/manifest.json) |
| Derived training/validation labels | `data/labels/v007_19f/{train,val}.slp` |
| Original collaborator file, unchanged backup | `data/labels/v007_19f/source/danlabels.v007.collaborator.slp` |
| NEW production models | `models/v007_19f/{centroid,centered}/` |
| Saved 75/100-epoch centroid snapshots | `models/v007_19f/centroid/milestones/epoch_{0075,0100}/` |
| Frozen models at pause | `models/v007_19f/pauses/step_00021212/` |
| Logs | [`logs/v007_19f/`](../logs/v007_19f/) |
| Isolated smoke-test artifacts (NOT production models) | `checks/v007_19f/` |

The root `danlabels.v007.slp` was relinked, at the user's request, to the local
`clip_cam2_8k.mp4`. Its annotations were checked against the untouched backup.
Preparation decoded every one of the 19 labeled frames. Do not delete/move the
raw video: SLPs reference it rather than embedding copies of every image.

### Training

Fresh models, 200-epoch ceiling, early-stopping patience 40, existing scheduler
patience 20. Native resolution, 192-pixel pose crops, network architectures,
augmentation settings, and the 200 training steps/epoch are retained from the
previous model configuration. No pretrained old weights are used.

Production batch entry points:

- `jobs/train_v007_centroid.sh`
- `jobs/train_v007_centered.sh`

Those original scripts retain launch-specific `afterok:15256367` dependencies.
Do not use them for the requested future resume. Use
`jobs/resume_v007_centroid.sh`, which has no expired Slurm dependency and still
requires matching saved preflight, extension, data/configuration and checkpoint
integrity checks. Submit it only when the user explicitly wants to resume.

Both use 58-minute gpu-test allocations, 80-GB GPUs, 64-GB host RAM, and a
three-minute stop warning. They retain intentional auto-requeue via `scontrol`;
only the worker's explicit exit 75 requests a next hop. Requeue rejection,
uncaught errors, OOM, invalid saves, and exit 0 without validated completion do
not trigger a replacement submission. Logs append across requeues of the same
job ID. Limits: 400 total hops and 3 non-progress hops.

`TRAINING_COMPLETE.json` is written only after a normal epoch-limit/early-stop
finish and checkpoint read-back. Each run has an exclusive writer lock and
pinned data/configuration/code manifests. Best weights are never hidden/renamed
for setup. Resume chooses valid checkpoints by global-step progress rather than
just file modification time; interrupt saves preserve a previous checkpoint.
Signals request stops, while checkpoint writes happen at batch hooks.

**Do not make unapproved edits to active configs or worker/helper code.** A
changed manifest is a hard error, not permission to combine incompatible
checkpoints. The user-requested 75/100-epoch snapshot addition has a separate,
tested, narrowly scoped extension receipt; original manifests remain unchanged
and actual runtime hashes are recorded separately. Unrelated changes still fail
integrity checks. For new data/settings, create a new run ID and reviewed scripts.

The inherited SLEAP/PyTorch dataloader is not exactly resumable mid-epoch: sampler
ordering is not guaranteed to replay identically across a hop. The preflight
checks actual optimizer progress, held-out validation, and checkpoint recovery;
it does not claim bit-identical interrupted/uninterrupted training. The previous
whole-frame post-training evaluation that OOMed remains disabled.

### Intermediate centroid snapshots

Both requested milestones have been saved: **75 completed epochs / step 15,245**
and **100 completed epochs / step 20,275**. These are epoch-end weights, not the
best weights seen so far. Each snapshot includes its model config, epoch, step,
validation loss, checksums and approximate timing. Pair them with the same
finished centered-instance model for comparisons. No intermediate predictions
have been launched. See [milestone details](milestone_checkpoints.md) and the
[pause/resume guide](RESUME_V007_TRAINING.md).

### Verification jobs

- `jobs/prepare_v007_19f.sh`: finite CPU backup/relink/split/config preparation.
- `jobs/preflight_v007_19f.sh`: regression checks plus tiny, isolated GPU training
  runs for both model types, each interrupted after one step and then resumed.
  Does not requeue or launch production. Production workers require its matching
  `checks/v007_19f/PREFLIGHT_COMPLETE.json` before starting.

- `jobs/check_milestone_extension.sh test`: finite CPU tests of the additive
  checkpoint callback (job `15389031`, all five tests passed).
- `jobs/check_milestone_extension.sh verify-install`: verify that the production
  edit matches the tested hashes and original data/configuration baseline.

All execution, including Python inspection and tests, is through Slurm.

## Predictions: historical outputs versus future new outputs

**No newly retrained prediction files exist yet.** The following are historical:

- [`predictions/Crab_Cam_2_2/`](../predictions/Crab_Cam_2_2/)
- [`predictions/Crab_Cam_2_3/`](../predictions/Crab_Cam_2_3/)
- Old model pair: `models/dan_centroid_native/`, `models/dan_centered_native/`.

Paused and milestone models can be evaluated before training finishes, with
explicit checkpoint selection and a separate approved prediction job. For later
full-video new-model outputs, the intended location is
`predictions/v007_19f/<video>/`, not either old prediction directory. Inputs:
`crab_vids/Crab_Cam_2_2.mp4` and `crab_vids/Crab_Cam_2_3.mp4`.

Next-stage requirements are recorded in the run config: 20-minute GPU prediction
allocations, resumable untracked chunks, a separate finite CPU merge/tracklet/save
stage, conservative tracklets without gap bridging, and independent identities
for the two videos. This new inference/finalization worker is **not implemented
or submitted yet**. Root-level legacy prediction scripts still use old models
and must not be mistaken for the new workflow.

Deliver tracked SLPs first. Overlay videos can be rendered later from those and
the original video files. Old artifacts will be archived only after replacements
are validated and the archive operation is approved. There is no `latest` link
that falsely presents old results as new ones.

## Original-prediction dataset organization: separate agent task

Before integrating new-model outputs, the user requested an audit of whether the
original predictions can support conservative tracklets, followed—if feasible—by
a pandas-readable dataset containing tracklet ID, frame, centroid, body/shoulder
information, and per-observation water-region membership. No inference rerun is
authorized until the existing predictions have been evaluated and a rerun approved.
Water labels here are downstream analysis metadata, not regional treatment in
training or prediction. Original artifacts and the new training run stay untouched.

- [Self-contained agent handoff](legacy_dataset_agent_handoff.md)
- [Starting prompt for a new agent session](legacy_dataset_agent_prompt.md)
- [Dataset notes and audit status](datasets/legacy_native_tracklets_v001/README.md)

The handoff is historical task context. The dataset notes record subsequent audit
results and user decisions; a completed audit is not a completed tracklet dataset.

## History

- [`docs/retraining_v007.md`](retraining_v007.md): original review and decisions.
- `data/labels/v007_combined/audit.json`: earlier source inventory. Its arithmetic
  sum assumed disjoint files and is superseded by the v007-only 19-frame decision.
- `danlabels.v004.slp`, historical root YAMLs/scripts, old models, logs and
  predictions are retained unchanged. Do not use `remap_dan_video_v004.sh` for
  the new run: that legacy script overwrites the old label file in place.
