# v007_19f: paused training, available models, and how to resume

[Documentation index](README.md) · [Project overview](PROJECT.md)

Project root: `/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap`.
All code-formatted paths below are relative to that root unless absolute.

## What happened

The user requested that centroid training stop so the existing models could be
examined before spending more training time. Job **15256499** was **PENDING
between allocations** when cancelled. Slurm accounting confirmed `CANCELLED`.
There was no running batch to interrupt, and the previous hop's durable save
was already available. No forced mid-write save or extra training was necessary.

Finite CPU job **15521654** completed successfully on **2026-10-14 UTC**. It:

- Read back and hash-verified the 75- and 100-completed-epoch snapshots.
- Verified the latest canonical checkpoint and its full training state.
- Preserved independent frozen copies of latest centroid, validation-best
  centroid, and completed pose-model best weights/configuration/provenance.
- Revalidated the unchanged dataset, video metadata, configs, baseline manifests,
  package versions, preflight and tested checkpoint-extension code hashes.
- Syntax-checked the future resume batch script and passed its CPU resume checks.

**No resume, new GPU training, or prediction job was submitted.** No training
completion marker was invented: this is a pause, not completed centroid training.
The script's future GPU execution has intentionally not been exercised here.

## Exact point at which training is paused

| Field | Value |
| --- | --- |
| Run | `v007_19f`, role `centroid` |
| Canonical resume checkpoint | `models/v007_19f/centroid/interrupt.ckpt` |
| Checkpoint epoch, zero-based | **104** |
| Human-readable position | **104 completed epochs, partway through epoch 105** |
| Optimizer step | **21,212** |
| Saved per-parameter Adam step counters | all **21,212** |
| Completed allocation hops | **122** |
| Consecutive no-progress hops | **0** |
| Configured total epoch ceiling | **200**, not 200 additional epochs |
| Saved learning rate | **0.0001** |
| Early-stopping patience / saved wait count | **40 / 30** validation checks |

The checkpoint contains model weights, optimizer moments, learning-rate scheduler,
early-stopping and model-checkpoint callbacks, and Lightning loop progress. These
are retained, not reset. Early stopping can therefore occur before epoch 200;
resumption does not give the run a fresh patience allowance.

**Observed scheduler caveat:** the serialized ReduceLROnPlateau state has
`last_epoch=0` and `best=Infinity`, despite the advanced training counter. This
state was preserved unchanged. The preservation checks establish that it is
present/readable, not that prior scheduler stepping behaved as intended. Review
that behavior separately before deciding to change the scientific training
policy; do not silently reset or repair it as part of a resume.

The inherited dataloader is not exactly resumable mid-epoch. Optimizer progress
and saved state are restored, but exact sampler replay / bit-identical results
against an uninterrupted run are not guaranteed. Do not translate step 21,212
into a different epoch by simply dividing by 200: interrupted hops can cause
those counters to differ.

## Models available for examination now

### Milestones: yes, both were saved

| Model | Completed epochs | Checkpoint step | Saved validation loss |
| --- | ---: | ---: | ---: |
| Centroid milestone 75 | 75 | 15,245 | 0.0000632621304 |
| Centroid milestone 100 | 100 | 20,275 | 0.0000552727615 |
| Validation-best centroid at pause | 73 (index 72) | 14,843 | 0.0000538005261 |
| Latest centroid at pause | partial epoch 105 (index 104) | 21,212 | not yet evaluated at this exact partial-epoch checkpoint |

The milestone directories are:

```text
models/v007_19f/centroid/milestones/epoch_0075/
models/v007_19f/centroid/milestones/epoch_0100/
```

The fixed pause bundle is:

```text
models/v007_19f/pauses/step_00021212/
    PAUSE_COMPLETE.json
    centroid_latest/
        best.ckpt
        training_config.yaml
        snapshot.json
        ...
    centroid_best/
        best.ckpt
        training_config.yaml
        snapshot.json
        ...
    centered_best/
        best.ckpt
        training_config.yaml
        snapshot.json
        ...
```

`best.ckpt` is the filename SLEAP expects within a model directory. In
`centroid_latest/` it is the **latest partial-epoch state**, NOT the best model
under validation loss. In `centroid_best/` it is the actual validation-best
checkpoint. Extra epochs do not automatically imply better predictions.

For an initial qualitative preview, use `centroid_best/` together with the
fixed `centered_best/`. For comparison, replace ONLY the centroid directory
with `centroid_latest/`, `epoch_0075/`, or `epoch_0100/` while keeping the pose
model and inference settings fixed. The pose model completed 200 epochs; its
selected best checkpoint is index 191 (epoch 192), step 38,400.

These directories contain the configuration and full checkpoint needed for
inference; no retraining is required to inspect them. Validation loss is a
training/model-selection measure, not a direct pose-accuracy assessment on a
large independent test set. No new prediction files or preview images were
created by this pause task. The next useful step is a separately requested,
bounded prediction preview on specified frames/clip(s), not an automatic
full-video rerun. Do not use legacy root prediction scripts: they select the
OLD `dan_*_native` model pair rather than these models.

## How to resume later

**Do not do this until the user wants training to continue.** First ensure there
is no other running or pending `v007_centroid` job. Submit exactly one job.

From the user's ordinary Della terminal:

```bash
cd /scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
sbatch jobs/resume_v007_centroid.sh
```

For an agent, read that script first and submit it with the Slurm submission
tool, using the project root as the working directory. Do not execute Python or
training directly on vis1.

What the resume script does:

1. Uses a NEW Slurm job ID. The cancelled job `15256499` is not revived.
2. Runs a finite first-start check of the frozen pause bundle, current data/code
   integrity, latest full checkpoint and remaining safety limits. Fails rather
   than silently starting fresh or moving backward behind the preserved pause.
3. Invokes the unchanged production training worker and requeue wrapper on the
   SAME `models/v007_19f/centroid/` run directory/configuration.
4. The worker selects the valid checkpoint with greatest saved progress. At this
   pause that is `interrupt.ckpt`, index 104 / step 21,212. Do not manually rename
   the older validation-best checkpoint over it or copy a frozen model over it.
5. Resumes optimizer/callback/scheduler/loop state through the existing SLEAP /
   Lightning resume path. The current 200-epoch ceiling and stopping rules stay
   in force. No new run ID, data split, or pretrained-weight-only restart.
6. Retains the original 58-minute `gputest` allocations, 80-GB GPU requirement,
   one CPU, 64-GB host RAM, and three-minute stop warning. Only explicit worker
   exit 75 requests a normal same-ID requeue. Fatal errors remain fatal.

The new batch entry point deliberately omits `afterok:15256367`, the historical
launch's Slurm dependency. It does NOT remove preflight/integrity validation:
the saved matching preflight and narrowly approved milestone extension are
still checked. Do not resubmit the original `jobs/train_v007_centroid.sh` with
its old dependency, and do not rerun preparation/preflight over pinned outputs.

After submission, note the NEW job ID. Its logs are:

```text
logs/v007_19f/centroid_<NEW_JOB_ID>.out
logs/v007_19f/centroid_<NEW_JOB_ID>.err
```

Check that its first `Hop ... resume=... progress=...` line resumes at least
step 21,212 rather than initializing fresh weights. Then use Slurm and durable
progress files for status. Preserve the original `centroid_15256499` logs as
history. The centered model is finished and should NOT be resubmitted.

## Preserve these files and rules

- Keep the entire canonical `models/v007_19f/centroid/` directory, including
  `interrupt.ckpt`, `interrupt.previous.ckpt`, `last.ckpt`, `best.ckpt`, config,
  manifests and hop state. Do not reset counters or delete `.worker.lock` to
  bypass an active process; the empty lock file is normally retained.
- Keep `models/v007_19f/pauses/step_00021212/` and both milestone directories
  immutable. Normal resumed training changes canonical rolling checkpoints,
  not these frozen copies.
- Keep labels, source video, run/resolved configs, original preflight and
  extension receipts, and pinned worker/helper code unchanged. A mismatch is a
  review condition, not permission to patch hashes or reset manifests.
- `models/v007_19f/TRAINING_PAUSED.json` is a historical receipt. It is not a
  live scheduler status or a flag that prevents explicit submission. Cancelling
  the pending Slurm job is what stopped continuation.
- Do not assume ordinary cancellation of a RUNNING job will provide enough time
  for a new checkpoint. This pause was especially clean because the job was
  pending after a completed safe-save hop. A future active-job pause needs a
  separately reviewed safe-stop procedure.

## Verification records

- [Frozen pause receipt](../models/v007_19f/pauses/step_00021212/PAUSE_COMPLETE.json)
- [CPU resume check](../checks/v007_19f/pause_15256499/RESUME_CHECK.json)
- [75-epoch snapshot record](../models/v007_19f/centroid/milestones/epoch_0075/snapshot.json)
- [100-epoch snapshot record](../models/v007_19f/centroid/milestones/epoch_0100/snapshot.json)
- [Future resume script](../jobs/resume_v007_centroid.sh)

`jobs/maintain_v007.sh verify-resume` can run the CPU-only checks again through
Slurm without starting training, if requested. Do not rerun `preserve` against
this already published bundle: that action intentionally refuses overwrites.
