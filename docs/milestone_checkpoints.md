# Centroid checkpoints after 75 and 100 completed epochs

**Status update:** both snapshots were saved and read-back/hash verified by
CPU Slurm job **15521654**. The 75-epoch snapshot is step **15,245**; the
100-epoch snapshot is step **20,275**. Centroid training is now paused after
104 completed epochs plus a partial 105th epoch, step **21,212**. See the
[pause/resume guide](RESUME_V007_TRAINING.md) for preserved models and resumption.

User request: keep the existing training run and its settings, but preserve
intermediate models for later accuracy-versus-training-time comparisons.

## Intended outputs

```text
models/v007_19f/centroid/milestones/
    epoch_0075/
        best.ckpt
        training_config.yaml
        snapshot.json
        manifest.json
        checkpoint_extension_active.json
        training_log.csv
    epoch_0100/
        ...
```

These directories now exist and passed checkpoint/config checksum and read-back
validation. Their `snapshot.json` records contain the exact timing and scores.
`best.ckpt` is the filename expected by SLEAP's model-directory loader. In these
milestone directories it contains the **weights at the end of that epoch**, NOT
the best-validation weights from an earlier epoch. The original training
`best.ckpt`, rolling checkpoints, and stopping policy are not replaced.

Epoch counts are human-readable completed epochs: 75 corresponds to checkpoint
`epoch=74`, and 100 to `epoch=99`. Optimizer steps are recorded rather than
assuming they equal exactly 200 times the epoch count under mid-epoch resumes.
A milestone directory is immutable and published only after checkpoint read-back
and epoch/step checks. It contains the model config needed for later inference.

## Comparing predictions

Pair each centroid milestone with the SAME finished pose model:

`models/v007_19f/centered/`

That isolates the effect of centroid training duration. This does not retroactively
create 75/100-epoch pose-model checkpoints: that model already completed its run.
Use the same videos, inference thresholds and tracking settings when comparing.
No prediction jobs are submitted as part of checkpoint retention.

## Timing and provenance

`snapshot.json` records completed epoch, optimizer step, current validation loss,
learning rate, job ID, timestamp, checkpoint/config checksums, and timing:

- Approximate cumulative fit-loop time: sums completed hops' existing log timing
  plus the current hop. Includes validation and checkpoint overhead, excludes
  most setup and queue waits. Prior durations were rounded in the old logs.
- Wall time since the first logged setup: includes intervening queue waits.

Neither is a measurement of GPU kernel time. The copied training CSV may not yet
contain the current epoch's final row because callback ordering controls when
that row is appended; the directly captured validation loss is in snapshot.json.

## How the addition is activated safely

The new helper is `scripts/milestone_checkpoints.py`. Its only training action is
an additional save at the requested epoch ends. No model parameters, input data,
augmentations, optimizer/scheduler settings, early stopping, epoch ceiling,
resource requests, or requeue rules change. It does not consume the training RNG.

The original training manifest and original preflight report remain unchanged
baseline records. A narrowly scoped tested extension receipt in
`checks/v007_19f/milestones_75_100/APPROVED.json` binds those baseline hashes to
exact new worker/helper hashes and the unchanged dataset/run configuration.
Any unrelated code change still fails integrity checks. The active worker writes
its actual hashes and receipt to `checkpoint_extension_active.json` on its next
normal start. That file distinguishes the original baseline from the extension;
no old preflight result is rewritten to pretend it tested the new callback.

A finite CPU Slurm job tests real Lightning epoch-end checkpoint behavior on a
tiny synthetic model, verifies unchanged optimization/RNG behavior, tests
atomic/idempotent output handling, and rejects unauthorized receipt changes.
The production worker edit is applied only after those checks, during an active
hop safely past startup. The already-running Python process keeps its loaded
code; the new callback takes effect at the next NATURAL requeue. There is no
forced stop/restart, polling service, recurring observer job, or additional
training experiment.

`checks/v007_19f/milestones_75_100/INSTALLED.json` records post-edit verification.
The files in this checks directory are audit/staging artifacts, not alternative
production entry points.

If auxiliary snapshot saving fails, it logs `MILESTONE_SAVE_FAILED` and writes an
`epoch_NNNN.FAILED.json` report where possible, but does not kill otherwise healthy
training. It never labels a later epoch as the requested one. If training stops
early before a target epoch, that future snapshot is not produced.
