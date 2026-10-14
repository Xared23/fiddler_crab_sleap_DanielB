# v007 combined-data retraining: decisions, inventory, and pending work

> **Update: source question resolved by the user.** Use only the 19 labeled
> frames in v007, treating it as cumulative; hold out two. The active run ID is
> `v007_19f`, not the provisional `v007_combined` inventory name. The user also
> authorized relinking the root v007 SLP to the local clip (with its untouched
> collaborator original backed up), and authorized production training. There
> is explicitly NO special water-patch treatment in training or prediction.
> See the root README and `data/labels/v007_19f/manifest.json` for current state.
> The original review below is retained as historical context; its blocker and
> initial implementation-status paragraphs are superseded by this update.

## Agreed scientific and workflow choices

- Fresh centroid and centered-instance training, not an optimizer-state resume of
  the previous completed models.
- Initial ceiling: 200 epochs. Coordinate early-stopping patience with the
  learning-rate scheduler (old values were 10 versus 20, respectively).
- Approximately 10% frame-level holdout, identical for both models and stable
  across restarts. Call it validation when used for early stopping/model
  selection; it is not an independent final test set.
- Primary quality problems: pose accuracy and missed males, particularly in the
  water patch. Body/shoulder localization was relatively good. Review water and
  land separately; do not optimize for track continuity at the expense of poses.
- Treat the two full prediction videos independently.
- Conservative tracklets: end continuity at missing/unreliable detections and
  ambiguous associations. No gap filling, re-identification, optical-flow/Kalman
  extrapolation across gaps, or post-hoc tracklet stitching in this first run.
  Pose detections alone cannot reliably identify every physical occlusion or
  burrow entry; document the operational confidence/matching rule honestly.
- GPU inference in resumable chunks, with 20-minute prediction allocations.
  Keep the existing intentional gpu-test chaining design, with bounded retries.
- Separate CPU merge/tracking/final-save stage with adequate walltime; no full
  merge inside repeated GPU inference hops.
- Deliver tracked SLPs first; retain original video paths so overlays can be
  rendered later. Do not render full MP4s now.
- Do not move/archive old artifacts until replacements are checked and the
  archive operation is approved.

## Audit performed

Slurm job `15255806` failed before Python because `module` was not defined in the
batch shell. Removed the unnecessary module command. The same finite audit
completed as job `15255830` (12 seconds, exit 0).

Report: `data/labels/v007_combined/audit.json`.

| Input | Stored frame records | Human-labeled frames | Human instances |
| --- | ---: | ---: | ---: |
| danlabels.v004.slp | 13 | 12 | 1,619 |
| danlabels.v007.slp | 20 | 19 | 2,573 |

The arithmetic sum is 31 labeled-frame records and 4,192 instance annotations;
this is not yet a verified count of distinct source images. Both inputs have the
same five-node skeleton. Neither contains predicted instances. Frames without
human instances are excluded from these training-frame counts; the audit does
not classify them as deliberate negative examples.

### Source-mapping blocker discovered in the inventory

Per the user's instruction, the audit trusted the collaborator's statement that
the sources were non-overlapping. No coordinate/image comparison or deduplication
was performed. Nevertheless, the frame inventory directly exposed this ambiguity:

- Both sources name `clip_cam2_8k.mp4` and record shape
  `[1820, 4320, 7680, 3]` at approximately 29.97 fps.
- v004 points to the project-local clip.
- v007 points to
  `/Users/krystianrzeczycki/LLP Proj/Daniel_videos/clip_cam2_8k.mp4`.
  That path was read as metadata only; the audit did not access that filesystem.
- All 12 old frame indices also appear among v007's 19 indices:
  `0, 151, 302, 453, 604, 755, 906, 1057, 1208, 1359, 1510, 1661`.
- The seven additional indices are:
  `84, 205, 682, 774, 984, 1593, 1762`.

If these are the same video, the union has **19 distinct labeled source frames**,
with seven additional frames in v007. If the clips differ despite sharing their
name and shape, there may instead be 31 distinct labeled source frames and the
second video must be supplied/mapped correctly.

Do NOT independently split the files or map the second file to the local clip
without resolving this question. Do NOT silently deduplicate, discard older
annotations, rewrite the originals, or assume repeated frames have identical
annotations. The report's proposed train/validation counts are conditional
arithmetic under the original non-overlap assumption, not an approved split.

Pending question to user/collaborator: is v007 a cumulative version of labels for
the SAME clip, or labels for a DIFFERENT clip that has the same filename?

## Implementation status

Only inventory tooling and an initial shared-helper scaffold have been added.
No label preparation, training, model evaluation, inference, tracking, or archive
jobs have been launched. Existing production scripts remain unchanged. The
20-minute setting, new epoch ceiling, and holdout are agreed requirements, not
settings already active in those old scripts.

## Implementation sequence after source confirmation

1. Resolve video mapping and choose a documented source policy. If same-image
   annotations occur in both files, keep them in the same split at minimum;
   resolve any cumulative-version choice with the user rather than guessing.
2. Create a run manifest and reproducible frame-level split; record original
   checksums, exact video/frame membership, package versions, and all settings.
   Do not combine raw originals destructively. With 19 distinct frames, a
   two-frame holdout is approximately 10.5%; with 31, three frames are about 9.7%.
   Either is a small validation set and cannot establish robust water/land
   generalization by itself.
3. Prepare fresh model configs and stable run directories. Verify source pixels,
   coordinates, skeleton, crop coverage (especially claw extremes), and GPU
   compatibility. The installed PyTorch is 2.13.0; the previous read-only source
   inspection found a CUDA 13.0 build, whereas project comments mention 12.8.
4. Implement/test safe restart handling:
   - Keep all parameters in a versioned manifest, not transient environment
     overrides lost through `sbatch --export=NONE`.
   - Only explicit transient/continuation statuses request another hop. Fatal
     worker errors stay fatal even when a preemption signal was also received.
   - An exit 0 without validated stage completion is an error, not success.
   - Bound total hops and consecutive hops without durable progress.
   - Lock each run against concurrent writers and write progress atomically.
   - Resume the newest valid checkpoint by training progress, not just mtime;
     audit epoch/global-step progress and the non-resumable dataloader warning.
   - Keep recoverable checkpoints and never erase a valid best checkpoint during
     setup. Do not claim exact mid-epoch replay without verifying sampler state.
5. Implement/test prediction and finalization:
   - Use explicitly selected NEW model checkpoints and validate their completion.
   - Bind raw chunks to video/model/config fingerprints and exact frame ranges.
   - Handle empty-detection frames without treating them as unprocessed frames.
   - Reject missing/duplicate/mismatched chunks before marking inference done.
   - Finalization is a separate finite CPU job, not a GPU requeue loop. Do not
     attach it naively to the first job ID of a chain that may submit new IDs.
     A manual explicit finalization submission after completion is the simplest
     initial handoff; any later automatic handoff must be exactly-once guarded.
   - Conservative adjacent-frame association must have confidence, displacement,
     and ambiguity rejection; a short SLEAP window alone still permits unsafe
     forced matches. Preserve every raw pose even if it gets no reliable track.
   - Use one tracker state for the entire video, not one per chunk. No reconnects
     across gaps and no cross-video identities.
   - Save to `.tmp.slp` with explicit SLP format, read back and validate, atomically
     replace the destination, then write completion. Keep prior good output
     until replacement validation succeeds. A final-save failure must stop for
     review rather than endlessly resubmit.
6. Run Slurm regression tests and a bounded GPU preflight before asking for
   production approval. Tests should cover signal/fatal-error precedence,
   checkpoint progress, no-progress caps, corrupt/missing chunks, interrupted
   saves, stale completion markers, tracklet gaps, ambiguous matches, and chunk
   boundary continuity. Do not rerun expensive whole-frame training evaluation
   that previously OOMed just to obtain a completion marker.
7. Train, review held-out pose accuracy/recall (especially water), predict both
   full videos, finalize tracked outputs, validate, and only then promote/archive.

No production launch is authorized by the existence of this plan.
