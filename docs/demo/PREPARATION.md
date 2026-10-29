# v004 archive, v007 demo preparation, and Git diagnosis

All paths are project-relative. **No Git commands, staging, commits, pushes or
branch/remotes changes were performed.** The user subsequently authorized
`.gitignore` changes: narrow demo-payload exceptions and exclusions for the
archive/incomplete builds are now applied. Reading Git's project-local metadata
and existing historical push receipts was sufficient for the initial diagnosis.

## Active data inventory

| Item | Current path |
|---|---|
| Raw full videos | `crab_vids/{Crab_Cam_2_2,Crab_Cam_2_3}.mp4` |
| Training clip (preserved) | `clip_cam2_8k.mp4` |
| Cumulative current labels | `danlabels.v007.slp`; fixed split/provenance under `data/labels/v007_19f/` |
| Current model family | `models/v007_19f/` |
| Frozen models used for full-video predictions | `models/v007_19f/pauses/step_00021212/{centroid_best,centered_best}/` |
| Current full-video raw/tracked SLPs | `predictions/v007_best_body_first_v001/<video>/{raw,tracklets}/` |
| Original finalizer tables | `predictions/v007_best_body_first_v001/<video>/tracklets/*.parquet` |
| Recommended standardized dataframe | `datasets/v007_best_body_first_dataframe_v001/` |
| Collaboration demo | `demo/crab_cam2_2_middle_300/` (require `DEMO_COMPLETE.json`) |
| v004 payload archive | `archive/v004_legacy/` (require `ARCHIVE_COMPLETE.json`) |

Centroid training remains paused; the archive/demo does not resume training,
run inference, alter thresholds, rerun tracking, or promote the later bounded
threshold/grace experiments. The full-video v007 dataset is the latest-label
production dataset, even though some later experiments use different settings.

## Archive

CPU Slurm job **16416656** moved **6,714 files / 5,295,645,123 bytes** into
`archive/v004_legacy/`, mirroring their original relative project paths.
All pre/post file hashes, mtimes and inodes matched. Active v007 artifacts,
raw videos/clip and Git metadata/ignore rules were unchanged.

Includes the v004 native centroid/centered pair, all earlier `models/dan_*`
attempts, v004 labels, both historical prediction directories, the legacy
dataframe, old-data body-first tracking pilot evidence, and associated logs.
Source code, root historical YAMLs/scripts and historical documentation were
retained, not silently rewritten or repurposed.

- `archive/v004_legacy/inventory.json`: every original file path, hash, stat,
  and root relocation mapping.
- `archive/v004_legacy/ARCHIVE_COMPLETE.json`: compact completion/verification.
- `datasets/legacy_native_tracklets_v001/`: **not an active dataset anymore**.
  It retains only a notice and a relative `config/` compatibility symlink to
  the archived approvals/ROI metadata. These bytes are pinned by v007 manifests
  and must remain reachable; no full old observations table is left here.
- Old manifests were not rewritten. For moved legacy source paths, prepend
  `archive/v004_legacy/`. Raw-video paths were not moved, so old SLPs retain
  their original raw-video links. Old-data audit/pilot regeneration would need
  explicit path adaptation; do not launch old source scripts unmodified.

## Demo selection and validation

The source has 28,800 frames at 7680 × 4320 and 30000/1001 fps. The centered
300-frame selection is **14250–14549 inclusive**. The MP4 and both SLPs use
clip-local 0–299; dataframe source columns retain original indices and IDs,
with explicit added `demo_frame`, `demo_time_s` and observation
`demo_instance_row`. No filtering on water, acceptance or track assignment.

The movie is a native-resolution H.264 demo copy, **not lossless pixels**.
SLP pose/score bytes and dataframe source-column values are preserved exactly.
All payload files must be below 100 MiB individually. The intended bundle is
MP4 + raw SLP + tracked SLP + observations/frame Parquet + frame mapping +
README + completion/provenance receipt. Neither full original video nor models
are required by collaborators.

Build job **16416826** passed all 8 synthetic controls, extracted the clip and
SLPs/dataframes, and passed video decode, every-pose/track/dataframe comparison,
Parquet readback and relocated SLP/video loading. It then failed on an
inappropriate final assertion that Git metadata was unchanged while the user
was configuring their tracking branch. This was a workflow error, not a data
or inference failure. Its generated files were retained, not discarded.

Existing-file validation/publication job **16446713 COMPLETED**, exit **0:0**,
in **59 seconds**, using `jobs/publish_existing_crab_demo.sh` /
`scripts/publish_existing_crab_demo.py`. It revalidated the existing payloads,
wrote `DEMO_COMPLETE.json` with `status=validated_complete`, and renamed staging
into **`demo/crab_cam2_2_middle_300/`**. No video encoding, inference or tracking
was rerun. The bundle has **300 frames / 36,573 observations**, including
**36,519 assigned / 54 unassigned** and **1,181 existing experimental tracklets**.
The MP4 is 77,624,774 bytes; each file is below 100 MiB. No desktop GUI is claimed
tested. It is now ready locally for a separately authorized commit/push; this
publication job did not stage, commit or upload anything.
The extraction script no longer treats unrelated user Git changes as a data
publication failure; neither script invokes Git.

Inspection job **16416619** completed successfully. Logs for these jobs:
`logs/demo_preparation/`. Generator: `scripts/prepare_crab_demo.py`; controls:
`tests/test_crab_demo.py`; archive implementation: `scripts/archive_v004.py`.
The generated bundle README is copied from `docs/demo/README.template.md`.

## Git diagnosis and user-completed repair

The user supplied the exact `git push` error: `The current branch main has no
upstream branch.` This confirmed missing local tracking configuration, not a
GitHub authentication failure. The user subsequently configured/pushed their
branch themselves. Inspection afterward confirmed local `my-code` with
`remote=origin` and `merge=refs/heads/my-code`. No agent Git commands or branch
configuration changes were performed. A successful code push does not include
new/uncommitted demo files automatically.

The following records the **pre-repair** inspection of `.git/config`, `.git/HEAD`,
local refs and packed refs:

- `origin` is already `git@github.com:Xared23/fiddler_crab_sleap_DanielB.git`.
- Remote named `upstream` is the original collaborator repository:
  `https://github.com/kryrze27/fiddler_crab_sleap.git`.
- The current local branch is **main**, commit
  `d9e650d4e559f6da6bbacfd62c9d67c56c9ca5ea` at inspection.
- The only other local ref is
  `backup/before-data-exclusion-2bcbcf5`. There is **no local `my-code`**.
- `branch.main.pushRemote=origin`, but `branch.main.remote` and
  `branch.main.merge` are **absent**. The VS Code merge-base setting is not a
  Git tracking configuration. With the ordinary default push policy this
  explains a plain `git push` reporting that main has no upstream branch.
- No `origin/*` remote-tracking refs are present locally. This is **not proof**
  that the remote has no branches; no network fetch/lookup was performed.

“upstream” is overloaded: the *remote named upstream* and a branch's *tracking
(upstream) branch* are different things. The original-repository remote is
normal for a fork and need not be removed to fix branch tracking.

Historical evidence, not a current connectivity test:

- `logs/github_publish/publish_15581291/push_result.json` records an older HTTPS
  **main:main** push failing with `Could not resolve host: github.com` from a
  compute node. That did not contact GitHub; it says nothing about current SSH
  authorization, the existence of remote `my-code`, or a non-fast-forward.
- The earlier job 15581215 failed on missing Git author identity before a
  replacement commit; 15581291 preserved the prior commit identity and got
  farther. Author identity is separate from SSH push authentication.
- A prior local cleanup removed large dataframe blobs from then-current main
  ancestry; the local backup ref deliberately retains the original history.
  Do not push that backup, all branches, tags or a mirror. The old audit is
  not a fresh size audit of the current October-29 commit.

The later exact error confirmed the missing-upstream diagnosis as noted above.
No SSH keys/agent credentials/global configuration were inspected, changed or
exposed. No agent push, even a dry-run, was attempted.

### Original repair proposal (user subsequently handled tracking/push)

First establish whether the public fork already has `my-code` and compare its
history to local work. Then either use a local `my-code` branch tracking
`origin/my-code`, or deliberately push the present branch with the explicit
refspec `HEAD:refs/heads/my-code`. An explicit refspec does not require an
existing tracking branch; it must still pass authentication and history checks.
Do **not** blindly set local main to track origin/main if the desired target is
my-code, or force-push/reset/rename branches to make the error disappear.
Remote inspection/fetch and any configuration/branch repair await approval.

## Authorized ignore changes — APPLIED

The user explicitly authorized making the demo Git-visible. The following rules
are now appended to `.gitignore` after its global payload exclusions:

```gitignore
# Local legacy payloads must never enter the collaborative code/demo history.
/archive/
/demo/.build_*/

# Narrow exception for this one bounded, portable collaboration demo.
!/demo/crab_cam2_2_middle_300/Crab_Cam_2_2.frames_014250_014549.mp4
!/demo/crab_cam2_2_middle_300/Crab_Cam_2_2.frames_014250_014549.predictions.slp
!/demo/crab_cam2_2_middle_300/Crab_Cam_2_2.frames_014250_014549.tracked.slp
!/demo/crab_cam2_2_middle_300/observations.parquet
!/demo/crab_cam2_2_middle_300/frames.parquet
```

The demo is outside `/datasets/`, so no ignored-parent exceptions are needed.
The other demo metadata/docs are already ordinarily eligible. Only this bundle's
five listed payloads are exempted; full datasets/videos/predictions stay ignored.
The `/archive/` exclusion is important because the former path-specific
`models/**/*.ckpt` rule does not protect relocated checkpoints. Review expected
legacy working-tree deletions and new demo/source/docs additions explicitly;
do not force-add archived payloads. Nothing has been staged, committed or
uploaded by this work.
