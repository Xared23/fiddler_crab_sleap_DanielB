# Replace the 300-frame GitHub demo with 2,000 frames

## Requested change

Replace, rather than add alongside, the current collaborator demo on
`origin/my-code` (`Xared23/fiddler_crab_sleap_DanielB`). The new active bundle is
**`demo/crab_cam2_2_middle_2000/`**: Cam 2_2 original frames **13400–15399**, centered
on the same source-video midpoint, lasting **66.733 seconds**. Both SLPs and all
dataframe/frame-map rows cover the same 2,000 frames. Predictions come from the
existing published frozen-best v007 run; no retraining, new inference, tracking,
threshold changes or experimental grace-track promotion are involved.

The original 300-frame bundle is only moved to the ignored local
`archive/demo/crab_cam2_2_middle_300/` **after** the new bundle validates. Its
files are hashed before/after the move. It is removed from the active Git tree,
not deleted from local storage or rewritten out of historical commits.

## Video size and fidelity

The old 300-frame clip was 77,624,774 bytes. Extending its previous encoding rate
would exceed GitHub's ordinary per-file limit. The replacement retains native
7680×4320 resolution, H.264/yuv420p and every frame at 30000/1001 fps, with CRF 18
constrained by a **10-Mbit/s maxrate/buffer** instead of 60 Mbit/s. This is stronger
lossy video compression, not a resize or coordinate transformation. Small visual
details may be degraded. It is a viewing demo, not a pixel-faithful analysis video.
The MP4 and every other payload must be individually below 100 MiB.

Original pose/score bytes, identities and dataframe source-column values remain
exact. The local SLP frame range is 0–1999; dataframe original indices remain
unchanged and get explicit `demo_frame`, `demo_time_s`, `demo_instance_row`.

## Build and validation

Finite Slurm entry point: `jobs/replace_crab_demo_2000.sh` (4 CPUs, 24 GB, two-hour
walltime ceiling; no automatic resubmission). All tests and encoding run there.
The reusable builder keeps the historical 300-frame defaults; the replacement
script explicitly selects the new profile. Historical completed artifacts are
not edited or silently regenerated.

The test gate includes the original eight extraction controls, plus 2,000-row
boundaries and indexing, exact centered selection, an all-empty selection,
publication path exclusions, template consistency, a temporary-repository
path-only commit control (preserving unrelated staged edits), and a small real
video encode/decode/frame-mapping control. Real-data extraction then verifies:

- Published source manifests, selected source table/SLP hashes and video stat pins.
- Exactly 2,000 decoded MP4 frames at original resolution and rational FPS.
- Selected decoded original-frame hashes and sampled source/output image errors.
- Every raw/tracked pose byte and existing track ID against the source dataframe.
- Parquet readback and exact preservation of source columns.
- Loading copied SLPs against relocated video files using only local basenames.
- Output sizes and SHA-256 hashes; old-bundle preservation before replacement.

A desktop GUI is not launched. Automated pixel-error checks are not a claim that
small crab details are visually indistinguishable from the full-quality source.

## Scoped Git publication

The same authorized job attempts a narrowly scoped commit and normal push only
**after successful validation**. It requires:

- The current branch is `my-code`, and `origin` is the expected SSH repository.
- No pre-existing staged changes, no branch/HEAD change during the build, and
  no changes to the planned source/doc files while building.
- The old demo video is tracked on the current branch.
- Remote `my-code` exactly matches the starting local HEAD, so the push cannot
  unintentionally upload unrelated pre-existing local commits.
- Only demo replacement files, demo-specific scripts/tests/docs, navigation and
  narrow `.gitignore` edits are staged and committed. No broad `git add .`.

No force push, history rewrite, branch/remote/global-config changes, credential
inspection/changes, archive upload, or upload of full videos/models/dataframes.
An SSH/network/authentication error leaves the validated local replacement
intact and stops without automatic retries. A failed push can leave its local
commit in place; inspect the receipt rather than rerunning the whole build.

## Completion and status

This document specifies the workflow; **it is not a completion claim**.

- Build/log stream: `logs/demo_preparation/replace_2000_<job-id>.{out,err}`.
- Tests and publication receipts: `logs/demo_preparation/replace_<job-id>/`.
- `TESTS.json` and `tests.txt`: test gate result.
- New bundle `DEMO_COMPLETE.json`: validated local data only, not a GitHub upload.
- `LOCAL_REPLACEMENT_COMPLETE.json`: old bundle archived, new bundle validated.
- `PUSH_RESULT.json`: commit ID and actual push exit status.
- `PUBLICATION_BLOCKED.json`: reason publication stopped; no automatic retry.
- Final `COMPLETE.json` in the log directory requires validation **and** a
  successful non-force push.

The active bundle README is generated from `docs/demo/README_2000.template.md`.
Previous preparation history remains in [PREPARATION.md](PREPARATION.md).
