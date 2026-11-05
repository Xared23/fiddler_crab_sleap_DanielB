# 2,000-frame demo replacement: lossless video chunks

> **On branch `sam`** this bundle was copied from `my-code` (commit 8edb144)
> *beside* the 300-frame demo rather than replacing it: the gap-bridging
> golden tests, hold-out sweeps and promotion floor are defined on
> `demo/crab_cam2_2_middle_300/`. The 300-frame demo's frames 14250–14549 are
> row-identical to the same frames of this bundle. Do not run
> `jobs/replace_crab_demo_2000.sh` on `sam`: it archives the 300-frame demo and
> commits and pushes to `my-code`.

## Current implementation

The active replacement is **`demo/crab_cam2_2_middle_2000/`**, containing exactly
2,000 consecutive source frames in independently playable, size-bounded MP4
chunks. It replaces the 300-frame demo on `origin/my-code`
(`Xared23/fiddler_crab_sleap_DanielB`).

Source metadata inspection job **16795763** found original **HEVC/H.265**, native
7680×4320, 30000/1001 fps, no reordered frames, and a keyframe every 30 frames.
The nearest suitable start to 13400 is **13410**; the 2,000-frame selection is
**13410–15409 inclusive**. This 10-frame shift allows direct packet copying.
Chunks end at keyframes, except the final partial GOP, and are chosen using
actual packet sizes with a 90-MiB packet budget and a strict 100-MiB file limit.

**No video encoder is used on research footage.** The new containers retain the
original video packets, resolution, frame rate, color metadata and pixel values.
This introduces no additional video compression. Playback requires HEVC support.

Each numbered chunk has its matching raw SLP, tracked SLP and two Parquet tables.
The root also contains combined observation/frame tables, `chunks.json`, a frame
map and `DEMO_COMPLETE.json`. All raw observations and original experimental
track names are retained. No training, inference, tracking or filtering occurs.

### Frame conventions

- Original `frame`, `time_s`, positions, scores and IDs remain unchanged.
- `demo_frame` is global within the 2,000-frame excerpt (0–1999).
- `chunk_id` and `chunk_frame` identify the correct chunk and local SLP/video frame.
- `chunk_instance_row` is local to the corresponding SLP; `demo_instance_row`
  indexes the combined selected table. Original source row ordinals are retained.
- `chunks.json` gives exact ranges and paths. No frames are missing or duplicated.

## Validation and preservation

`jobs/replace_crab_demo_2000.sh` runs a finite 4-CPU Slurm job. Existing extraction
controls and new tests cover chunk planning, exact 2,000-frame coverage, oversized
GOP refusal, reordered-frame rejection, index mapping, real synthetic video
packet/pixel identity (including a partial final GOP), overwrite refusal, and a
path-only Git commit that preserves unrelated staged work.

The real bundle is checked for:

1. Published source manifest/table/SLP hashes and original video stat pins.
2. Identical SHA-256 hashes for **every original/copied encoded video packet**.
3. **Pixel-identical decoded BGR frames for all 2,000 frames**, with no omissions.
4. Original codec/geometry/color metadata and frame-rate preservation.
5. Every SLP pose/score byte and assigned ID against the source dataframe.
6. Exact original dataframe columns and Parquet readback.
7. Complete chunk-local SLP frame coverage and relocated basename video loading.
8. Every payload below 100 MiB and every file hashed in the receipt.

Only after validation is the old bundle moved into ignored local
`archive/demo/crab_cam2_2_middle_300/`, with pre/post hashes. Original videos,
predictions and datasets are unchanged. Failed historical staging directories
are not published or overwritten. Desktop-GUI playback is not claimed tested.

## Git publication: local commit does not depend on network access

The job stages only this demo's replacement, its source/tests/docs, navigation,
and narrow `.gitignore` exceptions. It refuses pre-existing staged work or a
branch/HEAD change during the build. No `git add .`, force push, history rewrite,
branch/remote/credential changes, or upload of full source datasets/models.

If the compute node lacks Git author configuration, the job preserves the
starting branch commit's existing author identity through process-local Git
environment variables. It does not invent an identity or edit Git configuration.
Job 16796620 stopped at this identity precheck before tests or extraction; the
process-local fallback addresses that specific failure.

The job **commits locally before trying the network**, and verifies every demo
payload's Git blob against its working-tree file. The old demo must be absent
from the new commit's tree. This prevents a later push from uploading only
preparation code while silently omitting the data bundle.

Inspection job 16795763 confirmed that the compute node cannot resolve
`github.com`. That is separate from authentication and from data generation.
The job still prepares a complete local commit, then attempts one ordinary
push to `origin/my-code`. If networking blocks it, the user's usual terminal can
publish the ready commit with `git push origin my-code`. No automatic retries.

## Receipts (not live status claims)

- Job stream: `logs/demo_preparation/replace_2000_<job-id>.{out,err}`.
- Run receipts: `logs/demo_preparation/replace_<job-id>/`.
- `TESTS.json`, `tests.txt`: pre-extraction controls.
- `CHUNK_PLAN.json`: chosen source interval, stream metadata and chunk boundaries.
- Bundle `DEMO_COMPLETE.json`: validated local bundle only.
- `LOCAL_REPLACEMENT_COMPLETE.json`: old bundle archived and new bundle validated.
- `LOCAL_COMMIT_COMPLETE.json`: commit SHA and verification that payloads are in Git.
- `PUSH_RESULT.json`, `push.txt`: actual push result, distinct from local success.
- Final `COMPLETE.json`: local validation, commit and successful push.

## Why the first attempt did not appear on GitHub

Job **16760391 FAILED (exit 1)** after video encoding, before SLP/dataframe
extraction and before a commit. The assistant had tried to keep one long MP4
below 100 MiB by lowering its encoding bitrate, an extra requirement the user
had not requested. Its first-frame mean absolute pixel error was 16.2005,
failing the <10 check. A subsequent user push therefore could not include a
completed new demo. That re-encoding approach is superseded by this direct-copy
chunk implementation; its failed video is not used.

The stopped diagnostic job 16773602 failed before executing Python because its
submission working directory made a relative interpreter path invalid. Current
jobs use absolute interpreter/script paths and an explicit project working
directory. Neither failure changed original research files.

## Entry points

- `scripts/chunk_crab_demo.py`: planning, packet copying, data extraction, validation.
- `scripts/replace_crab_demo_2000.py`: tests, archive, scoped commit and one push.
- `scripts/prepare_crab_demo.py`: reused index-only SLP/dataframe helpers; the
  historical 300-frame encoder is not called by the new workflow.
- `tests/test_crab_demo_chunks.py` and `tests/test_crab_demo.py`.
- `docs/demo/README_2000.template.md`: generated bundle instructions.

Earlier 300-frame/archive history: [PREPARATION.md](PREPARATION.md).
