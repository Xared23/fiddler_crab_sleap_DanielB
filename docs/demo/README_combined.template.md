# Fiddler-crab demo: males, females, and fixed burrows in 2,000 frames

This bundle covers **Crab_Cam_2_2 original frames 13410–15409 inclusive**
(zero-based), in the **same 10 lossless video chunks** as the original male demo.
Exactly 2,000 consecutive frames, native 7680×4320, 30000/1001 fps. Existing
videos and male-only SLPs are byte-identical: no video re-encoding, resizing,
smoothing, new neural inference, or male retracking.

## See all three layers together in SLEAP

Open this file for the first chunk:

**`chunk_001/Crab_Cam_2_2.frames_013410_013619.combined.tracked.slp`**

Each subsequent chunk has its own matching **`.combined.tracked.slp`**. These
contain every existing male prediction/ID, every female prediction (including
unassigned detections), and all **282 fixed burrow markers on every frame**.
Keep the MP4 and SLP together. If asked to relink, choose the same chunk's MP4,
not the full movie or another chunk. Playback requires HEVC/H.265 support.
Enable both predictions and user instances in the GUI; burrows are stored as
static-reference user-type instances, while crabs retain predicted-instance type.
Turn on node labels or node-based coloring to distinguish `FBody` and `Burrow`
from male `Shell`/limb nodes. Track coloring alone is not sex coloring.

For compatibility with single-skeleton GUI workflows, the combined file has a
**display-only seven-node union skeleton**: `Shell`, `Shoulder`, `Elbow`, `Claw`,
`Arm`, `FBody`, `Burrow`. It retains only the original male edges. Each instance
has coordinates only for its own original landmarks; every non-applicable node
is **NaN and invisible**. No female claws/shoulders or cross-entity edges are
invented. **Do not use these combined display files for model training or treat
the propagated burrows as independent manual labels on 2,000 frames.** Native
male, female, and burrow skeletons are retained in separate files.

SLP data readback and portable video loading were validated on Della, including
all saved instance coordinates/scores/visibility/IDs. **A desktop GUI session has
not been tested**; exact GUI behavior still depends on your SLEAP/video backend.

## Files

Each `chunk_001/` through `chunk_010/` contains the same original MP4 plus:

| SLP suffix | Contents |
|---|---|
| `.predictions.slp` | Original male raw poses, unchanged |
| `.tracked.slp` | Original male poses/experimental full-video IDs, unchanged |
| `.female.predictions.slp` | Matching female `FBody` detections, no IDs |
| `.female.tracked.slp` | Same female detections, demo-only experimental tracklets |
| `.burrows.slp` | Fixed `Burrow` positions repeated at each chunk-local frame |
| **`.combined.tracked.slp`** | **Males + females + fixed burrows together** |

The chunk's `observations.parquet` now contains **male and female predictions**;
`frames.parquet` contains their frame summaries. Root files:

- `observations.parquet`: all male/female observations across the 2,000 frames.
- `frames.parquet`: exactly 2,000 frame records, with explicit male/female/total
  crab counts and a static-burrow count.
- `burrows.parquet`: one row per fixed burrow (282 rows), with its reference
  frame/coordinates/provenance and applicable original-frame interval. This is
  intentionally a separate static table, **not 564,000 independent detections**.
- `female_tracklets.parquet`: demo-only female track starts, ends and lengths.
- **`burrows/BurrowLabels3.slp`**: byte-identical **copy**, not a move, of the
  uploaded labels, retaining both annotated frames (563 and 948) and the empty
  frame record. It retains its original video reference; use chunk `.burrows.slp`
  or `.combined.tracked.slp` to inspect positions over demo movies.
- `chunks.json`: original exact chunk/frame mapping, plus the new SLP paths.
- `frame_map.csv`: unchanged original/global/chunk-local mapping and video hashes.
- `DEMO_COMPLETE.json`, `AUGMENTATION_COMPLETE.json`: current validation receipts.
- `provenance/MALE_*`: historical male-only manifest, instructions and receipt.
  Historical hashes describe the old male-only tables, not the augmented tables.

Every payload stays below GitHub's ordinary 100-MiB per-file limit. Full research
videos, models and prediction runs are not required to use this bundle. This
update is local; a completion receipt does not imply a Git commit or push.

## Dataframe use and indexing

```python
from pathlib import Path
import pandas as pd

root = Path('demo/crab_cam2_2_middle_2000')
df = pd.read_parquet(root / 'observations.parquet', dtype_backend='numpy_nullable')
frames = pd.read_parquet(root / 'frames.parquet', dtype_backend='numpy_nullable')
burrows = pd.read_parquet(root / 'burrows.parquet', dtype_backend='numpy_nullable')

males = df.loc[df['entity_type'].eq('male')]
females = df.loc[df['entity_type'].eq('female')]
poses = df.loc[df['demo_frame'].eq(1000)]
print(poses[['entity_type', 'frame', 'chunk_id', 'chunk_frame', 'tracklet_id']])
print(burrows[['burrow_id', 'burrow_x', 'burrow_y', 'source_label_frame']])
# Every burrow applies throughout the interval in valid_source_frame_*.
# Open poses['combined_slp'].iloc[0] at poses['chunk_frame'].iloc[0].
```

- `entity_type` / `sex` distinguish **model classes** `male` and `female`, not
  independently confirmed biological sex. No cross-model deduplication, matching
  or arbitration is performed: an animal could appear in both model outputs.
- `body_x`, `body_y`, `body_score` are original `Shell` for males, `FBody` for
  females. `centroid_source` identifies the proxy (`shell_proxy` / `FBody_proxy`),
  not recovered centroid-detector crop centers. Female raw-node columns are
  `fbody_x`, `fbody_y`, `fbody_score`, `fbody_visible`. Male-only anatomy is null
  in female rows, never filled with `FBody` values.
- Original male columns and all male measurements/IDs/flags/provenance retain
  their values. Filter `entity_type == 'male'` for the old male table.
- `frame`, `time_s` refer to the full source video. `demo_frame` is 0–1999;
  `chunk_frame` is local to the chunk SLP/movie. **Original frame = demo_frame +
  13410.** Coordinates are unscaled image pixels, x right and y down.
- `layer_slp` and `chunk_instance_row` locate the native male/female tracked
  file's flattened instance row. `combined_slp`, `combined_instance_index` and
  `combined_chunk_instance_row` locate the combined file (within-frame index and
  flattened row, respectively). `source_instance_row` retains the full raw SLP
  row. Keys include model/dataset scope; do not join by `instance_index` alone.
- New `observation_row` indexes the augmented root table. Historical
  `demo_instance_row` remains the original **male-only** ordinal and is null for
  females. Original `frames.source_count` / `assigned_count` remain male-only;
  use `male_source_count`, `female_source_count`, `female_assigned_count`,
  `all_crab_source_count` and `static_burrow_count` for explicit augmented counts.
- Female `source_chunk_path`, `source_chunk_instance_row`, and
  `source_observations_path` are null: these rows are sliced directly from the
  merged female SLP, not a pre-existing male-style observation export.

## Female identities and static-burrow assumptions

The completed `female_v002` full-video output had **no track IDs**. This demo
adds experimental **body-first adjacent-frame tracklets** using `FBody`, the
existing male body-first implementation and its predeclared distance/score/
ambiguity settings. Shoulder cues are disabled because females have no Shoulder
landmark. No parameter sweep, inference, gap filling, re-identification,
fixed-animal count, or pose modification. State continues across chunk boundaries
but starts at the demo's first frame. IDs begin `female_demo_body_first_v001/`;
they are **not** validated unique animals or full-video female tracks. Rejected
low-score/duplicate detections remain in the SLP/dataframe with null IDs. Male
full-video track names are unchanged. These are SLEAP-format tracks generated by
the project's body-first association, not a new run of SLEAP's built-in tracker.

The original burrow upload contains **two slightly different 282-point maps**,
not a certified 564-burrow catalog. The fixed demo map uses the **first annotated
frame, 563**, preserving its coordinates and source instance order. Frame 948 is
retained in the copied source; maps are **not unioned, averaged, or cross-frame
matched**. Stable names `burrow_static/f000563/b0000`, etc., identify reference
points, not validated cross-frame identities. Positions are assumed constant,
as requested. Transfer from the label video `clip_cam2_8k.mp4` to the full demo
assumes the same fixed camera coordinates; **no independent image registration
or validation of all burrows at the demo time was performed**.

The original male-only bundle is preserved in the ignored local `archive/demo/`
location recorded in `DEMO_COMPLETE.json`. Uploaded source labels, full-video
predictions, models, and source research videos are unchanged.
