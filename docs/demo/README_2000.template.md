# Male fiddler-crab demo: 2,000 consecutive frames, lossless video chunks

This bundle contains **Crab_Cam_2_2 original frames @START@–@LAST@ inclusive**
(zero-based), about **66.733 seconds at 30000/1001 fps**, in **@CHUNKS@ numbered
chunks**. It replaces the former 300-frame demo. The first frame is a source
keyframe nearest the middle-excerpt start, so every chunk is independently
playable without re-encoding.

**The video is copied, not re-encoded.** Original HEVC/H.265 compressed packets,
native 7680×4320 resolution, color metadata and decoded pixel values are retained.
There is no added compression, resize, crop, interpolation, smoothing, inference
or tracking. Each payload is below GitHub's ordinary 100-MiB limit; no Git LFS or
full-video/model download is needed. Playback needs HEVC support.

## Layout

Each `chunk_001/`, `chunk_002/`, … contains:

- `Crab_Cam_2_2.frames_<first>_<last>.mp4`: the original encoded video packets for
  that interval, in a new MP4 container.
- Same basename with `.predictions.slp`: every original v007 pose, without IDs.
- Same basename with `.tracked.slp`: identical poses and existing experimental IDs.
- `observations.parquet` and `frames.parquet`: matching dataframe rows.

Bundle-level files:

- `chunks.json`: chunk order, exact original/global frame ranges, paths and counts.
- `observations.parquet`: all selected observations, including their chunk IDs.
- `frames.parquet`: all 2,000 frame-summary rows in order.
- `frame_map.csv`: exact original/global/chunk-local frame mapping and decoded
  pixel/source packet hashes for every frame.
- `DEMO_COMPLETE.json`: source provenance, artifact hashes and validation receipt.

There are **exactly 2,000 frames total**, with no gaps or duplicates across chunks.
Chunk lengths can differ because cuts use keyframes and actual encoded sizes.

## Indices and unchanged research data

- Each MP4 and its two SLPs use **chunk-local** frame numbers starting at zero.
- `chunk_frame` indexes the chunk's movie and SLPs.
- `demo_frame` runs from 0 to 1999 across the whole selection.
- Original `frame`, `time_s`, coordinates, scores, detection/tracklet IDs,
  nullable values, acceptance flags and provenance columns are unchanged.
- `chunk_instance_row` indexes the instance row in that chunk's SLP.
  `demo_instance_row` indexes the combined selected observation table.
  Original `source_instance_row` / `raw_instance_id` still refer to the full source.
- `original frame = demo_frame + @START@`. See `chunks.json` for each chunk's start.
- All detections are retained, including water-region and unassigned poses.
- Full-video track names are unchanged, including across chunk boundaries;
  only internal per-SLP track-catalog indices are compacted.
- IDs are experimental, not validated unique animals. Chunk/selection boundaries
  do not indicate track birth/death. No new tracking is performed here.
- Coordinates are original image pixels, x right/y down. `centroid_source` is
  `shell_proxy`; `water` is fixed-ROI analysis metadata, not segmentation.

Sources: frozen-best `v007_best_body_first_v001` predictions and the published
`v007_best_body_first_dataframe_v001` dataframe. These are not the later
threshold/grace experiments or archived v004 outputs. Full-source paths in the
dataframe are provenance, not files included in this bundle.

## Open a chunk locally

Keep each chunk's MP4 and SLPs together. Open an SLP in a compatible SLEAP GUI;
it references the MP4 basename in its own folder. If asked to relink, choose
that chunk's MP4—not another chunk or the full original movie. Start with the
untracked `.predictions.slp` for a simpler pose view. The GUI/video backend must
support HEVC/H.265; the original codec has deliberately not been converted.

```python
from pathlib import Path
import pandas as pd

root = Path('demo/crab_cam2_2_middle_2000')
df = pd.read_parquet(root / 'observations.parquet', dtype_backend='numpy_nullable')
frames = pd.read_parquet(root / 'frames.parquet', dtype_backend='numpy_nullable')
poses = df.loc[df['demo_frame'].eq(1000)]
print(poses[['frame', 'demo_frame', 'chunk_id', 'chunk_frame', 'tracklet_id']])
# Open the SLP under poses['chunk_id'].iloc[0], at its chunk_frame value.
```

On Della, computation/playback still requires Slurm. The build verifies every
copied packet hash, every decoded frame pixel-for-pixel, every pose/score/ID,
all source dataframe columns, SLP frame coverage, and relocated video links.
A desktop GUI is not claimed tested.

## Publication

The old 300-frame demo is preserved locally under ignored
`archive/demo/crab_cam2_2_middle_300/`, rather than as a second active demo.
Original research files are never changed. Git history is not rewritten.

`DEMO_COMPLETE.json` means validated local data, not a GitHub upload. The build
job separately verifies that every payload is committed to `my-code`, then
attempts one normal push. If compute-node networking blocks GitHub, the local
commit is still ready for a normal `git push origin my-code` from the user's
network-enabled terminal. Receipts: `logs/demo_preparation/replace_<job-id>/`.
See `docs/demo/REPLACEMENT_2000.md` for the workflow and prior failure explanation.
