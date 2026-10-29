# Male fiddler-crab demo: 300 middle frames, v007 predictions

This self-contained sample uses **Crab_Cam_2_2 original frames 14250–14549
inclusive** (zero-based), centered in the 28,800-frame source video. It lasts
10.01 seconds at 30000/1001 fps. No model files or full original videos are
needed to explore this folder.

## Files

| File | Purpose |
|---|---|
| `Crab_Cam_2_2.frames_014250_014549.mp4` | Exactly 300 consecutive frames; native 7680 × 4320; H.264 demo copy |
| `Crab_Cam_2_2.frames_014250_014549.predictions.slp` | Every v007 pose in those frames, without identities |
| `Crab_Cam_2_2.frames_014250_014549.tracked.slp` | Identical poses plus the existing optional experimental tracklet IDs |
| `observations.parquet` | Exact selected rows/columns from the published full-video dataframe, plus demo-index columns |
| `frames.parquet` | Corresponding 300 frame-summary rows, plus demo-index columns |
| `frame_map.csv` | Local/original frame and time mapping; SHA-256 of every selected decoded original BGR frame |
| `DEMO_COMPLETE.json` | Counts, input provenance, output hashes, versions, encoding settings and validation receipt |
| `encoding.log` | Encoder diagnostics |

**Video pixels are lossy re-encoded** (CRF 18, 60-Mbit/s VBV ceiling) to make an
ordinary-Git-sized sample. There is no resize, crop, stabilization, rotation,
overlay, temporal interpolation or change to prediction coordinates/scores.
The movie is for demonstration, not a byte-identical scientific pixel archive.
All payload files are checked to be below 100 MiB individually.

## Frame and identity conventions

- The demo movie and **both SLPs** use **local frames 0–299**.
- In the dataframes, the original `frame`, `time_s`, detection/tracklet IDs,
  coordinates, acceptance flags, nullable values and provenance columns remain
  unchanged. Added `demo_frame` and `demo_time_s` align rows to the clip.
- `original frame = demo_frame + 14250`. No frames or detections are dropped;
  unassigned observations remain present with null tracklet IDs.
- `demo_instance_row` is the zero-based instance row within either demo SLP.
  Original `raw_instance_id` / `source_instance_row` still refer to the **full**
  source SLP. `instance_index` remains the within-frame ordinal.
- Original full-video tracklet **names** are retained, not retracked/renamed.
  Only the internal SLP track-catalog indices are compacted to used IDs.
  Tracks may start before or end after this excerpt. Do not interpret clip
  support as the full lifetime, or tracklet counts as unique animal counts.
- Original `source_*path` columns are historical project-relative provenance;
  those full-video sources are deliberately **not** included. For local work,
  use this folder's MP4/SLPs and `demo_frame` / `demo_instance_row` instead.
- Positions are native-image pixels (x right, y down). `centroid_source` is
  explicitly `shell_proxy`, not a retained centroid-detector output. `water`
  is fixed-ROI analysis metadata, not dynamic segmentation.

The source is `v007_best_body_first_v001`, using the frozen validation-best
v007 centroid/centered pair, exported as
`v007_best_body_first_dataframe_v001`. It is not the archived v004 model,
nor the later bounded threshold/grace-period experiments. Experimental IDs
are **not ground-truth validated**; missed/false/duplicate detections remain
possible. No neural inference, smoothing or tracking was rerun for this demo.

## Open locally

Keep the MP4 and SLP files side by side. Open either SLP in a compatible SLEAP
desktop GUI. It references the **local MP4 basename**, not a Della absolute path.
If your GUI requests relinking, select the MP4 in this folder. Start with the
untracked `.predictions.slp` for a simpler pose view.

For pandas, install/use pandas and a Parquet engine such as pyarrow in your own
normal environment; no project models or GPU are needed. From the project root:

```python
from pathlib import Path
import pandas as pd

root = Path('demo/crab_cam2_2_middle_300')
df = pd.read_parquet(root / 'observations.parquet', dtype_backend='numpy_nullable')
frames = pd.read_parquet(root / 'frames.parquet', dtype_backend='numpy_nullable')
local_frame = 150  # original frame 14400
poses = df.loc[df['demo_frame'].eq(local_frame)]
print(poses[['frame', 'demo_frame', 'tracklet_id', 'centroid_x', 'centroid_y', 'water']])
```

On Della, any execution or playback workload still needs a Slurm allocation.
Simply reading this documentation requires no computation. The finite build
entry point is `jobs/prepare_crab_demo.sh`; it refuses to overwrite an existing
published demo. Its synthetic tests precede extraction, and its validation
compares every pose/ID with the dataframe, decodes all 300 output frames and
opens relocated SLP/video copies with sleap-io. A desktop GUI was not launched.

## Git state

The user authorized narrowly scoped `.gitignore` exceptions for this bundle's
MP4, SLP and Parquet files. The v004 archive and incomplete demo builds remain
ignored. Generating this folder does **not stage, commit or push it**. The demo
scripts do not change Git branch/remote configuration; the user manages tracking
and publication to GitHub separately. See
`docs/demo/PREPARATION.md`; do not use a broad force-add of datasets or archives.
