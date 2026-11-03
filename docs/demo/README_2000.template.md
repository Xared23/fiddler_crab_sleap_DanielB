# Male fiddler-crab demo: 2,000 middle frames, v007 predictions

This self-contained sample contains **Crab_Cam_2_2 original frames 13400–15399
inclusive** (zero-based), centered in the 28,800-frame source video. It lasts
**66.733 seconds at 30000/1001 fps**. It replaces the former 300-frame demo.
No models or full original videos are needed to explore this folder.

## Files

| File | Purpose |
|---|---|
| `Crab_Cam_2_2.frames_013400_015399.mp4` | Exactly 2,000 consecutive frames; native 7680 × 4320; H.264 demo copy |
| `Crab_Cam_2_2.frames_013400_015399.predictions.slp` | Every v007 pose in those frames, without identities |
| `Crab_Cam_2_2.frames_013400_015399.tracked.slp` | Identical poses plus the existing optional experimental tracklet IDs |
| `observations.parquet` | Exact selected published dataframe rows, plus demo-index columns |
| `frames.parquet` | All 2,000 frame-summary rows, plus demo-index columns |
| `frame_map.csv` | Local/original frame/time mapping and hashes of decoded original BGR frames |
| `DEMO_COMPLETE.json` | Counts, source/output hashes, encoding settings and validation receipt |
| `encoding.log` | Encoder diagnostics |

**Video pixels are lossy re-encoded**, using H.264 CRF 18 constrained by a
10-Mbit/s VBV ceiling/buffer. This is stronger compression than the old short
demo, to keep the longer MP4 below ordinary GitHub's 100-MiB per-file limit.
There is no resize, crop, stabilization, rotation, overlay, dropped frame or
interpolation. Prediction coordinates and scores are unchanged. This is a
viewing/demo copy, not a pixel-identical scientific archive; small details may
lose fidelity. Each payload's size is checked before publication.

## Frame and identity conventions

- The movie and **both SLPs use local frames 0–1999**.
- `original frame = demo_frame + 13400`.
- Dataframe source columns (`frame`, `time_s`, detection/tracklet IDs,
  coordinates, acceptance flags, nullable values and provenance) are unchanged.
  Added `demo_frame` and `demo_time_s` align rows to the clip.
- `demo_instance_row` is the zero-based instance row within either demo SLP.
  Original `raw_instance_id` / `source_instance_row` reference the **full source
  SLP**; `instance_index` is still the within-frame ordinal.
- All detections are retained, including water-region and unassigned poses.
  Unassigned observations retain null tracklet IDs.
- Original full-video tracklet names are retained. Only the internal SLP track
  catalog indices are compacted; no tracking or inference is rerun. Tracks can
  extend beyond this excerpt. Tracklets are not validated unique animal IDs.
- Original `source_*path` columns are provenance, not files included here.
  Use the local MP4/SLPs and `demo_frame` / `demo_instance_row` for this sample.
- Coordinates are native-image pixels, x right and y down. `centroid_source`
  is `shell_proxy`, not a retained centroid-detector output. `water` is fixed-ROI
  analysis metadata, not dynamic segmentation.

The source is the frozen-best **v007_best_body_first_v001** prediction run and
**v007_best_body_first_dataframe_v001** dataframe. This is not archived v004,
nor the later threshold/grace experiments. Identities are experimental;
false/missed/duplicate detections remain possible.

## Open locally

Keep the MP4 and SLPs side by side. Open either SLP in a compatible SLEAP desktop
GUI; its video reference is the local MP4 basename, not a Della absolute path.
If prompted, relink to the MP4 in this folder. Start with `.predictions.slp`
for a view without track identities.

For pandas, use pandas and pyarrow in your local environment:

```python
from pathlib import Path
import pandas as pd

root = Path('demo/crab_cam2_2_middle_2000')
df = pd.read_parquet(root / 'observations.parquet', dtype_backend='numpy_nullable')
frames = pd.read_parquet(root / 'frames.parquet', dtype_backend='numpy_nullable')
local_frame = 1000  # original frame 14400
poses = df.loc[df['demo_frame'].eq(local_frame)]
print(poses[['frame', 'demo_frame', 'tracklet_id', 'centroid_x', 'centroid_y', 'water']])
```

On Della, computation/playback still requires a Slurm allocation. The finite
replacement entry point is `jobs/replace_crab_demo_2000.sh`. Controls precede
extraction; validation decodes all 2,000 output frames, compares every pose/ID
and dataframe value with the source, and opens relocated SLP/video copies.
Desktop-GUI playback has not been tested by the build job.

## Repository replacement

The old 300-frame bundle is preserved locally under the ignored
`archive/demo/crab_cam2_2_middle_300/`, not kept as a second active demo.
Only the new demo's MP4, two SLPs and two Parquet files are exempted from the
repository's payload ignores. Original videos, models and full datasets remain
excluded. The old bundle remains in Git history; history is not rewritten.

A successful local `DEMO_COMPLETE.json` verifies the data bundle, **not a GitHub
push**. The replacement job records its separate commit/push result under
`logs/demo_preparation/replace_<job-id>/`. It only attempts a normal push to
`origin/my-code`, refuses unrelated staged changes or divergent remote history,
and never force-pushes or changes branches/remotes/credentials.
See `docs/demo/REPLACEMENT_2000.md` for the repository workflow.
