# Inspecting the male-crab SLEAP predictions

[Documentation index](README.md) · [Prediction provenance](PREDICTION_V007_BEST.md)

All code-formatted data paths below are relative to
`/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/`.

## Recommended: local SLEAP desktop GUI

Use the SLEAP desktop labeling/viewing application, commonly launched with
`sleap-label`, rather than assuming `sleap-io` or `sleap-nn` installs a GUI.
Install a GUI release supporting these SLP prediction files using the official
[SLEAP installation guidance](https://sleap.ai/) appropriate for your OS.
Exact local installation steps depend on OS and existing environment; this
project has not installed or tested a desktop GUI in the current session.
No model files or GPU inference are needed to inspect existing predictions.

Start with **one video and its merged raw predictions**:

| Video | Matching video file | Predictions (without experimental IDs) |
| --- | --- | --- |
| Cam 2_2 | `crab_vids/Crab_Cam_2_2.mp4` | `predictions/v007_best_body_first_v001/Crab_Cam_2_2/raw/Crab_Cam_2_2.predictions.slp` |
| Cam 2_3 | `crab_vids/Crab_Cam_2_3.mp4` | `predictions/v007_best_body_first_v001/Crab_Cam_2_3/raw/Crab_Cam_2_3.predictions.slp` |

1. Transfer the matching `.mp4` and `.slp` to your local computer, using your
   usual approved Della file-transfer method. Keep the filenames recognizable.
2. Open the `.slp` in the SLEAP GUI (File/Open, or `sleap-label <local-file.slp>`).
3. When the GUI cannot find the original Della video path, use its missing-video
   / relinking dialog to select the downloaded **matching full video**. The SLP
   stores predictions and video references, not a copy of the movie.
4. Scrub the timeline, zoom in and step through adjacent frames. Enable display
   of predicted instances/skeletons if your GUI settings hide them. These poses
   contain Shell, Shoulder, Elbow, Claw and Arm landmarks. Compare overlays to
   actual visible bodies rather than treating the overlay as ground truth.
5. Save any relinked or edited review project as a **new local copy**. Do not
   replace published SLP files on Della.

If you want to inspect the experimental track identities, open instead:

- `predictions/v007_best_body_first_v001/Crab_Cam_2_2/tracklets/Crab_Cam_2_2.tracked.slp`
- `predictions/v007_best_body_first_v001/Crab_Cam_2_3/tracklets/Crab_Cam_2_3.tracked.slp`

These contain the same pose measurements plus optional identities; they retain
unassigned detections too. IDs are short experimental tracklets, not validated
unique animals. Each video has over 100,000 tracklets, so the GUI's track list
may be unwieldy. Start with the raw/untracked file for a simpler pose review.
The `danlabels.v007.slp` training project is NOT the full-video prediction file.

### Size and performance

These are 7680 x 4320 videos with millions of detections. Movies are large and
merged SLPs are hundreds of MB; GUI loading, decoding and memory use may be
substantial. A local GUI is usually easier than forwarding an 8K GUI over SSH,
but full-file responsiveness has not been benchmarked here. If it is sluggish,
a short matched video/SLP review excerpt is a better next step. Do not simply
relink full-resolution coordinates to a resized video or a time-cropped clip:
spatial coordinates and original frame indices would no longer align unless
both video and labels are transformed consistently.

## Remote alternative

A site-supported **Slurm-backed interactive desktop** could avoid transferring
the original videos, if such an application is available in your Della portal.
Availability and GUI dependencies have not been verified in this session.
Use the site's supported interactive workflow; do not run playback, Python,
Jupyter kernels or rendering workloads directly on vis1/login nodes. Nothing
has been installed and no remote GUI/server has been launched by this task.

## Simplest viewing alternative: a short burned-in overlay MP4

A finite CPU Slurm render can produce a bounded excerpt with pose overlays that
plays in any ordinary video player. This avoids GUI setup but loses editable
labels and interactive node/track toggles. Full-video overlays have not been
rendered. Choose a video, start time and duration before requesting an excerpt;
scaling and original-frame mapping must be handled together.

Existing bounded static QC previews are already available, for example:
[Cam 2_2 adjacent-frame crop at frames 14400/14401](../datasets/v007_best_body_first_dataframe_v001/qc/Crab_Cam_2_2.frames_014400_014401.png).
Those shoreline-biased images are not full-scene or representative accuracy
validation.

## Count notebook

[In-frame detection count notebook](../analyses/crab_counts_v001/crab_counts_over_time.ipynb)
shows the full per-frame time series and nonoverlapping time-bin means for each
video. Its saved output can be viewed without executing a kernel. See
[count notebook notes](analyses/crab_counts_v001.md) for definitions and execution.
