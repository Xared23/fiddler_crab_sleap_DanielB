# In-frame male-detection counts notebook

[Documentation index](../README.md) · [Viewing pose overlays](../VISUALIZING_PREDICTIONS.md)

Notebook: [analyses/crab_counts_v001/crab_counts_over_time.ipynb](../../analyses/crab_counts_v001/crab_counts_over_time.ipynb).

## Content

- Original per-frame counts for each of the two videos, with nonoverlapping
  10-second mean counts for readability; includes all frames, including zeros.
- Comparison of all finite in-image Shell detections, tracking-accepted totals,
  and accepted land/water subsets. Raw outputs are an accounting reference.
- Summary table, synthetic aggregation tests, input-checksum and count validations.
- Embedded plots plus standalone PNG/PDF exports and CSV summaries.

The main plotted count includes all tracking-accepted detections, including
singletons and water; it is not the number of track IDs over a time window.
Unassigned finite in-image detections appear in the less-filtered comparison.
Neither selection is ground-truth animal abundance. No inference, retracking,
video processing or source dataset modification is performed.

This lightweight notebook reuses the validated full-frame CSV tables from
`analyses/male_particle_eda_v001/run_15600144/tables/`. It verifies the original
analysis artifact hashes and frame/time/count consistency before plotting.
Aggregation tests cover known bin means, bin boundaries, zero values, the
partial final bin, weighted mean conservation, constants and invalid widths.
A deliberately clipped-curve control verifies the plot-bounds guard; every
saved curve must lie within its displayed y-limits.

## Execution

`analyses/crab_counts_v001/run.sh` is a finite 1-CPU, 4-GB, 5-minute Slurm job.
`execute_notebook.py` executes with the project Python through nbclient, stops
on errors, checks embedded figures and saved PNGs, and retains input/executed
notebook copies and a completion receipt under `run_<job-id>/`. The convenient
notebook at the path above receives the executed outputs. No recurring job or
Jupyter server is installed. Any later execution on Della needs a Slurm-backed
kernel; simply viewing the saved output does not require kernel execution.

## Completed and reviewed

Final job **15635974** completed successfully (exit 0:0) in **17 seconds**, one
CPU, peak RSS 284,096 KiB (~277 MiB). All five code cells executed, all synthetic
and real-data checks passed, and both plot outputs are embedded in the notebook.
The finite kernel uses same-node IPC and is shut down after execution.

- [Main count plot](../../analyses/crab_counts_v001/run_15635974/figures/in_frame_crab_counts.png)
- [Filtering/land/water comparison](../../analyses/crab_counts_v001/run_15635974/figures/count_selection_comparison.png)
- [Summary CSV](../../analyses/crab_counts_v001/run_15635974/figures/count_summary.csv)
- [Completion and artifact hashes](../../analyses/crab_counts_v001/run_15635974/COMPLETE.json)

Both final PNGs were opened and visually reviewed. All original data are retained;
accepted per-frame counts span 93–161 in Cam 2_2 and 114–193 in Cam 2_3. Mean
accepted totals are 127.152 and 149.135; means of the less-filtered finite-in-image
selection are 127.511 and 149.643.

Initial job **15635747** passed the numerical checks but visual review caught
shared-axis limits clipping the higher Cam 2_3 counts. Its job-scoped output is
retained as superseded evidence. The final version sets a common upper bound
from BOTH videos, includes a clipping regression test, and retains the same
numerical count tables. Use the final notebook and `run_15635974` plots, not the
initial figures. No additional jobs are running as part of this request.
