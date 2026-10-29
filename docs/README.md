# Project documentation and notes

All canonical project README/Markdown notes live in this `docs/` directory.
The project-root and dataset-root README entry points are short navigation
pointers, not duplicate copies of the notes. Dependency documentation and hidden
agent/tool configuration are deliberately excluded from this reorganization.

## Current training and available models

- **[Paused training: model locations and exact resume procedure](RESUME_V007_TRAINING.md)**
  — centroid paused after 104 completed epochs plus a partial 105th, step 21,212;
  75/100-epoch snapshots verified; frozen latest/best model pairs available.
- [Project overview and layout](PROJECT.md)
- [75/100-epoch checkpoint mechanism and provenance](milestone_checkpoints.md)
- [Original retraining decisions and historical review](retraining_v007.md)
  — contains clearly marked superseded sections; not a live job-status report.

## New-model predictions and optional experimental tracking

- [1,000-frame continuation-threshold / five-missed-frame tests](prediction_grace_tests.md)
  — independent 0.2/0.1 births with 0.05 continuation; no interpolation,
  no fixed count, two final SLPs and a complete raw-candidate audit archive.
- [200-frame centroid-threshold comparison](prediction_threshold_tests.md)
  — timestamped 0.1/0.05 tests against existing 0.2 predictions; no retraining,
  no saved frame chunks, unchanged variable-count body-first tracking.
- [Viewing predictions in the SLEAP GUI or as overlays](VISUALIZING_PREDICTIONS.md)
  — matching video/SLP paths, local relinking, remote-workflow caveats and excerpt options.
- **[Frozen-best prediction run: separate raw poses and optional identities](PREDICTION_V007_BEST.md)**
  — both full videos; new output root; the raw predictions remain independently
  usable even if the body-first identities are discarded.
- **[Standardized v007 dataframe: schema, provenance, and pandas loading](datasets/v007_best_body_first_dataframe_v001/README.md)**
  — **validated and published**: 7,985,830 observations and 260,385 experimental
  tracklets; every source detection and existing ID retained. Includes explicit
  centroid-proxy coordinates, fixed-water annotation and source references.
- [Body-first tracking method, pilot results, and original review plan](BODY_FIRST_TRACKLETS.md)
  — numerical continuity checks passed; scientific identity accuracy remains
  unvalidated. The user explicitly authorized experimental identities.

## Preliminary male-only analyses

- [In-frame detection counts notebook](analyses/crab_counts_v001.md)
  — per-frame series, 10-second means, land/water and acceptance comparisons,
  with embedded figures and aggregation/data validations.
- [Particle-system exploratory analysis and technical report](analyses/male_particle_eda_v001.md)
  — isolated CPU workflow with known-answer tests, occupancy/pair statistics,
  displacement distributions and temporal correlations; experimental-identity caveats.

## Original-model dataset organization

- **[Completed legacy dataset: status and results](datasets/legacy_native_tracklets_v001/README.md)**
- [Dataframe schema, conservative tracking rules and pandas loading](datasets/legacy_native_tracklets_v001/DATAFRAME.md)
- [Feasibility assessment](datasets/legacy_native_tracklets_v001/ASSESSMENT.md)
- [Proposed dataframe schema](datasets/legacy_native_tracklets_v001/SCHEMA_PROPOSAL.md)
- [Legacy dataset job/log notes](datasets/legacy_native_tracklets_v001/JOBS.md)
- [Original agent handoff / project primer](legacy_dataset_agent_handoff.md)
- [Starting prompt for the dataset agent](legacy_dataset_agent_prompt.md)

Dataset artifacts remain under `datasets/legacy_native_tracklets_v001/` at the
project root. Moving the explanatory notes did not move or modify videos, SLPs,
model checkpoints, data/config manifests, ROI decisions, or hashed audit evidence.

## Organization convention

Put future project notes here, with topic/dataset subdirectories as needed.
Use relative Markdown links from each document's actual location. Clearly state
whether code-formatted artifact paths are project-relative or dataset-relative.
Keep dated/job-specific receipts separate from live status claims. Avoid copying
large README texts back into dataset/model/output directories; link here instead.

`organization.json` records the relocated files and their post-edit checksums,
and the finite Slurm job that verified the central documents' local links.
