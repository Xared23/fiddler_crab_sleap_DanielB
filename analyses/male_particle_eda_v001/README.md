# Male-only particle-system exploratory analysis

Canonical methods, execution status and report links:
[analysis notes](../../docs/analyses/male_particle_eda_v001.md).

This directory contains isolated analysis code and job-scoped outputs. It does
not modify the published dataset, models, predictions or tracker. All Python,
unit tests, analysis and PDF generation must run through the finite Slurm script
`run.sh`. A `run_<job-id>/COMPLETE.json` marks successful completion, not scientific
validation of identities.
