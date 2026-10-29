# Original-model dataset jobs

[Dataset notes](README.md) · [Documentation index](../../README.md)

This note is centralized in `docs/datasets/legacy_native_tracklets_v001/`.
The actual job logs remain under `logs/legacy_dataset/` at the project root.
Code-formatted script paths below are project-relative.

Finite CPU-only jobs for the separate legacy dataset task. No new-model training
or prediction scripts run here. Slurm stdout/stderr are retained by job ID.

| Entry point | Job | Result |
|---|---|---|
| `jobs/audit_legacy_dataset.sh` | 15428610 | Read-only source inventory/coverage/hash audit; completed, exit 0, 8m40s |
| `jobs/test_legacy_audit.sh` | 15428784 | Five synthetic audit tests passed; exit 0 |
| `jobs/report_legacy_audit.sh` | 15429096 | Compact assessment and static ROI proposal previews; exit 0 |
| `jobs/build_legacy_dataset.sh` | 15521792 | 21 synthetic tests, streaming conversion, full read-back and bounded QC PNGs; exit 0, 7m22s |
| `jobs/publish_legacy_dataset.sh 15521792` | 15522072 | Reverified hashes/counts/schema/review and published completion marker; exit 0, 7s |

All allocations were finite CPU Slurm jobs, without requeue or automatic follow-on
submission. The publication job was submitted only after bounded visual review.
No training/inference was launched and no job remains running for this task.

The build staging directory retains its original manifest and validation receipt
as provenance; tables/QC were atomically moved to their final locations.
Published consumers must require
[`DATASET_COMPLETE.json`](../../../datasets/legacy_native_tracklets_v001/DATASET_COMPLETE.json).
The scripts deliberately refuse to overwrite existing validated outputs; future
changes or reruns require a new reviewed version/destination.

See [DATAFRAME.md](DATAFRAME.md) for schema, loading, threshold rationale and
quality limitations, including the large number of preserved singletons.
