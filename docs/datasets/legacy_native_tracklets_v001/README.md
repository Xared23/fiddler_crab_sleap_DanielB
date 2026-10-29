# Original-model predictions: legacy_native_tracklets_v001

[Documentation index](../../README.md)

> **ARCHIVED at the user's request.** This dataset is historical, not the active
> male-crab dataframe. Its payload now lives at
> `archive/v004_legacy/datasets/legacy_native_tracklets_v001/` relative to the
> project root. Old models, predictions, logs and pilot evidence were relocated
> under the same archive prefix. Historical paths/links below describe the
> pre-archive layout; prepend `archive/v004_legacy/` for moved artifacts. Only
> the shared `config/` metadata remains reachable via a compatibility link at
> the old dataset location. Use the
> [v007 dataframe](../v007_best_body_first_dataframe_v001/README.md) for new work;
> see [archive verification and demo notes](../../demo/PREPARATION.md).

This note is stored centrally in `docs/datasets/legacy_native_tracklets_v001/`.
Code-formatted dataset-relative paths below still refer to
`datasets/legacy_native_tracklets_v001/` under the project root; clickable links
point to the actual artifacts. No data or hashed audit evidence was relocated.

**Status: validated dataset published.** Build/test **15521792** and publication
**15522072** completed successfully. Body = centered-model Shell, the fixed
water polygon, and the explicit Shell centroid proxy are approved. The
[completion marker](../../../datasets/legacy_native_tracklets_v001/DATASET_COMPLETE.json)
records **9,635,190 accepted observations**, **9,004 rejected observations** and
**3,611,722 conservative tracklets**. Most tracklets are singletons (median length
one frame); they are not persistent animal identities.

**[Dataframe schema, tracking rules, validation and pandas loading](DATAFRAME.md)**
is the implementation guide. [ASSESSMENT.md](ASSESSMENT.md) retains the earlier
input-audit findings; its pending-decision text is historical.

This task concerns only `models/dan_centroid_native/` and
`models/dan_centered_native/` predictions for `Crab_Cam_2_2` and `Crab_Cam_2_3`.
It does not incorporate `v007_19f`, change the environment, or run inference.
Original videos, models, predictions and completion markers remain in place.

## Audit entry point

`jobs/audit_legacy_dataset.sh` runs `scripts/legacy_dataset/audit.py` in a finite
CPU Slurm job. **15428610 completed successfully** in 8m40s. Logs are in
`logs/legacy_dataset/audit_15428610.{out,err}`. Job **15428784** passed all five
synthetic audit checks; job **15429096** successfully summarized provenance and
rendered the fixed-ROI proposals. All jobs exited 0; none requeued.

### Results and previews

- [Short assessment](ASSESSMENT.md)
- [Machine-readable compact assessment](../../../datasets/legacy_native_tracklets_v001/qc/assessment.json)
- [Full source/hash inventory](../../../datasets/legacy_native_tracklets_v001/audit/inventory.json)
- [Audit summary and selected-frame evidence](../../../datasets/legacy_native_tracklets_v001/audit/summary.json)
- [Cam 2_2 water proposal](../../../datasets/legacy_native_tracklets_v001/qc/Crab_Cam_2_2.water_proposal.frame_014400.png)
- [Cam 2_3 water proposal](../../../datasets/legacy_native_tracklets_v001/qc/Crab_Cam_2_3.water_proposal.frame_014400.png)
- [Versioned, unconfirmed ROI polygons](../../../datasets/legacy_native_tracklets_v001/config/water_regions.json)
- [Proposed dataframe contract](SCHEMA_PROPOSAL.md)

The user confirmed the region is **the same in every frame**. Proposed ROI
coordinates therefore remain fixed; they are not redetected or changed per
frame. The user subsequently accepted the polygon for initial use. See
[the decision receipt](../../../datasets/legacy_native_tracklets_v001/config/user_decisions_001.json). The previously hashed
proposal/previews retain their original unconfirmed labels; the receipt records
later approval without rewriting that evidence. The subsequent
[proxy approval](../../../datasets/legacy_native_tracklets_v001/config/user_decisions_002.json)
uses Shell-point membership with boundary included. The earlier unconfirmed
30-pixel uncertainty-band proposal is not applied.

The audit writes to a job-specific staging directory, with an exclusive writer
lock, and publishes `audit/` by directory rename only after success. It refuses
to overwrite an existing `audit/`. A failed staging directory is evidence of an
incomplete attempt, not a complete audit or dataset. No automatic retries.

Checks include SHA-256 inventory; SLP HDF5 frame/instance/point ranges; chunk
SLEAP-IO read-back with video backends disabled; exact per-frame raw payload
comparison between chunks and merged files; skeleton/video references;
coverage, saved identities and fields; trusted-coordinate/score diagnostics;
and bounded local-video decoding. Missing records remain unknown, not empty.
Video frame count and FPS come from container metadata, supplemented by bounded
sample decoding, not an exhaustive decode or timestamp validation.

Previews cover early/middle/late consecutive pairs, the frame 39/40 chunk
boundary, a maximum-detection-count pair, and a minimum-Shell-separation pair.
These are selected review examples, not a random accuracy evaluation. Cyan is
Shell; yellow is Shoulder. Close-pair examples include full-resolution crops.
No automated diagnostic establishes physical identity or the semantic water
boundary.

## Decisions required before schema/annotation finalization

1. **Resolved:** body means the centered-instance model's `Shell` landmark
   position and score. The labels do not encode body outlines or size.
2. **Resolved:** detector centroid coordinates were not retained; the user
   approved `centroid_source="shell_proxy"`. The instance score retains its
   source semantics and is not automatically a centroid-confidence column.
3. **Resolved for initial use:** the accepted polygon is fixed in every frame;
   membership uses the Shell point, boundary included, with no uncertainty band.
   Unknown/invalid positions must never be automatically labeled land.

The input audit finished before these decisions. Conservative conversion,
21 synthetic tests, full saved-table read-back and bounded visual review have
now passed; the final `DATASET_COMPLETE.json` gate is present. Existing track
identities were not reused. Source hashes record files as found now; they do
not establish historical checkpoint hashes unless those were saved at inference
time. See [DATAFRAME.md](DATAFRAME.md) for loading examples and all limitations.
