# Original-prediction feasibility assessment

[Dataset notes](README.md) · [Documentation index](../../README.md)

This historical note is centralized under `docs/`. The v004 dataset and hashed
audit artifacts have since been archived, unchanged, under
`archive/v004_legacy/datasets/legacy_native_tracklets_v001/`. Dataset-relative
paths below refer to that archived root. See [archive notes](../../demo/PREPARATION.md).

**Subsequent user decisions:** body means the centered-model Shell position,
the fixed water polygon is accepted for initial use, and the explicitly labeled
Shell centroid proxy is approved. See `config/user_decisions_001.json` and
`config/user_decisions_002.json`. Pending-decision descriptions below are
historical and superseded by those receipts and [DATAFRAME.md](DATAFRAME.md).
The original hashed audit/QC artifacts are preserved.

## Disposition

**Both videos are structurally usable for conservative post-hoc tracklets. No
new inference is indicated by this audit.** Neither saved file contains usable
track identities, and the detector's centroid coordinates were not retained.
Tracklets and the requested dataframe have **not** yet been generated: body
semantics and permission for a centroid proxy remain unresolved. The static
nature of the water region was explicitly confirmed by the user; its polygon
boundary and point-membership convention have not yet been confirmed.

| Check | Crab_Cam_2_2 | Crab_Cam_2_3 |
|---|---:|---:|
| Original dimensions | 7680 × 4320 | 7680 × 4320 |
| Container FPS | 30000/1001 | 30000/1001 |
| Original frames, zero based | 0–28799 | 0–28799 |
| Readable 40-frame chunks | 720 | 720 |
| Saved detections | 4,459,295 | 5,184,899 |
| Missing / overlapping frame records | 0 / 0 | 0 / 0 |
| Explicit zero-detection frames | 0 | 0 |
| Merged/chunk raw frame payloads | Exact match | Exact match |
| Missing Shell positions | 1 | 0 |
| Out-of-image Shell positions | 0 | 0 |
| Missing Shoulder positions | 1,159,688 (~26.0%) | 1,270,589 (~24.5%) |
| Out-of-image Shoulder positions | 7 | 1 |
| Both nodes finite/inbounds, scores ≥0.2 | 3,299,599 | 3,914,309 |
| Within-frame Shell pairs <5 pixels | 1,767 | 2,420 |

The last two rows are diagnostics, **not final acceptance/rejection rules**.
Close pairs are candidate duplicates/ambiguities, not automatically proven
physical duplicates. Every source detection remains in its original SLP.
Missing frame records would have remained unknown rather than being counted
as empty; neither video has missing records in this audit.

## Serialized information and provenance

- Five pose landmarks and node scores are saved, with full-image pixel
  coordinates, plus instance and tracking-score fields. No per-observation
  detector centroid positions or crop transforms are available. Full-image
  positioning is supported by the source's crop-offset conversion, coordinate
  distributions and overlays, not an assumption that crop coordinates are global.
- Track catalogs are empty; serialized track indices are `-1`, and tracking
  scores are all nonfinite. There are no existing IDs to trust or restart.
- Keep the serialized instance score named `instance_score`. The inspected
  top-down conversion in the installed `predictors.py` passes `centroid_val`
  into that field, but a score is not a retained centroid position. The audit
  does not rename it to centroid confidence, nor assume it is a whole-pose mean.
- Source metadata records the original model paths, native scales and 192-pixel
  crops, with peak threshold 0.2. The inventory contains metadata and SHA-256
  hashes for every chunk and merged SLP, both raw videos, the old model/config
  artifacts and the audit/provenance code. Current model hashes are not proof
  of historical checkpoint hashes; those were not pinned in the saved metadata.
- Scores are not calibrated probabilities. Shell scores can exceed 1; Shoulder
  scores are typically much lower (median about 0.225–0.228 including zeros).
  A generic 0.5 threshold on both nodes would discard essentially all shoulders.
- Median nearest-Shell displacement between adjacent frames is about 0.11 and
  0.14 pixels, but the 99th percentile is about 230 pixels. This is not an
  identity measurement: disappearing/appearing detections can select unrelated
  neighbors. Do not use that tail to justify permissive matching or gap bridging.
- Python 3.13.13, sleap-io 0.9.2, sleap-nn 0.3.3, pandas 3.0.5 and pyarrow 25.0.1
  were available. **No Parquet dependency installation is needed.** Full versions
  are in `audit/summary.json`.

Frame counts and timing were checked against OpenCV and ffprobe container
metadata, with successful bounded decoding including the last frame. This was
not an exhaustive decode or full per-frame PTS scan. If exported, `time_s` should
be explicitly nominal `frame × 1001 / 30000`; frame remains the primary key.

## Bounded visual review

Generated and reviewed contact sheets for these explicitly selected pairs:

- `Crab_Cam_2_2`: 0/1, 39/40, 5026/5027 (largest detection count),
  14400/14401, 27794/27795 (closest Shell pair), 28798/28799.
- `Crab_Cam_2_3`: 0/1, 39/40, 14400/14401, 22344/22345 (largest detection
  count), 25892/25893 (closest Shell pair), 28798/28799.

Full-preview early/middle/late frames and first-video 39/40 and dense frames
were additionally examined individually. The closest-pair examples were
examined as 640×640 original-resolution crops for both consecutive frames.
Cyan marks Shell, yellow marks Shoulder; numeric crop labels are **source-frame
instance indices, not animal or tracklet IDs**.

Findings: the pose coordinates overlay crab-sized structures across the full
image rather than clustering in a 192-pixel crop. The scene contains dense
burrows, close neighbors, apparent duplicate predictions on the same crab and
missing Shoulder points. Examples in both the water channel and dry bank make
ambiguity handling important. At the reviewed chunk boundary there is no
obvious coordinate reset. The close-pair crops illustrate detections appearing
or disappearing over a single frame; they must not be forced into persistent
identities. These are selected examples, not a quantified accuracy evaluation.

## Fixed water ROI proposal

`config/water_regions.json` holds a **single fixed polygon per video, reused
unchanged in every frame**. After separate early/middle/late visual checks of
the banks and pole, the proposed coordinates are the same for both videos;
confirmation remains explicit for each video. No automated registration was
performed. There is no frame-adaptive water estimator.

The proposed polygon covers the winding visible channel and indents around the
exposed central bank. The wet-mud boundary, small bank pools and narrow shadowed
lower continuation are semantically uncertain. Please confirm/edit the orange
proposal previews in `qc/`, or supply an annotated screenshot. A 30-pixel
uncertainty band is also only a proposal. The suggested membership rule uses
the chosen Shell/body reference point, not any part of the crab overlapping
water. This reference point requires user agreement.

Until those decisions are resolved, water must be **NA**, never automatic land.
Once confirmed, membership is evaluated at each observation against the same
static polygon; a moving crab can change membership. Water does not change
inference, canonical tracking or observation retention.

## Minimal remaining decisions and work

1. Confirm whether **body = Shell landmark position**, rather than outline,
   size or another quantity not represented by the labels.
2. Since detector positions were not saved, approve **`centroid_source="shell_proxy"`**
   or explicitly choose to leave centroid coordinates missing. No silent
   duplication of Shell into detector-centroid columns; no limb averaging.
3. Confirm/edit the proposed static polygon and reference-point rule. Water
   annotation can remain nullable while the rest of the dataset is built.

With those semantics settled, implement the bounded conversion with deterministic
adjacent-frame one-to-one/ambiguity gating using trusted positions, no bridging
or re-identification, singleton retention, explicit rejection accounting and
video-isolated IDs. Freeze numerical parameters after this audit, run the
required synthetic association/ROI fixtures and table read-back checks, then
review real association examples and publish atomically. The absence of saved
IDs does not require new predictions. Missing Shoulder coordinates imply
fragmentation or an explicitly justified body-only rule, not invented points.

No tracklet/observation Parquet, derived tracked SLP or `DATASET_COMPLETE.json`
has been produced. `SCHEMA_PROPOSAL.md` is a design document, not a dataset.
Original files and the separate `v007_19f` run were left untouched.

## Jobs and reproducibility

- **15428610** — complete input audit, exit 0, 8m40s, about 1.6 GiB maximum RSS.
- **15428784** — five synthetic audit-primitive tests, all passed, exit 0.
- **15429096** — compact provenance assessment and fixed-ROI proposal rendering,
  completed successfully, exit 0. Polygon geometry and assessment JSON read-back
  passed. ROI previews were visually reviewed after rendering. This is not
  dataset completion; see `qc/REPORT_COMPLETE.json`.

Entry points are `jobs/{audit_legacy_dataset,test_legacy_audit,report_legacy_audit}.sh`.
All computation and decoding ran in finite CPU Slurm jobs. There is no requeue,
retry loop, new training or inference. `audit/AUDIT_COMPLETE.json` records the
inventory's SHA-256 at its staging path; the directory was then atomically
renamed to `audit/`. The inventory now lives at `audit/inventory.json` with the
same digest. Audit/report scripts intentionally refuse to overwrite published
outputs; a future rerun must use a new reviewed destination/version.
