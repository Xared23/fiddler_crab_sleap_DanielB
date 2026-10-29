# Dataframe contract — proposal, not a published dataset

[Dataset notes](README.md) · [Documentation index](../../README.md)

The implementation must follow the [agent handoff](../../legacy_dataset_agent_handoff.md). This
proposal deliberately leaves unresolved semantics explicit; it does not grant
approval for a Shell-based centroid proxy or a water ROI.

## Observation table

One row per accepted source detection, never one row per landmark. All original
landmarks remain in the unchanged source SLPs and are linked by source path,
frame and instance index; an optional raw-points table may make those easier to
query. Confidence scores are measurements, not calibrated probabilities.

| Column | Pandas dtype | Meaning |
|---|---|---|
| `video_id` | `string` | Original-video identity, not camera-global animal identity |
| `detection_id` | `string` | Dataset/video/frame/source-instance scoped key |
| `tracklet_id` | `string` | Dataset/video scoped consecutive-observation sequence |
| `frame` | `int64` | Zero-based original-video index |
| `time_s` | `Float64` | Optional nominal `frame / verified container FPS`, explicitly not measured timestamps |
| `centroid_x`, `centroid_y` | `Float64` | Detector coordinates if actually saved; approved proxy only if explicitly labeled |
| `centroid_source` | `string` | `detector`, explicitly approved `shell_proxy`, or `not_retained` |
| `body_x`, `body_y`, `body_score` | `Float64` | Pending confirmation that body means Shell position |
| `shoulder_x`, `shoulder_y`, `shoulder_score` | `Float64` | Shoulder landmark and original node score |
| `instance_score` | `Float64` | Original serialized instance score; no silent relabeling |
| `source_tracking_score` | `Float64` | Serialized value, not evidence of valid identities |
| `water` | `boolean` | Nullable per-observation membership: True / False / NA |
| `source_path` | `string` | Relative canonical chunk path, pinned by SHA-256 |
| `source_instance_index` | `int64` | Instance's index within the source frame |
| `source_track` | `string` | Original track label if present, never reused as authoritative identity |

Coordinates are original-resolution pixels, x right and y down. Missing
coordinates/scores remain missing, not zeros. Detector and pose coordinates
must not be conflated. Non-body/shoulder landmarks are untrusted for association.
A single Shell/Shoulder pair does not measure a body outline or guarantee heading.

## Association contract

Use only adjacent original frames and confirmed trusted position features.
Carry the preceding frame across chunk boundaries, but reset between videos.
Require coordinate validity, documented per-node confidence, displacement gates,
one-to-one matching and explicit competitor/ambiguity checks. Ambiguous links
are broken rather than assigned; reliable observations can still be singletons.
Do not use a whole-pose mean score to reject a good Shell/Shoulder solely because
limbs are poor. Do not bridge missing records, recorded empty frames, rejected
observations, occlusions or later reappearances. Do not infer lasting identity.

Numerical parameters will be fixed only after the input audit. There is no
parameter sweep in this task. Rejected detections need source keys and explicit
reasons, so accepted + rejected reconciles to the pinned source observations.

## ROI and filtering

Each video needs original-resolution polygon vertices, source frame, dimensions,
confirmation status and an explicit membership reference point. A recommended
rule is point inclusion with boundary included; an uncertainty band, if used,
is NA. Missing/unconfirmed ROI or invalid reference position always yields NA.
Water is labeled per observation, never per tracklet, and does not affect
canonical tracking or retain/drop decisions.

After a validated table exists, a pandas land-only analysis should use an
explicit NA policy, e.g. `df.loc[df['water'].eq(False).fillna(False)]`. This
excludes unknown water status from that analysis without deleting canonical
rows or inventing continuity across the gaps created by filtering.

## Publication conditions

Use an exclusive writer lock, a new staging directory, Parquet read-back and
atomic publication. Preserve rejected observations and raw-point provenance.
Verify count reconciliation, key uniqueness, consecutive frames per tracklet,
video isolation, deterministic IDs and dtypes. Synthetic fixtures must cover
empty/missing frames, invalid/low-confidence points, ambiguous collisions,
chunk boundaries and ROI inside/outside/boundary/unknown behavior. Review bounded
real association examples and length/rejection summaries; structural tests alone
do not establish scientific accuracy. Do not write a dataset completion marker
while required schema decisions or conversion/validation are unresolved.
