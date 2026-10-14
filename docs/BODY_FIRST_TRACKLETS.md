# Body-first tracklets: method, bounded pilot, and production gate

[Documentation index](README.md) · [Frozen best models](RESUME_V007_TRAINING.md)

## Request and current stage

**Latest authorization supersedes the original scientific-review launch gate:**
the user explicitly requested both full-video runs with the frozen best centroid
and centered models and this experimental tracker. Identities may be discarded
later; raw pose predictions must remain independently usable. The user did NOT
ask us to certify tracking accuracy, change tracking parameters, bridge gaps, or
resume training.

See **[New-model prediction run and separate raw/tracklet outputs](PREDICTION_V007_BEST.md)**
for the current execution plan and submission receipt. Original predictions,
published legacy data, frozen models, and paused training stay untouched.

Pilot **15525082** completed: 20 synthetic tests and both videos' numerical gates
passed on 1,080 frames. The evidence-only rendering correction **15525273** includes
the predecessor of each selected link; it replayed the same identities rather
than changing the method. The image review found useful continuity gains but also
remaining missed/duplicate-detection problems in the old predictions. It has NOT
established a ground-truth ID-switch rate or scientific validation of identities.

The original conditional pilot/review protocol below is retained as historical
context and a guide for future scientific validation. It is no longer a blocker
to the user's explicitly authorized raw predictions plus optional identities.

## Why change the previous method?

The published legacy table passed structural/read-back checks but approximately
90% of its tracklets were singletons. The old association rule required usable
Shoulder at both endpoints and exactly one body candidate in a 20-pixel
neighborhood. Missing Shoulder or a second nearby candidate could terminate an
otherwise clear body trajectory. Passing those software rules did not establish
useful identity continuity or ground-truth tracking accuracy.

This version changes association, not the detector, raw observations, or water
annotation. It does not infer long-lived animal identity through occlusion.

## Exact proposed method: `body_first_v001`

Implementation: `scripts/body_first_tracklets.py`.
Frozen candidate parameters: `configs/tracking/body_first_v001.yaml`.
The published baseline `scripts/legacy_dataset/tracklets.py` is imported for its
Frame and validity primitives but is NOT modified.

1. **Body acceptance stays the same:** finite, in-image, visible Shell with
   score >=0.5. Body pairs within 5 pixels remain duplicate candidates and both
   are rejected rather than choosing a potentially incorrect winner. Rejected
   detections are retained in source data/accounting, not erased.
2. **Shoulder is optional:** score >=0.2, in-image/visible, and body–shoulder
   distance 10–65 pixels determine whether Shoulder contributes to association.
   An unavailable/unusable Shoulder does not prevent an otherwise convincing
   body match. Its missing coordinates stay missing.
3. **Adjacent-frame candidate gates:** measured Shell displacement <=12 pixels;
   error relative to the short-history predicted Shell position <=18 pixels.
   All distances use the original 7680x4320 image coordinates.
4. **Association cost:**

   ```text
   cost = 0.7 * measured_body_displacement
        + 0.3 * error_from_predicted_body_position
        + optional_shoulder_penalty

   optional_shoulder_penalty = 0.15 * min(body_to_shoulder_vector_change, 12)
   ```

   The Shoulder term is zero unless Shoulder is usable at both endpoints. Its
   capped contribution is supporting evidence, not an obligatory pose match.
   Elbow, Claw and Arm are never association features.
5. **One-to-one assignment with unmatched options:** Hungarian matching includes
   explicit birth/death choices (cost 7 each). Real matches above cost 12 are
   disallowed. The algorithm is never required to attach every detection.
6. **Ambiguity rejection after assignment:** a selected match must beat every
   other feasible candidate at BOTH endpoints by at least 2 cost units. Even
   rejected/low-confidence finite in-image bodies count as alternatives. Their
   mere presence does not veto a much clearer match; truly similar alternatives
   do. Rejected ambiguous assignments are not greedily reassigned afterward.
   These margins are deterministic heuristics, not calibrated probabilities.
7. **Bounded motion history:** each matched displacement updates velocity with
   weight 0.5, capped at 6 pixels/frame. The next prediction is last measured
   Shell plus that velocity. A new track starts at zero velocity. Raw measured
   positions are not smoothed or replaced in the output.
8. **No gaps/re-ID:** a missing or empty frame, rejected/unmatched body, or
   unresolved association terminates the old ID. Reappearance gets a new ID.
   Motion prediction is only a matching cue for an actually detected next-frame
   crab; it does not create rows, bridge occlusion, or extrapolate trajectories.
9. **State spans chunks, not videos:** the same tracker instance must process
   every frame chronologically across 40-frame inference chunks. An ID cannot
   cross videos. Reliable singleton observations remain valid observations.
10. **Water is metadata only:** it never changes acceptance or association.
    A track can cross the fixed water boundary without changing identity.

The local margin checks deliberately remain cautious. They can still fragment
real close interactions. This is not a claim that Hungarian matching or motion
prediction guarantees correct identities.

## Predeclared bounded pilot

Use ONLY existing old-model predictions. No neural inference is needed here.
Compare baseline and candidate on identical observations from six 180-frame
clips (1,080 total frames, approximately 36 seconds across both videos):

| Video | Frames, inclusive | Reason |
| --- | --- | --- |
| Cam 2_2 | 0–179 | Early scene, multiple chunk boundaries |
| Cam 2_2 | 4960–5139 | Audited high-density neighborhood |
| Cam 2_2 | 27760–27939 | Audited closest-pair neighborhood |
| Cam 2_3 | 0–179 | Early scene, multiple chunk boundaries |
| Cam 2_3 | 22280–22459 | Audited high-density neighborhood |
| Cam 2_3 | 25840–26019 | Audited closest-pair neighborhood |

This is one fixed parameter set, NOT an autonomous tuning sweep. IDs reset at
clip starts for BOTH methods; clip boundaries truncate lifetime estimates.
Selected chunks are rehashed against the published legacy manifest. Video
size/mtime are checked against that manifest; their old full-video hashes are
provenance, not a claim that this pilot rehashed all raw-video bytes.

Entry point: `jobs/validate_body_first_tracklets.sh`.
Evidence: `checks/body_first_v001/pilot_<JOB_ID>/`.

### Automated checks

- Synthetic cases for missing Shoulder, shuffled order, distinguishable nearby
  bodies, equal alternatives, low-score competitors, forced far matches,
  missing/empty frames, chunk boundaries, duplicates, invalid bodies, unusable
  Shoulder geometry, velocity cap, unchanged raw measurements, deterministic
  IDs, video isolation, chronological order and births/deaths.
- Identical accepted observations in both methods, no ID collisions, no gap
  links, and complete count reconciliation on each real clip.
- Exact Parquet read-back of the comparison observations.
- Compare singleton-observation fraction, length quantiles, and fraction of
  observations in tracks lasting at least 30 / 150 frames (roughly 1 / 5 sec).

**Necessary numerical gates, separately for each video:**

- At least 50% reduction in observations relegated to singleton tracks.
- At least a 10 percentage-point increase in observation coverage by tracks of
  >=30 frames, with unchanged accepted-observation counts.

These criteria measure continuity only. Longer wrong tracks must NOT pass the
scientific review merely because these numbers improved.

### Required image/identity review

For each clip, up to three deliberately challenging focal examples are selected:
newly enabled body-only/previously-vetoed links, water, and motion/close-neighbor
cases. Each example has 18 CONSECUTIVE full-resolution crops, presented both
without overlays and with baseline/new IDs and landmarks. The raw montage is
for checking physical bodies rather than trusting the proposed IDs.

Review whether the focal identity follows the same visible crab, whether an ID
jumps to a neighbor/background, whether visible disappearance ends continuity,
and whether obvious unnecessary breaks remain. Record specific example/frame
references and uncertain cases. This limited diagnostic review is not a random
accuracy estimate or exhaustive ground-truth labeling. Do not report an ID-switch
rate for unannotated/unreviewed data.

Under the ORIGINAL conditional launch plan, a clear identity error, unresolved
critical case, failed invariant, or poor numerical improvement meant HOLD. The
latest user authorization permits experimental identities without that scientific
approval, but failed software/data-integrity invariants still block execution. A correction or expanded test is a new, explicit decision, not an
automatic sweep or endless retry loop. Preserve the published baseline and every
pilot artifact regardless of the outcome.

## If the pilot is acceptable: new-model prediction plan

- Reuse the project's existing prediction entry point/machinery with necessary
  explicit model/output/tracking changes; do not silently run the legacy model
  defaults or an unrelated parallel pipeline.
- Model directories:
  `models/v007_19f/pauses/step_00021212/centroid_best/` and
  `models/v007_19f/pauses/step_00021212/centered_best/`.
- Inputs: `crab_vids/Crab_Cam_2_2.mp4` and `crab_vids/Crab_Cam_2_3.mp4`.
- First perform a finite new-model compatibility/small-output check: the old-data
  pilot does not by itself verify inference loading, serialization, or behavior
  of new detections. Do not change training or calibration to make a test pass.
- Use a new, versioned output root; preserve historical SLPs and the published
  dataset. Keep both raw predictions and derived tracklet IDs/provenance.
- GPU inference remains chunked; merge/tracking/final-save is a separate finite
  CPU stage with chronological state across chunks. Keep no-progress/retry bounds
  and atomic, read-back-validated publication; do not recreate final-save loops.
- The resulting run must include the tested tracklet method, not stop at poses
  lacking identities. Water remains downstream metadata, not inference filtering.
- Record model/config/code/source hashes and the passed pilot/review references.
  If true detector centroids are available during inference, preserve them with
  explicit semantics rather than silently substituting or discarding them.

The numerical pilot is not scientific identity validation. Full-video execution
is now authorized by the user's explicit acceptance of optional, experimental
identities, not by an invented visual-review pass. Software/integrity checks still
apply. No training restart is part of this request.
