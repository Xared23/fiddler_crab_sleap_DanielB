# Adding female predictions and fixed burrows to the existing demo

[Documentation index](../README.md) · [Active bundle instructions](../../demo/crab_cam2_2_middle_2000/README.md)

## Completed result

Corrected CPU Slurm job **17190860 COMPLETED**, exit **0:0**, in **3m42s**.
All **32 tests** passed (12 augmentation tests + 20 existing tracker tests).
The active bundle's `AUGMENTATION_COMPLETE.json`, updated `DEMO_COMPLETE.json`,
and `checks/demo_female_burrows_17190860/PUBLISHED.json` confirm publication.

- Exactly **2,000 frames** in the original **10 chunks**.
- **243,588 unchanged male observations** + **107,311 female observations** =
  **350,899 dataframe observation rows**.
- **7,834 experimental female tracklets**, with **434 links across chunk
  boundaries**. All female observations received IDs under the fixed settings;
  tracklet count is not a count of unique animals.
- **282 fixed burrows**, using source annotation frame 563, shown at every frame
  in the combined/static GUI files; the relational table has just 282 rows.
- Original male-only bundle retained at
  `archive/demo/crab_cam2_2_middle_2000_male_only_17190860/`. Source burrow SLP
  retained at its original location and copied byte-identically into the demo.
- No Git commit or push performed. No desktop SLEAP GUI tested.

Open `demo/crab_cam2_2_middle_2000/chunk_001/Crab_Cam_2_2.frames_013410_013619.combined.tracked.slp`
for all three layers over the first chunk's unchanged movie.

## Scope

User-requested local extension of `demo/crab_cam2_2_middle_2000/`, using its
unchanged 2,000 frames (Crab_Cam_2_2 frames 13410–15409) and 10 lossless video
chunks. This is not a new neural-inference run, full-video female tracking run,
video conversion, training dataset, or Git publication.

The female production completion receipts exist for both videos, each covering
28,800 frames with the single `FBody` landmark and `tracking_enabled: false`.
The demo build checks the Cam 2_2 merged-SLP SHA-256 against both its raw and
prediction completion receipts before extraction. Cam 2_3 is not used here.

## Contents and semantics

- Male-only raw/tracked SLPs and movies are retained byte-identically. Every male
  dataframe source column, measurement, flag and full-video experimental ID is
  preserved. Augmented root/chunk observation tables add female rows, class and
  file/index columns. Frame summaries gain explicit female and all-crab counts;
  original count columns remain male-only for backward compatibility. Likewise,
  legacy `chunks.json` fields `observations` / `assigned_observations` remain
  male-only; new `male_observations`, `female_observations`, and
  `female_assigned_observations` identify the per-layer counts. Sum male and
  female counts for all predicted crabs; burrows are a separate static layer.
- Female raw points/scores/visibility are copied from the completed merged SLP.
  A separate tracked copy adds demo-scoped experimental IDs using the existing
  body-first tracker, with the same numeric parameters and no Shoulder cues.
  State crosses all chunk boundaries; no temporal gaps are bridged. No new
  observations, threshold sweep or full-video tracking are performed.
- The uploaded burrow SLP is **copied** into `burrows/BurrowLabels3.slp` with its
  original hash. Both annotated frames remain there. The first annotated map,
  frame **563** (282 points), is the explicit constant-position reference. It is
  not merged/averaged with frame 948. Its point coordinates/order are retained.
  Stable reference IDs are not a claim of validated inter-frame association.
- `burrows.parquet` has 282 static reference rows rather than one independent
  detection per frame. The GUI SLPs repeat those static positions on every frame,
  explicitly as propagated reference instances, not repeated human annotation.
  Camera-coordinate transfer from `clip_cam2_8k.mp4` is an assumption, not a newly
  validated registration.
- Per-chunk `.combined.tracked.slp` files display males, females and burrows
  together. To avoid depending on multiple-skeleton GUI support, they use one
  display-only union skeleton with the original five male landmarks, `FBody`
  and `Burrow`. Only native nodes have positions; all other coordinates are NaN
  and invisible. Original male edges only. Separate native-skeleton files are
  included for each layer. Combined files must not be used for training.
- Model classes are not independently confirmed sex. No male/female matching or
  duplicate arbitration is introduced. Experimental IDs are not validated
  physical-animal identities; false detections and fragmentation may remain.

## GUI representation and evidence

The GUI's instance-selection helper includes user instances plus unused
predictions. Separate male/female/burrow track namespaces avoid having a static
burrow suppress a prediction through a shared track. Source inspection:
[`get_instances_to_show` / `get_unused_predictions`](https://github.com/talmolab/sleap/blob/develop/sleap/sleap_io_adaptors/lf_labels_utils.py),
retrieved directly over HTTPS from the official repository on 2026-11-09, not
through a third-party documentation proxy. This is source-code evidence, not a
desktop-GUI test or a guarantee for every installed GUI version. The union
skeleton additionally avoids depending on multi-skeleton editing support.

## Build, tests and publication

Entry points:

- `scripts/augment_crab_demo.py`
- `tests/test_augment_crab_demo.py`
- `jobs/augment_crab_demo.sh`
- `docs/demo/README_combined.template.md`

The finite CPU Slurm job runs new known-answer tests plus the existing body-first
tracker tests. It then verifies original demo artifact hashes, all raw female
point bytes, tracked/raw equality outside IDs, all combined readback instance
coordinates/scores/visibility/completion flags and IDs, missing non-applicable
anatomy, static reference coordinates, no within-frame duplicate assigned IDs,
no female tracklet gaps, dataframe row/SLP mapping and original male-column
preservation. Every chunk's new tracked/static/combined SLP is loaded via its
portable basename link and its first/last video frames decoded. This is a data/
backend check, **not a desktop SLEAP GUI test**.

Build occurs under an ignored staging directory. Only after validation is the
unchanged old demo archived under `archive/demo/` and the staged directory
published to the original active path. Publication restores the old path if the
second rename fails. Models, original burrow labels and full prediction runs
are never moved or edited. Payloads remain below 100 MiB per file.

Receipts:

- `checks/demo_augmentation_tests_<job-id>.json` and matching test text logs.
- `logs/demo_female_burrows_<job-id>.{out,err}`.
- `checks/demo_female_burrows_<job-id>/PUBLISHED.json`.
- Active bundle `AUGMENTATION_COMPLETE.json` and updated `DEMO_COMPLETE.json`.
- `provenance/MALE_DEMO_COMPLETE.json`: historical original receipt, not current
  augmented-table hashes.

Initial job **17190767** stopped at the synthetic-test gate: sleap-io emitted
absolute media paths. No production staging or active-demo modification occurred.
The writer now explicitly uses adjacent-video basenames before readback.
Corrected build **17190860** subsequently completed and published successfully,
as recorded above. Submission alone was not treated as completion: Slurm exit
status and the active bundle's updated durable receipts were checked.
No Git commit, push, recurring job or follow-on full-video tracking was performed.
