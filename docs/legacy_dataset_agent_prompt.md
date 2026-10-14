Work in `/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap`.

Read `docs/PROJECT.md`, `docs/legacy_dataset_agent_handoff.md`, and the latest
`docs/datasets/legacy_native_tracklets_v001/README.md` completely, along with
applicable workspace instructions. The handoff is historical task context;
consult the dataset notes/decision receipts before repeating already completed
work or asking questions the user has already resolved.

Your job is to organize predictions from the ORIGINAL SLEAP model pair into a
clean, reproducible, pandas-readable dataset that can serve as the template for
future model outputs. Do not integrate the new `v007_19f` models in this task.

First run a finite, non-destructive Slurm audit to determine whether existing
per-frame predictions can support conservative adjacent-frame tracklets. Do
not assume saved tracks exist or are valid, and do not rerun inference merely
because identities are absent. Report feasibility and specific deficiencies.

If feasible, implement/test the requested conservative tracklets and dataframe:
tracklet ID, original-video frame, centroid position, body information, shoulder
information, and nullable water-region membership, with video/source provenance.
No gap bridging, re-identification, cross-video identities, or forced ambiguous
matches. Treat centroid/body/shoulder as the trusted features; preserve other
landmarks but do not rely on them for association.

Confirm whether "body" means the `Shell` landmark. Check whether detector
centroids were retained; do not silently substitute Shell or average unreliable
limbs. Propose a static water ROI from representative frames and obtain user
confirmation, or request an annotated region if unclear. Water is a per-frame
analysis annotation, not a reason to delete observations or alter inference.
The input audit can proceed while those clarifications are pending.

Follow the handoff's versioned organization, provenance, read-back checks,
quality-review requirements and acceptance criteria. Keep all original files
and the separate training run untouched. All computation, tests and decoding
must use finite Slurm jobs; report their IDs and results. No prediction rerun
without explicit approval after evaluation, no new training, no autonomous
continuation or recurring jobs. Finish this task and become idle.
