# Defect log

## D-001 — staged linking lets a margin-vetoed endpoint link at a later stage
- **Found:** 2026-10-30, final review of long-gap bridging (I-1)
- **Status:** fixed
- **Where:** scripts/gap_bridging/candidates.py:604-621 (`link_tracklets`)
- **Symptom:** an END/START whose stage-k pair was `refused_margin` stays free; at stage k+1 its runner-up only sees stage-(k+1) candidates, so it links to a farther long-gap candidate with "no competitor"
- **Repro:** A ends f10; B, B2 start f12 at (0, ±1.5); C starts f20 at (3, 0); sqrt, max_gap 15 → A→C linked at stage 2
- **Impact:** sqrt runs at max_gap > 5 can contain links the margin veto should have refused (none on the demo)
- **Regression test:** tests/test_gap_bridging_candidates.py::StagedLinkingTests::test_regression_D001_vetoed_endpoint_not_relinked
- **Fix:** vetoed END/START removed from later stages (`vetoed_end`/`vetoed_start` in `link_tracklets`); later gate-passing pairs are `refused_assignment` "endpoint vetoed at stage s" (fix wave, uncommitted)

## D-002 — `gate_model: linear` no longer reproduces v001 interpolated paths
- **Found:** 2026-10-30, final review (I-2)
- **Status:** fixed
- **Where:** scripts/gap_bridging/geometry.py:419-420 (`fill_bridge`)
- **Symptom:** tangent damping for dt > prediction_horizon_frames is applied under both gate models; linear demo fills differ from v001 by up to 1.23 px (max_gap 5) and 27.8 px (max_gap 30)
- **Repro:** `fill_bridge(a, b, ..., BridgeConfig(gate_model='linear'))` with dt = 60 and non-chord endpoint velocities differs from the undamped Hermite
- **Impact:** linear-model outputs since 91b5d6c have different interpolated coordinates from v001
- **Regression test:** tests/test_gap_bridging_geometry.py::DampedFillTests::test_regression_D002_linear_fill_not_damped; tests/test_gap_bridging_pipeline.py::LinearGoldenTests
- **Fix:** `fill_bridge` damps only when `cfg.gate_model == "sqrt"`; linear outputs equal 3233ca6 on the demo at max_gap 5 and 30 (fix wave, uncommitted)

## D-003 — no golden test pins the linear model to v001 on the demo
- **Found:** 2026-10-30, final review (I-3)
- **Status:** fixed
- **Where:** tests/test_gap_bridging_pipeline.py
- **Symptom:** D-002 was not caught by any test
- **Repro:** n/a (missing test)
- **Impact:** legacy reproducibility unprotected
- **Regression test:** tests/test_gap_bridging_pipeline.py::LinearGoldenTests::test_regression_D003_linear_demo_golden
- **Fix:** golden test pins counts, interpolated-row sha256 d459e647… and audit-table sha256 882625ca… (fix wave, uncommitted)

## D-004 — candidate enumeration is O(time-window pairs) Python objects
- **Found:** 2026-10-30, final review (I-4)
- **Status:** fixed
- **Where:** scripts/gap_bridging/candidates.py:85-117 (`_enumerate_pairs`), :623-628
- **Symptom:** every (END, START) pair within max_gap frames gets a LinkDecision and a links row although ~99 % fail the displacement gate; ~570 B per pair, ≈ 30 GB at max_gap 60 on the full dataset
- **Repro:** demo, sqrt, max_gap 60: 201,887 decision rows, 198,570 refused_gate
- **Impact:** memory and audit volume of the full-dataset sweep
- **Regression test:** tests/test_gap_bridging_candidates.py::RadiusPrefilterTests
- **Fix:** sqrt prefilter `_prune_radius` + per-START-frame vectorised `_enumerate_pairs`; pruned pairs counted in `link_tracklets(stats=)` and marker `counts.candidate_pairs`; linear unpruned (fix wave, uncommitted)

## D-005 — `--margin` is a silent no-op under the sqrt gate model
- **Found:** 2026-10-30, final review (I-5)
- **Status:** fixed
- **Where:** scripts/bridge_gaps.py:306-307, :360
- **Symptom:** `--margin` sets `minimum_cost_margin`, which the sqrt model ignores
- **Repro:** `bridge_gaps.py --config gap_bridging_v001.yaml --margin 0.5 ...` → marker gate margin 0.25
- **Impact:** margin scans under sqrt silently return the default
- **Regression test:** tests/test_gap_bridging_pipeline.py::CliTests::test_regression_D005_margin_sets_active_model
- **Fix:** `with_margin` in scripts/bridge_gaps.py routes `--margin` to the active model's field (fix wave, uncommitted)

## D-006 — YAML comment says the sqrt cost divides by r(dt)
- **Found:** 2026-10-30, final review (M-6)
- **Status:** fixed
- **Where:** configs/tracking/gap_bridging_v001.yaml:18
- **Symptom:** comment says `dist / r(dt)`; code divides by the extended radius R
- **Repro:** read the comment
- **Impact:** documentation only
- **Regression test:** n/a (comment)
- **Fix:** comment now says the cost divides by the extended radius R (fix wave, uncommitted)

## D-007 — hold-out never plants a gap longer than the horizon, so the default rests on a blind spot
- **Found:** 2026-10-31, final review of gap bridging v003 (C1, I3)
- **Status:** fixed
- **Where:** scripts/bridge_gaps.py `sweep` (`gaps = [g for g in holdout_gaps if 1 <= g <= horizon]`); pooling and rule only in an untracked /tmp script
- **Symptom:** at horizon H no crab is absent longer than H, so a long absence linked to a neighbour within H is never scored; the shipped default 45 passed the rule although the sweep's only false link (seed 0, planted gap 60, linked at gap 38 in stage 4) is within reach of 45
- **Repro:** `run_holdout` at max_gap 45, seed 0, gaps 1..60, `--per-gap 8 --decoys 4 --flicker 2` → plant t000042944 linked to t000043141 at gap 38, cost 0.86
- **Impact:** default horizon chosen on unscored cases (45 instead of 30); the derivation was not reproducible from the repository
- **Regression test:** tests/test_gap_bridging_holdout.py::RuleTests::test_false_link_at_38_under_45_fails_45_and_30_passes, ::RuleTests::test_capped_sweep_is_refused, ::RuleTests::test_false_link_from_a_plant_beyond_the_horizon, ::PoolTests; tests/test_gap_bridging_sweep.py::SweepTests::test_holdout_by_gap, ::SweepTests::test_pool_cli
- **Fix:** the sweep plants every gap at every horizon (a plant beyond the horizon is a miss unless linked, and its links are false links); `holdout.pool_sweeps` / `holdout.apply_rule` and `bridge_gaps.py pool` pool seeds and apply the rule (refusing horizons scored on different gaps); default `max_gap` 45 → 30 in `BridgeConfig` and the YAML (fix wave, uncommitted)

## D-008 — claw-side veto fires on Shoulder flips of one crab
- **Found:** 2026-10-31, final review of gap bridging v003 (I1)
- **Status:** fixed (guarded)
- **Where:** scripts/gap_bridging/candidates.py `_cue_terms`
- **Symptom:** the side sign `sign(u × w)` flips when the Shoulder keypoint jumps to the other end of the body axis; all 18 demo vetoes at max_gap 45 had antiparallel Shell→Shoulder and parallel Shell→Claw vectors, and one vetoed pair (41951→42378) is chained into one trajectory through a one-frame tracklet
- **Repro:** demo, default config, max_gap 45 → 18 `refused_gate claw_side` rows before the fix
- **Impact:** fragmentation (true links refused) wherever a fragment carries a flipped Shoulder; no link changed on the demo
- **Regression test:** tests/test_gap_bridging_cues.py::AxisFlipTests (test_axis_reversed_claw_aligned_is_not_vetoed, test_genuine_opposite_claw_is_vetoed, test_true_turn_keeps_the_ordinary_rule, test_unknown_vectors_never_flip, test_suppression_iff_axis_reversed_and_claw_kept), ::DemoCueTests::test_cues_change_few_sqrt_decisions
- **Fix:** `_axis_flip` (`u_a·u_b < 0` and `c_a·c_b > 0`) sets the side term to 0 and lifts the veto; such pairs that would have been vetoed get `; axis_flip` in their reason and are counted in marker `counts.claw_side_suppressed` (demo 45: 18 → 0 vetoes, 18 suppressed, link set unchanged). The rule also lifts the veto for different-handed crabs facing opposite ways (hold-out decoys), see docs/GAP_BRIDGING.md (fix wave, uncommitted)

## D-009 — hold-out positives are long-tracklet halves, so a scorer trained on them under-scores natural breaks
- **Found:** 2026-11-01, gap bridging v004 integration
- **Status:** mitigated in v005 (`sweep --profile short` plants halves matching the natural length mix; the retrained scorer gives p >= 0.5 to 97-98% of natural links in every length bin; the learned model stays an option because the threshold recipe, not the confound, now limits demo links; see docs/GAP_BRIDGING.md v005)
- **Where:** scripts/gap_bridging/holdout.py `plant_gaps` (`min_tracklet_len` 40, `margin` 3); scripts/gap_bridging/train.py
- **Symptom:** every planted positive has both halves ≥ 4 frames (seed 0, horizon 30: min 4, median 42), whereas 75% of the demo's 827 natural links have a 1-frame side. The first trained scorer (pre fold-correction, threshold 0.98) gave `p >= 0.98` to 3% of natural 1-frame links and 86% of ≥ 20-frame links; the committed fold-corrected scorer (threshold 0.9944) keeps 26 of the demo's 827 links at horizon 30 and 704 at `p >= 0.5`, while its hold-out recall over all plants is 0.33 (trained threshold) to 0.97 (`p >= 0.5`, horizon 60)
- **Repro:** `pipeline.pair_feature_export` on the demo + `scorer.load_scorer('configs/tracking/link_scorer_v001.joblib').predict_proba`, grouped by `min(n_frames_a, n_frames_b)`
- **Impact:** the frame-deletion hold-out cannot validate a learned scorer on the breaks that matter (short natural fragments); the promotion rule is blind to the demo loss (801 links at the trained threshold, 123 at `p >= 0.5`)
- **Regression test:** n/a (data property); the comparison tables in docs/gap_bridging_v004_comparison*.md and the demo link counts in docs/GAP_BRIDGING.md document it
- **Fix (proposed):** plant gaps that leave short halves (1 to 3 frames) or drop the tracklet-length features (`n_frames_*`, `size_n_*`, `claw_side_n_*`, `vel_known_*`) and retrain; judge any learned variant on demo link counts as well as the hold-out

## D-010 — isotonic calibration puts the precision-1.0 threshold at p = 1
- **Found:** 2026-11-01, gap bridging v004 training
- **Status:** fixed (sigmoid is the default; a threshold ≥ 1 or an unreachable target is refused unless `--allow-degenerate-threshold`)
- **Where:** scripts/gap_bridging/train.py `fit_scorer`, `choose_threshold`
- **Symptom:** with isotonic calibration the top plateau of the OOF probabilities is exactly 1.0, so the smallest threshold with OOF precision 1.0 is 1.0; every accepted pair then has cost `-log 1 = 0`, the assignment cannot rank accepted candidates and `learned_margin` has nothing to measure
- **Repro:** `bridge_gaps.py train --sweeps logs/gap_sweep_v004/current/seed_0..4 --out /tmp/x.joblib` → manifest `threshold 1.0`
- **Impact:** degenerate costs under the learned model
- **Regression test:** tests/test_gap_bridging_train.py (degenerate and unreachable thresholds refused and allowed; isotonic on separable data raises)
- **Fix:** sigmoid (Platt) calibration keeps the ranking (threshold 0.9805 on the pre-correction labels, 0.9944 on the corrected ones) and `train_command` refuses degenerate or unreachable thresholds

## D-011 — single global assignment loses short links that staging keeps; a stronger gap penalty does not recover them
- **Found:** 2026-11-01, gap bridging v004 comparison
- **Status:** documented (`assignment: staged` stays the default)
- **Where:** scripts/gap_bridging/candidates.py `_stages`, `_assign`
- **Symptom:** demo at max_gap 30: staged 827 links, global 781 (fixed dummy), 772 (adaptive dummy); 59 staged links (36 of them 1 to 2 frame gaps) become `refused_assignment` because a cheaper long-gap pair claims an endpoint, while only 4 new links appear. Raising `gap_penalty` from 0.06 to 0.12/0.2/0.3/0.5 under global gives 783/765/722/665 links: the penalty pushes long pairs over `max_cost` faster than it reorders contested components. On the planted hold-out global and staged are identical at every gap, horizon and fold
- **Repro:** `bridge_gaps.py run --config configs/tracking/gap_bridging_v004_global.yaml` vs defaults on the demo
- **Impact:** the hold-out cannot adjudicate the 55 demo links (both variants score alike); staging remains the better-supported choice on this clip
- **Regression test:** tests/test_gap_bridging_assignment.py (global equals staged with a single stage; adaptive dummy properties); tests/test_gap_bridging_pipeline.py::FeatureExportTests::test_global_adaptive_variant_runs_with_invariants
- **Fix:** none needed for the default; a joint formulation that prices chain consistency (min-cost flow over tracklets, next-steps item 5) is the principled alternative

## D-012 — compare selected the full TRUTH_SCHEMA from older truth tables
- **Found:** 2026-11-01, planning gap bridging v005
- **Status:** fixed
- **Where:** scripts/gap_bridging/compare.py `fold_scores` (`.select(holdout.TRUTH_SCHEMA.names)`)
- **Symptom:** any new truth column would have made `fold_scores` raise on every hold-out written before the column existed (including `logs/gap_sweep`, which `RealSweepTests` reads)
- **Repro:** add a column to TRUTH_SCHEMA and run tests/test_gap_bridging_compare.py::RealSweepTests
- **Impact:** v004 sweeps could not have been compared by v005 code
- **Regression test:** tests/test_gap_bridging_compare.py::TruthSchemaToleranceTests
- **Fix:** select the intersection of the schema's names and the table's columns
