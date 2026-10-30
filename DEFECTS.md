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
