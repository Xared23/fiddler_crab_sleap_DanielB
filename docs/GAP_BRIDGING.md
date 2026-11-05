# Gap bridging between body-first tracklets

[Documentation index](README.md) · [Body-first tracker](BODY_FIRST_TRACKLETS.md) ·
[v007 dataframe](datasets/v007_best_body_first_dataframe_v001/README.md)

**Status:** experimental post-processing, validated on the 300-frame demo clip
only. The full dataset is not in this checkout and has not been processed. Like
the tracklets it starts from, the bridged identities have **not** been checked
against ground truth.

## Purpose

The body-first tracker (`scripts/body_first_tracklets.py`) links detections only
between adjacent frames. When SLEAP misses a crab at frame i+1, or the tracker
rejects that detection, the identity ends at frame i and a new tracklet starts at
i+2 or later. As a result, tracklet lengths in the published dataframe vary
widely (demo clip: 1,181 tracklets, median length 3 frames).

`scripts/bridge_gaps.py` links pairs of tracklets separated by 1 to `max_gap`
missing frames (default 30 since v003; v001/v002 shipped 5 and v002 recommended 15; see
[Default horizon (v003)](#default-horizon-v003)). It fills each missing frame with either a real
detection the tracker had rejected (**recovered**) or a synthesized pose
(**interpolated**). It writes new tables and an audit row for every candidate
link. The input table is never modified, and its rows are carried into the
output value- and type-identical (pyarrow `Table.equals`).

Since v003 the run also extends trajectories at their ends with rejected
low-score detections (**recovered_tail**), uses carapace size and claw side as
appearance cues when linking, and flags one-frame low-score units as
`spurious` without removing them; see [v003](#v003-tail-recovery-identity-filter-appearance-cues).

Implementation: `scripts/gap_bridging/` (`geometry.py`, `candidates.py`,
`chains.py`, `recover.py`, `tails.py`, `identity.py`, `tables.py`,
`pipeline.py`; the evaluation tools `fit.py`, `holdout.py` and `figures.py`;
the weak-label export `export.py`), plus `scripts/reinfer_low_threshold.py`
and `jobs/reinfer_low_threshold.sh` (re-inference scaffold).
Parameters: `configs/tracking/gap_bridging_v001.yaml`.
Tests: `tests/test_gap_bridging_*.py`.

## Method

Notation: tracklet A ends at frame e and tracklet B starts at frame s, in the
same video. `dt = s - e`, and the number of missing frames is `gap = dt - 1`.
Units are native-image pixels, px/frame and integer frames.

Steps 2 to 5 below describe the v001 **linear** gate model
(`gate_model: linear`). The default is now the **sqrt** gate model, which
replaces the gates, the cost and the thresholds of steps 2 to 5 and keeps the
assignment, margin veto and chain-aware runner-up unchanged; see
[Gate model v002](#gate-model-v002). Under the sqrt model only, step 7 damps
the tangents for gaps longer than `prediction_horizon_frames`; see
[Damped fill](#damped-fill). With `gate_model: linear` every step, the fill
included, is the v001 method, and the outputs reproduce v001 exactly.

1. **Endpoints.** The method reads only accepted rows that carry a
   `tracklet_id`. For each tracklet it computes the following at the START and at
   the END:
   - Shell position.
   - Velocity over the last (first) `k = min(velocity_window, n - 1)` steps.
     The speed is clipped to 6 px/frame. With n = 1 the velocity is unknown and
     set to zero.
   - Heading: `atan2` of Shoulder − Shell at the endpoint frame. If that frame's
     Shoulder is not usable, the nearest usable frame in the window is taken.
     Shoulder is usable when it is visible and finite and its distance from Shell
     is 10–65 px.
   - Body-frame offsets of every node: `R(−heading) · (node − Shell)`, averaged
     over the window frames.
2. **Candidates and gates.** Each (END a, START b) pair in the same video with
   `1 <= gap <= max_gap` is a candidate. It must pass two gates:
   - `dist = |pb − pa| <= 12 · dt`.
   - `err <= 18 · dt`, where `err` is the mean of the forward error
     `|pb − (pa + va·dt)|` and the backward error `|pa − (pb − vb·dt)|` over the
     endpoints with a known velocity. When neither velocity is known, `err = dist`.

   With the defaults, the prediction-error gate can never fail. Endpoint speeds
   are clipped to 6 px/frame and the displacement gate allows at most `12·dt`,
   so `err <= dist + 6·dt <= 18·dt`. It matters only when
   `max_velocity_px_per_frame` or `max_body_displacement_px` is raised.
3. **Cost** (a per-frame equivalent of the tracker's cost):

   ```text
   cost = 0.7 * dist/dt + 0.3 * err/dt + 0.15 * min(|u_a − u_b|, 12)
   ```

   `u` is the Shell→Shoulder vector in image coordinates. The Shoulder term is
   zero unless both headings are known. A pair is assignable only when
   `cost <= 12`.
4. **Assignment.** Per video, Hungarian matching on the `(m+n) × (m+n)` matrix
   used by `body_first_tracklets.association`:
   - Non-candidate cells are 1e9.
   - Leaving an end or a start unmatched costs 7 (dummy diagonal).
   - The dummy–dummy block is 0.

   The solve is done per connected component of the candidate graph. This gives
   the same optimum with bounded memory.
5. **Margin veto.** An assigned pair is refused (`refused_margin`) when another
   gated pair in its row or column costs less than `cost + 4`. Neither side is
   then linked, and the pair is not retried. Under staged linking (sqrt model),
   both endpoints of a vetoed pair also stay unlinked at every later stage: a
   later pair that touches one of them and passes the gates is
   `refused_assignment` with reason `endpoint vetoed at stage s`.

   Competitors that are reachable along the chain are not counted. For an
   assigned pair (a, b):
   - A row competitor (a, c) is ignored when (b, c) passes the motion gates,
     because c is then a plausible successor of b.
   - A column competitor (d, b) is ignored when (d, a) passes the motion gates,
     because d is then a plausible predecessor of a.

   Without this rule, the cost is per frame, so skipping one of a crab's own
   short tracklets would cost about the same as linking to it. The skip pair
   would then veto the true link. The `reason` of a `linked` or
   `refused_margin` row ends with `; N chain competitors ignored` when N > 0,
   and the same N is in the `n_chain_ignored` column.

   The other outcomes of the audit are:
   - `refused_assignment`: the pair passed the gates but lost the assignment.
   - `refused_gate`: the pair failed a gate. The reason is `displacement`,
     `prediction_error` or `max_cost`.
6. **Chains.** Links are one-to-one, so they form disjoint paths. Each path,
   singletons included, becomes one trajectory. Trajectories are ordered by
   (video, start frame, first tracklet) and named
   `<video_id>/trajectory0000001`, numbered per video. The k-th gap of a
   trajectory is `<trajectory_id>/b00k`.
7. **Hermite fill of the Shell path.** For each missing frame f:

   ```text
   t = (f − e) / dt
   p(t) = h00·pa + h10·dt·va + h01·pb + h11·dt·vb
   ```

   This is the cubic Hermite curve, which reproduces constant-velocity motion
   exactly. If either endpoint has an unknown velocity, both tangents are set to
   the chord velocity, which makes the fill linear. Under the sqrt gate model,
   for `dt` above `prediction_horizon_frames` the tangents are damped towards
   the chord ([Damped fill](#damped-fill)); under the linear model they are
   never damped.
8. **Recovery.** At each bridged frame, the method takes the closest rejected row
   to `p(f)` that is within 8 px and meets these conditions:
   - `accepted_for_tracking == False`.
   - Its `rejection_reason` is one of `low_body_score`, `missing_position`,
     `outside_image`, `invisible` or `duplicate_candidate`.
   - Its body position is finite.
   - It has not already been adopted.

   Ties are broken by `detection_id`. Bridges are processed in (video, first gap
   frame, trajectory) order. The adopted row is labelled in place and no synthetic
   row is created for that frame. The path is **not** refit through recovered
   detections.
9. **Heading.** The heading is interpolated along the shortest arc:
   `h(t) = ha + t · wrap(hb − ha)`, with the difference wrapped into (−π, π]. If
   only one endpoint heading is known it is held constant. If neither is known
   the heading is undefined.
10. **Limbs (rigid transform).** Each node is placed at `p + R(h) · offset`.
    The offset is the mean of the two endpoints' body-frame offsets, or one side's
    offset if the other is unknown. A node with no heading or no offset is left
    null, with `*_visible = False`.
11. **Tail recovery (v003).** After the bridges are filled, each trajectory is
    extended backwards from its first frame and forwards from its last frame
    with rejected `low_body_score` rows; see [Tail recovery](#tail-recovery).
12. **Identity flag (v003).** Every trajectory is classified `identity` or
    `spurious` and the class is appended to both output tables
    (`trajectory_status`); see [Identity filter](#identity-filter). Linking
    is not affected.

Under the sqrt model the cost of step 3 also carries two appearance terms and
a claw-side veto ([Appearance cues](#appearance-cues)).

## Parameters

From `configs/tracking/gap_bridging_v001.yaml`. With no `--config`, the built-in
defaults are used; they are identical to this file. The rows below are the v001
parameters; the gate-model parameters are listed under
[Gate model v002](#gate-model-v002). Under the default sqrt model the
px-scale thresholds `max_assignment_cost`, `unmatched_cost` and
`minimum_cost_margin` are not used; their `*_sqrt` counterparts are. The CLI
`--margin` sets the margin of the active gate model: `minimum_cost_margin`
under linear, `minimum_cost_margin_sqrt` under sqrt. The marker's `gate` block
records the value in force.

| Parameter | Default | Meaning |
|---|---:|---|
| `max_gap` | 30 | Bridge `1 <= gap <= max_gap` missing frames; 0 disables linking (CLI `--max-gap`); 5 until v002, 45 briefly in v003, 30 since the corrected v003 rule |
| `velocity_window` | 3 | Steps used for endpoint velocity, heading search and offsets |
| `max_velocity_px_per_frame` | 6.0 | Endpoint speed clip (tracker value) |
| `max_body_displacement_px` | 12.0 | Gate: `dist <= this · dt` |
| `max_prediction_error_px` | 18.0 | Gate: `err <= this · dt` |
| `motion_weight` | 0.3 | Weight of `err/dt` (and `1 − w` of `dist/dt`) |
| `shoulder_vector_weight` | 0.15 | Weight of the Shoulder-vector change |
| `shoulder_vector_cost_cap_px` | 12.0 | Cap on the Shoulder-vector change |
| `max_assignment_cost` | 12.0 | Pairs above this cost are not assignable |
| `unmatched_cost` | 7.0 | Cost of leaving an end or start unmatched |
| `minimum_cost_margin` | 4.0 | Veto when the runner-up is within this (CLI `--margin` under linear) |
| `recovery_radius_px` | 8.0 | Radius for adopting a rejected detection (CLI `--recovery-radius`) |
| `recoverable_reasons` | 5 reasons (see method, step 8) | Rejection reasons eligible for recovery |
| `fps` | 30000/1001 | Sets `time_s = frame / fps` on interpolated rows when the input has `time_s` |
| `fill_confidence_gaps` | [5, 15] | `fill_confidence` of a bridge: `high` for gap ≤ 5, `medium` for gap ≤ 15, else `low` |

v003 parameters (all in the YAML):

| Parameter | Default | Meaning |
|---|---:|---|
| `tail_recovery` | true | Extend trajectories at their ends with rejected rows |
| `tail_radius_px` | 8.0 | Adoption radius around the predicted tail position |
| `tail_max_frames` | 15 | Maximum length of each tail |
| `tail_reasons` | [low_body_score] | Rejection reasons eligible for a tail |
| `tail_exclusion_px` | 5.0 | A candidate within this of another trajectory's row at that frame is never adopted (the tracker's duplicate-Shell radius) |
| `min_identity_frames` | 2 | A trajectory with fewer real frames ... |
| `min_identity_score` | 0.6 | ... and a median `instance_score` below this is `spurious` |
| `appearance_cues` | true | Size and claw-side terms and the claw-side veto (sqrt model only) |
| `size_scale_px`, `size_noise_px`, `size_cost_cap`, `size_weight` | 3.0, 1.0, 3.0, 0.1 | Size term (see [Appearance cues](#appearance-cues)) |
| `claw_side_weight`, `claw_side_confidence`, `claw_side_min_frames` | 0.15, 0.9, 5 | Side term and veto |

v004 parameters (all in the YAML; defaults reproduce v003 byte for byte; see
[v004](#v004-learned-link-scorer-and-single-global-assignment)):

| Parameter | Default | Meaning |
|---|---:|---|
| `cost_model` | hand | `hand`: the sqrt/linear cost; `learned`: `-log p` from the scorer at `scorer_path` (sqrt gates and claw veto unchanged) |
| `scorer_path` | null | `.joblib` model with a `.json` manifest beside it; a relative path is resolved against the YAML's directory |
| `learned_min_probability` | 0.5 | `max_cost` gate under `learned`: a pair with `p` below this is refused; `train --write-config` writes the trained threshold |
| `learned_unmatched_probability` | 0.5 | Dummy diagonal `-log p` under `learned` |
| `learned_margin` | 0.0 | Runner-up margin under `learned`, in `-log p` units (CLI `--margin`) |
| `assignment` | staged | `staged`: `link_stages`; `global`: one assignment over gaps `1..max_gap` |
| `unmatched_cost_mode` | fixed | `adaptive`: each connected component's dummy diagonal is `adaptive_factor` × its largest finite cost (Jaqaman et al. 2008) |
| `adaptive_factor` | 1.05 | Multiplier of the adaptive dummy |
| `export_pair_features` | false | Also write `pair_features.parquet` (training input of the scorer) |

## Outputs

These files are written to `--out`. Each parquet is first written to
`.<name>.tmp` and then moved into place, and the marker is written last. Existing
outputs are refused unless `--force` is given.

**`observations.parquet`**. All input rows in their original order, with the
original columns unchanged, followed by six new columns (an input that already
has any of them is refused; the published `identity_status` column is
provenance and unrelated):

| Column | Type | Content |
|---|---|---|
| `trajectory_id` | string | Trajectory of the row. Null for rejected rows that were not recovered. |
| `source` | string | `observed`, `recovered`, `recovered_tail` or `interpolated` |
| `bridge_id` | string | `<trajectory_id>/bNNN` for recovered and interpolated rows, `<trajectory_id>/tail_back` or `/tail_fwd` for tail rows, otherwise null |
| `gap_frames` | int64 | Number of missing frames of the row's bridge (recovered frames included), on recovered and interpolated rows; the tail's length on tail rows; null on observed and unassigned rows |
| `fill_confidence` | string | `high` / `medium` / `low` from `gap_frames` and `fill_confidence_gaps`, on recovered, tail and interpolated rows; otherwise null |
| `trajectory_status` | string | `identity` or `spurious` ([Identity filter](#identity-filter)) of the row's trajectory; null on unassigned rows |

With `export_pair_features: true` (the default inside `sweep` hold-outs) a
fourth parquet, **`pair_features.parquet`**, holds one row per links-audit row
in the same order: `video_id`, `tracklet_a`, `tracklet_b`, `gap_frames`,
`gate` and the 28 float64 features of `features.FEATURE_NAMES` (v004). The
marker's `timings` block gives the wall seconds of each stage before the write
and their `total`; its `gate` block also records `cost_model`, `assignment`
and `unmatched_cost_mode`.

Interpolated rows come after the input rows, sorted by (video_id, frame,
trajectory_id). Every column of an interpolated row is null except the
following:

| Column | Value |
|---|---|
| `video_id`, `frame` | Bridged frame |
| `body_x/body_y`, `centroid_x/centroid_y`, `shell_x/shell_y` | Hermite path |
| `<node>_x/_y` | Rigidly placed nodes; null where unknown |
| `<node>_visible` | True where coordinates exist, else False |
| `centroid_source` | `'interpolated'` |
| `accepted_for_tracking` | False |
| `incoming_reason` | `'bridged'` |
| `instance_index`, `raw_instance_id` | −1 |
| `detection_id` | `<video_id>/f<frame:06d>/<bridge_id>` |
| `time_s` | `frame / fps`, when the input has this column |

All scores are null on interpolated rows, and `tracklet_id` is null. A recovered
row keeps all of its original values, including `accepted_for_tracking = False`
and its `rejection_reason`. Only the new columns mark it as adopted; the same
holds for a `recovered_tail` row.

**`trajectories.parquet`**. One row per trajectory, with these columns:
`trajectory_id`, `video_id`, `start_frame`, `end_frame`, `n_observed`,
`n_recovered`, `n_interpolated`, `n_recovered_tail`, `n_tracklets`,
`member_tracklet_ids` (list<string>, in time order) and `trajectory_status`.
Every trajectory is contiguous: `end_frame − start_frame + 1 = n_observed +
n_recovered + n_interpolated + n_recovered_tail`.

**`links.parquet`**. One audit row per candidate pair, with these columns
(under the sqrt model, pairs beyond the prefilter radius `R_max` are not
candidates and have no row; see [Gate model v002](#gate-model-v002)):
`video_id`, `tracklet_a`, `tracklet_b`, `gap_frames`, `cost` (NaN when a motion
gate or the claw-side veto failed), `runner_up_cost` (NaN when there is no
competitor), `decision`, `reason`, `n_chain_ignored`, `stage`, `size_term` and
`side_term` (the appearance terms added to the cost; NaN under linear or with
`appearance_cues: false`, 0 when a cue is unknown). `stage` (int64) is the 1-based
linking stage whose gap range holds the pair (always 1 under the linear model;
see [staged linking](#gate-model-v002)). `n_chain_ignored` (int64) is the number of
chain-reachable competitors left out of the runner-up (method, step 5). It is
set on `linked` and `refused_margin` rows and is 0 on all other rows. Filter or
sum it directly; there is no need to parse `reason`.

**`GAP_BRIDGING_COMPLETE.json`** contains:
- `config`: the full parameter set.
- `gate`: `gate_model` and the thresholds in force on that model's cost scale
  (`max_assignment_cost`, `unmatched_cost`, `minimum_cost_margin`; the
  `*_sqrt` values under sqrt).
- `input`: `path`, `sha256`, `rows` and `tracklets`. For a directory input it
  also lists the `files` with their hashes, and `sha256` is the hash of the
  sorted listing `"<relpath>\t<sha256>\n"`.
- `counts`: `videos`, `tracklets`, `links`, `trajectories`, `bridges`,
  `n_observed`, `n_recovered`, `n_interpolated`, `n_recovered_tail`,
  `n_unassigned`, `decisions`, `candidate_pairs` (`enumerated`: pairs with an
  audit row; `pruned_by_radius`: pairs inside the time window but beyond
  `R_max`, which were not enumerated; always 0 under linear),
  `spurious_trajectories`, `identity_starts` (trajectories starting after
  their video's first frame), `identity_starts_identity_only` (the same,
  identities only) and `claw_side_suppressed` (audit rows that would have
  been claw-side vetoed but for an axis flip; reason suffix `; axis_flip`).
- `versions`: the numpy, scipy and pyarrow versions.
- `outputs`: the `sha256` and `rows` of each parquet.

Before writing, the run checks these invariants and raises an error on any
violation:
- Every tracklet is in exactly one trajectory.
- `trajectories = tracklets − links`.
- Output rows = input rows + interpolated rows.
- Every trajectory is contiguous (tails included), with no duplicate frame.
- Every `recovered_tail` row lies outside the span of its trajectory's
  tracklets and bridges and inside the trajectory's span.
- `trajectory_status` is set on exactly the assigned rows and agrees between
  the two tables.

## Limitations

- **Gap-0 re-labels are not addressed.** A crab that is detected at frame i+1
  but given a new ID (for example because the tracker's ambiguity veto refused
  the adjacent-frame link) stays split. `max_gap` cannot go below 1, and this tool
  never re-links tracklets that are adjacent in time. This is a candidate
  follow-up.
  - On the demo clip, 1,058 tracklets end before the clip's last frame. For none
    of them does another tracklet start one frame later within the 12 px
    displacement gate.
  - Taking for each such end the smallest gap that has a start within the gate
    (`12 · dt` px): 682 ends have one at gaps 1–5, 267 at gaps 6–29, and 109
    have none within 29 frames.
  - These figures come from a nearest-start heuristic. They are not identity
    truth.
- **Chain-aware runner-up (residual risk).** The chain-aware rule (method,
  step 5) stops a crab's own later or earlier tracklets from vetoing its links.
  Without the rule, 214 links on the demo were margin-vetoed; with it, 10 are.
  The residual risk is that a genuinely different crab is not counted as a
  competitor when it is a plausible successor of b or predecessor of a, so a
  real ambiguity can go unvetoed. The audit `reason` records how many
  competitors were ignored for each link, and so does the `n_chain_ignored`
  column. On the demo, 206 links had at least one, 241 in total over the
  `linked` rows.
- **Not refit through recoveries.** The interpolated path is not refit through
  recovered detections, so a recovered detection and the interpolated frames
  next to it can be inconsistent by up to `recovery_radius_px`.
- **Interpolated poses are a model, not measurements.** Limbs are placed rigidly,
  so claw waving inside a gap is not represented. Filter on `source` when
  measured positions are required.
- **No identity validation.** As with the tracker, the costs and margins are
  deterministic heuristics, not probabilities.

## Running

Demo clip (about 3 s on a login node):

```bash
.venv/bin/python scripts/bridge_gaps.py \
    --observations demo/crab_cam2_2_middle_300/observations.parquet \
    --out outputs/gap_bridging/demo_v001 \
    --config configs/tracking/gap_bridging_v001.yaml
```

Add `--force` to overwrite. The CLI prints one progress line per stage and then
the marker as JSON. It exits non-zero on a missing input, an invalid config or
refused outputs.

**Full dataset (not run; the data is not in this checkout).** `--observations`
also accepts a directory. Every `*.parquet` under it is read in sorted path
order; hidden files are skipped. Point it at the published observations
directory, not at `unassigned_observations/`, which duplicates rows. Paths
are relative to the project root that holds the dataset:

```bash
sbatch --time=01:00:00 --mem=48G --cpus-per-task=4 --wrap \
  ".venv/bin/python scripts/bridge_gaps.py \
     --observations datasets/v007_best_body_first_dataframe_v001/tables/observations \
     --out outputs/gap_bridging/v007_best_body_first_v001 \
     --config configs/tracking/gap_bridging_v001.yaml"
```

The full dataframe has 7,985,830 rows, about 220 times the demo. The memory and
time requests above are estimates and have not been measured on the full data.

Memory of the link step (estimated by tiling the demo endpoints 16 times in
time, 18,896 tracklets, and scaling to about 260,000 tracklets with 0–15 %
higher endpoint density). With the sqrt prefilter, only pairs within `R_max`
become Python objects and audit rows:

| `max_gap` | 5 | 10 | 15 | 30 | 45 | 60 |
|---|---:|---:|---:|---:|---:|---:|
| audit rows, demo | 824 | 1,280 | 1,674 | 2,716 | 3,622 | 4,421 |
| audit rows, 16× tiled | 13,409 | 20,975 | 27,504 | 45,706 | 62,467 | 77,876 |
| audit rows, full data (est.) | 0.18–0.21 M | 0.29–0.33 M | 0.38–0.44 M | 0.63–0.72 M | 0.86–0.99 M | 1.07–1.23 M |
| link-step memory, full data (est.) | ≤ 0.2 GB | ≤ 0.3 GB | ≤ 0.4 GB | ≤ 0.7 GB | ≤ 1.0 GB | ≤ 1.2 GB |

The memory row assumes at most 1 kB per audit row; `tracemalloc` peaks on the
tiled input were 510–850 B per row, including the links table. Without the
prefilter, the same 16× input at `max_gap` 60 enumerated 4.45 M pairs (22 s,
2.25 GB of RSS growth), against 77,876 rows and 1.3 s with it, and the
full data would have needed about 45–55 M pairs and 25–32 GB for the link step
alone. The link step is therefore no longer the memory driver at any horizon.
The observation table is: 31 MB in memory on the demo, about 7 GB on the full
data, and the demo pipeline peaks at 3–5 times the table size above the
interpreter baseline. The 48 GB request is sized for that and is unchanged.
A sweep runs its horizons and hold-outs one after another in one process.

## Demo results

These numbers come from the v001 config (`gap_bridging_v001.yaml` with
`gate_model: linear`, i.e. the v001 default before the sqrt gate model) applied to
`demo/crab_cam2_2_middle_300/observations.parquet`; the linear model reproduces
them exactly, interpolated coordinates included (the original columns of the
output equal those of a run of the pre-gate-model code, 3233ca6, at `max_gap` 5
and 30). `tests/test_gap_bridging_pipeline.py::LinearGoldenTests` pins the counts
below and the SHA-256 of the interpolated rows and of the audit table. The
v002 default (sqrt, `max_gap` 5) is the first row of the
[sweep table](#horizon-sweep-results); the current default (v003, `max_gap`
45) is in [v003 results](#v003-results).
Input: 36,573 rows, 300 frames,
`Crab_Cam_2_2` frames 14250–14549. Input sha256
`739c2daf087bd1967d53dcfd8d41653166d7d30b9cc713d5ba5203a41432e1ef`. The run took
1.5 s, and a forced rerun gave byte-identical parquet files.

| Quantity | Tracklets (input) | Trajectories (bridged) |
|---|---:|---:|
| Count | 1,181 | 527 |
| Median length (frames) | 3 | 13 |
| Mean length (frames) | 30.9 | 71.8 |
| 90th percentile length | 99 | 300 |
| Length 1 frame | 398 | 113 |
| Spanning all 300 frames | 62 | 72 |
| Fraction of frames in units ≥ 30 frames | 0.871 | 0.938 |
| Fraction of frames in units ≥ 150 frames | 0.709 | 0.789 |

| Count | Value |
|---|---:|
| Links | 654 |
| Trajectories with ≥ 2 tracklets | 217 (max 47 tracklets) |
| Bridged frames | 1,338 |
| Recovered detections (all `low_body_score`; 45 of the 54 rejected rows are recoverable, the other 9 are `missing_position` with no body position) | 15 |
| Interpolated rows | 1,323 |
| `gap_frames` / `fill_confidence` on bridged rows | 1–5 (median 3) / all `high` (1,338 rows) |
| Output rows (36,573 + 1,323) | 37,896 |
| Rejected rows left unassigned | 39 |

| Gap (frames) | linked | refused_margin | refused_assignment | refused_gate |
|---:|---:|---:|---:|---:|
| 1 | 306 | 1 | 0 | 3,733 |
| 2 | 160 | 5 | 5 | 3,761 |
| 3 | 83 | 1 | 39 | 3,643 |
| 4 | 62 | 2 | 54 | 3,662 |
| 5 | 43 | 1 | 48 | 3,695 |
| **Total** | **654** | **10** | **146** | **18,494** |

- **Gate failures:** 18,480 failed on displacement and 14 on `max_cost`.
- **Linked costs:** median 0.59, maximum 8.76 (px-equivalent per frame).
- **Chain competitors:** 206 links ignored at least one chain-reachable
  competitor, 241 in total (sum of `n_chain_ignored` over `linked` rows). Of the
  10 `refused_margin` decisions, 3 also ignored chain competitors (5 in total);
  their remaining runner-up was still within the margin.
- **Interpolated poses:** every interpolated row has a heading and all five
  nodes, because every bridged endpoint had a usable Shoulder.

The median length rises from 3 to 13 frames. The number of 1-frame units drops
from 398 to 113, and the fraction of frames in units of 30 frames or more rises
from 0.871 to 0.938. These are continuity statistics only; they do not measure
identity accuracy.

## Gate model v002

**Why.** Crabs on the demo clip are nearly stationary across detection losses:
the median per-frame shell step is 0.11 px, the 30-frame net displacement has a
median of 1.4 px and a 99th percentile of 20 px. The v001 displacement gate
grows linearly, `12 · dt` px, so at a 30-frame gap it admits any start within
372 px: far more than any crab moves and enough to reach many neighbours. The
particle-tracking literature closes gaps with a maximum frame gap, a search
radius that grows sub-linearly with the gap and a gap-length cost penalty:
u-track (Jaqaman et al. 2008, *Nature Methods* 5, 695,
<https://pmc.ncbi.nlm.nih.gov/articles/PMC2747604/>, and the u-track
documentation: the Brownian search radius grows as `sqrt(dt)`, with a
gap-closing penalty), TrackMate's LAP trackers
(<https://imagej.net/plugins/trackmate/trackers/lap-trackers>: gap-closing
max distance and max frame gap), LapTrack (Fukai & Kawaguchi 2023,
*Bioinformatics* 39, btac799,
<https://academic.oup.com/bioinformatics/article/39/1/btac799/6887138>) and
trackpy's `memory` (a particle may vanish for up to `memory` frames and is
re-linked within `search_range`). Field observations of fiddler crabs
(Brodie et al., <https://par.nsf.gov/servlets/purl/10454816>) put burrow
retreats at a median of 2.7 min, so losses longer than 1 to 2 s (30 to 60
frames) are a different regime that interpolation should not reach into.

**Radius.** With `dt = s − e` (frames between the linked END and START):

```text
r(dt) = max(radius_min_px, radius_safety · (radius_r0_px + radius_k_px · sqrt(dt)))
R     = r(dt) + max(|v_a|, |v_b|) · min(dt, h)          h = prediction_horizon_frames
```

`|v|` is an endpoint's clipped speed (0 when its velocity is unknown). The
extension term lets a crab that is walking at either endpoint travel up to
`h` frames' worth of its speed; beyond that, only the diffusive radius grows.
With the defaults, `r = 6.59, 11.42, 18.65, 25.96, 36.42` px at
`dt = 2, 6, 16, 31, 61` (gaps 1, 5, 15, 30, 60), versus `24 … 732` px for the
linear gate.

**Gates.** `dist = |p_b − p_a| <= R`, and `err <= prediction_error_ratio · R`
(1.5), where `err` is the mean forward/backward prediction error with each
velocity extrapolated over `min(dt, h)` frames, not the full `dt`.

**Prefilter.** Because `r` and `min(dt, h)` do not decrease with `dt`, no pair
of a run can pass the displacement gate beyond

```text
R_max = r(max_gap + 1) + v_max · min(max_gap + 1, h)
```

where `v_max` is `max_velocity_px_per_frame` (6 px/frame), or the fastest
known endpoint speed if that is higher (only possible for endpoints built
outside the pipeline). With the defaults, `R_max` is 29.4 px at `max_gap` 5, 36.6 at 15,
44.0 at 30 and 54.4 at 60. Candidate pairs are enumerated per START frame
(the ENDs 2 to `max_gap + 1` frames earlier, one vectorised distance block),
and pairs with body distance above `R_max` are dropped before any per-pair
work. They get no `links.parquet` row and are counted in the marker's
`counts.candidate_pairs.pruned_by_radius`. Every dropped pair would have been
`refused_gate` with reason `displacement`, so links, costs and every other
audit row are unchanged. The linear model does not prune: its v001 audit table
keeps a `displacement` row for every pair in the time window.

**Cost** (dimensionless, a fraction of the radius):

```text
c = [(1 − w) · dist + w · err] / R · (1 + gap_penalty)^(gap − 1) + w_s · min(|u_a − u_b|, cap) / cap
```

`w = motion_weight` (0.3), `w_s = shoulder_vector_weight` (0.15),
`cap = shoulder_vector_cost_cap_px` (12), `gap = dt − 1`. A 1-frame gap has no
penalty; with `gap_penalty = 0.06` a 30-frame gap costs 5.4 times as much at
equal `dist / R`, which outweighs the larger radius (`r(31) / r(2) = 3.9`), so
at equal pixel distance a short gap wins.

**Thresholds** on this scale: `max_assignment_cost_sqrt = 1.0` (pairs above are
`refused_gate` with reason `max_cost`), `unmatched_cost_sqrt = 0.6`,
`minimum_cost_margin_sqrt = 0.25`. Assignment, margin veto, chain-aware
runner-up and the audit labels are those of the method above.

**Staged linking.** A single assignment over all gaps up to `max_gap` let a
cheaper long-gap candidate take an endpoint whose true partner was a few frames
away; the margin veto then refused the long pair, and the short link was never
made (on the demo, up to 50 short-gap links were lost at long horizons). Under
the sqrt model, linking therefore runs in stages of increasing gap: with
`link_stages = [5, 15, 30, 45]` (stages at or above `max_gap` are dropped and
`max_gap` is always the last stage), stage 1 assigns gaps 1–5, stage 2 gaps
6–15, and so on. Each stage solves the assignment and margin veto over its own
candidates whose END and START are both still free, fixes its links, and
removes the matched endpoints before the next stage. The END and START of a
pair refused by the margin veto are removed as well, so a vetoed endpoint is
not linkable at a later stage (the veto means "ambiguous; leave unlinked"). Without
this rule, a later stage computed the runner-up only over its own candidates,
so the earlier ambiguous alternatives were forgotten and a farther long-gap
candidate could link with "no competitor". Gates and costs do not
depend on the stage; chain reachability is still taken over all motion-gated
pairs up to `max_gap`. A gate-passing pair whose endpoint was linked in an
earlier stage is `refused_assignment` with reason `endpoint linked at stage s`
and that link's cost as runner-up. If instead an endpoint was vetoed at an
earlier stage, the reason is `endpoint vetoed at stage s` and the runner-up is
the vetoed pair's cost. The `stage` column of `links.parquet` gives each pair's
1-based stage. The linear model has a single stage.

| Parameter | Default | Meaning |
|---|---:|---|
| `gate_model` | `sqrt` | `sqrt` (v002) or `linear` (v001; reproduces earlier results exactly, fill included, pinned by a golden test) |
| `radius_r0_px` | 0.00 | Fitted intercept `r0` (px) |
| `radius_k_px` | 3.73 | Fitted `sqrt(dt)` coefficient `k` (px) |
| `radius_safety` | 1.25 | Multiplies the fitted radius |
| `radius_min_px` | 6.0 | Floor on `r(dt)` |
| `prediction_error_ratio` | 1.5 | Prediction gate `err <= ratio · R` |
| `prediction_horizon_frames` | 3 | Extrapolation cap `h`; also the fill damping horizon `H` (sqrt only) |
| `gap_penalty` | 0.06 | Cost factor `(1 + p)^(gap − 1)` |
| `max_assignment_cost_sqrt` | 1.0 | Pairs above are not assignable |
| `unmatched_cost_sqrt` | 0.6 | Cost of leaving an end or start unmatched |
| `minimum_cost_margin_sqrt` | 0.25 | Veto when the runner-up is within this (CLI `--margin` under sqrt) |
| `fill_confidence_gaps` | [5, 15] | `high` ≤ 5 < `medium` ≤ 15 < `low` |
| `link_stages` | [5, 15, 30, 45] | Upper gap of each linking stage (sqrt only); stages ≥ `max_gap` dropped, `max_gap` is the last stage |

**Fit.** `bridge_gaps.py fit` (`scripts/gap_bridging/fit.py`) takes every
unbroken accepted tracklet (consecutive, unique frames), pools the net shell
displacements `|p(f + dt) − p(f)|` over all frame pairs inside a tracklet, and
takes their 99th percentile `D(dt)` for `dt = 1 … 60` (a `dt` with fewer than
200 samples is dropped). It then fits `D(dt) = r0 + k · sqrt(dt)` by
non-negative least squares (`scipy.optimize.nnls`). On the demo clip:
`r0 = 0.00 px`, `k = 3.73 px`, RMSE 1.36 px over 60 values of `dt`
(`D(1) = 2.5`, `D(5) = 6.1`, `D(30) = 19.8`, `D(60) = 30.7` px; 35,338 samples
at `dt = 1`, 21,901 at `dt = 60`). These are the committed defaults. The fit
under-shoots the measured quantile slightly for `dt > 45`, where the samples
come from a few dozen long tracklets; `radius_safety` covers it
([radius figure](#horizon-sweep-results)). Rerun the fit per dataset:

```bash
.venv/bin/python scripts/bridge_gaps.py fit \
    --observations demo/crab_cam2_2_middle_300/observations.parquet \
    [--max-dt 60] [--quantile 0.99] [--min-samples 200] [--write-yaml radius.yaml]
```

`--write-yaml` writes `radius_r0_px` and `radius_k_px` (2 decimals) as a
config overlay that `--config` accepts.

## Damped fill

Under the sqrt gate model, for a bridge with `dt > H`
(`H = prediction_horizon_frames`), the Hermite tangents (method, step 7) are
pulled towards the chord velocity `c = (p_b − p_a) / dt`:

```text
λ = min(1, H / dt)
v_a' = c + λ · (v_a − c)        v_b' = c + λ · (v_b − c)
```

For `dt <= H` the tangents are unchanged (the v001 fill, bit for bit); as
`dt → ∞` the fill tends to the uniform-time chord; a constant-velocity crab
(`v_a = v_b = c`) is reproduced exactly at every `dt`. The deviation from the
uniform-time chord is bounded by

```text
|p(t) − (p_a + t (p_b − p_a))| <= (4/27) · min(dt, H) · (|v_a − c| + |v_b − c|)
```

independent of the gap length (for `H = 3`, `(4/9) · (|v_a − c| + |v_b − c|)`
px, e.g. 5.3 px when both tangents deviate from the chord by 6 px/frame), so a
long fill never overshoots by the endpoint speed times the gap length. The
chord fallback (unknown velocity) is unchanged. Under `gate_model: linear` the
tangents are never damped, so the fill is the v001 fill bit for bit at every
`dt`. Damping both models made linear runs differ from v001 by up to 1.2 px
at `max_gap` 5 and 27.8 px at 30.

## Gap length and fill confidence columns

Recovered and interpolated rows carry `gap_frames` (the bridge's
missing-frame count, recovered frames included, so all rows of one bridge have
the same value) and `fill_confidence` (`high` for gap ≤ 5, `medium` for
≤ 15, `low` above; `fill_confidence_gaps`). Observed rows and unassigned
rejected rows have nulls. Filter on these columns to keep only short fills,
e.g. `fill_confidence == 'high'` keeps only fills of at most 5 frames.

## Hold-out evaluation

`scripts/gap_bridging/holdout.py` deletes runs of `g` consecutive frames from
the interior of real tracklets and splits each tracklet in two, exactly as the
adjacent-frame tracker would have done had the detections been missed:

- Eligible tracklets have consecutive, unique accepted frames (no real gap
  inside) and at least `max(40, g + 6)` frames; at least 3 frames are kept on
  each side of the deleted run. A tracklet is planted at most once.
- The draw is deterministic in `--seed`. For the gap lengths of one call the
  plants of the shorter lengths do not depend on the longer ones, so the
  sweep's horizons share their short-gap plants.
- The planted table is bridged with `pipeline.run` and scored from
  `links.parquet`: a plant is `recovered` when its left half is linked to its
  right half, `false_link` when either half is linked to anything else, else
  `missed`. `recall = recovered / planted`,
  `precision = recovered / (recovered + false_link)` (NaN for 0/0).

Caveats:

- Plants come from long, unbroken tracklets, i.e. crabs the detector tracks
  well, and these are mostly stationary or slow. Recall and precision are
  therefore optimistic for walking crabs and for crabs in crowded spots.
- A false link of an unplanted tracklet is not seen; precision covers only the
  planted halves. `holdout_unplanted_links` counts the links among the rest.
- 8 plants per gap length is thin: a per-gap precision of 1.0 from 8 plants
  has a one-sided 95% lower bound of 0.69, and from 40 plants of 0.93.
- The demo clip has 156 eligible tracklets, so several seeds reuse
  tracklets; seeds are not independent samples.

## Horizon sweep

`bridge_gaps.py sweep` runs the pipeline at several horizons and, with
`--holdout`, the hold-out at each (every planted gap length at every horizon
since the v003 fix wave; before, they were capped at the horizon), then
writes the tables and figures:

```bash
.venv/bin/python scripts/bridge_gaps.py sweep \
    --observations demo/crab_cam2_2_middle_300/observations.parquet \
    --out logs/gap_sweep --max-gaps 5,10,15,30,45,60 \
    --holdout [--holdout-gaps 1,2,3,5,10,15,20,30,45,60] [--per-gap 8] [--seed 0] \
    [--decoys 4] [--flicker 2] \
    [--config YAML] [--video MP4] [--bridges-from N] [--no-figures] [--force]
```

Outputs under `--out`:

- `max_gap_<N>/`: pipeline outputs per horizon; `holdout_<N>/`: hold-out
  outputs (`planted_observations.parquet`, `holdout_truth.parquet`,
  `bridged/`, `holdout_scores.csv`, `holdout_<N>.json`).
- `sweep_summary.csv`: one row per horizon: `max_gap`, `links`,
  `trajectories`, `mean_length`, `median_length`, `p90_length`, `units_len1`,
  `frac_frames_in_units_ge_30`, `recovered`, `interpolated`,
  `interpolated_{high,medium,low}`, `decision_<label>` for the four labels,
  `n_recovered_tail`, `claw_side_refusals` (`refused_gate` rows with reason
  `claw_side`), `spurious`, `identity_starts`, `identity_starts_filtered`,
  `units_len1_filtered`, `mean_length_filtered`, `median_length_filtered`
  (the `_filtered` statistics over `identity` trajectories only),
  `holdout_planted`, `holdout_precision`, `holdout_recall`,
  `holdout_unplanted_links`, `holdout_decoy_planted`,
  `holdout_decoy_false_link`, `holdout_flicker_planted`,
  `holdout_flicker_spurious`. Lengths are `end_frame − start_frame + 1` of the
  trajectories (interpolated and tail frames included).
- `holdout_by_gap.csv`: long form `max_gap, gap, planted, recovered,
  false_link, missed, recall, precision, decoy_planted, decoy_recovered,
  decoy_false_link, flicker_planted, flicker_recovered, flicker_spurious`
  ([Decoys and flicker singletons](#decoys-and-flicker-singletons)).
- `figures/`: `sweep.pdf` (links and trajectory length vs horizon; dashed:
  identities only), `start_curve.pdf` (cumulative identity starts over the
  clip per horizon, all and identities only, and the input tracklets),
  `holdout.pdf` (precision and recall vs planted gap, one line per horizon),
  `radius_fit.pdf` (measured 99th percentile, fit, configured `r(dt)`, legacy
  `12 dt`), each with a PNG preview, and `bridges_max_gap_<N>_p<k>.png`
  (20 rendered bridges with gap ≥ 10, five per page, cyan observed, yellow
  recovered, magenta interpolated). The video defaults to the single `*.mp4`
  next to a file input; `--bridges-from` picks the horizon (default the
  largest).

The full sweep with hold-out on the demo takes 40 s on a login node (3.0 GB
peak RSS, most of it from rendering the bridge figures).

### Horizon sweep results

Demo clip, default config (sqrt gate, demo fit, staged linking
`link_stages = [5, 15, 30, 45]`), `--per-gap 8 --seed 0` (`logs/gap_sweep/`,
not committed):

| `max_gap` | links | trajectories | mean L | median L | p90 L | 1-frame units | frames in units ≥ 30 | recovered | interpolated (high / medium / low) | refused_margin | refused_assignment | refused_gate | hold-out planted | precision | recall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | 638 | 543 | 69.6 | 13 | 300 | 116 | 0.933 | 19 | 1,273 (1,273 / 0 / 0) | 1 | 118 | 67 | 32 | 1.000 | 1.000 |
| 10 | 739 | 442 | 87.3 | 21 | 300 | 82 | 0.952 | 19 | 2,041 (1,273 / 768 / 0) | 1 | 422 | 118 | 40 | 1.000 | 1.000 |
| **15** | **781** | **400** | **97.8** | **28** | **300** | **71** | **0.959** | **19** | **2,576 (1,273 / 1,303 / 0)** | **1** | **711** | **181** | **48** | **1.000** | **1.000** |
| 30 | 827 | 354 | 113.3 | 52 | 300 | 63 | 0.974 | 19 | 3,566 (1,273 / 1,303 / 990) | 1 | 1,522 | 366 | 64 | 1.000 | 0.969 |
| 45 | 843 | 338 | 120.4 | 61 | 300 | 59 | 0.978 | 21 | 4,158 (1,273 / 1,303 / 1,582) | 1 | 2,163 | 615 | 72 | 1.000 | 0.944 |
| 60 | 849 | 332 | 123.5 | 63.5 | 300 | 56 | 0.979 | 21 | 4,461 (1,273 / 1,303 / 1,885) | 1 | 2,467 | 1,104 | 80 | 1.000 | 0.863 |

`refused_gate` counts only the pairs inside the prefilter radius `R_max`
([Gate model v002](#gate-model-v002)). Of these, the `displacement` and
`max_cost` failures are 63/4, 105/13, 148/33, 183/183, 212/403 and 255/849.
Another 18,480 / 36,535 / 54,445 / 106,012 / 154,002 / 197,466 pairs lie beyond
`R_max`. They are not enumerated and appear only in the marker's
`counts.candidate_pairs.pruned_by_radius`. Before the prefilter, `refused_gate`
was 18,547 / 36,653 / 54,626 / 106,378 / 154,617 / 198,570. Links,
trajectories, every other decision count and the hold-out are unchanged by the
prefilter and by persisting margin vetoes across stages. For comparison, the
v001 linear gate at `max_gap` 5 gives 654 links and 527 trajectories
([Demo results](#demo-results)).

With staged linking, a horizon only adds links: the 638 links at gaps 1–5 are
the same at every horizon, and the high-confidence fill count stays at 1,273.
Links by gap at `max_gap` 60: 638 at gaps 1–5, 101 at 6–10, 42 at 11–15, 46 at
16–30, 16 at 31–45, 6 at 46–60; per stage 638 / 143 / 46 / 16 / 6. Links that
ignored at least one chain-reachable competitor: 198 at `max_gap` 5, 202 at
15, 30 and 60. The one stage-1 veto removes its endpoints from later stages,
so a competitor that used to count as chain-ignored for one stage-3 link is no
longer in that stage (it was 203 at 30 and 60 before vetoes persisted). Before staged linking (one assignment over all gaps)
the same sweep gave 755 links at `max_gap` 15 and 803 at 60, lost up to 50
short-gap links to longer competitors that were then margin-vetoed
(`refused_margin` 48 at `max_gap` 30, now 1), and the chain-ignored count grew
to 763.

Hold-out per planted gap (seed 0, 8 plants each). Precision is 1.000 for every
gap at every horizon (no false link); recall:

| planted gap | 1 | 2 | 3 | 5 | 10 | 15 | 20 | 30 | 45 | 60 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recall (every horizon ≥ gap) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.875 | 0.875 | 0.750 | 0.125 |

Because seed 0 alone shows no false link anywhere, it cannot separate the
horizons. Four more seeds (1–4, same settings, `logs/gap_sweep_seeds/`) were
pooled with seed 0, 40 plants per gap and horizon; the values are the same at
every horizon that covers the gap:

| planted gap | 1–10 | 15 | 20 | 30 | 45 | 60 |
|---|---:|---:|---:|---:|---:|---:|
| recall | 1.000 | 0.975 | 0.850 | 0.750 | 0.675 | 0.375 |
| precision | 1.000 | 1.000 | 1.000 | 0.968 (1 false link) | 1.000 | 1.000 |

| `max_gap` | plants | pooled precision | pooled recall |
|---:|---:|---:|---:|
| 5 | 160 | 1.000 | 1.000 |
| 10 | 200 | 1.000 | 1.000 |
| 15 | 240 | 1.000 | 0.996 |
| 30 | 320 | 0.997 | 0.947 |
| 45 | 360 | 0.997 | 0.917 |
| 60 | 400 | 0.997 | 0.863 |

The pooled hold-out counts per gap are identical to those measured before
staged linking, so the planted pairs were not among the short-gap links that
the single assignment had lost. The one false link (seed 1, gap 30) is another tracklet that ended
21 frames before the planted right half and was linked to it in stage 3 at
cost 0.93, while the true pair (also stage 3) cost 1.22 and failed `max_cost`
(not inspected in the video: a neighbour or a duplicate detection of the same
crab).

Figures: `logs/gap_sweep/figures/sweep.pdf`, `holdout.pdf`,
`radius_fit.pdf` and `bridges_max_gap_60_p1.png` … `_p4.png`. In the 20
rendered bridges (gaps 10–40, inspected by eye at crop scale), the crab is
visible and stationary throughout the gap in every row, often at a burrow
mouth or half inside it, and the interpolated shell stays on it: the long
losses on the demo are detector misses on still crabs, not crabs walking away.

### Recommended default

*Superseded by [Default horizon (v003)](#default-horizon-v003): with the
appearance cues and the decoy hold-out the default is now 30. The v002
reasoning is kept below.*

**`max_gap = 15`** (0.5 s), unchanged by staged linking. The rule: the
highest horizon whose pooled hold-out precision is at least 0.95 and whose
precision does not decline at the longest planted gaps. On seed 0 alone every
horizon passes (precision 1.000 throughout), which only shows that 8 plants
per gap cannot resolve the question. With the five seeds pooled, precision
first drops below 1 at a planted gap of 30 (0.968), so every horizon ≥ 30
shows a decline at its longest gaps; 15 is the highest horizon with precision
1.000 at every planted gap (240 plants, recall 0.996). Because the hold-out is
unchanged, so is the decision.

What staged linking did change is the price of stopping at 15: going from 5
to 15 now adds 143 links (+22%) and raises the median trajectory length from
13 to 28 frames, and going from 15 to 30 adds 46 links and raises the median
to 52 frames (before: +21 links, median 24.5 → 28), at the cost of 990
low-confidence interpolated rows and the first hold-out false link. Beyond 30
the gain is small (+22 links from 30 to 60) and hold-out recall at gap 60 is
0.375. 15 keeps every fill in the `high`/`medium` classes and well below the
1–2 s at which burrow retreats begin; 30 is the candidate to re-test on the
full dataset, where the hold-out has enough plants to measure precision at
gaps 20–30. The YAML default stays at 5 until this is confirmed there (rerun
`fit`, then the sweep with `--holdout`).

### Residual risks

- **Burrow clusters.** Neighbouring crabs at adjacent burrows sit within a few
  radii of each other; the only hold-out false link may be of this kind. The
  chain-aware runner-up rule leaves some competitors out of the margin veto
  (about 200 links at every horizon with staged linking), so a real ambiguity
  can go unvetoed.
- **Crabs that keep walking beyond the horizon.** The sqrt radius admits at
  most `h = 3` frames of directed motion; a crab that walks steadily through a
  long loss is refused (missed, not mislinked). Hold-out plants are mostly
  still crabs, so recall for walking crabs is not measured.
- **Thin statistics beyond `dt` 30 on the demo.** The fit and the hold-out
  rest on one 300-frame clip; above `dt` 30 the displacement samples come from
  few long tracklets and the measured quantile exceeds the fit for
  `dt > 45`. Refit and resweep on the full dataset before adopting a horizon
  above 15.
- **Interpolation inside burrow retreats.** A crab that retreats into its
  burrow and re-emerges at the same spot within the horizon is bridged; the
  interpolated pose is then a placeholder for an invisible crab. Filter on
  `fill_confidence` or `gap_frames` when that matters.

## v003: tail recovery, identity filter, appearance cues

v003 adds three remedies for the excess identity starts left after v002 (at
`max_gap` 15: 400 trajectories, 277 starts after frame 0, of which 70 were
one-frame units with median instance score 0.33):

- tail recovery: miss fewer detections at trajectory ends;
- a flag-only identity filter: count and report without spurious one-frame units;
- appearance cues: refuse links between visibly different crabs.

It also adds decoys and flicker singletons to the hold-out and re-derives the
default horizon (now **30**, see [Default horizon (v003)](#default-horizon-v003)).
Every v003 stage can be switched off from the config (`tail_recovery`,
`appearance_cues`), and the identity filter only adds a column. With
`gate_model: linear` the cues are ignored, so the linear golden test (D-003)
still passes; its interpolated rows and audit table are unchanged.

### Tail recovery

`scripts/gap_bridging/tails.py`, run after the gap fill and recovery (method,
step 11). This is a continuation-only relaxation of the tracker's Shell-score
gate. Trajectories are processed in (video, start frame, trajectory) order:
first the backward tail from the START endpoint, then the forward tail from
the END.

- Anchor `p` = the endpoint's Shell position; velocity `v` = the endpoint
  velocity (0 when unknown). Step `j = 1 .. tail_max_frames` probes frame
  `start − j` (or `end + j`) and predicts `p ∓ v`. The anchor then moves to
  the adopted row, so the extrapolation is always one frame (`dt = 1`) and
  `prediction_horizon_frames` never shortens it.
- The candidates at that frame are rejected rows with a `rejection_reason` in
  `tail_reasons` (`low_body_score`) and a finite body position whose detection
  is not already adopted by a bridge or another tail. A candidate within
  `tail_exclusion_px` = 5 px (the tracker's duplicate-Shell radius) of a row
  owned by another trajectory at that frame is excluded. Owned rows are
  accepted rows, bridge recoveries and tails adopted so far.
- The nearest remaining candidate within `tail_radius_px` = 8 px is adopted
  (ties by `detection_id`). The tail stops at the first frame without an
  adoption, after `tail_max_frames`, or at the video's first or last frame.
- Adopted rows keep every original value and get `source='recovered_tail'`,
  `bridge_id = <trajectory_id>/tail_back` or `/tail_fwd`, `gap_frames` = the
  tail's length and `fill_confidence` from that length.

Tails run after linking, so they never change `links.parquet`
(`TailSceneTests`, `HypothesisTailTests`). Demo yield: 3 rows at `max_gap` 15
and 1 at 45. Of the 26 `low_body_score` rows still unassigned after bridging,
only these lie within 8 px of a trajectory-end prediction. The rest lie 100 to
1,300 px from every trajectory end, e.g. a run at frames 14326–14335 about
300 px from any accepted row. They are isolated low-score objects, not missed
tails. Unassigned rejected rows: 32 at both horizons (9 of them
`missing_position`, never recoverable).

### Identity filter

`scripts/gap_bridging/identity.py`, run on the final trajectories (method,
step 12). The real frames of a trajectory are its rows with `source` in
{`observed`, `recovered`, `recovered_tail`}. A trajectory is `spurious` when
it has fewer than `min_identity_frames` = 2 real frames **and** the median
`instance_score` over them is below `min_identity_score` = 0.6. A trajectory
without any scored real row counts as low-score. Every other trajectory is an
`identity`. The class is written to both tables as `trajectory_status`.

The filter is a flag only, for three reasons:

- Nothing is deleted, so row counts and every v002 invariant are unchanged.
- The flag is computed after linking, so a one-frame tracklet bridged into a
  chain is an identity, and the flag cannot affect linking.
- The 0.6 threshold is tuned on one clip's bimodal scores (one-frame units:
  median 0.33; long trajectories: 0.86). A consumer who disagrees can
  recompute it from the published columns.

Filter with `trajectory_status == 'identity'` to drop spurious units.

The sweep reports both variants of every count. On the demo:

| `max_gap` | trajectories | spurious | starts after frame 0 (all / identities) | 1-frame units (all / identities) | mean L (all / identities) | median L (all / identities) |
|---:|---:|---:|---|---|---|---|
| 5 | 542 | 95 | 419 / 326 | 115 / 20 | 69.8 / 84.4 | 13 / 20 |
| 10 | 441 | 61 | 318 / 258 | 81 / 20 | 87.5 / 101.4 | 21 / 31.5 |
| 15 | 399 | 52 | 276 / 225 | 70 / 18 | 98.1 / 112.6 | 28 / 40 |
| **30** | **354** | **47** | **231 / 185** | **63 / 16** | **113.2 / 130.4** | **51.5 / 78** |
| 45 | 339 | 44 | 216 / 173 | 59 / 15 | 119.9 / 137.6 | 61 / 100 |
| 60 | 333 | 41 | 210 / 170 | 56 / 15 | 123.0 / 140.1 | 64 / 105 |

Every spurious unit is one frame long. The one-frame units left as
identities have scores of 0.6 or more (18 at `max_gap` 15).

`figures/start_curve.pdf` plots the cumulative number of starts against the
frame of the clip, one colour per horizon (solid: all trajectories; dashed:
identities only), and the 1,058 starts of the input tracklets (dotted).
At every horizon the start rate stays roughly constant over the clip, both
with and without the spurious units. The remaining starts are therefore
neither a transient at the clip start nor concentrated in a few frames.

Interaction with tail recovery: a tail adds a real frame. A one-frame
low-score unit that gains a one-frame tail (two low-score detections at the
same spot) becomes a two-frame identity. On the demo this happens once at
`max_gap` 15 (`trajectory0000212`, median score 0.33), which is why the
spurious count is 52 with tails and 53 without (identity-only starts 225
and 224).

### Appearance cues

`geometry.tracklet_cues` measures two whole-tracklet statistics, which are
stored identically on the START and END endpoints:

- **Size.** `L` = the median Shell–Shoulder length `|u|`, `u = shoulder −
  shell`. Only frames with Shell and Shoulder valid and `|u|` in [10, 65] px
  count; `n` is their number. On the demo the within-trajectory std is
  0.94 px and the between-trajectory std 4.9 px.
- **Claw side.** `s` = the mean over valid frames of
  `sign(u_x w_y − u_y w_x)`, with `w = claw − shell` in image coordinates (y
  down); `n_s` is the number of frames. Valid frames are size frames with the
  Claw also valid and a non-zero sign. The sign convention: a claw at body
  offset (40, −14) with the shoulder at (22, 0) (the fixtures'
  `DEFAULT_OFFSETS`) gives −1; its mirror image gives +1. The statistic needs
  no heading, and it is invariant under rotation and translation and flips
  under reflection. On the demo it is consistent in 96% of long trajectories,
  with a 48/52 split across crabs. That agreement is measured inside
  tracklets, where the tracker already enforces Shoulder continuity (the final
  review counted 31 Shell→Shoulder reversals in 35,301 consecutive frames), so
  it overstates how reliable the sign is across a gap (see the axis-flip rule
  below). The project docs list the Claw as an untrusted keypoint and not an
  association feature (`docs/PREDICTION_V007_BEST.md`); v003 departs from that
  deliberately, as a soft cost term plus a veto guarded by the axis-flip rule.

Under the sqrt model (`appearance_cues: true`), each candidate pair (END a,
START b) gets two terms:

```text
size_term = min(3, |L_a − L_b| / sqrt(3² + 1²/n_a + 1²/n_b))      0 unless n_a, n_b > 0
side_term = max(0, −s_a s_b) · min(1, n_s,a/5) · min(1, n_s,b/5)   0 unless n_s,a, n_s,b > 0
cost     += 0.1 · size_term + 0.15 · side_term
```

The parameters are `size_scale_px`, `size_noise_px`, `size_cost_cap`,
`claw_side_min_frames`, `size_weight` and `claw_side_weight`. The
confidence factor `min(1, n_s/5)` keeps a side measured on one to four frames
from carrying the full weight. Both terms are symmetric in a and b, and both
are zero when a cue is unknown. They are added after the gap-penalised motion
cost, like the Shoulder term. The audit table records them (`size_term`,
`side_term`).

**Veto.** A pair is refused when both sides are confident and opposite:
`|s_a|, |s_b| ≥ claw_side_confidence` (0.9),
`n_s,a, n_s,b ≥ claw_side_min_frames` (5) and `s_a s_b < 0`. The refusal is
`refused_gate` with reason `claw_side` and cost NaN. The veto ranks after the
`displacement` and `prediction_error` gates and before `max_cost`. A vetoed
pair is neither assignable nor chain-reachable, and it is no runner-up.

**Axis-flip rule** (`candidates._axis_flip`, fix wave I1). The side sign
`sign(u × w)` also flips when the Shoulder keypoint lands on the other end of
the body axis while the claw stays where it was. Per endpoint, `u = (cos h,
sin h)` is the image-frame Shell→Shoulder unit vector (`h` =
`Endpoint.heading`) and `c` the image-frame Shell→Claw unit vector
(`R(h)·offsets[claw]`, normalised). When `u_a·u_b < 0` and `c_a·c_b > 0` (axis
reversed, claw in place) the pair is treated as a Shoulder flip of one crab:
its `side_term` is 0 and it is not vetoed. A pair that passed the motion gates
and would otherwise have been vetoed keeps its ordinary decision with
`; axis_flip` appended to its reason, and the marker counts it as
`counts.claw_side_suppressed`. An unknown heading or claw offset means no
flip. A genuine 180° turn (axis **and** claw reversed) and a genuine
opposite-claw pair (axes aligned) keep the ordinary rule.

On the demo the rule removes **every** veto: `claw_side` refusals were
1 / 1 / 3 / 11 / 18 / 25 at horizons 5 / 10 / 15 / 30 / 45 / 60 and are now
0, with `claw_side_suppressed` equal to the old count at every horizon. The
earlier 18 vetoes at `max_gap` 45 were therefore Shoulder flips, not
different-handed crabs: all had antiparallel Shell→Shoulder and parallel
Shell→Claw vectors at the gap boundary. For example, 41951 → 42378 (gap 5,
shells 4 px apart) is one crab that the pipeline chains anyway, through the
one-frame tracklet 42365. With the rule, 16 of the 18 pairs are
`refused_gate max_cost` and 2 are `refused_assignment` (41951 → 42378: the
assignment chose another pair; 42346 → 42491: endpoint linked at stage 1).
The link set is identical with and without the rule at 30, 45 and 60 (827,
842 and 848 links); the side term drops to 0 on 217, 308 and 406 audit rows
without changing any other decision. The three former vetoes at `max_gap` 15
(tracklet ids `…/t0000NNNNN`):

| pair | gap | size term | outcome now (axis flip) |
|---|---:|---:|---|
| 41951 → 42378 | 5 | 1.74 | `refused_assignment` |
| 42409 → 42525 | 14 | 0.66 | `refused_gate max_cost` |
| 42600 → 42661 | 15 | 0.41 | `refused_gate max_cost` |

The rule cannot tell a Shoulder flip from two different-handed crabs that
face opposite ways with their claws pointing the same way in the image. The
hold-out decoys show this: decoy donors are different crabs chosen for
differing cues, and at horizon 30 the rule lifts the veto on 56 of the 94
target–donor pairs the old veto refused. No decoy was linked (the size term
and the margin veto still refuse them), but 7 decoy targets at planted gaps
≤ 30 that the veto used to recover became margin-refused misses (2 each at
gaps 3 and 5, 1 at gap 20, 2 at gap 30). The veto now fires only on
axis-aligned opposite-claw pairs; none occurs on the demo.

Overall, the cues change the demo at `max_gap` 15 from 781 links / 400
trajectories (cues off) to 782 / 399, and at `max_gap` 45 from 843 / 338 to
842 / 339 (soft terms only). On the v002 hold-out draw (no decoys, seeds 0–4),
the cues remove the one false link of v002 (seed 1, planted gap 30), which
becomes a miss. Re-run with the axis-flip rule and every gap scored at every
horizon (scratch), this draw still has precision 1.000 at every planted gap
at horizons 30, 45 and 60 (304, 331 and 346 links).

### Decoys and flicker singletons

`holdout.plant_gaps(..., decoys, decoy_offset_px=3.0, flicker)` and
`sweep --decoys N --flicker M` add two kinds of plants to the hold-out. Both
are drawn per gap length, after that length's ordinary plants and from the
tracklets not yet used. The `role` column of `holdout_truth.parquet` records
them.

- **Decoy pair** (`decoy_target` C, `decoy_donor` A). C is drawn like a
  plant. The same run of `g` frames is deleted in C and in A, so both right
  halves start on the same frame `s`. A is any unused contiguous tracklet of
  the same video that covers 3 frames on each side of the run. A's right half
  must differ in appearance from C's left half: `|ΔL| ≥ 2 · size_scale_px`,
  or confidently opposite claw sides (the veto condition). Donors that are not
  eligible as plants are preferred. Every coordinate column (`*_x`, `*_y`:
  body, nodes, centroid) of A's right half is translated by
  `shell_C(s) − shell_A(s) + (decoy_offset_px, 0)`. A's right half thus starts
  3 px from C's right half and competes with it for C's left end. Translation
  preserves both cues.
- **Scoring.** Targets are scored like plants and count in recall and
  precision. A target whose left half links to the donor's right half is a
  `false_link` and also a `decoy_false_link`. Donors are excluded.
- **Flicker singleton.** `g` frames are deleted on each side of one kept frame
  of a tracklet with at least `max(40, 2g + 7)` frames. This leaves a left
  part, a one-frame middle tracklet and a right part. A flicker is
  `flicker_recovered` when all three parts end in one trajectory, and
  `flicker_spurious` when the middle's trajectory is flagged spurious.
- **Pool limits.** Plants are mandatory: the run raises an error when the
  pool is short. Decoys and flicker singletons are best effort. On the demo
  the 156 eligible tracklets cannot supply the full 4 + 2 at planted gap 60,
  where 7 of 20 decoys and 4 of 10 flickers were planted over the five seeds;
  every shorter gap got the full count. `decoy_planted` and `flicker_planted`
  in the scores show what was planted.
- **Reproducibility.** With `decoys = flicker = 0` the draw is the v002 plant
  draw. With decoys or flicker the plants of each gap differ from v002's,
  because the extras share the generator. The draws of shorter gaps still never
  depend on the longer ones, so a sweep's horizons share their short-gap
  plants, decoys and flickers.

Decoys test the cues, not motion plausibility: the donor's right half is real
motion translated next to the target. On straight synthetic crabs the motion
model alone already refuses a 3-px decoy, by the margin veto. Cues-off then
loses recall but not precision. `DecoyVetoExampleTests` therefore moves the
target's right half 6 px off its prediction and puts the decoy on it. Without
cues the decoy is linked; with cues the pair is vetoed. On the demo,
flicker singletons come from long, high-score tracklets, so they cannot be
spurious whether linked or not. The meaningful check is
`FlickerPropertyTests`, with all scores 0.3: every flicker with
`g ≤ max_gap` is re-linked and none is spurious, while a flicker beyond the
horizon is.

### Weak-label export and re-inference scaffold

The detector misses that gap bridging interpolates over can only be fixed
upstream. v003 ships two ready-to-run but, in this checkout, untested pieces
for that (no checkpoints, raw videos or GPU here):

- `bridge_gaps.py export --bridged DIR --out DIR [--frame-offset N]
  [--max-gap-frames N] [--crop-size 256] [--confidences high medium]`
  (`gap_bridging/export.py`) writes the interpolated rows of a bridging run as
  `weak_labels.parquet`, plus a README and a completion marker. The table
  holds the video and frame, a `local_frame` index into the video file, the
  trajectory and bridge, `gap_frames`, `fill_confidence`, all node
  coordinates, a crop window clipped to 7680×4320 and
  `label_status = 'weak_interpolated'`.
  These are **fine-tuning candidates, not labels**: every pose is a rigid
  Hermite interpolation that has to be reviewed before training a centroid
  model on it. The export refuses a run whose `observations.parquet` does not
  match its marker. It does not filter on `trajectory_status`; with
  `min_identity_frames = 2` no bridged trajectory can be spurious.
- `scripts/reinfer_low_threshold.py` and `jobs/reinfer_low_threshold.sh`
  re-run SLEAP inference on a frame range with a lower centroid threshold
  (default 0.05) and keypoint threshold (0.2). They write an untracked
  `<video>.frames_<start>_<stop-1>.predictions.slp`, `raw_candidates.npz` and
  `RUN_MANIFEST.json` (checkpoint hashes, thresholds, versions). `--dry-run`
  validates the arguments and paths without importing torch; it exits 2 when
  checkpoints are missing, as they are here. The job takes its root from
  `SLURM_SUBMIT_DIR`, requests one 80 GB GPU and fails on a node without one.
  **Caveat:** the SLP keeps the video's frame numbers (`start..stop−1`), as
  the SLP validator requires. `prediction_finalize.derive_tracks` assumes
  frames `0..n−1`, so it can consume the output unchanged only when
  `start == 0` (e.g. a range cut into its own clip, as the demo was); a range
  of a full video needs renumbering first. Centroid thresholds below 0.05 are
  refused because `unpack_output` asserts scores ≥ 0.05.

### v003 results

Demo clip, default config except `max_gap`. The commands (fix wave):

```bash
.venv/bin/python scripts/bridge_gaps.py sweep \
    --observations demo/crab_cam2_2_middle_300/observations.parquet \
    --out logs/gap_sweep --max-gaps 5,10,15,30,45,60 --holdout \
    --per-gap 8 --decoys 4 --flicker 2 --force          # seed 0, with figures
# seeds k = 1..4: the same with --seed k --no-figures --out logs/gap_sweep_seeds/seed_k
.venv/bin/python scripts/bridge_gaps.py pool \
    --sweeps logs/gap_sweep logs/gap_sweep_seeds/seed_1 logs/gap_sweep_seeds/seed_2 \
             logs/gap_sweep_seeds/seed_3 logs/gap_sweep_seeds/seed_4 \
    --out logs/gap_sweep_seeds --rule-precision 1.0 --force
```

`pool` (`holdout.pool_sweeps`, `holdout.apply_rule`) writes
`logs/gap_sweep_seeds/pooled_by_gap.csv` and `pooled_by_horizon.csv` (not
committed) and prints the per-horizon table and the chosen horizon. The
`refused_gate` column has no `claw_side` refusal left; the bracket gives the
pairs whose veto the axis-flip rule lifted (`claw_side_suppressed`):

| `max_gap` | links | trajectories | mean L | median L | 1-frame units | frames in units ≥ 30 | recovered | tail | interpolated (low) | refused_assignment | refused_gate (axis flips lifted) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| 5 | 639 | 542 | 69.8 | 13 | 115 | 0.934 | 19 | 3 | 1,278 (0) | 118 | 67 (1) |
| 10 | 740 | 441 | 87.5 | 21 | 81 | 0.952 | 19 | 3 | 2,046 (0) | 421 | 119 (1) |
| 15 | 782 | 399 | 98.1 | 28 | 70 | 0.959 | 19 | 3 | 2,581 (0) | 708 | 184 (3) |
| **30** | **827** | **354** | **113.2** | **51.5** | **63** | **0.974** | **19** | **3** | **3,546 (965)** | **1,514** | **375 (11)** |
| 45 | 842 | 339 | 119.9 | 61 | 59 | 0.978 | 21 | 1 | 4,100 (1,519) | 2,144 | 636 (18) |
| 60 | 848 | 333 | 123.0 | 64 | 56 | 0.979 | 21 | 1 | 4,403 (1,822) | 2,438 | 1,135 (25) |

Links, trajectories and every length statistic are unchanged by the
axis-flip rule; only the former `claw_side` rows moved to `max_cost` or
`refused_assignment`. `refused_margin` is 0 at every horizon (v002: 1).

**Hold-out, every gap scored at every horizon** (fix wave C1). The sweep now
plants all ten gap lengths at every horizon. A plant longer than the horizon
is scored like any other: recovered only if its halves are linked to each
other, a false link if either half is linked to anything else, otherwise a
miss. Pooled over seeds 0–4, each gap length has 40 plants plus 20 decoy
targets (60 scored pairs) and 10 flicker singletons; at planted gap 60 the
pool limit gives 47 scored pairs, 7 decoys and 4 flickers. For a gap within
the horizon the counts are the same at every horizon that covers it.
"links" counts the recovered plus falsely linked pairs, the denominator of
precision; "lower 95%" is the one-sided exact (Clopper–Pearson) bound
computed by `pool` from those links:

| planted gap | 1 | 2 | 3 | 5 | 10 | 15 | 20 | 30 | 45 | 60 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recall (horizon ≥ gap) | 1.000 | 0.983 | 0.950 | 0.950 | 1.000 | 0.950 | 0.883 | 0.750 | 0.567 | 0.170 |
| links (of which decoy targets) | 60 (20) | 59 (19) | 57 (18) | 57 (17) | 60 (20) | 57 (19) | 53 (16) | 45 (14) | 34 (10) | 9 (0) |
| precision (horizon ≥ gap) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.889 |
| lower 95% (all links) | 0.951 | 0.950 | 0.949 | 0.949 | 0.951 | 0.949 | 0.945 | 0.936 | 0.916 | 0.571 |
| lower 95% (plants only) | 0.928 | 0.928 | 0.926 | 0.928 | 0.928 | 0.924 | 0.922 | 0.908 | 0.883 | – |
| decoys recovered / planted | 20/20 | 19/20 | 18/20 | 17/20 | 20/20 | 19/20 | 16/20 | 14/20 | 10/20 | 0/7 |
| decoy false links | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| flickers recovered / planted | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 9/10 | 9/10 | 6/10 | 5/10 | 1/4 |

Plants beyond the horizon: under every horizon up to 30 none of the 107
scored pairs at gaps 45 and 60 was linked to anything (all misses). Under
horizon 45, one plant at gap 60 was linked to another crab (precision 0 at
gap 60, 1 link). The only flicker flagged spurious is one at gap 30 under
horizons 5–15 (beyond the horizon, as expected). Per horizon:

| `max_gap` | scored pairs | links | false links | pooled precision (lower 95%) | pooled recall | decoy false links | precision 1.0 at every gap |
|---:|---:|---:|---:|---|---:|---:|:---:|
| 5 | 587 | 233 | 0 | 1.000 (0.987) | 0.397 | 0 | yes |
| 10 | 587 | 293 | 0 | 1.000 (0.990) | 0.499 | 0 | yes |
| 15 | 587 | 350 | 0 | 1.000 (0.991) | 0.596 | 0 | yes |
| **30** | **587** | **448** | **0** | **1.000 (0.993)** | **0.763** | **0** | **yes** |
| 45 | 587 | 483 | 1 | 0.998 (0.990) | 0.821 | 0 | no (gap 60: 0.000) |
| 60 | 587 | 491 | 1 | 0.998 (0.990) | 0.835 | 0 | no (gap 60: 0.889) |

Recall per horizon now counts every plant beyond the horizon as a miss, so it
is not comparable with the v003 values before the fix wave.

No decoy was ever linked. The one false link (seed 0, plant
`t000042944`, planted gap 60) is a stage-4 link at gap 38 from the plant's
left half to another crab's tracklet `t000043141`. It is the same link under
horizons 45 and 60. Its cost is 0.86: the size term 2.27 corresponds to about
7 px of carapace difference, and the claw sides were unknown. The true pair
(gap 60, cost 6.6) failed `max_cost`. A size difference that large adds only
0.23 to the cost, so the size cue alone does not refuse it.

Figures: `logs/gap_sweep/figures/sweep.pdf` (dashed: identities only),
`start_curve.pdf`, `holdout.pdf`, `radius_fit.pdf` and
`bridges_max_gap_60_p*.png`.

### Default horizon (v003)

**`max_gap = 30`** (1.0 s), in `BridgeConfig` and
`configs/tracking/gap_bridging_v001.yaml`. The rule
(`holdout.apply_rule`, `bridge_gaps.py pool --rule-precision 1.0`): the
highest horizon whose pooled hold-out precision is 1.0 at every planted gap,
with decoy false links counted and with every planted gap, also those longer
than the horizon, scored at every horizon.

- With decoys, pooled over seeds 0–4: precision 1.000 at every planted gap
  for horizons 5, 10, 15 and 30. At horizon 45 a 60-frame plant is linked at
  gap 38 to another crab (gap 60: precision 0, 1 link); at 60 the same link
  gives 0.889 at gap 60 (8 recovered, 1 false). Hence **30**.
- Before the fix wave the sweep planted only gaps up to the horizon, so this
  false link could not be scored under 45, and the rule returned 45 (the
  earlier derivation, "at horizon 60, gap 60 drops to 0.889; hence 45", was
  wrong in substance). `apply_rule` now refuses horizons scored on different
  planted gaps.
- On the v002 draw (no decoys, seeds 0–4, 40 plants per gap) with the fix-wave
  code: cues on gives precision 1.000 at every planted gap at horizons 30, 45
  and 60; cues off reproduces v002's false link at gap 30 (0.968). This draw
  alone does not separate 30 from 45; the decoy draw does.

What 30 buys over 15 on the demo:

- +45 links (782 → 827) and −45 trajectories (399 → 354).
- Median trajectory length 28 → 51.5 frames; 78 frames for identities only.
- Identity starts 276 → 231 (225 → 185 for identities only).

What it costs:

- 965 interpolated rows with `fill_confidence = 'low'` (bridges of 16–30
  frames), 27% of the 3,546.
- Hold-out recall of 0.88 at gap 20 and 0.75 at gap 30. Long losses are mostly
  missed rather than mislinked.

45 remains a candidate pending the full-dataset sweep: it adds 15 links on
the demo but admits the one long-absence mislink above. Keep only short
fills with `fill_confidence in ('high', 'medium')` or `gap_frames <= 15`, or
run with `--max-gap 15` for the horizon v002 recommended (v002 shipped 5).

### Residual risks (v003)

- **Thin evidence at long gaps.** The default rests on one 300-frame clip.
  Precision is recovered / (recovered + false links), so its bound depends on
  the number of links, not of scored pairs. With no false link, the one-sided
  exact 95% lower bound per gap is 0.951 / 0.950 / 0.949 / 0.949 / 0.951 /
  0.949 / 0.945 / 0.936 from 60 / 59 / 57 / 57 / 60 / 57 / 53 / 45 links at
  gaps 1 / 2 / 3 / 5 / 10 / 15 / 20 / 30, and 0.916 from 34 links at gap 45.
  These bounds are optimistic for three reasons:
  - Decoys only add successes. Decoy targets are chosen to differ in
    appearance from their donor, so by construction they can end recovered or
    missed, and on this clip none was ever linked to its donor. Without the
    decoy targets the plant-only bounds are 0.908 at gap 30 (31 links) and
    0.883 at gap 45 (24 links).
  - The seeds are not independent. All five redraw from the same 156 eligible
    tracklets, so the same crabs and frames recur across seeds.
  - The hold-out does not measure the veto's false refusals. Plant halves
    come from long, high-score tracklets with Shoulder continuity enforced, so
    they cannot produce a cross-gap Shoulder flip; how often the claw-side
    veto (or the axis-flip rule) refuses a true link is unmeasured.

  Above `dt` 45 the measured displacement quantile also exceeds the radius fit
  (few long tracklets). Re-run `fit` and the decoy sweep, then `pool`, on the
  full dataset before relying on 30, let alone 45, there.
- **Burrow retreats.** At 1.0 s the horizon reaches the 1–2 s range at
  which burrow retreats begin. A crab that disappears into its burrow and
  re-emerges is bridged with a placeholder pose (`low` confidence), and a
  neighbour emerging from the same burrow can be linked instead.
- **Size cue is soft.** The one hold-out false link had a 7 px size
  difference and unknown claw sides. Only the claw side has a hard veto.
  Claw waving can flip the side sign within a tracklet (4% of long
  trajectories fall below 0.9 agreement); the veto needs 0.9 over 5 frames
  on both sides. With the axis-flip rule the veto no longer fires on the
  demo, and it is lifted for most different-handed decoy donors too, so on
  this clip the claw side acts as a soft term only.
- **Decoys translate real rows.** They test the appearance cues, not motion
  plausibility. A real neighbour with the same size and side is not covered.
- **Tail adoption at burrow clusters.** A tail can adopt a neighbour's
  low-score detection; the 5 px exclusion radius mitigates this but does not
  prove identity. A tail can also turn a spurious one-frame unit into a
  two-frame identity (see [Identity filter](#identity-filter)).
- **Identity threshold.** `min_identity_score = 0.6` is tuned on one clip's
  bimodal scores; the filter is a flag, so it can be re-applied.
- **Export and re-inference are unvalidated here**: only unit tests,
  `--dry-run` and `bash -n` were run.

## v004: learned link scorer and single global assignment

v004 (2026-11-01) implements the two changes that a comparison with the
tracklet-linking literature ranked highest, benchmarks them against v003 with
the hold-out and on the demo, and ships both as **options**: neither passed
the promotion rule, so every default is unchanged and the default
configuration reproduces v003 byte for byte (demo `links.parquet`,
`trajectories.parquet` and `observations.parquet` compared, default and
`gate_model: linear`). The literature background: Lauer et al. 2022
(multi-animal DeepLabCut) and Boenisch et al. 2018 (BeesBook) learn the
pairwise tracklet-link score from labelled pairs instead of hand-weighting it;
Jaqaman et al. 2008 (u-track), Bise et al. 2011 and Zhang et al. 2008 solve
one global assignment over all gaps, with the price of leaving an endpoint
unmatched set to 1.05 times the largest cost.

### Per-pair features and labels

`features.pair_feature_table` writes one row per links-audit row (same
enumeration, same R_max prefilter) with the 28 features of
`features.FEATURE_NAMES`: the gap, the body distance, the prediction error,
the extended radius and both ratios, the endpoint speeds and whether they are
known, the Shoulder-vector term, the heading change, the size and claw-side
terms with their raw differences (NaN rather than 0 when a side is unknown),
the cue sample sizes, the tracklet lengths, the number of rival pairs (gate
`''` or `max_cost`, so claw-side-vetoed pairs are not rivals) from the same
END and into the same START with a gap no longer than the pair's own (so the
count does not depend on the horizon), the axis-flip flag and the hand cost. `sweep --holdout` exports it as
`holdout_<h>/bridged/pair_features.parquet` (off with
`--no-export-features`); `run` exports it with `export_pair_features: true`.

`labels.label_pairs` joins the features of a hold-out run to its truth table.
A pair is a **positive** when it rejoins the two halves of a plant or decoy
target, or a flicker's left–middle or middle–right halves; **unlabelled**
when neither side is a planted half, when it is a flicker's left–right pair,
a decoy donor's own pair (its right half was translated) or when its two
sides' originals fall in different folds; and a **negative** otherwise
(including the decoy target → donor hard negative). Every side has an
ORIGINAL: a planted left, right or middle half maps to its truth
`original_id` whichever end the pair uses, any other tracklet is its own
original. Folds are a deterministic function of the original,
`sha256('gap_bridging_v004' NUL id)[:8] / 2^64 < 0.4 → test`, and a labelled
pair's fold is the common fold of its two originals, so no test-fold
tracklet (planted under any seed, or unplanted) ever enters training; the
price is that most decoy target → donor hard negatives (two originals) are
unlabelled. `labels.dedupe_pairs` keeps one row per pair across the horizons of a
sweep (the planted tables are identical across horizons; a feature mismatch
raises).

### Training (`bridge_gaps.py train`)

`train.train_command` collects the labelled, deduplicated, motion-gated pairs
of every sweep, fits a `HistGradientBoostingClassifier` (depth 3, 200
iterations, learning rate 0.05, 20 samples per leaf, L2 1.0) with monotonic
constraints (the probability of a true link may not rise with the gap, the
distance, the prediction error, their ratios, the Shoulder term, the size or
side term, or the hand cost), calibrates it (sigmoid by default) on
out-of-fold predictions of a group-aware 5-fold split (groups = planted
original), chooses the smallest threshold with out-of-fold precision 1.0,
refits on the whole train fold and saves
`configs/tracking/link_scorer_v001.joblib` with a JSON manifest
(scikit-learn version, feature order, constraints, hyperparameters, the
feature-relevant configuration subset `features.FEATURE_CONFIG_KEYS`,
threshold, data hash, fold sizes, metrics) and a per-gap metrics CSV. The
trainer refuses a sweep whose `pair_features.parquet` is not listed in its
marker with a matching sha256 and row count, sweeps that disagree on the
feature configuration, and a threshold that is degenerate (≥ 1, every
accepted cost 0) or does not reach the target precision
(`--allow-degenerate-threshold` overrides). `scorer.load_scorer` refuses a
model saved under another scikit-learn version or feature order, and
`resolve_scorer` refuses a run whose feature configuration differs from the
manifest's. `--write-config` writes the threshold into the learned overlay.

Committed model (five demo sweeps, seeds 0–4, `--per-gap 8 --decoys 4
--flicker 2`): 924 training pairs (468 positive), 484 test pairs (306
positive), sigmoid calibration, threshold 0.9944; test-fold pairwise
precision 1.0, recall 0.35, ROC AUC 0.985. Before the fold correction the
same sweeps gave 1,201 / 792 pairs (the extra 585 were cross-fold
negatives), threshold 0.9805 and pairwise recall 0.62. Isotonic calibration
puts the threshold at exactly 1.0, which zeroes every accepted cost (D-010),
and is now refused unless overridden.

### Learned cost model

Under `cost_model: learned` the sqrt motion gates and the claw-side veto are
unchanged; the pairs that pass them are scored and their cost is
`-log p`. The thresholds are probabilities: `max_cost = -log
learned_min_probability` (so a pair is refused `max_cost` iff
`p < learned_min_probability`), the dummy diagonal `-log
learned_unmatched_probability`, and `learned_margin` is a log-probability
ratio (CLI `--margin`). Chain reachability, runner-up and margin veto work
on that scale. The linear model and the hand sqrt cost are untouched.

### Global assignment and adaptive dummy

`assignment: global` makes `_stages` a single stage over gaps `1..max_gap`,
so no pair is excluded by an earlier stage and no reason says "at stage".
`unmatched_cost_mode: adaptive` prices each connected component's dummy
diagonal at `adaptive_factor` (1.05) times the component's largest finite
cost. A lone gated pair then always links (`c < 2 · 1.05 c`); with the
shipped fixed values (dummy 0.6, max cost 1.0) a lone pair already always
links, so the modes differ only inside contested components.

### Comparison (`bridge_gaps.py compare`)

`compare.compare_variants` re-scores every hold-out directory of every seed
of each variant on one fold of planted originals (`fold_scores`, keeping a
decoy donor whenever its target is kept), pools the seeds
(`holdout.pool_sweeps`), applies the horizon rule and checks each variant
against the baseline with `promotion_check`: precision 1.0 (or no links) at
every planted gap up to `--through-gap` and recall not below the baseline at
every such gap. The generated tables are
`docs/gap_bridging_v004_comparison.md` (test fold) and
`docs/gap_bridging_v004_comparison_all.md` (all plants).

Pooled hold-out, five seeds, every gap 1–60 planted at every horizon
(587 pairs per horizon over all plants, 232 on the test fold); `recovered /
false links`:

| `max_gap` | current (all) | global (all) | learned (all) | learned p ≥ 0.5 (all) | current (test) | learned (test) | learned p ≥ 0.5 (test) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 233 / 0 | 233 / 0 | 159 / 0 | 239 / 0 | 92 / 0 | 64 / 0 | 93 / 0 |
| 10 | 293 / 0 | 293 / 0 | 163 / 0 | 298 / 0 | 119 / 0 | 68 / 0 | 119 / 0 |
| 15 | 350 / 0 | 350 / 0 | 172 / 0 | 357 / 0 | 142 / 0 | 73 / 0 | 142 / 0 |
| 30 | 448 / 0 | 448 / 0 | 191 / 0 | 471 / 1 | 179 / 0 | 82 / 0 | 186 / 0 |
| 45 | 482 / 1 | 482 / 1 | 210 / 0 | 525 / 1 | 188 / 0 | 88 / 0 | 203 / 0 |
| 60 | 490 / 1 | 490 / 1 | 226 / 0 | 568 / 3 | 190 / 0 | 94 / 0 | 222 / 0 |

`learned_global` equals `learned` row for row. "learned p ≥ 0.5" is an
exploratory variant (`learned_min_probability: 0.5`, overlay not committed)
run to see how far the trained threshold limits recall. All learned rows use
the retrained (fold-corrected) model.

Demo clip, seed-0 sweep runs (`links / trajectories`):

| `max_gap` | current | global | learned | learned p ≥ 0.5 |
|---:|---:|---:|---:|---:|
| 5 | 639 / 542 | 639 / 542 | 26 / 1,155 | 623 / 558 |
| 15 | 782 / 399 | 747 / 434 | 26 / 1,155 | 690 / 491 |
| 30 | 827 / 354 | 772 / 409 | 26 / 1,155 | 704 / 477 |
| 60 | 848 / 333 | 810 / 371 | 26 / 1,155 | 729 / 452 |

### Promotion decision (v004)

- **Global assignment: not promoted.** It passes the rule trivially because
  it is identical to the staged linker on every planted gap, horizon and
  fold; the rule's tie-break (highest test recall) does not separate them.
  On the demo it makes 55 fewer links at horizon 30 (772 vs 827; 781 with
  the fixed dummy): 59 staged links, 36 of them 1 to 2 frame gaps, are
  refused because a cheaper long-gap pair claims an endpoint and only 4
  new links appear. A stronger gap penalty does not recover them
  (`gap_penalty` 0.12 / 0.2 / 0.3 / 0.5 → 783 / 765 / 722 / 665 links): the
  penalty pushes long pairs over `max_cost` before it reorders the contested
  components. The planted hold-out cannot adjudicate these 55 links because
  both variants score alike on it (D-011). Staging stays the default.
- **Learned scorer: not promoted.** At its trained threshold (0.9944) it
  keeps precision 1.0 everywhere, including the two long-horizon false links
  of the hand cost, but recovers far fewer plants (191 vs 448 at horizon 30
  over all plants; 82 vs 179 on the test fold) and keeps only 26 of the
  demo's 827 links. With the threshold relaxed to 0.5 (exploratory overlay,
  not committed) it matches or exceeds the hand cost at every horizon on the
  test fold with zero false links (186 vs 179 at horizon 30, 222 vs 190 at
  60) and over all plants (471 vs 448, 568 vs 490), but it fails the rule
  twice: one plant short at gap 10 on the test fold (26 of 27), and over all
  plants 1 false link at horizon 30 (planted gap
  30) and 3 at horizon 60 (planted gaps
  30, 60). On the demo it makes 704 links at horizon 30
  against 827. The gap between the hold-out and the demo has one cause:
  hold-out positives are halves of tracklets of at least 40 frames (shorter
  half ≥ 4 frames, median 42), whereas 75% of the demo's natural links have a
  1-frame side; the first committed scorer gave `p ≥ 0.98` to 3% of natural
  1-frame links and to 86% of links whose shorter tracklet has 20 frames or
  more (D-009). The learned model has learned, correctly for its training
  set, that long fragments link and short fragments do not. Until the
  hold-out plants short halves (or the tracklet-length features are removed
  and the model retrained), no learned variant can be promoted from these
  data, and a threshold chosen on planted pairs does not transfer to natural
  breaks.
- **Horizon default: 30, unchanged.** No component was promoted, so the
  rule is not re-derived. For the record, the test fold alone would choose
  60, because the single gap-38 false link of the hand cost falls on a
  train-fold original; the all-plants rule, which is the right one for a
  model with no training set, keeps 30.

### Runtime benchmark

`scripts/bench_gap_bridging.py` times `pipeline.run` (and `--link-only` the
endpoints plus linking) on the demo and on spatially tiled copies
(`tile_observations`: `ceil(sqrt n)` grid, tiles separated by the clip's
extent plus 400 px so they never interact; string ids suffixed `/tileNN`,
integer ids offset per tile). Five repeats, median wall seconds, login node
della9 (192 logical CPUs, `OMP_NUM_THREADS=8`), max_gap 30:

| config | tiles | rows | tracklets | links | total s | link s | assemble s |
|---|---:|---:|---:|---:|---:|---:|---:|
| current | 1 | 36,573 | 1,181 | 827 | 1.23 | 0.08 | 0.85 |
| global | 1 | 36,573 | 1,181 | 772 | 1.12 | 0.07 | 0.77 |
| learned | 1 | 36,573 | 1,181 | 26 | 0.39 | 0.12 | 0.09 |
| learned_global | 1 | 36,573 | 1,181 | 26 | 0.39 | 0.12 | 0.09 |
| current | 4 | 146,292 | 4,724 | 3,308 | 4.78 | 0.38 | 3.42 |
| global | 4 | 146,292 | 4,724 | 3,088 | 4.49 | 0.34 | 3.17 |
| learned | 4 | 146,292 | 4,724 | 104 | 1.49 | 0.55 | 0.36 |
| learned_global | 4 | 146,292 | 4,724 | 104 | 1.45 | 0.55 | 0.36 |
| current | 16 | 585,168 | 18,896 | 13,232 | 20.26 | 2.43 | 13.84 |
| global | 16 | 585,168 | 18,896 | 12,352 | 18.82 | 2.34 | 12.78 |
| learned | 16 | 585,168 | 18,896 | 416 | 6.74 | 3.19 | 1.48 |
| learned_global | 16 | 585,168 | 18,896 | 416 | 6.78 | 3.17 | 1.53 |

Linking (endpoint matching, Hungarian per component) is under 15% of the wall time; table assembly dominates. The learned variants are faster only because they make far fewer links (fewer bridges to fill and assemble), not because scoring is cheaper: their `link` stage is 1.3 to 1.5 times the hand cost's. Runtime scales linearly in the number of tiles (crabs) at fixed density. CSV: `logs/bench/gap_bridging_v004.csv` (git-ignored).

### Residual risks (v004)

- The learned scorer is trained on one clip's planted gaps and inherits the
  survivorship bias of the gate fit; it is shipped as an overlay only.
- The exact scikit-learn version is pinned in the manifest; a lock update
  requires `bridge_gaps.py train` again (deterministic given the seed).
- Calibration and threshold both use out-of-fold train predictions (slightly
  optimistic); promotion uses the test fold through the full pipeline.
- Mixed-fold decoy hard negatives are unlabelled (target and donor come from
  different originals), which removes most of them from training; the
  full-pipeline test fold still scores every decoy whose target is in the fold.
- The adaptive dummy weakens the "leave unlinked" option inside contested
  components; it is compared, not shipped.

### Commands (v004)

```bash
# current method, five seeds, with feature export (default)
for k in 0 1 2 3 4; do
  .venv/bin/python scripts/bridge_gaps.py sweep \
      --observations demo/crab_cam2_2_middle_300/observations.parquet \
      --out logs/gap_sweep_v004/current/seed_$k --max-gaps 5,10,15,30,45,60 --holdout \
      --per-gap 8 --decoys 4 --flicker 2 --seed $k --no-figures --force
done
# train the scorer, write its threshold into the learned overlay
.venv/bin/python scripts/bridge_gaps.py train \
    --sweeps logs/gap_sweep_v004/current/seed_{0,1,2,3,4} \
    --out configs/tracking/link_scorer_v001.joblib --calibration sigmoid \
    --write-config configs/tracking/gap_bridging_v004_learned.yaml
# the same sweeps with --config configs/tracking/gap_bridging_v004_{global,learned,learned_global}.yaml
# compare on the test fold (and --fold all), promotion rule through gap 30
.venv/bin/python scripts/bridge_gaps.py compare \
    --variant current=logs/gap_sweep_v004/current/seed_0,...,seed_4 \
    --variant global=... --variant learned=... --variant learned_global=... \
    --out logs/gap_sweep_v004/compare_test --fold test --through-gap 30 \
    --docs docs/gap_bridging_v004_comparison.md
# runtime
.venv/bin/python scripts/bench_gap_bridging.py --tiles 1,4,16 --repeats 5 --max-gap 30 \
    --configs current=configs/tracking/gap_bridging_v001.yaml global=... learned=... learned_global=... \
    --out logs/bench/gap_bridging_v004.csv --markdown logs/bench/gap_bridging_v004.md
```

## v005: short-half hold-out profile and feature ablation

v005 (2026-11-01) addresses D-009: a learned link scorer trained on the
standard hold-out learns tracklet length because every planted positive is a
half of a tracklet of at least 40 frames with at least 3 frames kept per
side, while natural breaks are mostly between short fragments. Two remedies
were built and judged by the same rules: a hold-out **planting profile** that
produces short halves matching the natural mix (the primary remedy), and a
**feature ablation** that removes the length-confounded features (a control).
Every default is unchanged; the standard profile is byte-identical to v004
(planted tables, links and CSVs compared for five seeds and six horizons).

### The natural mix

`holdout.link_side_lengths` / `holdout.link_mix` measure, over the 827 links
of the default demo run at horizon 30, the lengths of the two tracklets each
link joins (shorter side `lo`, longer side `hi`):

| category | definition | demo fraction |
|---|---|---:|
| both_short | hi ≤ 3 | 0.401 |
| one_short | lo ≤ 3, hi ≥ 10 | 0.177 |
| short_mid | lo ≤ 3, hi in 4..9 | 0.214 |
| mid | lo in 4..9 | 0.127 |
| both_long | lo ≥ 10 | 0.081 |

Among sides of at most 3 frames the lengths 1 / 2 / 3 occur with weights
0.58 / 0.27 / 0.15. The standard planting (shorter half ≥ 4 frames, median
42) covers only the last row.

### Planting profile (`sweep --profile short`)

`holdout.HalfProfile` fixes the draw: per plant, a category from the
probabilities above (0.40 / 0.18 / 0.21 / 0.13 / 0.08), then the kept length
of each side (short: 1..3 with the weights above; mid: uniform 4..9; a
"keep-all" side keeps every frame on that side and must have at least 10, or
4 when the other side is mid; which side is short is a coin flip), then an
eligible unused tracklet (at least 10 frames, enough for gap + both kept
lengths) and the run offset, uniform over the admissible positions. The rows
outside the kept window are deleted together with the gap run (a kept
remainder would be a same-crab neighbour labelled negative); the trimmed left
half's first row gets `incoming_reason 'no_candidate'` like a right half.
Decoy targets are drawn like plants (their left cue is measured on the
trimmed rows, so most have too few claw frames for a veto); donors and
flicker singletons are not trimmed. Shorter gap lengths never depend on
longer ones, and the `standard` profile runs the v004 code path verbatim.
The truth table gains `profile`, `category`, `left_frames` and
`right_frames` under both profiles; `holdout_<h>.json` records the profile.
On the demo (8 plants, 4 decoys, 2 flicker per gap, seeds 0–4) every gap up
to 60 is fully planted; the pooled 400 plants fall 0.40 / 0.19 / 0.21 /
0.11 / 0.08 into the five categories.

### Feature ablation (`train --features`)

`features.FEATURE_SETS` names ordered subsets of `FEATURE_NAMES`: `all`,
`no_length` (drops `n_frames_*`, `size_n_*`, `claw_side_n_*`, `vel_known_*`;
20 features) and `no_length_no_speed` (also `speed_*`; 18). The manifest
records the subset (`feature_names`, `feature_set`), the loader accepts any
ordered subset, and the learned cost path selects the columns by name. The
ablation cannot blind the model completely (speeds are 0 when unknown, the
radius uses them, `abs_side_*` is discrete for n ≤ 3, and the size term's
noise term depends on n), so it is a control, not a candidate.

### Scorers

All trained with the v004 recipe (sigmoid calibration, threshold at
out-of-fold precision 1.0, folds by original tracklet id):

| scorer | features | training profile | pairs train / test (positive) | threshold | pairwise test precision / recall |
|---|---|---|---:|---:|---:|
| S1 (`link_scorer_v005_s1`) | all | short | 1,368 / 537 (510 / 287) | 0.989 | 0.989 / 0.33 |
| S2 (`link_scorer_v005_s2`) | no_length | short | 1,368 / 537 (510 / 287) | 0.985 | 0.990 / 0.36 |
| S3 (`link_scorer_v005_s3`) | no_length | standard | 924 / 484 (468 / 306) | 0.995 | 1.000 / 0.12 |

The D-009 diagnostic (probability given to the demo's 827 natural links,
by the shorter tracklet's length) shows the confound is gone: S1 gives
p ≥ 0.5 to 98% of links whose shorter side has 1 frame and to 97% of those
with 20 or more (v004 model: 84% and 97%, median p 0.89 vs 0.99 for the
1-frame bin); S2 behaves alike, and even S3, trained on long halves without
length features, reaches 96% in the 1-frame bin. At the trained thresholds
(about 0.99) the fractions drop to 16–58%, so the threshold, not the model,
now limits the demo links.

### Comparison on the short profile

Baseline `current` = the hand cost re-scored on the short profile. 600
planted pairs per horizon over all plants, 223 on the test fold; `recovered /
false links`; `_p50` variants use `learned_min_probability: 0.5`
(exploratory overlays, not committed):

| `max_gap` | current (all) | S1 (all) | S2 (all) | S3 (all) | S1 p ≥ 0.5 (all) | S2 p ≥ 0.5 (all) | current (test) | S1 p ≥ 0.5 (test) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 239 / 1 | 120 / 1 | 159 / 0 | 65 / 0 | 239 / 1 | 239 / 1 | 98 / 1 | 97 / 1 |
| 10 | 298 / 1 | 122 / 1 | 164 / 0 | 65 / 0 | 299 / 1 | 299 / 1 | 118 / 1 | 118 / 1 |
| 15 | 355 / 1 | 125 / 1 | 164 / 0 | 65 / 0 | 357 / 1 | 357 / 1 | 138 / 1 | 138 / 1 |
| 30 | 452 / 2 | 126 / 1 | 172 / 0 | 65 / 0 | 472 / 1 | 471 / 1 | 172 / 1 | 178 / 1 |
| 45 | 482 / 2 | 134 / 1 | 193 / 0 | 65 / 0 | 526 / 1 | 525 / 1 | 183 / 1 | 201 / 1 |
| 60 | 497 / 2 | 155 / 1 | 215 / 0 | 65 / 0 | 582 / 1 | 581 / 1 | 187 / 1 | 216 / 1 |

Every variant's single false link at every horizon is the same planted
gap-60 crab (absent for two seconds after trimming) whose short half links
to a neighbour within 5 frames; the hand cost adds one more at planted gap
30 under horizons ≥ 30, so **the hand baseline fails its own precision rule
on the short profile** (the demo's natural breaks are harder than the
standard plants), and the promotion check ran at the fallback horizon 30.
Demo links at horizon 30: current 827, S1 292, S2 361, S3 123, S1 p ≥ 0.5
816, S2 p ≥ 0.5 813, S3 p ≥ 0.5 797. Full tables:
`docs/gap_bridging_v005_comparison.md` (test fold) and
`docs/gap_bridging_v005_comparison_all.md` (all plants).

### Visual audit

`bridge_gaps.py audit` renders the links a candidate gained and lost against
the current run (`docs/figures/gap_bridging_v005_audit_s1_p50_{gained,lost}_p*.png`
and the CSV beside them; S1 p ≥ 0.5 vs current, horizon 30, seed-0 runs):
8 gained (gaps 8–29) and 19 lost (gaps 1–30, median 9). On the video both
sets are overwhelmingly the same crab sitting at a burrow: the lost links are
short-gap pairs with a 1 to 3 frame side that the hand cost links at
0.3–0.9 and the scorer puts just below 0.5; the gained links are 8 to 25
frame absences at a burrow that the hand cost refused above `max_cost`
(1.04–1.48) and the scorer accepts at 0.02–0.39. The audit therefore does
not separate the two linkers in quality; it shows that the learned scorer
trades short-gap burrow links for long-gap ones.

### Promotion decision (v005)

- **Not promoted.** S1 at p ≥ 0.5 is the strongest variant: precision 1.0 at
  every planted gap through 30 on both folds (better than the baseline's
  own 452 / 2 at horizon 30), recall above the baseline at every horizon
  (472 vs 452 recovered at 30, 582 vs 497 at 60 over all plants; 178 vs 172
  and 216 vs 187 on the test fold), yet it fails the rule twice: one plant
  short of the baseline at gap 2 on the test fold (26 of 27) and 816 demo
  links against the floor of 827. The trained-threshold variants (S1, S2,
  S3) lose most of the recall and the demo links; the control S3 at p ≥ 0.5
  adds a false link at gap 10. By the agreed rule no default changes; the
  horizon stays 30 on the standard rule.
- **What changed in substance.** The length confound of D-009 is removed
  by the short profile (and largely by the ablation), the learned scorer is
  now a credible alternative at a relaxed threshold rather than a failure,
  and the hand cost is shown to make false links on natural-shaped plants
  that it never made on the standard ones. The remaining obstacle is the
  threshold recipe: out-of-fold precision 1.0 on planted pairs sets 0.99,
  which refuses half of the natural links; a threshold chosen against the
  demo link count, or a precision target below 1.0, would be a different
  rule and is left for a decision.

### Residual risks (v005)

- Planted short halves are confident detections of well-tracked crabs
  (median instance score of natural 1-frame tracklets 0.29); the scorer still
  never sees a low-score fragment as a positive.
- The gap-60 false link shared by every linker suggests a neighbour
  fragment within reach of a trimmed half; the hold-out cannot tell a wrong
  link from a crab that genuinely returned (the trimmed rows are deleted).
- S1/S2 pairwise test precision is 0.99, not 1.0: the out-of-fold threshold
  does not transfer perfectly to held-out short halves.
- Mix probabilities are demo constants, measured on the links the hand-cost
  linker accepted at horizon 30 and pooled over all gaps, so they describe
  the mix conditional on acceptance, not that of all true breaks, and they
  are applied identically at every planted gap; re-measure with `link_mix`
  on the full dataset before planting there.
- Short halves are cut from tracklets drawn uniformly among those long
  enough for the plant, i.e. mostly well-tracked crabs, and the rest of the
  tracklet is deleted around them; a natural 1-frame fragment is usually a
  poorly tracked crab with neighbours still present.
- Decoy categories are conditional on a successful draw (a target whose cue
  differs from a donor), which skews planted decoys towards categories whose
  short left half still carries a usable cue.

### Commands (v005)

```bash
OBS=demo/crab_cam2_2_middle_300/observations.parquet
COMMON="--observations $OBS --max-gaps 5,10,15,30,45,60 --holdout --per-gap 8 --decoys 4 --flicker 2 --no-figures --force"
for k in 0 1 2 3 4; do .venv/bin/python scripts/bridge_gaps.py sweep $COMMON --seed $k --out logs/gap_sweep_v005/current_std/seed_$k; done
for k in 0 1 2 3 4; do .venv/bin/python scripts/bridge_gaps.py sweep $COMMON --profile short --seed $k --out logs/gap_sweep_v005/current_short/seed_$k; done
.venv/bin/python scripts/bridge_gaps.py train --sweeps logs/gap_sweep_v005/current_short/seed_{0,1,2,3,4} --features all \
    --out configs/tracking/link_scorer_v005_s1.joblib --write-config configs/tracking/gap_bridging_v005_s1.yaml   # S2: --features no_length; S3: current_std sweeps
# learned sweeps: the same sweep line with --profile short --config configs/tracking/gap_bridging_v005_<s>.yaml (and p>=0.5 overlays)
.venv/bin/python scripts/bridge_gaps.py compare --variant current=logs/gap_sweep_v005/current_short/seed_0,... \
    --variant s1=... --variant s1_p50=... --fold test --through-gap 30 --demo-horizon 30 \
    --out logs/gap_sweep_v005/compare_test --docs docs/gap_bridging_v005_comparison.md
.venv/bin/python scripts/bridge_gaps.py audit --current logs/gap_sweep_v005/current_short/seed_0/max_gap_30 \
    --candidate logs/gap_sweep_v005/s1_p50/seed_0/max_gap_30 --observations $OBS \
    --out docs/figures/gap_bridging_v005_audit_s1_p50 --n 20
```

## v006: pipeline-level threshold rule for learned scorers

v006 (2026-11-02) addresses D-013: the v004 threshold recipe chooses the
smallest out-of-fold probability at which every candidate **pair** above it is
a planted positive. Re-scoring the v005 short-profile hold-outs with S1 shows
that this metric, not its target, is the obstacle: pair precision is flat near
0.97 from p = 0.5 down to 0.2 (train fold: 10 of 402 negatives above 0.5,
13 above 0.2, 19 above 0.1 against 402 to 410 positives), so the rule lands at
0.989 and refuses two thirds of the demo's natural links, while the full
linker at p ≥ 0.5 makes one false link in 600 plants. The one-to-one
assignment and the margin veto remove almost every negative the pair count
charges to the scorer. A second coupling made any threshold below 0.5 inert:
the dummy diagonal is −log(`learned_unmatched_probability`) with 0.5 fixed,
so a pair with p < 0.5 lost to "unmatched" whatever `learned_min_probability`
said. Every default is unchanged (the default and `linear` demo runs are
byte-identical to v005, and the `pair_oof` rule remains the training default).

### The pipeline rule (`train --threshold-rule pipeline`)

`gap_bridging.threshold` chooses the threshold where the promotion rule is
evaluated: after assignment, per planted gap, on the **train-fold** plants of
each training sweep's `holdout_<curve_horizon>` run (default 30; the test
fold stays reserved for `compare`), **cross-fitted** so that no plant is
linked by a model that saw its pairs.

1. Candidates: the ascending unique out-of-fold probabilities of the
   labelled train pairs inside (0, 1), thinned to at most `--max-candidates`
   (200) by evenly spaced indices that keep both extremes.
2. Scene: each run's `planted_observations.parquet`, its own hand-cost
   config (from the run's JSON; `gate_model` must be `sqrt` and
   `unmatched_cost_mode` `fixed`) and its truth table restricted to rows
   whose original tracklet is in the train fold (`labels.truth_folds`; a gap
   left with no plant is dropped rather than failed).
3. Cross-fitting (`train.fit_fold_models`, `threshold.crossfit_scenes`): the
   classifier is refitted on each of the `StratifiedGroupKFold` splits of
   `fit_scorer` (same seed and fold count), the train-fold truth is
   partitioned by the split that held out each plant's original tracklet,
   and each part is linked with that split's model through the calibrator
   fitted on the out-of-fold scores. The curve's `run` column names the
   part (`<run>#fold<k>`). The first v006 draft scored the plants with the
   final model, which had seen them; the review found this optimistic (S2's
   in-sample choice 0.103 against an out-of-sample 0.3), hence the refit.
4. Curve: for every candidate `t`, `pipeline.link_videos` with
   `cost_model: learned`, the fold model as scorer and
   `learned_min_probability = learned_unmatched_probability = t` (gates,
   staging, margin and chain reachability unchanged), then
   `holdout.score_links` per gap; pooled over sweeps and folds.
5. Choice: a candidate **qualifies** when precision is exactly 1.0 at every
   gap through `--through-gap` (30; `recovered > 0` and `false_link == 0`);
   it is **clean** when it makes no false link there (recall may vanish).
   The threshold is the smallest qualifying candidate above which every
   candidate is clean, so a lucky clean dip below a false-linking candidate
   cannot be chosen and vanishing recall at the top (where the highest
   candidates recover nothing) does not block the choice. No such candidate
   refuses the training with the offending candidate and gap named.
6. Outputs: `<model>_threshold_curve.csv` (pooled per threshold and gap) and
   `<model>_threshold_curve_runs.csv` (per fold part) beside the model; the
   manifest records `threshold_rule`, the pair-level value as
   `pair_threshold` and a `threshold_summary` (candidates, counts through
   the rule's gap at the chosen and at the pair-level threshold;
   `hyperparameters.target_reached` describes the pair rule); the overlay
   written by `--write-config` sets both probabilities to the threshold, and
   a later `pair_oof` write removes the dummy key and any stale curve files.
   Old manifests load as `pair_oof` with no pair threshold.

Cost: 200 candidates × 5 sweeps × 5 fold parts at about 0.3 s each, about
25 minutes of CPU per scorer, which exceeds the login node's per-process
limit: run it under Slurm (`logs/gap_sweep_v006/slurm/train_v006.slurm`
is the array script used; 8 CPUs, 16 GB, under 30 minutes per task).
Features are recomputed per candidate; caching them would not change the
result. The candidate grid is a proxy for where the pipeline curve steps
(the pairs' probabilities under the fold models), so the choice can sit one
plant of recall above the true boundary; the grid near the choice is
recorded in the curve CSV.

### Scorers (v006)

Retrained on the five v005 short-profile hand-cost sweeps under the pipeline
rule (same classifier, calibration and folds as v005):

| scorer | features | rule threshold | pair-level threshold | train-fold plants through gap 30 at the threshold (recovered / false of 300) | largest false-linking candidate below | pairwise test precision / recall |
|---|---|---:|---:|---:|---:|---:|
| S1 (`link_scorer_v006_s1`) | all | 0.413 | 0.989 | 289 / 0 (pair threshold: 86 / 0) | 0.259 (one at gap 30) | 0.965 / 0.972 |
| S2 (`link_scorer_v006_s2`) | no_length | 0.478 | 0.985 | 289 / 0 (pair threshold: 112 / 0) | 0.265 (one at gap 30) | 0.965 / 0.965 |

An independent script (not using `gap_bridging.threshold`; it refits the
fold models from the manifest's recipe with scikit-learn directly) rebuilt
the five scenes from the files, re-linked each fold part at the chosen
threshold and at the largest candidate below it, and reproduced the pooled
curve rows and the verdicts exactly for both scorers. The D-009 diagnostic
on the demo's 827 natural links: S1 gives p ≥ 0.413 to 98.0% of links whose
shorter side has 1 frame, 97.9% (2), 98.4% (3), 97.1% (4–9) and 98.5% (10 or
more); S2 at 0.478 gives 97.1%, 97.2%, 98.4%, 97.1% and 98.5%; the v005 S1
at 0.989 gave 48%, 27%, 16%, 16% and 19%.

### Comparison on the short profile (v006)

Baseline `current` = the hand cost on the short profile (v005 sweeps);
`S1 p ≥ 0.5` is the v005 exploratory overlay. Test fold, horizon 30,
`recovered / false` per planted gap (`docs/gap_bridging_v006_comparison.md`;
all plants in `docs/gap_bridging_v006_comparison_all.md`):

| planted gap | current | S1 p ≥ 0.5 | S1 (0.413) | S2 (0.478) |
|---:|---:|---:|---:|---:|
| 1 | 18 / 0 of 18 | 18 / 0 | 18 / 0 | 18 / 0 |
| 2 | 27 / 0 of 27 | 26 / 0 | 27 / 0 | 26 / 0 |
| 3 | 31 / 0 of 31 | 31 / 0 | 31 / 0 | 31 / 0 |
| 5 | 22 / 0 of 22 | 22 / 0 | 22 / 0 | 22 / 0 |
| 10 | 20 / 0 of 21 | 21 / 0 | 21 / 0 | 21 / 0 |
| 15 | 20 / 0 of 20 | 20 / 0 | 20 / 0 | 20 / 0 |
| 20 | 16 / 0 of 20 | 19 / 0 | 19 / 0 | 18 / 0 |
| 30 | 18 / 0 of 21 | 21 / 0 | 21 / 0 | 21 / 0 |
| all plants, horizon 30 (of 600) | 452 / 2 | 472 / 1 | 474 / 2 | 471 / 1 |
| all plants, horizon 60 (of 600) | 497 / 2 | 582 / 1 | 585 / 2 | 582 / 1 |
| demo links at horizon 30 (trajectories) | 827 (354) | 816 (365) | 819 (362) | 814 (367) |

S1 under the cross-fitted rule recovers every test-fold plant through gap
30 that the hand cost recovers, with precision 1.0, and is the first
learned variant to pass the hold-out half of the rule; it reaches 819 demo
links against the floor of 827. Over all plants it makes one false link at
planted gap 30 under horizons ≥ 30 (a train-fold plant; the cross-fitted
curve put the boundary at 0.259 for the fold models, and the final model
differs) besides the gap-60 plant every linker gets wrong. S2 misses the
gap-2 test plant (p = 0.47 against its 0.478) and 13 demo links. The horizon
rule itself is not satisfied by any variant on the short profile (the hand
baseline fails precision 1.0 at planted gap 30 over all plants, as in
v005), so the promotion check ran at the fallback horizon 30.

### Visual audit (v006)

`docs/figures/gap_bridging_v006_audit_s1_{gained,lost}_p*.png` (S1 vs
current, horizon 30, seed-0 runs; CSV beside them): 9 gained (gaps 8 to 29,
seven with a 1 to 2 frame side) and 17 lost (gaps 1 to 30, median 6). On the
video every gained link joins the same crab at the same burrow or on the
same patch of mud (refused by the hand cost above `max_cost` at 1.04 to
2.82, accepted by S1 at p 0.48 to 0.98), and every lost link is just as
plausible: a 59- and a 77-frame tracklet of one crab in its burrow 6 frames
apart (hand cost 0.45, S1 p = 0.04), a 6- and a 109-frame tracklet 2 frames
apart (p = 0.30), and short burrow fragments at gaps 1 to 30 (p 0.01 to
0.33). The audit shows no visibly wrong link on either side; it shows that
the scorer's ordering of stationary burrow links differs from the hand
cost's, and that the hold-out contains no case of this kind.

### Promotion decision (v006)

- **Not promoted.** S1 (threshold 0.413) passes the hold-out half of the
  rule on the test fold (precision 1.0 and recall ≥ the hand baseline at
  every planted gap through 30) but reaches 819 demo links against the floor
  of 827, and over all plants it adds one false link at planted gap 30. S2
  (0.478) misses one gap-2 test plant and 13 demo links. No default changes.
- **The in-sample draft would have promoted S2.** Before the review, the
  curve scored the train-fold plants with the final model (which had seen
  them): S2's threshold came out at 0.103 and S2 cleared every part of the
  rule (test fold 160 of 160, 831 demo links). Cross-fitting moved it to
  0.478 and the verdict to a fail. The rule's guarantee ("no false link
  through gap 30") now holds out of sample on the fold models; the final
  model's one extra false link on all plants shows the remaining gap
  between fold models and the model shipped.
- **What the rule changed.** The thresholds moved from 0.99 to 0.41 (S1)
  and 0.48 (S2), recovering 289 of 300 train-fold plants instead of 86 and
  112, and S1 now matches the hand cost's recall on the test fold at every
  gap. The demo floor is the remaining obstacle: the 17 lost links sit at
  S1 probabilities of 0.01 to 0.33, below any threshold the rule can choose
  (its boundary is 0.259), so a learned scorer trained on planted gaps
  cannot reach 827 without a different training signal for stationary
  burrow links. The pair-level rule is kept as the default and recorded in
  every manifest for comparison.

### Runtime benchmark (v006)

`scripts/bench_gap_bridging.py --tiles 1,16 --repeats 5` (full pipeline,
`max_gap` 30, login node, median of 5; `logs/bench/gap_bridging_v006.csv`):

| config | demo total (link stage) | links | 16 tiles total (link stage) | links |
|---|---:|---:|---:|---:|
| current (hand cost) | 1.30 s (0.09 s) | 827 | 20.7 s (2.47 s) | 13,232 |
| S1 (0.413) | 1.28 s (0.15 s) | 819 | 21.2 s (3.57 s) | 13,104 |
| S2 (0.478) | 1.25 s (0.15 s) | 814 | 22.3 s (3.78 s) | 13,024 |

The learned cost adds about 0.06 s to the link stage on the demo and 1.1 to
1.3 s on 16 tiles (feature computation and scoring of all candidate pairs);
the assembly stage dominates either way, and the totals differ by less than
the run-to-run spread. The v004 learned bench row (0.39 s total) reflected
its 26 links, not a faster linker.

### Residual risks (v006)

- The curve is chosen on about 300 train-fold plants of five seeds at one
  horizon through five fold models; the shipped model is the refit on all
  train rows and can differ from them near the boundary (S1's extra false
  link at gap 30 on all plants). A new dataset should re-run the curve.
- The candidate grid is thinned from the OOF probabilities, which cluster
  near 0 and 1; the choice can sit one plant of recall above the true
  boundary.
- Plants beyond the horizon (absences longer than `max_gap`) are not in the
  rule; `compare` still reports them.
- Lowering the dummy with the threshold lets low-p pairs compete that were
  previously absorbed by "unmatched"; the curve measures the consequence per
  gap only where plants exist.
- The v005 residual risks (planted halves are confident detections, demo
  constants for the mix) apply unchanged.

### Commands (v006)

```bash
SW=logs/gap_sweep_v005/current_short/seed_{0,1,2,3,4}
# about 25 min CPU per scorer: submit logs/gap_sweep_v006/slurm/train_v006.slurm (array 1-2) rather than running on a login node
.venv/bin/python scripts/bridge_gaps.py train --sweeps $SW --features all --threshold-rule pipeline \
    --out configs/tracking/link_scorer_v006_s1.joblib --write-config configs/tracking/gap_bridging_v006_s1.yaml   # S2: --features no_length
# sweeps: the v005 sweep line with --profile short --config configs/tracking/gap_bridging_v006_<s>.yaml -> logs/gap_sweep_v006/<s>/seed_k
.venv/bin/python scripts/bridge_gaps.py compare --variant current=logs/gap_sweep_v005/current_short/seed_0,... \
    --variant s1_p50=... --variant s1=logs/gap_sweep_v006/s1/seed_0,... --variant s2=... \
    --fold test --through-gap 30 --demo-horizon 30 --out logs/gap_sweep_v006/compare_test --docs docs/gap_bridging_v006_comparison.md
.venv/bin/python scripts/bridge_gaps.py audit --current logs/gap_sweep_v005/current_short/seed_0/max_gap_30 \
    --candidate logs/gap_sweep_v006/s1/seed_0/max_gap_30 --observations $OBS --out docs/figures/gap_bridging_v006_audit_s1 --n 20
.venv/bin/python scripts/bench_gap_bridging.py --tiles 1,16 --repeats 5 --configs current=configs/tracking/gap_bridging_v001.yaml \
    s1=configs/tracking/gap_bridging_v006_s1.yaml s2=configs/tracking/gap_bridging_v006_s2.yaml --out logs/bench/gap_bridging_v006.csv
```

## v007: scaling to the 2,000-frame demo

v007 (2026-11-05) runs the three strongest linkers on the 2,000-frame demo
(`demo/crab_cam2_2_middle_2000/`, original Cam 2_2 frames 13410–15409, copied
from `my-code` beside the 300-frame demo) and measures how their accuracy,
natural links and cost change with the amount of data. The linkers are the
v003 hand cost (the default), v006 S1 at its pipeline threshold (0.413) and
v005 S1 at p ≥ 0.5 with the dummy tied (the v005 exploratory overlay). Each
learned scorer is run twice: frozen (the committed model, trained on the
300-frame demo) and retrained on the window's own hand-cost sweeps
(`train --threshold-rule pipeline --features all`; the p ≥ 0.5 variant is the
same refit at 0.5, since the threshold rule does not change the fit). No code
path of the pipeline changed and no default moved; the study adds
`scripts/gap_scaling.py` (windows, overlays, natural links, timing, report),
`scripts/gap_scaling_report.py` and the Slurm chain
`jobs/gap_scaling_v007{.slurm,_submit.sh}`. Generated tables:
[gap_bridging_v007_scaling.md](gap_bridging_v007_scaling.md).

### Design

- **Nested windows** of 300, 600, 1000 and 2000 frames, centred on the
  300-frame demo (14250–14549) and shifted to stay inside the bundle. The
  300-frame window is row-identical to `demo/crab_cam2_2_middle_300` on every
  source column, so it reproduces earlier results exactly: all 15 frozen and
  hand-cost sweeps (5 seeds each) are byte-identical to the committed v005 /
  v006 sweeps, and the retrained S1 on that window chooses 0.412646, the
  committed v006 threshold.
- **Hold-out sweeps** with the v005 short profile, horizons 5–60, 5 seeds.
  Constant plant density (8 plants per gap length per 300 frames) is
  infeasible beyond 300 frames: each plant needs its own unused tracklet with
  enough consecutive accepted frames (56 for gap 45), and long tracklets grow
  more slowly than the window. `gap_scaling.py windows` probes the largest
  count that plants for every seed: **8, 15, 19, 34** plants per gap length
  (targets 8, 16, 27, 53), decoys and flicker at the recipe's 1/2 and 1/4
  (rounded down). Over 5 seeds this gives 600, 1074, 1385 and 2485 scored
  plants (decoy targets included) and 223, 403, 550 and 1002 on the test fold.
- **Promotion rule** as in v005/v006, applied at **horizon 30 on every
  window**: test-fold precision 1.0 and recall ≥ the hand cost at every
  planted gap through 30, and natural links at horizon 30 ≥ the hand cost's
  on that window. `compare` itself checks at the baseline's chosen horizon,
  which drifts (30 fallback, 10, 30 fallback, 5 on the four test folds), so its verdicts
  are not comparable across windows; both are tabulated.
- **Plants only.** The scored plants include decoy targets: a donor right half
  placed 3 px from the true right half. When both true halves are 1 frame
  long no cue separates them, and every linker takes the decoy; this is the
  one gap-30 false link the hand cost makes on the 300-frame demo over all
  plants. The report therefore also scores plants only (`gap_scaling.py
  natural`, `role_scores`; with decoy targets included it reproduces
  `compare`'s counts exactly).
- **Runtime and memory**: `bridge_gaps.py run` at max_gap 30 under
  `/usr/bin/time -v`, 3 repeats per window and linker, one Slurm node.

### Results

Test fold, horizon 30, gaps 1–60 pooled (recovered / false links of planted;
decoy targets included):

| frames | hand cost | v006 S1 frozen | v005 S1 p ≥ 0.5 frozen | v006 S1 retrained | v005 S1 p ≥ 0.5 retrained |
|---:|---|---|---|---|---|
| 300 | 172 / 1 of 223 | 179 / 1 | 178 / 1 | 179 / 1 | 178 / 1 |
| 600 | 312 / 2 of 403 | 317 / 1 | 317 / 1 | 317 / 1 | 319 / 1 |
| 1000 | 399 / 3 of 550 | 422 / 0 | 418 / 0 | 217 / 0 | 426 / 0 |
| 2000 | 734 / 9 of 1002 | 774 / 5 | 770 / 5 | 434 / 0 | 781 / 8 |
| natural links, 2000 frames (h = 30) | 5095 | 5095 | 5066 | 3484 | 5114 |

- **The hand cost loses its precision guarantee with more data.** At
  horizon 30 it fails precision 1.0 on the test fold at 600 (gaps 5, 30),
  1000 (gaps 3, 30) and 2000 frames (gaps 10–30). On plants only (no decoys)
  it makes 0, 0, 4 and 4 false links over all plants through gap 30 on the
  four windows (1360 plants at 2000 frames), and its own horizon rule picks 5
  frames on the 2000-frame test fold. The 300-frame demo was the easiest
  window it has been judged on.
- **Frozen learned scorers keep their recall advantage.** Trained on the
  300-frame demo, v006 S1 and v005 S1 p ≥ 0.5 keep it on 6.7 times more data
  (77.2% and 76.8% test recall at 2000 frames vs 73.3%; at gap 30 they
  recover 88 and 87 of 99 test plants vs 61) with fewer false links than the
  hand cost on the test fold (5 vs 9) but more over all plants through gap 30
  without decoys (7 and 6 vs 4 of 1360). The natural links match the hand
  cost's count (5095 for S1, Jaccard 0.95; 127 gained, 127 lost) and the
  trajectory lengths closely match the hand cost's (median 16 vs 15 frames,
  89% of trajectory frames in trajectories of 150 frames or more for both;
  figure `gap_bridging_v007_scaling_lengths.png`; no formal test). Their
  advantage is in recall only: on plants only they make more false links,
  and the larger windows contain the 300 frames they were trained on (see
  residual risks). They fail the rule at
  2000 frames on precision (gaps 10–20) and on one plant of recall at gaps
  1, 2 and 10.
- **Retraining improves the model but the pipeline threshold rule breaks.**
  The retrained thresholds are 0.413, 0.706, 0.990 and 0.990: the rule asks
  for zero false links on every train-fold plant through gap 30, and with
  more plants one confident false link appears and pins the threshold near 1.
  On the 1000-frame window it is a short_mid plant at gap 30 (seed 3, frames
  14749–14778 deleted, halves of 4 and 1 frames): a separate natural 1-frame
  detection inside the deleted interval, 6 px from the plant at the same
  burrow (frame 14759), is linked to the right half (p 0.94 for the final
  refit, ≥ 0.989 for the fold-0 model) while the true left half is refused.
  The truth table calls this false; it is most likely the same crab, i.e. a
  labelling artefact of planting a gap over a fragmented track rather than a
  scorer error. The retrained v006 S1 then recovers 43% of the
  test plants and loses a third of the natural links (3484 of 5095 at 2000
  frames). The same refit at p ≥ 0.5 has the highest recall on every window
  above 300 frames: 77.9% test recall at 2000 frames (781 vs 770 frozen and
  734 hand), 95.8% over all plants at horizon 60 (vs 80.9% hand), and,
  with the frozen v006 S1, at or above the natural-link floor on the 600,
  1000 and 2000-frame windows (1708, 2756, 5114 links; frozen S1 1699, 2748,
  5095). The gain is small on 600 frames (319 vs 317 of 403) and costs false
  links on 2000 frames: 8 on the test fold (5 for each frozen variant, 9 for
  the hand cost) and 11 over all plants (9 frozen p ≥ 0.5, 14 hand); 5 of
  1360 plants-only, against 4 for the hand cost.
- **No variant passes the rule on any window above 300 frames**, the hand
  cost included. On the 1000-frame window both p ≥ 0.5 variants and the
  frozen S1 pass precision and fail recall at one gap: the frozen variants by
  two plants at gap 2 (58 vs 60), the retrained p ≥ 0.5 variant by one at
  gap 5 (47 vs 48); the frozen p ≥ 0.5 variant also misses the natural-link
  floor (2733 vs 2748).
- **Cost grows roughly linearly or slower** (four windows, 300 to 2000
  frames). Pipeline time grows as frames^0.88–0.96 (wall time as
  frames^0.69–0.73, with a fixed start-up cost)
  (link stage frames^0.90–0.99) and peak memory as frames^0.6: 12.6 s and
  1.1 GiB for the hand cost on 2000 frames, 13.1 and 13.9 s for the learned
  scorers (their link stage costs 1.7 times the hand cost's, 1.2 vs 0.7 s,
  but assembly dominates). Retraining under the pipeline rule took 19, 37,
  51 and 100 min (peak 0.34 to 0.66 GiB) on the four windows.

### Visual audit (v007)

`bridge_gaps.py audit` on the 2000-frame window (seed-0 runs at horizon 30,
hand cost as current, video stitched losslessly from the 10 chunks; all 2000
decoded frames match the bundle's `decoded_bgr_sha256`), 20 links rendered per
side. Frozen v006 S1 gains 127 and loses 127 natural links (pages 1–4 of each
reviewed): every gained link joins the same crab at the same burrow or claw
position across 1 to 30 frames, refused by the hand cost above `max_cost`
(0.97 to 4.4) and accepted at cost 0.02 to 0.83 (p 0.44 to 0.98); every lost
link (1 to 30 frames, hand cost 0.09 to 0.96) is just as plausible. Frozen
v005 S1 p ≥ 0.5 (117 gained, 146 lost) has the same character. As on the
300-frame demo, the audit shows no visibly wrong link on either side; the
linkers order stationary burrow links differently. The retrained v006 S1 at
0.990 gains 1 link and loses 1612: the lost links are plainly correct 1 to
5-frame continuations of one crab that it scores at p ≈ 0.9 and refuses at
the threshold, which is the collapse seen in the hold-out. Committed pages:
`docs/figures/gap_bridging_v007_audit_w2000_s1_{gained_p4,lost_p1}.png`,
`..._s1_retrained_lost_p1.png` and the three audit CSVs beside them (all
pages in `logs/gap_scaling_v007/audit/`).

### Verdict (v007)

- **Defaults unchanged.** No linker meets the promotion rule beyond 300
  frames, so by the agreed rule nothing is promoted.
- **What the scaling shows.** The rule's precision-1.0 requirement is a
  zero-error requirement on a sample that grows with the data, and on larger
  windows it also meets labelling artefacts (decoys with 1-frame halves,
  plants over fragmented tracks); on 2000 frames the hand cost fails it and
  the three learned variants that keep a moderate threshold fail it by a
  comparable margin, while recovering 3.5–4.6 points more test plants at
  horizon 30 and 12–15 points more of all plants at horizon 60. Retraining on
  more data raises the model's recall (the p ≥ 0.5 refit recovers more than
  the frozen one on every window above 300 frames, at a few more false links
  on 2000 frames), but the
  pipeline threshold rule, which chose well on 300 plants, collapses to 0.99
  on more. Any promotion on the larger data set needs a rule decision first:
  a precision target below 1.0 (or a false-link rate per plant), decoy
  targets judged separately, or a recall floor pooled over gaps instead of
  per gap.
- **Recommended candidate if the rule is relaxed:** v005 S1 at p ≥ 0.5
  retrained on the 2000-frame sweeps
  (`logs/gap_scaling_v007/models/w2000/gap_bridging_s1p50.yaml`, git-ignored;
  re-create with the commands below).

### Residual risks (v007)

- Windows are nested, not independent: the 300-frame demo is inside every
  window, and the frozen scorers were trained on its plants, so their
  results on larger windows are partly in-sample (the share falls from 100%
  to 15% of frames).
- Plant density falls with window length (8 per 300 frames to 34 per 2000),
  and plants require long, cleanly tracked tracklets, so the hold-out
  over-represents easy crabs more on larger windows; the natural-link
  agreement (Jaccard 0.95) is the check that does not depend on plants.
- The short-profile half-length mix (1/2/3 frames at 0.58/0.27/0.15) was
  measured on the 300-frame demo and is not re-measured per window.
- The audit is one seed and 20 links per side on the 2000-frame window.

### Commands (v007)

```bash
.venv/bin/python scripts/gap_scaling.py windows --source demo/crab_cam2_2_middle_2000/observations.parquet \
    --out logs/gap_scaling_v007/windows --sizes 300,600,1000,2000        # probes the feasible plant counts
.venv/bin/python scripts/gap_scaling.py overlay --scorer configs/tracking/link_scorer_v005_s1.joblib \
    --probability 0.5 --out logs/gap_scaling_v007/configs/s1p50_frozen.yaml
bash jobs/gap_scaling_v007_submit.sh          # sweeps, retraining, timing, compare, natural links, audit (Slurm)
.venv/bin/python scripts/gap_scaling.py report --root logs/gap_scaling_v007 --sizes 300,600,1000,2000 \
    --docs docs/gap_bridging_v007_scaling.md --figures docs/figures
```
