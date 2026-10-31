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
| `max_gap` | 45 | Bridge `1 <= gap <= max_gap` missing frames; 0 disables linking (CLI `--max-gap`); 5 until v002 |
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
