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
missing frames (default 5; the horizon sweep below recommends 15). It fills each missing frame with either a real
detection the tracker had rejected (**recovered**) or a synthesized pose
(**interpolated**). It writes new tables and an audit row for every candidate
link. The input table is never modified, and its rows are carried into the
output value- and type-identical (pyarrow `Table.equals`).

Implementation: `scripts/gap_bridging/` (`geometry.py`, `candidates.py`,
`chains.py`, `recover.py`, `tables.py`, `pipeline.py`; the evaluation tools
`fit.py`, `holdout.py` and `figures.py`).
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
| `max_gap` | 5 | Bridge `1 <= gap <= max_gap` missing frames; 0 disables linking (CLI `--max-gap`) |
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

## Outputs

These files are written to `--out`. Each parquet is first written to
`.<name>.tmp` and then moved into place, and the marker is written last. Existing
outputs are refused unless `--force` is given.

**`observations.parquet`**. All input rows in their original order, with the
original columns unchanged, followed by five new columns:

| Column | Type | Content |
|---|---|---|
| `trajectory_id` | string | Trajectory of the row. Null for rejected rows that were not recovered. |
| `source` | string | `observed`, `recovered` or `interpolated` |
| `bridge_id` | string | `<trajectory_id>/bNNN` for recovered and interpolated rows, otherwise null |
| `gap_frames` | int64 | Number of missing frames of the row's bridge (recovered frames included), on recovered and interpolated rows; null on observed and unassigned rows |
| `fill_confidence` | string | `high` / `medium` / `low` from `gap_frames` and `fill_confidence_gaps`, on recovered and interpolated rows; otherwise null |

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
and its `rejection_reason`. Only the five new columns mark it as adopted.

**`trajectories.parquet`**. One row per trajectory, with these columns:
`trajectory_id`, `video_id`, `start_frame`, `end_frame`, `n_observed`,
`n_recovered`, `n_interpolated`, `n_tracklets`, and `member_tracklet_ids`
(list<string>, in time order). Every trajectory is contiguous:
`end_frame − start_frame + 1 = n_observed + n_recovered + n_interpolated`.

**`links.parquet`**. One audit row per candidate pair, with these columns
(under the sqrt model, pairs beyond the prefilter radius `R_max` are not
candidates and have no row; see [Gate model v002](#gate-model-v002)):
`video_id`, `tracklet_a`, `tracklet_b`, `gap_frames`, `cost` (NaN when a motion
gate failed), `runner_up_cost` (NaN when there is no competitor), `decision`,
`reason`, `n_chain_ignored` and `stage`. `stage` (int64) is the 1-based
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
  `n_observed`, `n_recovered`, `n_interpolated`, `n_unassigned`,
  `decisions` and `candidate_pairs` (`enumerated`: pairs with an audit row;
  `pruned_by_radius`: pairs inside the time window but beyond `R_max`, which
  were not enumerated; always 0 under linear).
- `versions`: the numpy, scipy and pyarrow versions.
- `outputs`: the `sha256` and `rows` of each parquet.

Before writing, the run checks four invariants and raises an error on any
violation:
- Every tracklet is in exactly one trajectory.
- `trajectories = tracklets − links`.
- Output rows = input rows + interpolated rows.
- Every trajectory is contiguous, with no duplicate frame.

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
current default (sqrt, `max_gap` 5) is the first row of the
[sweep table](#horizon-sweep-results).
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
`--holdout`, the hold-out at each (planted gap lengths capped at the
horizon), then writes the tables and figures:

```bash
.venv/bin/python scripts/bridge_gaps.py sweep \
    --observations demo/crab_cam2_2_middle_300/observations.parquet \
    --out logs/gap_sweep --max-gaps 5,10,15,30,45,60 \
    --holdout [--holdout-gaps 1,2,3,5,10,15,20,30,45,60] [--per-gap 8] [--seed 0] \
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
  `holdout_planted`, `holdout_precision`, `holdout_recall`,
  `holdout_unplanted_links`. Lengths are `end_frame − start_frame + 1` of the
  trajectories (interpolated frames included).
- `holdout_by_gap.csv`: long form `max_gap, gap, planted, recovered,
  false_link, missed, recall, precision`.
- `figures/`: `sweep.pdf` (links and trajectory length vs horizon),
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
