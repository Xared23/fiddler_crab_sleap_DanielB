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
missing frames (default 5). It fills each missing frame with either a real
detection the tracker had rejected (**recovered**) or a synthesized pose
(**interpolated**). It writes new tables and an audit row for every candidate
link. The input table is never modified, and its rows are carried into the
output byte-identical.

Implementation: `scripts/gap_bridging/` (`geometry.py`, `candidates.py`,
`chains.py`, `recover.py`, `tables.py`, `pipeline.py`).
Parameters: `configs/tracking/gap_bridging_v001.yaml`.
Tests: `tests/test_gap_bridging_*.py`.

## Method

Notation: tracklet A ends at frame e and tracklet B starts at frame s, in the
same video. `dt = s - e`, and the number of missing frames is `gap = dt - 1`.
Units are native-image pixels, px/frame and integer frames.

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
   then linked, and the pair is not retried. The other outcomes of the audit are:
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
   the chord velocity, which makes the fill linear.
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
defaults are used; they are identical to this file.

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
| `minimum_cost_margin` | 4.0 | Veto when the runner-up is within this (CLI `--margin`) |
| `recovery_radius_px` | 8.0 | Radius for adopting a rejected detection (CLI `--recovery-radius`) |
| `recoverable_reasons` | 5 reasons (see method, step 8) | Rejection reasons eligible for recovery |
| `fps` | 30000/1001 | Sets `time_s = frame / fps` on interpolated rows when the input has `time_s` |

## Outputs

These files are written to `--out`. Each parquet is first written to
`.<name>.tmp` and then moved into place, and the marker is written last. Existing
outputs are refused unless `--force` is given.

**`observations.parquet`**. All input rows in their original order, with the
original columns unchanged, followed by three new columns:

| Column | Type | Content |
|---|---|---|
| `trajectory_id` | string | Trajectory of the row. Null for rejected rows that were not recovered. |
| `source` | string | `observed`, `recovered` or `interpolated` |
| `bridge_id` | string | `<trajectory_id>/bNNN` for recovered and interpolated rows, otherwise null |

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
and its `rejection_reason`. Only the three new columns mark it as adopted.

**`trajectories.parquet`**. One row per trajectory, with these columns:
`trajectory_id`, `video_id`, `start_frame`, `end_frame`, `n_observed`,
`n_recovered`, `n_interpolated`, `n_tracklets`, and `member_tracklet_ids`
(list<string>, in time order). Every trajectory is contiguous:
`end_frame − start_frame + 1 = n_observed + n_recovered + n_interpolated`.

**`links.parquet`**. One audit row per candidate pair, with these columns:
`video_id`, `tracklet_a`, `tracklet_b`, `gap_frames`, `cost` (NaN when a motion
gate failed), `runner_up_cost` (NaN when there is no competitor), `decision` and
`reason`.

**`GAP_BRIDGING_COMPLETE.json`** contains:
- `config`: the full parameter set.
- `input`: `path`, `sha256`, `rows` and `tracklets`. For a directory input it
  also lists the `files` with their hashes, and `sha256` is the hash of the
  sorted listing `"<relpath>\t<sha256>\n"`.
- `counts`: `videos`, `tracklets`, `links`, `trajectories`, `bridges`,
  `n_observed`, `n_recovered`, `n_interpolated`, `n_unassigned` and
  `decisions`.
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
- **Self-competition vetoes many true gap-1 links.** The cost is per frame, so
  skipping one of a crab's own short tracklets costs about the same as linking
  to it. Example: tracklet A, then a 1-frame tracklet B two frames later, then C.
  The pair A→C competes with A→B at nearly equal cost, so the margin veto
  refuses A→B.
  - On the demo, 208 of the 214 `refused_margin` decisions have such a "skip"
    pair as their runner-up.
  - The veto is conservative: it never creates a wrong link, but it leaves
    these breaks unbridged.
  - A chain-aware runner-up rule is a candidate follow-up.
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
time requests above are estimates and have not been measured.

## Demo results

These numbers come from the default config (`gap_bridging_v001.yaml`) applied to
`demo/crab_cam2_2_middle_300/observations.parquet`: 36,573 rows, 300 frames,
`Crab_Cam_2_2` frames 14250–14549. Input sha256
`739c2daf087bd1967d53dcfd8d41653166d7d30b9cc713d5ba5203a41432e1ef`. The run took
1.4 s, and a forced rerun gave byte-identical parquet files.

| Quantity | Tracklets (input) | Trajectories (bridged) |
|---|---:|---:|
| Count | 1,181 | 731 |
| Median length (frames) | 3 | 6 |
| Mean length (frames) | 30.9 | 51.4 |
| 90th percentile length | 99 | 280 |
| Length 1 frame | 398 | 227 |
| Spanning all 300 frames | 62 | 69 |
| Fraction of frames in units ≥ 30 frames | 0.871 | 0.920 |
| Fraction of frames in units ≥ 150 frames | 0.709 | 0.767 |

| Count | Value |
|---|---:|
| Links | 450 |
| Trajectories with ≥ 2 tracklets | 224 (max 10 tracklets) |
| Bridged frames | 1,055 |
| Recovered detections (all `low_body_score`; 54 rejected rows available) | 12 |
| Interpolated rows | 1,043 |
| Output rows (36,573 + 1,043) | 37,616 |
| Rejected rows left unassigned | 42 |

| Gap (frames) | linked | refused_margin | refused_assignment | refused_gate |
|---:|---:|---:|---:|---:|
| 1 | 170 | 137 | 0 | 3,733 |
| 2 | 103 | 62 | 5 | 3,761 |
| 3 | 72 | 12 | 39 | 3,643 |
| 4 | 62 | 2 | 54 | 3,662 |
| 5 | 43 | 1 | 48 | 3,695 |
| **Total** | **450** | **214** | **146** | **18,494** |

- **Gate failures:** 18,480 failed on displacement and 14 on `max_cost`.
- **Linked costs:** median 0.65, maximum 8.76 (px-equivalent per frame).
- **Interpolated poses:** every interpolated row has a heading and all five
  nodes, because every bridged endpoint had a usable Shoulder.

The gains are modest because most `refused_margin` decisions come from the
self-competition described under Limitations. The median length rises from 3 to
6 frames, the number of 1-frame units drops from 398 to 227, and the fraction of
frames in units of 30 frames or more rises from 0.871 to 0.920. These are
continuity statistics only; they do not measure identity accuracy.
