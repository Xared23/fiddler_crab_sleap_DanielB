#!/usr/bin/env python3

"""
Frame-deletion hold-out evaluation of gap bridging.

``plant_gaps`` deletes a run of ``g`` consecutive frames from the interior of
real, contiguous tracklets and splits each into a left half (old id) and a
right half (new id), exactly as the adjacent-frame tracker would have done had
the detections been missed. Optionally (v003) it also plants decoys (the same
gap in a target and a donor tracklet whose appearance cues differ, with the
donor's right half moved next to the target's right half) and flicker
singletons (``g`` frames deleted on each side of one kept frame).
``score_links`` compares a ``links.parquet`` audit table with the planted
truth: per gap length, how many planted pairs were re-linked (recovered), how
many planted halves were linked to something else (false_link) and how many
were left alone (missed), plus the decoy and flicker outcomes.
``run_holdout`` plants, runs :func:`gap_bridging.pipeline.run` and writes the
scores.

All table work is pyarrow; rows and columns not touched by planting are kept
byte-identical.

Sam
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# ----- numerics -----
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

# ----- local modules -----
from gap_bridging import geometry, pipeline
from gap_bridging.config import BridgeConfig

# ============================================================
# CONSTANTS
# ============================================================

ROLES = ('plant', 'decoy_target', 'decoy_donor', 'flicker')
TRUTH_SCHEMA = pa.schema([
    ('left_id', pa.string()), ('right_id', pa.string()), ('original_id', pa.string()),
    ('gap_frames', pa.int64()), ('deleted_first_frame', pa.int64()),
    ('deleted_last_frame', pa.int64()),
    ('role', pa.string()),          # one of ROLES
    ('middle_id', pa.string()),     # flicker: the kept one-frame tracklet; else null
    ('partner_id', pa.string()),    # decoy_target: donor right half; decoy_donor: target left half
])
SCORES_SCHEMA = pa.schema([
    ('gap_frames', pa.int64()), ('planted', pa.int64()), ('recovered', pa.int64()),
    ('false_link', pa.int64()), ('missed', pa.int64()), ('recall', pa.float64()),
    ('precision', pa.float64()),
    ('decoy_planted', pa.int64()), ('decoy_recovered', pa.int64()),
    ('decoy_false_link', pa.int64()),
    ('flicker_planted', pa.int64()), ('flicker_recovered', pa.int64()),
    ('flicker_spurious', pa.int64()),
])
COORDINATE_SUFFIXES = ('_x', '_y')
ALL_GAPS = -1
PLANTED_NAME = 'planted_observations.parquet'
TRUTH_NAME = 'holdout_truth.parquet'
SCORES_NAME = 'holdout_scores.csv'
BRIDGED_DIR = 'bridged'
NEW_INCOMING_REASON = 'no_candidate'

# ============================================================
# PLANTING
# ============================================================


def _tracklet_spans(obs: pa.Table) -> Tuple[np.ndarray, List[str], Dict[int, Tuple[int, int]]]:
    """Dictionary codes of ``tracklet_id`` and the contiguous span of every tracklet.

    Parameters
    ----------
    obs : pa.Table
        Observation table with ``tracklet_id``, ``frame`` and
        ``accepted_for_tracking``.

    Returns
    -------
    codes : np.ndarray
        ``(n_rows,)`` int64 code into ``ids``; -1 where ``tracklet_id`` is null.
    ids : list of str
        Distinct tracklet ids (code order).
    spans : dict
        ``code -> (first_frame, n_frames)`` for every tracklet whose accepted
        rows cover consecutive, unique frames (a real gap inside a tracklet
        excludes it, so planted gaps never overlap one).
    """
    for name in ('tracklet_id', 'frame', 'accepted_for_tracking'):
        if name not in obs.schema.names:
            raise ValueError(f"observation table has no column {name!r}")
    encoded = pc.dictionary_encode(obs['tracklet_id'].combine_chunks())
    codes = encoded.indices.fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int64)
    ids = encoded.dictionary.to_pylist()
    frames = obs['frame'].fill_null(-1).to_numpy().astype(np.int64)
    accepted = obs['accepted_for_tracking'].fill_null(False).to_numpy(zero_copy_only=False)
    rows = np.flatnonzero(accepted & (codes >= 0))
    order = rows[np.lexsort((frames[rows], codes[rows]))]
    c, f = codes[order], frames[order]
    same = c[1:] == c[:-1]
    broken = set(c[1:][same & (np.diff(f) != 1)].tolist())
    first = np.flatnonzero(np.r_[True, ~same])
    counts = np.diff(np.r_[first, c.size])
    spans = {int(c[i]): (int(f[i]), int(n)) for i, n in zip(first, counts)
             if int(c[i]) not in broken}
    if len(ids) != np.unique(codes[codes >= 0]).size:
        raise RuntimeError("tracklet id encoding is inconsistent")
    return codes, ids, spans


Cue = Tuple[float, int, float, int]          # (size_px, size_n, claw_side, claw_side_n)
Operation = Tuple[str, int, int, List[Tuple[int, int]], int]
# (role, code, gap, deleted (first, last) runs, partner operation index or -1)


class _Cues:
    """Appearance cues (:func:`geometry.tracklet_cues`) of arbitrary row subsets."""

    def __init__(self, obs: pa.Table):
        self.xy, self.valid = geometry._node_arrays(obs)

    def of(self, rows: np.ndarray) -> Cue:
        if rows.size == 0:
            return (math.nan, 0, 0.0, 0)
        size, size_n, side, side_n = geometry.tracklet_cues(
            self.xy[rows], self.valid[rows], np.array([0]), np.array([rows.size - 1]))
        return float(size[0]), int(size_n[0]), float(side[0]), int(side_n[0])

    def shell(self, row: int) -> np.ndarray:
        return self.xy[row, geometry.SHELL]


def cues_differ(a: Cue, b: Cue, cfg: BridgeConfig) -> bool:
    """True when two cue sets differ enough for a decoy.

    Sizes differ by at least ``2 * cfg.size_scale_px`` (both known), or the
    claw sides are confidently opposite (both ``|side| >= claw_side_confidence``
    over at least ``claw_side_min_frames`` frames, opposite signs: the
    linker's veto condition).
    """
    size = (a[1] > 0 and b[1] > 0 and math.isfinite(a[0]) and math.isfinite(b[0])
            and abs(a[0] - b[0]) >= 2.0 * cfg.size_scale_px)
    side = (abs(a[2]) >= cfg.claw_side_confidence and abs(b[2]) >= cfg.claw_side_confidence
            and a[3] >= cfg.claw_side_min_frames and b[3] >= cfg.claw_side_min_frames
            and a[2] * b[2] < 0)
    return bool(size or side)


def _choose(spans: Dict[int, Tuple[int, int]], ids: List[str], gap_lengths: Sequence[int],
            per_gap: int, min_tracklet_len: int, margin: int, seed: int, *,
            decoys: int = 0, flicker: int = 0, rows_of: Dict[int, np.ndarray] | None = None,
            frames: np.ndarray | None = None, videos: Dict[int, str] | None = None,
            cues: _Cues | None = None, cfg: BridgeConfig | None = None) -> List[Operation]:
    """Draw the plants, decoys and flicker singletons; no tracklet twice.

    Per gap length (in the given order): ``per_gap`` plants, then ``decoys``
    target/donor pairs, then ``flicker`` singletons, all from the tracklets
    not used so far. Plants are mandatory (ValueError when the pool is
    short); decoys and flicker singletons are best effort: fewer are drawn
    when no further eligible tracklet (or decoy pair with differing cues)
    remains, and the truth table records what was planted. Eligible tracklets are sorted by id so the draw depends
    only on ``seed`` and the eligible set, not on row order; with
    ``decoys == flicker == 0`` the draw equals the v002 plant draw, and the
    draws of the shorter gap lengths never depend on the longer ones.
    """
    rng = np.random.default_rng(seed)
    remaining = sorted(spans, key=lambda code: ids[code])
    ops: List[Operation] = []

    def eligible(need: int) -> List[int]:
        return [code for code in remaining if spans[code][1] >= need]

    def offset(code: int, length: int) -> int:
        first, n = spans[code]
        return first + int(rng.integers(margin, n - length - margin + 1))

    for gap in gap_lengths:
        need = max(min_tracklet_len, gap + 2 * margin)
        pool = eligible(need)
        if len(pool) < per_gap:
            raise ValueError(f"only {len(pool)} eligible tracklets (>= {need} frames, unused) "
                             f"for gap {gap}; need {per_gap}")
        picked = [pool[i] for i in rng.choice(len(pool), size=per_gap, replace=False)]
        for code in picked:
            first_del = offset(code, gap)
            ops.append(('plant', code, int(gap), [(first_del, first_del + gap - 1)], -1))
        taken = set(picked)
        remaining = [code for code in remaining if code not in taken]

        for _ in range(decoys):
            pair = _choose_decoy(eligible(need), remaining, spans, ids, gap, margin, rng,
                                 rows_of, frames, videos, cues, cfg)
            if pair is None:      # best effort: fewer decoys for this gap (see truth/scores)
                break
            target, donor, first_del = pair
            run = [(first_del, first_del + gap - 1)]
            ops.append(('decoy_target', target, int(gap), run, len(ops) + 1))
            ops.append(('decoy_donor', donor, int(gap), run, len(ops) - 1))
            remaining = [code for code in remaining if code not in (target, donor)]

        if flicker:
            span = 2 * gap + 1
            need_f = max(min_tracklet_len, span + 2 * margin)
            pool = eligible(need_f)       # best effort: fewer when the pool is short
            picked = [pool[i] for i in rng.choice(len(pool), size=min(flicker, len(pool)),
                                                  replace=False)]
            for code in picked:
                first_del = offset(code, span)
                kept = first_del + gap
                ops.append(('flicker', code, int(gap),
                            [(first_del, kept - 1), (kept + 1, kept + gap)], -1))
            taken = set(picked)
            remaining = [code for code in remaining if code not in taken]
    codes = [op[1] for op in ops]
    if len(set(codes)) != len(codes):
        raise RuntimeError("a tracklet was planted twice")
    return ops


def _choose_decoy(targets: List[int], remaining: List[int], spans: Dict[int, Tuple[int, int]],
                  ids: List[str], gap: int, margin: int, rng: np.random.Generator,
                  rows_of: Dict[int, np.ndarray], frames: np.ndarray, videos: Dict[int, str],
                  cues: _Cues, cfg: BridgeConfig) -> Tuple[int, int, int] | None:
    """One ``(target, donor, deleted_first_frame)`` with differing cues, or None.

    Targets are tried in a random order. For a target C the deleted run
    starts at a uniform offset as for a plant; donors are the unused
    tracklets of the same video covering ``margin`` frames on each side of
    the run, whose right half's cues differ (:func:`cues_differ`) from C's
    left half; one is drawn uniformly, from the donors that are not eligible
    targets themselves when there are any.
    """
    eligible_set = set(targets)
    for t in rng.permutation(len(targets)):
        target = targets[int(t)]
        first, n = spans[target]
        first_del = first + int(rng.integers(margin, n - gap - margin + 1))
        lo, hi = first_del - margin, first_del + gap - 1 + margin
        rows_c = rows_of[target]
        left_c = cues.of(rows_c[frames[rows_c] < first_del])
        donors = []
        for code in remaining:
            a_first, a_n = spans[code]
            if code == target or videos[code] != videos[target] or a_first > lo \
                    or a_first + a_n - 1 < hi:
                continue
            rows_a = rows_of[code]
            right = rows_a[frames[rows_a] > first_del + gap - 1]
            if not np.isfinite(cues.shell(right[0])).all():
                continue
            if cues_differ(left_c, cues.of(right), cfg):
                donors.append(code)
        if donors:
            # Donors need not be eligible plants; draw from the ineligible ones when any
            # qualify, so decoys use up as few plant-eligible tracklets as possible.
            spare = [code for code in donors if code not in eligible_set]
            donors = spare or donors
            donor = donors[int(rng.integers(len(donors)))]
            return target, donor, first_del
    return None


def _replace(column: pa.ChunkedArray, rows: np.ndarray, values: list, n_rows: int) -> pa.Array:
    """``column`` with ``values`` written at the sorted row indices ``rows``."""
    if len(values) != rows.size:
        raise ValueError("one replacement value per row is required")
    array = column.combine_chunks()
    if rows.size == 0:
        return array
    mask = np.zeros(n_rows, dtype=bool)
    mask[rows] = True
    return pc.replace_with_mask(array, pa.array(mask), pa.array(values, type=array.type))


def plant_gaps(obs: pa.Table, gap_lengths: Sequence[int], per_gap: int,
               min_tracklet_len: int = 40, margin: int = 3,
               seed: int = 0, decoys: int = 0, decoy_offset_px: float = 3.0,
               flicker: int = 0,
               cue_config: BridgeConfig | None = None) -> Tuple[pa.Table, pa.Table]:
    """Delete interior frame runs from real tracklets and split them.

    Parameters
    ----------
    obs : pa.Table
        Published observation table (not modified).
    gap_lengths : sequence of int
        Gap lengths to plant, each >= 1; planted in the given order.
    per_gap : int
        Number of distinct tracklets per gap length (>= 1).
    min_tracklet_len : int
        Minimum number of accepted frames of an eligible tracklet. A tracklet
        must also have at least ``g + 2 * margin`` frames for gap ``g``, and
        its accepted frames must be consecutive and unique.
    margin : int
        Minimum number of frames kept on each side of the deleted run (>= 1);
        the run starts at a uniform offset in ``[margin, len - g - margin]``.
    seed : int
        Seed of the ``numpy`` generator; equal inputs give equal outputs.
    decoys : int
        Decoy pairs per gap length. A target C (eligible as a plant) and a
        donor A (any unused contiguous tracklet of the same video covering
        ``margin`` frames around the run) get the same deleted run, so their
        right halves start on the same frame ``s``; A's right half must
        differ in appearance from C's left half (:func:`cues_differ` under
        ``cue_config``). Every coordinate column (``*_x``, ``*_y``) of A's
        right half is translated by ``shell_C(s) - shell_A(s) +
        (decoy_offset_px, 0)``, so it starts ``decoy_offset_px`` from C's
        right half and competes with it for C's left end.
    decoy_offset_px : float
        Distance of the moved donor start from the target start, along +x.
    flicker : int
        Flicker singletons per gap length: ``g`` frames are deleted on each
        side of one kept frame of an unused tracklet with at least
        ``max(min_tracklet_len, 2 g + 1 + 2 margin)`` frames, leaving a left
        part, a one-frame middle tracklet and a right part.
    cue_config : BridgeConfig, optional
        Cue thresholds for the decoy draw (default :class:`BridgeConfig`).

    Returns
    -------
    planted : pa.Table
        ``obs`` without the deleted rows. The first part of a split tracklet
        keeps its id; every later part (the n-th new tracklet, n from 1, in
        truth order) gets ``tracklet_id = f"{old}/holdout{n:03d}"``,
        ``tracklet_local_id = max(tracklet_local_id) + n`` and its first row
        ``incoming_reason = 'no_candidate'``. Decoy donor right halves are
        translated as above. Every other value is unchanged.
    truth : pa.Table
        One row per planted tracklet (:data:`TRUTH_SCHEMA`); ``left_id ==
        original_id``; ``role`` is ``plant``, ``decoy_target``,
        ``decoy_donor`` or ``flicker``. For a flicker ``right_id`` is the
        outer right part, ``middle_id`` the kept frame and the deleted frames
        are ``deleted_first_frame .. deleted_last_frame`` minus the middle.

    Raises
    ------
    ValueError
        On invalid arguments or when fewer than ``per_gap`` unused eligible
        tracklets remain for some gap length. Decoys and flicker singletons
        are best effort: when the pool runs short, fewer are planted for that
        gap length (count the truth rows by ``role``).
    """
    if per_gap < 1 or margin < 1 or min_tracklet_len < 1:
        raise ValueError("per_gap, margin and min_tracklet_len must be >= 1")
    if decoys < 0 or flicker < 0 or not math.isfinite(decoy_offset_px):
        raise ValueError("decoys and flicker must be >= 0 and decoy_offset_px finite")
    if not gap_lengths or any(int(g) != g or g < 1 for g in gap_lengths):
        raise ValueError(f"gap lengths must be integers >= 1, got {list(gap_lengths)}")
    codes, ids, spans = _tracklet_spans(obs)
    frames = obs['frame'].fill_null(-1).to_numpy().astype(np.int64)
    accepted = obs['accepted_for_tracking'].fill_null(False).to_numpy(zero_copy_only=False)
    extra = {}
    if decoys:
        rows = np.flatnonzero(accepted & (codes >= 0))
        rows = rows[np.lexsort((frames[rows], codes[rows]))]
        split = np.flatnonzero(np.diff(codes[rows])) + 1
        rows_of = {int(codes[r[0]]): r for r in np.split(rows, split) if r.size}
        video_col = obs['video_id'].to_pylist()
        extra = dict(rows_of=rows_of, frames=frames,
                     videos={code: video_col[r[0]] for code, r in rows_of.items()},
                     cues=_Cues(obs), cfg=cue_config or BridgeConfig())
    ops = _choose(spans, ids, [int(g) for g in gap_lengths], per_gap, min_tracklet_len,
                  margin, seed, decoys=decoys, flicker=flicker, **extra)
    local_max = pc.max(obs['tracklet_local_id']).as_py() or 0
    keep = np.ones(obs.num_rows, dtype=bool)
    new_id = np.full(obs.num_rows, -1, dtype=np.int64)    # new-tracklet number per row
    heads: List[int] = []
    new_names: List[str] = []
    parts: List[List[str]] = []        # per operation: ids of its parts in frame order
    rows_by_op: List[np.ndarray] = []
    for role, code, gap, runs, _ in ops:
        rows = np.flatnonzero(codes == code)
        old = ids[code]
        names = [old]
        for first_del, last_del in runs:
            keep[rows[(frames[rows] >= first_del) & (frames[rows] <= last_del)]] = False
        bounds = [last for _, last in runs]
        for k, last_del in enumerate(bounds):
            upper = runs[k + 1][0] if k + 1 < len(runs) else None
            part = rows[(frames[rows] > last_del)
                        & ((frames[rows] < upper) if upper is not None else True)]
            new_names.append(f"{old}/holdout{len(new_names) + 1:03d}")
            new_id[part] = len(new_names)
            heads.extend(part[(frames[part] == last_del + 1) & accepted[part]].tolist())
            names.append(new_names[-1])
        parts.append(names)
        rows_by_op.append(rows)
    n_deleted = sum(last - first + 1 for op in ops for first, last in op[3])
    if int((~keep).sum()) != n_deleted or len(heads) != len(new_names):
        raise RuntimeError("planted rows do not match the requested gaps")

    truth: Dict[str, list] = {name: [] for name in TRUTH_SCHEMA.names}
    for (role, code, gap, runs, partner), names in zip(ops, parts):
        if role == 'decoy_target':
            partner_id = parts[partner][-1]
        elif role == 'decoy_donor':
            partner_id = parts[partner][0]
        else:
            partner_id = None
        values = (names[0], names[-1], ids[code], gap, runs[0][0], runs[-1][1], role,
                  names[1] if role == 'flicker' else None, partner_id)
        for name, value in zip(TRUTH_SCHEMA.names, values):
            truth[name].append(value)

    right_rows = np.flatnonzero(new_id > 0)
    planted = obs
    for name, values in (
            ('tracklet_id', [new_names[i - 1] for i in new_id[right_rows]]),
            ('tracklet_local_id', [int(local_max) + int(i) for i in new_id[right_rows]])):
        index = planted.schema.get_field_index(name)
        planted = planted.set_column(index, planted.schema.field(index),
                                     _replace(planted[name], right_rows, values, obs.num_rows))
    index = planted.schema.get_field_index('incoming_reason')
    heads_sorted = np.sort(np.asarray(heads, dtype=np.int64))
    planted = planted.set_column(index, planted.schema.field(index), _replace(
        planted['incoming_reason'], heads_sorted, [NEW_INCOMING_REASON] * len(heads),
        obs.num_rows))
    planted = _translate_donors(planted, ops, rows_by_op, frames, accepted, decoy_offset_px,
                                extra.get('cues'))
    return planted.filter(pa.array(keep)), pa.Table.from_pydict(truth, schema=TRUTH_SCHEMA)


def coordinate_columns(schema: pa.Schema) -> List[str]:
    """Floating-point columns named ``*_x`` / ``*_y`` (body, nodes, centroid)."""
    return [f.name for f in schema if f.name.endswith(COORDINATE_SUFFIXES)
            and pa.types.is_floating(f.type)]


def _translate_donors(planted: pa.Table, ops: List[Operation], rows_by_op: List[np.ndarray],
                      frames: np.ndarray, accepted: np.ndarray, offset_px: float,
                      cues: _Cues | None) -> pa.Table:
    """Move every decoy donor's right half next to its target's right-half start."""
    shifts: Dict[int, Tuple[float, float]] = {}
    for k, (role, _, gap, runs, partner) in enumerate(ops):
        if role != 'decoy_donor':
            continue
        start = runs[0][1] + 1
        rows_a, rows_c = rows_by_op[k], rows_by_op[partner]
        head_a = rows_a[(frames[rows_a] == start) & accepted[rows_a]][0]
        head_c = rows_c[(frames[rows_c] == start) & accepted[rows_c]][0]
        delta = cues.shell(head_c) - cues.shell(head_a) + np.array([offset_px, 0.0])
        for row in rows_a[frames[rows_a] >= start]:
            shifts[int(row)] = (float(delta[0]), float(delta[1]))
    if not shifts:
        return planted
    rows = np.asarray(sorted(shifts), dtype=np.int64)
    for name in coordinate_columns(planted.schema):
        axis = 0 if name.endswith('_x') else 1
        values = planted[name].combine_chunks()
        old = values.take(pa.array(rows)).to_pylist()
        new = [None if v is None else v + shifts[int(r)][axis] for v, r in zip(old, rows)]
        index = planted.schema.get_field_index(name)
        planted = planted.set_column(index, planted.schema.field(index),
                                     _replace(planted[name], rows, new, planted.num_rows))
    return planted


# ============================================================
# SCORING
# ============================================================


def _ratio(num: int, den: int) -> float:
    """``num / den``, NaN when ``den`` is 0."""
    return num / den if den else math.nan


def _score_row(gap: int, outcomes: List[str], decoy: List[str], flicker: List[str]) -> dict:
    """One scores row from the per-plant outcomes of one gap length (or all).

    ``outcomes`` covers plants and decoy targets; ``decoy`` the decoy targets
    only (``recovered``, ``decoy_false_link``, ``false_link`` or ``missed``);
    ``flicker`` the flicker singletons (``recovered`` and/or ``spurious``
    tags joined by ``+``, or ``''``).
    """
    planted = len(outcomes)
    recovered = outcomes.count('recovered')
    false_link = outcomes.count('false_link')
    missed = outcomes.count('missed')
    if recovered + false_link + missed != planted:
        raise RuntimeError("unknown plant outcome")
    return {'gap_frames': gap, 'planted': planted, 'recovered': recovered,
            'false_link': false_link, 'missed': missed, 'recall': _ratio(recovered, planted),
            'precision': _ratio(recovered, recovered + false_link),
            'decoy_planted': len(decoy), 'decoy_recovered': decoy.count('recovered'),
            'decoy_false_link': decoy.count('decoy_false_link'),
            'flicker_planted': len(flicker),
            'flicker_recovered': sum('recovered' in f.split('+') for f in flicker),
            'flicker_spurious': sum('spurious' in f.split('+') for f in flicker)}


def score_links(links: pa.Table, truth: pa.Table,
                tracklet_status: Dict[str, str] | None = None) -> pa.Table:
    """Score a link audit table against the planted truth.

    Parameters
    ----------
    links : pa.Table
        ``links.parquet`` of a bridging run (``tracklet_a``, ``tracklet_b``,
        ``decision``); only ``decision == 'linked'`` rows count.
    truth : pa.Table
        Truth table of :func:`plant_gaps` (a table without ``role`` counts
        every row as a plant).
    tracklet_status : dict, optional
        ``tracklet_id -> trajectory_status`` of the bridged run; needed for
        ``flicker_spurious`` (0 without it).

    Returns
    -------
    pa.Table
        :data:`SCORES_SCHEMA`, one row per gap length (ascending) and a last
        row ``gap_frames = -1`` over all plants. Plants and decoy targets are
        scored alike: ``recovered`` when the left half is linked to its right
        half, ``false_link`` when the left half is linked to another start or
        the right half from another end, else ``missed``. ``recall =
        recovered / planted``, ``precision = recovered / (recovered +
        false_link)``, NaN on a zero denominator. Decoy donors are excluded.
        ``decoy_false_link`` counts targets whose left half is linked to their
        donor's right half; ``decoy_recovered`` targets recovered. A flicker
        is recovered when its left part, middle and right part end in one
        trajectory (one chain of links) and spurious when its middle's
        trajectory is flagged ``spurious``.
    """
    for name in ('tracklet_a', 'tracklet_b', 'decision'):
        if name not in links.schema.names:
            raise ValueError(f"links table has no column {name!r}")
    linked = links.filter(pc.equal(links['decision'], 'linked'))
    pairs = list(zip(linked['tracklet_a'].to_pylist(), linked['tracklet_b'].to_pylist()))
    successor, predecessor = dict(pairs), {b: a for a, b in pairs}
    if len(successor) != len(pairs) or len(predecessor) != len(pairs):
        raise ValueError("linked pairs are not one-to-one")

    def head(tracklet: str) -> str:
        seen = set()
        while tracklet in predecessor:
            if tracklet in seen:
                raise ValueError("linked pairs form a cycle")
            seen.add(tracklet)
            tracklet = predecessor[tracklet]
        return tracklet

    n = truth.num_rows
    t = truth.to_pydict()
    roles = t.get('role') or ['plant'] * n
    middles = t.get('middle_id') or [None] * n
    partners = t.get('partner_id') or [None] * n
    status = tracklet_status or {}
    by_gap: Dict[int, Tuple[List[str], List[str], List[str]]] = {}
    for left, right, gap, role, middle, partner in zip(t['left_id'], t['right_id'],
                                                       t['gap_frames'], roles, middles,
                                                       partners):
        if role not in ROLES:
            raise ValueError(f"unknown truth role {role!r}")
        outcomes, decoy, flicker = by_gap.setdefault(int(gap), ([], [], []))
        if role == 'decoy_donor':
            continue
        if role == 'flicker':
            tags = []
            if head(left) == head(middle) == head(right):
                tags.append('recovered')
            if status.get(middle) == 'spurious':
                tags.append('spurious')
            flicker.append('+'.join(tags))
            continue
        if successor.get(left) == right:
            outcome = 'recovered'
        elif left in successor or right in predecessor:
            outcome = 'false_link'
        else:
            outcome = 'missed'
        outcomes.append(outcome)
        if role == 'decoy_target':
            decoy.append('decoy_false_link' if partner is not None
                         and successor.get(left) == partner else outcome)
    rows = [_score_row(gap, *by_gap[gap]) for gap in sorted(by_gap)]
    rows.append(_score_row(ALL_GAPS, *([x for gap in sorted(by_gap) for x in by_gap[gap][i]]
                                       for i in range(3))))
    return pa.Table.from_pylist(rows, schema=SCORES_SCHEMA)


# ============================================================
# RUN
# ============================================================


def _json_safe(row: dict) -> dict:
    """Replace NaN floats by None so the JSON stays standard."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}


def run_holdout(observations_path: Path, out_dir: Path, cfg: BridgeConfig,
                gap_lengths: Sequence[int], per_gap: int, min_tracklet_len: int = 40,
                seed: int = 0, force: bool = False, margin: int = 3, decoys: int = 0,
                flicker: int = 0, decoy_offset_px: float = 3.0) -> dict:
    """Plant gaps, bridge the planted table and score the links.

    Parameters
    ----------
    observations_path : Path
        Observation parquet file or directory of parts (as for
        :func:`pipeline.run`).
    out_dir : Path
        Receives ``planted_observations.parquet``, ``holdout_truth.parquet``,
        ``bridged/`` (the pipeline outputs), ``holdout_scores.csv`` and
        ``holdout_<max_gap>.json``.
    cfg : BridgeConfig
        Bridging configuration; ``cfg.max_gap`` names the JSON file.
    gap_lengths, per_gap, min_tracklet_len, seed, margin, decoys, flicker, decoy_offset_px
        Passed to :func:`plant_gaps` (with ``cfg`` as its ``cue_config``).
    force : bool
        Overwrite existing outputs (also passed to :func:`pipeline.run`).

    Returns
    -------
    dict
        Content of ``holdout_<max_gap>.json``: ``config``, ``seed``,
        ``gap_lengths``, ``per_gap``, ``min_tracklet_len``, ``margin``,
        ``decoys``, ``flicker``, ``decoy_offset_px``, ``input``, ``scores``
        (per-gap rows then the ``-1`` row, NaN as null), ``overall``
        (precision, recall), ``n_unplanted_links`` (linked pairs with neither
        side a planted tracklet of any role) and ``outputs``.

    Raises
    ------
    FileExistsError
        If hold-out outputs exist and ``force`` is False (before any work).
    """
    out_dir = Path(out_dir)
    json_name = f'holdout_{cfg.max_gap}.json'
    names = (PLANTED_NAME, TRUTH_NAME, SCORES_NAME, json_name)
    existing = [n for n in names if (out_dir / n).exists()]
    if existing and not force:
        raise FileExistsError(f"hold-out outputs already exist in {out_dir}: {existing}")
    obs = pipeline.read_observations(pipeline.input_files(observations_path))
    planted, truth = plant_gaps(obs, gap_lengths, per_gap, min_tracklet_len, margin, seed,
                                decoys=decoys, decoy_offset_px=decoy_offset_px,
                                flicker=flicker, cue_config=cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(planted, out_dir / PLANTED_NAME)
    pq.write_table(truth, out_dir / TRUTH_NAME)
    pipeline.run(out_dir / PLANTED_NAME, out_dir / BRIDGED_DIR, cfg, force)
    links = pq.read_table(out_dir / BRIDGED_DIR / 'links.parquet')
    bridged = pq.read_table(out_dir / BRIDGED_DIR / 'observations.parquet',
                            columns=['tracklet_id', 'trajectory_status'])
    bridged = bridged.filter(pc.is_valid(bridged['tracklet_id']))
    status = dict(zip(bridged['tracklet_id'].to_pylist(),
                      bridged['trajectory_status'].to_pylist()))
    scores = score_links(links, truth, status)
    pacsv.write_csv(scores, out_dir / SCORES_NAME)

    halves = (set(truth['left_id'].to_pylist()) | set(truth['right_id'].to_pylist())
              | set(truth['middle_id'].drop_null().to_pylist()))
    linked = links.filter(pc.equal(links['decision'], 'linked'))
    n_unplanted = sum(a not in halves and b not in halves for a, b in
                      zip(linked['tracklet_a'].to_pylist(), linked['tracklet_b'].to_pylist()))
    rows = [_json_safe(r) for r in scores.to_pylist()]
    summary = {
        'config': cfg.to_dict(), 'seed': int(seed), 'gap_lengths': [int(g) for g in gap_lengths],
        'per_gap': int(per_gap), 'min_tracklet_len': int(min_tracklet_len), 'margin': int(margin),
        'decoys': int(decoys), 'flicker': int(flicker), 'decoy_offset_px': float(decoy_offset_px),
        'input': {'path': str(observations_path), 'rows': obs.num_rows,
                  'planted_rows': planted.num_rows},
        'scores': rows,
        'overall': {'precision': rows[-1]['precision'], 'recall': rows[-1]['recall']},
        'n_unplanted_links': int(n_unplanted),
        'outputs': {'planted': PLANTED_NAME, 'truth': TRUTH_NAME, 'scores': SCORES_NAME,
                    'bridged': BRIDGED_DIR},
    }
    if rows[-1]['gap_frames'] != ALL_GAPS or scores.num_rows < 2:
        raise RuntimeError("scores table lacks the per-gap or overall rows")
    (out_dir / json_name).write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    return summary
