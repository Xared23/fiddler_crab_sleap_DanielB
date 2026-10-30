#!/usr/bin/env python3

"""
Frame-deletion hold-out evaluation of gap bridging.

``plant_gaps`` deletes a run of ``g`` consecutive frames from the interior of
real, contiguous tracklets and splits each into a left half (old id) and a
right half (new id), exactly as the adjacent-frame tracker would have done had
the detections been missed. ``score_links`` compares a ``links.parquet``
audit table with the planted truth: per gap length, how many planted pairs
were re-linked (recovered), how many planted halves were linked to something
else (false_link) and how many were left alone (missed). ``run_holdout``
plants, runs :func:`gap_bridging.pipeline.run` and writes the scores.

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
from gap_bridging import pipeline
from gap_bridging.config import BridgeConfig

# ============================================================
# CONSTANTS
# ============================================================

TRUTH_SCHEMA = pa.schema([
    ('left_id', pa.string()), ('right_id', pa.string()), ('original_id', pa.string()),
    ('gap_frames', pa.int64()), ('deleted_first_frame', pa.int64()),
    ('deleted_last_frame', pa.int64()),
])
SCORES_SCHEMA = pa.schema([
    ('gap_frames', pa.int64()), ('planted', pa.int64()), ('recovered', pa.int64()),
    ('false_link', pa.int64()), ('missed', pa.int64()), ('recall', pa.float64()),
    ('precision', pa.float64()),
])
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


def _choose(spans: Dict[int, Tuple[int, int]], ids: List[str], gap_lengths: Sequence[int],
            per_gap: int, min_tracklet_len: int, margin: int,
            seed: int) -> List[Tuple[int, int, int]]:
    """Pick ``(code, gap, deleted_first_frame)`` plants; no tracklet twice.

    Eligible tracklets are sorted by id so the draw depends only on ``seed``
    and the eligible set, not on row order.
    """
    rng = np.random.default_rng(seed)
    remaining = sorted(spans, key=lambda code: ids[code])
    plants: List[Tuple[int, int, int]] = []
    for gap in gap_lengths:
        need = max(min_tracklet_len, gap + 2 * margin)
        pool = [code for code in remaining if spans[code][1] >= need]
        if len(pool) < per_gap:
            raise ValueError(f"only {len(pool)} eligible tracklets (>= {need} frames, unused) "
                             f"for gap {gap}; need {per_gap}")
        picked = [pool[i] for i in rng.choice(len(pool), size=per_gap, replace=False)]
        for code in picked:
            first, n = spans[code]
            offset = int(rng.integers(margin, n - gap - margin + 1))
            plants.append((code, int(gap), first + offset))
        taken = set(picked)
        remaining = [code for code in remaining if code not in taken]
    if len({p[0] for p in plants}) != len(plants):
        raise RuntimeError("a tracklet was planted twice")
    return plants


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
               seed: int = 0) -> Tuple[pa.Table, pa.Table]:
    """Delete interior frame runs from real tracklets and split them in two.

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

    Returns
    -------
    planted : pa.Table
        ``obs`` without the deleted rows; the right half of the n-th planted
        tracklet (n from 1) gets ``tracklet_id = f"{old}/holdout{n:03d}"``,
        ``tracklet_local_id = max(tracklet_local_id) + n`` and its first row
        ``incoming_reason = 'no_candidate'``. Every other value is unchanged.
    truth : pa.Table
        One row per plant (:data:`TRUTH_SCHEMA`); ``left_id == original_id``.

    Raises
    ------
    ValueError
        On invalid arguments or when fewer than ``per_gap`` unused eligible
        tracklets remain for some gap length.
    """
    if per_gap < 1 or margin < 1 or min_tracklet_len < 1:
        raise ValueError("per_gap, margin and min_tracklet_len must be >= 1")
    if not gap_lengths or any(int(g) != g or g < 1 for g in gap_lengths):
        raise ValueError(f"gap lengths must be integers >= 1, got {list(gap_lengths)}")
    codes, ids, spans = _tracklet_spans(obs)
    plants = _choose(spans, ids, [int(g) for g in gap_lengths], per_gap, min_tracklet_len,
                     margin, seed)
    frames = obs['frame'].fill_null(-1).to_numpy().astype(np.int64)
    accepted = obs['accepted_for_tracking'].fill_null(False).to_numpy(zero_copy_only=False)
    local_max = pc.max(obs['tracklet_local_id']).as_py() or 0
    keep = np.ones(obs.num_rows, dtype=bool)
    new_id = np.full(obs.num_rows, -1, dtype=np.int64)    # plant index per right-half row
    heads: List[int] = []
    truth: Dict[str, list] = {name: [] for name in TRUTH_SCHEMA.names}
    for n, (code, gap, first_del) in enumerate(plants, 1):
        last_del = first_del + gap - 1
        rows = np.flatnonzero(codes == code)
        keep[rows[(frames[rows] >= first_del) & (frames[rows] <= last_del)]] = False
        right = rows[frames[rows] > last_del]
        new_id[right] = n
        heads.extend(right[(frames[right] == last_del + 1) & accepted[right]].tolist())
        old = ids[code]
        for name, value in zip(TRUTH_SCHEMA.names, (old, f"{old}/holdout{n:03d}", old, gap,
                                                    first_del, last_del)):
            truth[name].append(value)
    if int((~keep).sum()) != sum(p[1] for p in plants) or len(heads) != len(plants):
        raise RuntimeError("planted rows do not match the requested gaps")
    right_rows = np.flatnonzero(new_id > 0)
    planted = obs
    for name, values in (
            ('tracklet_id', [truth['right_id'][i - 1] for i in new_id[right_rows]]),
            ('tracklet_local_id', [int(local_max) + int(i) for i in new_id[right_rows]])):
        index = planted.schema.get_field_index(name)
        planted = planted.set_column(index, planted.schema.field(index),
                                     _replace(planted[name], right_rows, values, obs.num_rows))
    index = planted.schema.get_field_index('incoming_reason')
    heads_sorted = np.sort(np.asarray(heads, dtype=np.int64))
    planted = planted.set_column(index, planted.schema.field(index), _replace(
        planted['incoming_reason'], heads_sorted, [NEW_INCOMING_REASON] * len(heads),
        obs.num_rows))
    return planted.filter(pa.array(keep)), pa.Table.from_pydict(truth, schema=TRUTH_SCHEMA)


# ============================================================
# SCORING
# ============================================================


def _ratio(num: int, den: int) -> float:
    """``num / den``, NaN when ``den`` is 0."""
    return num / den if den else math.nan


def _score_row(gap: int, outcomes: List[str]) -> dict:
    """One scores row from the per-plant outcomes of one gap length (or all)."""
    planted = len(outcomes)
    recovered = outcomes.count('recovered')
    false_link = outcomes.count('false_link')
    missed = outcomes.count('missed')
    if recovered + false_link + missed != planted:
        raise RuntimeError("unknown plant outcome")
    return {'gap_frames': gap, 'planted': planted, 'recovered': recovered,
            'false_link': false_link, 'missed': missed, 'recall': _ratio(recovered, planted),
            'precision': _ratio(recovered, recovered + false_link)}


def score_links(links: pa.Table, truth: pa.Table) -> pa.Table:
    """Score a link audit table against the planted truth.

    Parameters
    ----------
    links : pa.Table
        ``links.parquet`` of a bridging run (``tracklet_a``, ``tracklet_b``,
        ``decision``); only ``decision == 'linked'`` rows count.
    truth : pa.Table
        Truth table of :func:`plant_gaps`.

    Returns
    -------
    pa.Table
        :data:`SCORES_SCHEMA`, one row per gap length (ascending) and a last
        row ``gap_frames = -1`` over all plants. A plant is ``recovered`` when
        its left half is linked to its right half, ``false_link`` when the
        left half is linked to another start or the right half from another
        end, else ``missed``. ``recall = recovered / planted`` and
        ``precision = recovered / (recovered + false_link)``, NaN on a zero
        denominator.
    """
    for name in ('tracklet_a', 'tracklet_b', 'decision'):
        if name not in links.schema.names:
            raise ValueError(f"links table has no column {name!r}")
    linked = links.filter(pc.equal(links['decision'], 'linked'))
    pairs = list(zip(linked['tracklet_a'].to_pylist(), linked['tracklet_b'].to_pylist()))
    successor, predecessor = dict(pairs), {b: a for a, b in pairs}
    if len(successor) != len(pairs) or len(predecessor) != len(pairs):
        raise ValueError("linked pairs are not one-to-one")
    by_gap: Dict[int, List[str]] = {}
    for left, right, gap in zip(truth['left_id'].to_pylist(), truth['right_id'].to_pylist(),
                                truth['gap_frames'].to_pylist()):
        if successor.get(left) == right:
            outcome = 'recovered'
        elif left in successor or right in predecessor:
            outcome = 'false_link'
        else:
            outcome = 'missed'
        by_gap.setdefault(int(gap), []).append(outcome)
    rows = [_score_row(gap, by_gap[gap]) for gap in sorted(by_gap)]
    rows.append(_score_row(ALL_GAPS, [o for gap in sorted(by_gap) for o in by_gap[gap]]))
    return pa.Table.from_pylist(rows, schema=SCORES_SCHEMA)


# ============================================================
# RUN
# ============================================================


def _json_safe(row: dict) -> dict:
    """Replace NaN floats by None so the JSON stays standard."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}


def run_holdout(observations_path: Path, out_dir: Path, cfg: BridgeConfig,
                gap_lengths: Sequence[int], per_gap: int, min_tracklet_len: int = 40,
                seed: int = 0, force: bool = False, margin: int = 3) -> dict:
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
    gap_lengths, per_gap, min_tracklet_len, seed, margin
        Passed to :func:`plant_gaps`.
    force : bool
        Overwrite existing outputs (also passed to :func:`pipeline.run`).

    Returns
    -------
    dict
        Content of ``holdout_<max_gap>.json``: ``config``, ``seed``,
        ``gap_lengths``, ``per_gap``, ``min_tracklet_len``, ``margin``,
        ``input``, ``scores`` (per-gap rows then the ``-1`` row, NaN as
        null), ``overall`` (precision, recall), ``n_unplanted_links``
        (linked pairs with neither side a planted half) and ``outputs``.

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
    planted, truth = plant_gaps(obs, gap_lengths, per_gap, min_tracklet_len, margin, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(planted, out_dir / PLANTED_NAME)
    pq.write_table(truth, out_dir / TRUTH_NAME)
    pipeline.run(out_dir / PLANTED_NAME, out_dir / BRIDGED_DIR, cfg, force)
    links = pq.read_table(out_dir / BRIDGED_DIR / 'links.parquet')
    scores = score_links(links, truth)
    pacsv.write_csv(scores, out_dir / SCORES_NAME)

    halves = set(truth['left_id'].to_pylist()) | set(truth['right_id'].to_pylist())
    linked = links.filter(pc.equal(links['decision'], 'linked'))
    n_unplanted = sum(a not in halves and b not in halves for a, b in
                      zip(linked['tracklet_a'].to_pylist(), linked['tracklet_b'].to_pylist()))
    rows = [_json_safe(r) for r in scores.to_pylist()]
    summary = {
        'config': cfg.to_dict(), 'seed': int(seed), 'gap_lengths': [int(g) for g in gap_lengths],
        'per_gap': int(per_gap), 'min_tracklet_len': int(min_tracklet_len), 'margin': int(margin),
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
