#!/usr/bin/env python3

"""
Pipeline-level threshold of the learned link scorer (v006).

The pair-level threshold (:func:`gap_bridging.train.choose_threshold`, the
smallest out-of-fold probability with pair precision 1) ignores what the
linker does after scoring: the one-to-one assignment and the margin veto
remove most of the negative pairs that score high, so it lands near 0.99.
This module chooses the threshold where the promotion rule is evaluated:
after assignment, per planted gap, on the TRAIN-fold plants of hold-out runs.

A hold-out run (``holdout_<h>/`` of ``bridge_gaps.py sweep --holdout``) is
loaded once (:func:`load_scene`: its own config, the endpoints of its planted
observation table, its train-fold truth rows). For each candidate threshold
``t`` (:func:`candidate_thresholds`, the OOF probabilities of the training
pairs), :func:`curve_point` links the scene with ``cost_model: learned``,
``learned_min_probability = learned_unmatched_probability = t``
(:func:`scene_config`; every other parameter is the run's) and scores the
links against the train-fold truth (:func:`gap_bridging.holdout.score_links`).
:func:`pipeline_curve` collects the per-run rows, :func:`pool_curve` sums
them over runs, and :func:`choose_pipeline_threshold` returns the smallest
candidate from which every larger candidate also has no false link and at
least one recovered plant at every planted gap up to ``through_gap``
(prefix-closed from the top, so a dip in the curve cannot be chosen). No
randomness is involved anywhere; each curve point recomputes the features.

Sam
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ----- numerics -----
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# ----- local modules -----
from gap_bridging import config, geometry, holdout, labels, pipeline
from gap_bridging.config import BridgeConfig

# ============================================================
# CONSTANTS
# ============================================================

CURVE_SCHEMA = pa.schema([('threshold', pa.float64()), ('run', pa.string()),
                          ('gap_frames', pa.int64()), ('planted', pa.int64()),
                          ('recovered', pa.int64()), ('false_link', pa.int64()),
                          ('missed', pa.int64())])
POOLED_SCHEMA = pa.schema([('threshold', pa.float64()), ('gap_frames', pa.int64()),
                           ('planted', pa.int64()), ('recovered', pa.int64()),
                           ('false_link', pa.int64()), ('missed', pa.int64()),
                           ('precision', pa.float64()), ('recall', pa.float64())])
DEFAULT_MAX_CANDIDATES = 200
CURVE_SUFFIX = '_threshold_curve.csv'            # pooled rows
CURVE_RUNS_SUFFIX = '_threshold_curve_runs.csv'  # per-run rows
COUNT_KEYS = ('planted', 'recovered', 'false_link', 'missed')
LABELS = (0, 1)
TRAIN_FOLD = 'train'
SQRT_GATE = 'sqrt'

# ============================================================
# CANDIDATES
# ============================================================


def candidate_thresholds(p_oof: np.ndarray, y: np.ndarray,
                         max_candidates: int = DEFAULT_MAX_CANDIDATES) -> np.ndarray:
    """Candidate pipeline thresholds from the out-of-fold pair probabilities.

    Parameters
    ----------
    p_oof : np.ndarray
        Calibrated out-of-fold probabilities of the training pairs.
    y : np.ndarray
        Their labels; only rows labelled 0 or 1 count.
    max_candidates : int
        Largest number of candidates returned (>= 2).

    Returns
    -------
    np.ndarray
        The ascending unique eligible values of ``p_oof`` strictly inside
        ``(0, 1)``; when there are more than ``max_candidates``, the values at
        the rounded indices ``linspace(0, n - 1, max_candidates)`` of the
        sorted unique values (the smallest and largest always kept).

    Raises
    ------
    ValueError
        If ``p_oof`` and ``y`` differ in shape, ``max_candidates < 2`` or no
        value is eligible.
    """
    p = np.asarray(p_oof, dtype=float)
    labels_ = np.asarray(y)
    if p.shape != labels_.shape:
        raise ValueError(f"p_oof and y must have equal shapes, got {p.shape} and {labels_.shape}")
    if int(max_candidates) < 2:
        raise ValueError(f"max_candidates must be >= 2, got {max_candidates}")
    eligible = np.isin(labels_, LABELS) & np.isfinite(p) & (p > 0.0) & (p < 1.0)
    values = np.unique(p[eligible])
    if values.size == 0:
        raise ValueError("no labelled out-of-fold probability strictly inside (0, 1)")
    if values.size > int(max_candidates):
        idx = np.unique(np.rint(np.linspace(0, values.size - 1, int(max_candidates)))
                        .astype(np.int64))
        values = values[idx]
    assert values.size <= max(int(max_candidates), 1), "candidate cap respected"
    assert np.all(np.diff(values) > 0), "candidates strictly ascending"
    return values


# ============================================================
# SCENES
# ============================================================


@dataclass(frozen=True)
class HoldoutScene:
    """One hold-out run, ready to be linked at any threshold.

    ``cfg`` is the run's own config (``scorer_path`` forced to None);
    ``endpoints`` are :func:`gap_bridging.geometry.endpoints` of its planted
    observation table; ``truth_train`` its train-fold truth rows (the
    :data:`gap_bridging.holdout.TRUTH_SCHEMA` columns it has); ``n_truth``
    the number of all its truth rows.
    """
    run_dir: Path
    cfg: BridgeConfig
    endpoints: dict
    videos: List[str]
    truth_train: pa.Table
    n_truth: int


def _run_config(run_dir: Path) -> BridgeConfig:
    """The ``config`` of the single ``holdout_*.json`` of ``run_dir``, ``scorer_path`` None."""
    jsons = sorted(run_dir.glob('holdout_*.json'))
    if not jsons:
        raise FileNotFoundError(f"{run_dir}: no holdout_<max_gap>.json (not a completed "
                                f"hold-out run)")
    if len(jsons) > 1:
        raise ValueError(f"{run_dir}: several hold-out JSON files {[j.name for j in jsons]}")
    values = dict(json.loads(jsons[0].read_text())['config'])
    values.pop('scorer_path', None)
    cfg = replace(config.load_config(None, **values), scorer_path=None)
    if cfg.gate_model != SQRT_GATE:
        raise ValueError(f"{run_dir}: gate_model {cfg.gate_model!r}; the learned cost model "
                         f"requires {SQRT_GATE!r}")
    return cfg


def _train_truth(truth: pa.Table, salt: str, test_fraction: float) -> pa.Table:
    """Train-fold truth rows with the TRUTH_SCHEMA columns present (as compare.fold_scores)."""
    with_folds = labels.truth_folds(truth, salt, test_fraction)
    if 'fold' not in with_folds.schema.names or with_folds.num_rows != truth.num_rows:
        raise RuntimeError("truth_folds must return the truth rows plus a 'fold' column")
    kept = with_folds.filter(pc.equal(with_folds['fold'], TRAIN_FOLD))
    return kept.select([n for n in holdout.TRUTH_SCHEMA.names if n in kept.schema.names])


def load_scene(run_dir: Path, salt: str = labels.DEFAULT_SALT,
               test_fraction: float = labels.DEFAULT_TEST_FRACTION) -> HoldoutScene:
    """Load a ``holdout_<h>/`` run as a :class:`HoldoutScene`.

    Parameters
    ----------
    run_dir : Path
        Hold-out run directory (``holdout_<h>.json``,
        ``planted_observations.parquet``, ``holdout_truth.parquet``).
    salt, test_fraction
        Fold assignment (:func:`gap_bridging.labels.truth_folds`).

    Returns
    -------
    HoldoutScene
        Endpoints computed with the run's config, exactly as
        :func:`gap_bridging.pipeline.run` computed them for the run.

    Raises
    ------
    FileNotFoundError
        If ``run_dir``, its hold-out JSON or one of its tables is missing.
    ValueError
        If the run's ``gate_model`` is not ``'sqrt'``.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"{run_dir}: hold-out run directory not found")
    cfg = _run_config(run_dir)
    paths = (run_dir / holdout.PLANTED_NAME, run_dir / holdout.TRUTH_NAME)
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir}: missing {missing}")
    planted, truth = pq.read_table(paths[0]), pq.read_table(paths[1])
    videos = sorted(set(planted['video_id'].unique().to_pylist()) - {None})
    endpoints = geometry.endpoints(planted, cfg)
    truth_train = _train_truth(truth, salt, test_fraction)
    assert truth_train.num_rows <= truth.num_rows, "train rows are a subset of the truth"
    return HoldoutScene(run_dir=run_dir, cfg=cfg, endpoints=endpoints, videos=videos,
                        truth_train=truth_train, n_truth=int(truth.num_rows))


def scene_config(cfg: BridgeConfig, threshold: float) -> BridgeConfig:
    """``cfg`` with the learned cost model at threshold ``t`` (gate and dummy), no scorer path."""
    t = float(threshold)
    if not 0.0 < t < 1.0:
        raise ValueError(f"threshold must be strictly inside (0, 1), got {threshold!r}")
    out = replace(cfg, cost_model='learned', scorer_path=None, learned_min_probability=t,
                  learned_unmatched_probability=t)
    assert out.learned_min_probability == out.learned_unmatched_probability == t
    return out


# ============================================================
# CURVE
# ============================================================


def curve_point(scene: HoldoutScene, scorer: Any, threshold: float) -> List[dict]:
    """Per-gap train-fold scores of one scene linked at one threshold.

    Parameters
    ----------
    scene : HoldoutScene
        Output of :func:`load_scene`.
    scorer : gap_bridging.scorer.Scorer
        The learned scorer (``feature_names``, ``cost``).
    threshold : float
        ``learned_min_probability`` and ``learned_unmatched_probability``,
        strictly inside ``(0, 1)``.

    Returns
    -------
    list of dict
        One row per planted gap length of ``scene.truth_train`` (ascending,
        the all-gaps row excluded) with the keys ``gap_frames``, ``planted``,
        ``recovered``, ``false_link`` and ``missed`` (plain ints). A gap whose
        train-fold rows are only decoy donors or flicker singletons (``planted
        == 0``: donor and target can fall in different folds) carries no
        evidence and is dropped, so it cannot fail :func:`qualifies`.
    """
    if scorer is None:
        raise ValueError("curve_point requires a scorer")
    cfg = scene_config(scene.cfg, threshold)
    _, links_table = pipeline.link_videos(scene.endpoints, scene.videos, cfg, scorer=scorer)
    scores = holdout.score_links(links_table, scene.truth_train, None)
    rows = [{'gap_frames': int(r['gap_frames']), **{k: int(r[k]) for k in COUNT_KEYS}}
            for r in scores.to_pylist() if int(r['gap_frames']) >= 1 and int(r['planted']) > 0]
    gaps = [r['gap_frames'] for r in rows]
    assert gaps == sorted(set(gaps)), "one row per gap length, ascending"
    assert all(r['recovered'] + r['false_link'] + r['missed'] == r['planted'] for r in rows)
    return rows


def pipeline_curve(scenes: Sequence[HoldoutScene], scorer: Any, thresholds: Sequence[float],
                   progress: Optional[Callable[[str], None]] = None) -> pa.Table:
    """Per-run curve over thresholds (:data:`CURVE_SCHEMA`).

    Parameters
    ----------
    scenes : sequence of HoldoutScene
        Hold-out runs, in output order.
    scorer : gap_bridging.scorer.Scorer
        The learned scorer.
    thresholds : sequence of float
        Candidate thresholds; evaluated in ascending order (duplicates once).
    progress : callable, optional
        Called once per threshold with a one-line summary.

    Returns
    -------
    pa.Table
        One row per (threshold ascending, scene in the given order, gap
        ascending); ``run`` is ``str(scene.run_dir)``.

    Raises
    ------
    ValueError
        If ``scenes`` or ``thresholds`` is empty.
    """
    scenes = list(scenes)
    ts = sorted(set(float(t) for t in thresholds))
    if not scenes or not ts:
        raise ValueError("pipeline_curve needs at least one scene and one threshold")
    rows: List[dict] = []
    for i, t in enumerate(ts):
        at_t = [{'threshold': t, 'run': str(scene.run_dir), **row}
                for scene in scenes for row in curve_point(scene, scorer, t)]
        rows.extend(at_t)
        if progress is not None:
            totals = {k: sum(r[k] for r in at_t) for k in COUNT_KEYS}
            progress(f"[threshold] {i + 1}/{len(ts)} t={t:.6g}: planted {totals['planted']}, "
                     f"recovered {totals['recovered']}, false {totals['false_link']}")
    table = pa.Table.from_pylist(rows, schema=CURVE_SCHEMA)
    assert table.num_rows == len(rows), "every row kept"
    return table


def _ratio(num: int, den: int) -> float:
    """``num / den``, NaN when ``den`` is 0."""
    return num / den if den else math.nan


def _check_names(table: pa.Table, schema: pa.Schema, what: str) -> None:
    """Raise ValueError unless ``table`` has every column of ``schema``."""
    missing = [n for n in schema.names if n not in table.schema.names]
    if missing:
        raise ValueError(f"{what} lacks the columns {missing}")


def pool_curve(curve: pa.Table) -> pa.Table:
    """Sum a per-run curve over runs (:data:`POOLED_SCHEMA`).

    One row per ``(threshold, gap_frames)``, ascending; ``precision =
    recovered / (recovered + false_link)`` and ``recall = recovered /
    planted``, NaN on a zero denominator.
    """
    _check_names(curve, CURVE_SCHEMA, "curve")
    sums: Dict[Tuple[float, int], List[int]] = {}
    for row in curve.select(list(CURVE_SCHEMA.names)).to_pylist():
        acc = sums.setdefault((float(row['threshold']), int(row['gap_frames'])),
                              [0] * len(COUNT_KEYS))
        for i, key in enumerate(COUNT_KEYS):
            acc[i] += int(row[key])
    rows = []
    for (t, gap), counts in sorted(sums.items()):
        row = {'threshold': t, 'gap_frames': gap, **dict(zip(COUNT_KEYS, counts))}
        row['precision'] = _ratio(row['recovered'], row['recovered'] + row['false_link'])
        row['recall'] = _ratio(row['recovered'], row['planted'])
        rows.append(row)
    pooled = pa.Table.from_pylist(rows, schema=POOLED_SCHEMA)
    assert pooled.num_rows == len(sums), "one row per (threshold, gap)"
    return pooled


# ============================================================
# CHOICE
# ============================================================


def _relevant(rows: Sequence[dict], through_gap: int) -> List[dict]:
    """Rows with ``gap_frames <= through_gap``, by ascending gap."""
    return sorted((r for r in rows if int(r['gap_frames']) <= int(through_gap)),
                  key=lambda r: int(r['gap_frames']))


def _fails(row: dict) -> bool:
    """True when a gap row has a false link or no recovered plant."""
    return int(row['false_link']) > 0 or int(row['recovered']) == 0


def clean(pooled_rows_at_t: Sequence[dict], through_gap: int) -> bool:
    """True iff no row with ``gap_frames <= through_gap`` has a false link (recovery may be 0)."""
    return not any(int(r['false_link']) > 0 for r in _relevant(pooled_rows_at_t, through_gap))


def qualifies(pooled_rows_at_t: Sequence[dict], through_gap: int) -> bool:
    """True iff every row with ``gap_frames <= through_gap`` has no false link and recovers one.

    False when no row has ``gap_frames <= through_gap``.
    """
    rows = _relevant(pooled_rows_at_t, through_gap)
    return bool(rows) and not any(_fails(r) for r in rows)


def _rows_by_threshold(pooled: pa.Table) -> Dict[float, List[dict]]:
    """Pooled rows grouped by threshold (POOLED_SCHEMA columns required)."""
    _check_names(pooled, POOLED_SCHEMA, "pooled curve")
    grouped: Dict[float, List[dict]] = {}
    for row in pooled.to_pylist():
        grouped.setdefault(float(row['threshold']), []).append(row)
    return grouped


def choose_pipeline_threshold(pooled: pa.Table, through_gap: int) -> float:
    """Smallest qualifying candidate above which no candidate makes a false link.

    Parameters
    ----------
    pooled : pa.Table
        Output of :func:`pool_curve`; its unique thresholds are the candidates.
    through_gap : int
        Largest planted gap the rule checks (:func:`qualifies`).

    Returns
    -------
    float
        The smallest candidate ``t`` that :func:`qualifies` (precision
        exactly 1.0 at every gap through ``through_gap``) such that every
        larger candidate is :func:`clean` (no false link; it may recover
        nothing at some gap, as the highest thresholds do). The false-link
        condition is prefix-closed from the top, so a lucky clean dip below a
        false-linking candidate cannot be chosen; vanishing recall above the
        choice does not block it.

    Raises
    ------
    ValueError
        If ``pooled`` is empty or no candidate qualifies above the largest
        false-linking one; the message names the offending candidate and the
        smallest failing ``gap_frames <= through_gap`` there.
    """
    grouped = _rows_by_threshold(pooled)
    if not grouped:
        raise ValueError("the pooled threshold curve is empty")
    candidates = sorted(grouped)
    chosen: Optional[float] = None
    barrier: Optional[float] = None
    for t in reversed(candidates):
        if not clean(grouped[t], through_gap):
            barrier = t
            break
        if qualifies(grouped[t], through_gap):
            chosen = t
    if chosen is None:
        at = candidates[-1] if barrier is None else barrier
        failing = [r for r in _relevant(grouped[at], through_gap) if _fails(r)]
        where = ("the largest candidate" if barrier is None
                 else "the largest false-linking candidate")
        if not failing:
            raise ValueError(f"no pipeline threshold qualifies: no row with gap_frames <= "
                             f"{through_gap} at {where} {at!r}")
        r = failing[0]
        raise ValueError(f"no pipeline threshold qualifies: at {where} {at!r}, gap_frames "
                         f"{int(r['gap_frames'])} has false_link {int(r['false_link'])} and "
                         f"recovered {int(r['recovered'])} (through_gap {through_gap}); no "
                         f"candidate above it reaches precision 1.0 at every gap")
    assert chosen in grouped, "the choice is a candidate"
    return float(chosen)


# ============================================================
# SUMMARY
# ============================================================


def _totals(rows: Sequence[dict], through_gap: int) -> Dict[str, int]:
    """``COUNT_KEYS`` summed over the rows with ``gap_frames <= through_gap``."""
    relevant = _relevant(rows, through_gap)
    return {k: int(sum(int(r[k]) for r in relevant)) for k in COUNT_KEYS}


def summary(pooled: pa.Table, chosen: float, through_gap: int,
            pair_threshold: Optional[float] = None) -> dict:
    """JSON-ready record of the choice for the scorer manifest.

    Parameters
    ----------
    pooled : pa.Table
        Output of :func:`pool_curve`.
    chosen : float
        The chosen threshold; must be one of the curve's candidates.
    through_gap : int
        The rule's gap.
    pair_threshold : float, optional
        The pair-level threshold (:func:`gap_bridging.train.choose_threshold`)
        for comparison.

    Returns
    -------
    dict
        ``n_candidates``, ``chosen``, ``through_gap``, ``at_chosen`` (counts
        summed over gaps through ``through_gap`` at ``chosen``) and
        ``at_pair_threshold`` (the same at the smallest candidate ``>=
        pair_threshold``; ``None`` when no pair threshold is given or it lies
        above every candidate).

    Raises
    ------
    ValueError
        If ``chosen`` is not a candidate.
    """
    grouped = _rows_by_threshold(pooled)
    chosen = float(chosen)
    if chosen not in grouped:
        raise ValueError(f"chosen threshold {chosen!r} is not a candidate of the curve")
    candidates = sorted(grouped)
    at_pair: Optional[Dict[str, int]] = None
    if pair_threshold is not None:
        above = [t for t in candidates if t >= float(pair_threshold)]
        if above:
            at_pair = _totals(grouped[above[0]], through_gap)
    return {'n_candidates': len(candidates), 'chosen': chosen, 'through_gap': int(through_gap),
            'at_chosen': _totals(grouped[chosen], through_gap), 'at_pair_threshold': at_pair}


# ============================================================
# FILES
# ============================================================


def curve_paths(model_path: Path) -> Tuple[Path, Path]:
    """``(<stem>_threshold_curve.csv, <stem>_threshold_curve_runs.csv)`` beside the model."""
    model_path = Path(model_path)
    return (model_path.with_name(model_path.stem + CURVE_SUFFIX),
            model_path.with_name(model_path.stem + CURVE_RUNS_SUFFIX))


def _write_csv(path: Path, table: pa.Table, schema: pa.Schema) -> None:
    """Write ``table``'s ``schema`` columns to ``path`` (floats as ``repr``)."""
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(schema.names))
        writer.writeheader()
        for row in table.select(list(schema.names)).to_pylist():
            writer.writerow({k: repr(v) if isinstance(v, float) else v for k, v in row.items()})


def write_curve(curve: pa.Table, pooled: pa.Table, model_path: Path) -> Tuple[Path, Path]:
    """Write the pooled and per-run curves beside ``model_path`` (:func:`curve_paths`).

    Returns
    -------
    (Path, Path)
        The pooled CSV and the per-run CSV.
    """
    _check_names(curve, CURVE_SCHEMA, "curve")
    _check_names(pooled, POOLED_SCHEMA, "pooled curve")
    pooled_path, runs_path = curve_paths(model_path)
    _write_csv(pooled_path, pooled, POOLED_SCHEMA)
    _write_csv(runs_path, curve, CURVE_SCHEMA)
    return pooled_path, runs_path


def _parse(value: str, kind: pa.DataType) -> Any:
    """One CSV cell as the Python value of an arrow type."""
    if pa.types.is_floating(kind):
        return float(value)
    if pa.types.is_integer(kind):
        return int(value)
    return value


def read_curve(path: Path) -> pa.Table:
    """Read a curve CSV of :func:`write_curve` into its schema (pooled or per-run).

    Raises
    ------
    ValueError
        If the header matches neither :data:`POOLED_SCHEMA` nor
        :data:`CURVE_SCHEMA`.
    """
    path = Path(path)
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        schema = {tuple(POOLED_SCHEMA.names): POOLED_SCHEMA,
                  tuple(CURVE_SCHEMA.names): CURVE_SCHEMA}.get(tuple(header))
        if schema is None:
            raise ValueError(f"{path}: header {header} is not a threshold curve")
        rows = [{f.name: _parse(row[f.name], f.type) for f in schema} for row in reader]
    table = pa.Table.from_pylist(rows, schema=schema)
    assert table.schema.equals(schema), "read back into the curve schema"
    return table
