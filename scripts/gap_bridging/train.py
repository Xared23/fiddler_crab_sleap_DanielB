#!/usr/bin/env python3

"""
Train the learned link scorer from hold-out sweeps (v004, ``bridge_gaps.py train``).

Data. Every completed ``holdout_<h>/`` of a sweep directory (``bridge_gaps.py
sweep --holdout``; completed means ``holdout_<h>.json`` is present, the file
:func:`gap_bridging.holdout.run_holdout` writes last) holds the candidate
pairs of the planted run (``bridged/pair_features.parquet``) and the planted
truth (``holdout_truth.parquet``). The pair features must be the ones of
that run: ``bridged/GAP_BRIDGING_COMPLETE.json`` must list them with their
sha256, and their row count must equal ``links.parquet``'s, so a stale file
of an earlier run is refused. Every hold-out's ``config`` must agree on the
feature keys (:data:`gap_bridging.features.FEATURE_CONFIG_KEYS`); that subset
is stored in the manifest so the scorer refuses a config that would compute
different features. :func:`gap_bridging.labels.label_pairs` labels each pair
(positive: rejoins a planted gap; negative: touches a planted half but is
not its partner; unlabelled otherwise, including pairs whose two sides'
originals fall in different folds) and assigns the train/test fold from the
originals; :func:`gap_bridging.labels.dedupe_pairs` keeps one row per pair
across the sweep's horizons. Only motion-gated pairs (``gate`` ``''`` or
``'max_cost'``, i.e. those the assignment can see) with a label are kept;
optionally the negatives whose START is a decoy donor are dropped
(``drop_donor_negatives``).

Model. A ``HistGradientBoostingClassifier`` with the monotonic constraints of
:data:`gap_bridging.features.MONOTONIC` (P(link) non-increasing in distance,
error, gap, the cue terms and the hand cost), shallow trees and no early
stopping. Out-of-fold (OOF) probabilities on the train fold come from
``StratifiedGroupKFold`` grouped by the original of the pair's planted side
(``original_a`` when the END is planted, else ``original_b``), so all pairs
of one plant, decoy or flicker stay in one CV fold (both originals of a
labelled pair are in the same train/test fold by construction); a
calibrator (sigmoid by default, or isotonic) is fitted on them, then the
classifier is refit on the whole train fold. ``features`` (``--features``,
v005) restricts the model to an ordered subset of
:data:`gap_bridging.features.FEATURE_NAMES` (a
:data:`gap_bridging.features.FEATURE_SETS` key such as ``no_length``, or a
comma-separated list); the manifest records the resolved names, their
constraints and the spec (``feature_set``), and inference selects those
columns by name. The ablation sets are a control for the length confound
(D-009), not a fix: see :mod:`gap_bridging.features`.

Threshold. The smallest calibrated OOF probability at which the OOF
precision reaches ``target_precision`` (recall maximal under that
constraint). Calibration and threshold both use the OOF predictions, which
is slightly optimistic; the test fold is reported at pair level in the
manifest, and the promotion decision uses the test fold through the full
pipeline (hold-out sweep with ``cost_model: learned``). Training refuses a
threshold of 1 (every accepted cost ``-log p`` would be 0, D-010; isotonic
calibration typically produces it) and a target precision that no OOF
threshold reaches, unless ``allow_degenerate_threshold``; the manifest
records ``target_reached`` in ``hyperparameters``.

Outputs: the model and manifest (:func:`gap_bridging.scorer.save_scorer`),
``<model stem>_metrics.csv`` (per planted gap length, test fold or OOF) and
optionally the config overlay (``--write-config``).

Sam
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ----- numerics -----
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

# ----- local modules -----
from gap_bridging import features, holdout, labels, tables
from gap_bridging.scorer import (CALIBRATIONS, THRESHOLD_RULES, LinkScorer, ScorerManifest,
                                 SigmoidCalibrator, calibrated_probability, manifest_path,
                                 save_scorer)
from pipeline_common import sha256

# ============================================================
# CONSTANTS
# ============================================================

#: Gates whose pairs reach the assignment (motion gates passed).
TRAIN_GATES = ('', 'max_cost')
DONOR_ROLE = 'decoy_donor'
HOLDOUT_DIR_RE = re.compile(r'^holdout_(\d+)$')
SWEEP_HINT = ("rerun `bridge_gaps.py sweep --holdout` (pair-feature export is on by "
              "default since v004)")
METRICS_FIELDS = ('fold', 'gap_frames', 'n', 'n_pos', 'precision', 'recall',
                  'mean_p_pos', 'mean_p_neg')
THRESHOLD_FLOOR = 1e-12          # learned_min_probability must be in (0, 1]
OOF_NOTE = ("calibration and threshold use train-fold out-of-fold predictions (slightly "
            "optimistic); promotion uses the test fold through the full pipeline")
ROOT = Path(__file__).resolve().parents[2]
LINKS_NAME = 'links.parquet'
DEGENERATE_HINT = "use --calibration sigmoid or lower --target-precision"
#: Pipeline threshold rule (v006): the hold-out horizon whose planted scenes feed the curve
#: and the largest planted gap the rule must hold through (the promotion rule's gap).
DEFAULT_CURVE_HORIZON = 30
DEFAULT_THROUGH_GAP = 30
DEFAULT_MAX_CANDIDATES = 200
#: Truth roles that :func:`gap_bridging.holdout.score_links` counts as planted.
PLANT_ROLES = ('plant', 'decoy_target')

# ============================================================
# PARAMETERS
# ============================================================


@dataclass(frozen=True)
class TrainParams:
    """Classifier, cross-validation and threshold settings.

    ``calibration`` defaults to ``'sigmoid'``: isotonic calibration maps the
    top raw scores to exactly 1, which puts a precision-1.0 threshold at
    ``p = 1`` (D-010). ``allow_degenerate_threshold`` lets
    :func:`train_command` save a model whose threshold is 1 or whose target
    precision is unreachable on the OOF predictions (refused otherwise).
    ``features`` is the feature-subset spec
    (:func:`gap_bridging.features.resolve_feature_set`), kept as given so
    that ``asdict`` stays JSON-ready; :attr:`feature_names` resolves it.
    ``threshold_rule`` (v006) selects how the threshold is chosen:
    ``'pair_oof'`` (:func:`choose_threshold` on the OOF pair predictions,
    the v004 rule) or ``'pipeline'`` (:mod:`gap_bridging.threshold`: the
    smallest candidate above which the full linker makes no false link on
    the train-fold plants of each sweep's ``holdout_<curve_horizon>`` run at
    every gap through ``through_gap``, with the dummy probability tied to the
    threshold). ``max_candidates`` caps the curve's candidate thresholds.
    """
    seed: int = 0
    max_depth: int = 3
    max_iter: int = 200
    learning_rate: float = 0.05
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0
    n_splits: int = 5
    calibration: str = 'sigmoid'            # or 'isotonic'
    target_precision: float = 1.0
    drop_donor_negatives: bool = False
    allow_degenerate_threshold: bool = False
    features: str = 'all'                   # FEATURE_SETS key or comma-separated names
    threshold_rule: str = 'pair_oof'        # or 'pipeline' (v006)
    curve_horizon: int = DEFAULT_CURVE_HORIZON
    through_gap: int = DEFAULT_THROUGH_GAP
    max_candidates: int = DEFAULT_MAX_CANDIDATES

    def __post_init__(self) -> None:
        if self.threshold_rule not in THRESHOLD_RULES:
            raise ValueError(f"threshold_rule must be one of {THRESHOLD_RULES}")
        if self.curve_horizon < 1 or self.through_gap < 1 or self.max_candidates < 2:
            raise ValueError("curve_horizon and through_gap must be >= 1, max_candidates >= 2")
        if self.through_gap > self.curve_horizon:
            raise ValueError(f"through_gap {self.through_gap} exceeds curve_horizon "
                             f"{self.curve_horizon}: the linker cannot bridge gaps above its "
                             f"horizon, so the rule could never hold there")
        if not isinstance(self.features, str):
            raise ValueError(f"features must be a string spec, got {self.features!r}")
        features.resolve_feature_set(self.features)        # raises ValueError if invalid
        if self.calibration not in CALIBRATIONS:
            raise ValueError(f"calibration must be one of {CALIBRATIONS}")
        if self.n_splits < 2 or self.max_depth < 1 or self.max_iter < 1 or self.min_samples_leaf < 1:
            raise ValueError("n_splits must be >= 2; max_depth, max_iter, min_samples_leaf >= 1")
        if not 0 < self.learning_rate or self.l2_regularization < 0:
            raise ValueError("learning_rate must be > 0 and l2_regularization >= 0")
        if not 0 < self.target_precision <= 1:
            raise ValueError("target_precision must be in (0, 1]")

    @property
    def feature_names(self) -> Tuple[str, ...]:
        """The resolved feature subset, in ``features.FEATURE_NAMES`` order."""
        return features.resolve_feature_set(self.features)


# ============================================================
# DATA
# ============================================================


def holdout_dirs(sweep_dir: Path) -> List[Tuple[int, Path]]:
    """``(horizon, dir)`` of the completed ``holdout_<h>/`` runs of a sweep, by horizon.

    A directory counts when it contains ``holdout_<h>.json`` (written last by
    :func:`gap_bridging.holdout.run_holdout`); this is the only completeness
    criterion of a hold-out run. Its pair features are then verified against
    the bridged run's marker (:func:`read_pair_features`).
    """
    sweep_dir = Path(sweep_dir)
    found = []
    for child in sweep_dir.iterdir() if sweep_dir.is_dir() else ():
        match = HOLDOUT_DIR_RE.match(child.name)
        if match and child.is_dir() and (child / f'{child.name}.json').is_file():
            found.append((int(match.group(1)), child))
    return sorted(found)


def read_pair_features(run_dir: Path) -> pa.Table:
    """The ``bridged/pair_features.parquet`` of a hold-out run, verified fresh.

    Parameters
    ----------
    run_dir : Path
        A completed ``holdout_<h>/`` directory.

    Returns
    -------
    pa.Table
        The pair features table.

    Raises
    ------
    FileNotFoundError
        If the pair features file is missing.
    ValueError
        If ``bridged/GAP_BRIDGING_COMPLETE.json`` is missing or does not list
        the file under ``outputs``, the recorded sha256 differs from the
        file's, or its row count differs from ``links.parquet``'s (a stale
        file from an earlier run).
    """
    bridged = Path(run_dir) / holdout.BRIDGED_DIR
    feature_path = bridged / features.PAIR_FEATURES_NAME
    if not feature_path.is_file():
        raise FileNotFoundError(f"pair features not found: {feature_path}; {SWEEP_HINT}")
    marker_path = bridged / tables.COMPLETE_NAME
    if not marker_path.is_file():
        raise ValueError(f"{bridged}: no {tables.COMPLETE_NAME}; the run did not complete; "
                         f"{SWEEP_HINT}")
    outputs = json.loads(marker_path.read_text()).get('outputs', {})
    entry = outputs.get(features.PAIR_FEATURES_NAME)
    if not isinstance(entry, dict) or 'sha256' not in entry:
        raise ValueError(f"{bridged}: {tables.COMPLETE_NAME} does not list "
                         f"{features.PAIR_FEATURES_NAME} (stale file of an earlier run?); "
                         f"{SWEEP_HINT}")
    if entry['sha256'] != sha256(feature_path):
        raise ValueError(f"{bridged}: {features.PAIR_FEATURES_NAME} does not match the sha256 "
                         f"in {tables.COMPLETE_NAME}; {SWEEP_HINT}")
    links_path = bridged / LINKS_NAME
    if not links_path.is_file():
        raise ValueError(f"{bridged}: {LINKS_NAME} is missing; {SWEEP_HINT}")
    pairs = pq.read_table(feature_path)
    n_links = pq.ParquetFile(links_path).metadata.num_rows
    if pairs.num_rows != n_links:
        raise ValueError(f"{bridged}: {features.PAIR_FEATURES_NAME} has {pairs.num_rows} rows "
                         f"but {LINKS_NAME} has {n_links}; {SWEEP_HINT}")
    return pairs


def read_feature_config(run_dir: Path) -> dict:
    """The feature keys (:func:`gap_bridging.features.feature_config`) of a hold-out run.

    Read from the ``config`` of ``holdout_<h>.json``.

    Raises
    ------
    ValueError
        If the JSON has no ``config`` or the config lacks a feature key.
    """
    run_dir = Path(run_dir)
    path = run_dir / f'{run_dir.name}.json'
    config = json.loads(path.read_text()).get('config')
    if not isinstance(config, dict):
        raise ValueError(f"{path} has no 'config'; {SWEEP_HINT}")
    try:
        return features.feature_config(config)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def read_profile(run_dir: Path) -> str:
    """Hold-out profile of one run: ``profile.name`` of ``holdout_<h>.json``.

    ``'standard'`` when the JSON has no ``profile`` (written before v005).
    """
    run_dir = Path(run_dir)
    profile = json.loads((run_dir / f'{run_dir.name}.json').read_text()).get('profile')
    return str(profile.get('name', 'standard')) if isinstance(profile, dict) else 'standard'


def training_profile(sweep_dirs: Sequence[Path]) -> str:
    """The hold-out profile shared by every completed run of every sweep.

    Read per run (:func:`holdout_dirs`, :func:`read_profile`); ``'standard'``
    for runs written before profiles existed, and when no sweep has a
    completed run (a sweep without one adds nothing). Raises ValueError when
    no sweep is given or two runs disagree (the message names each sweep
    directory and its profiles).
    """
    if not sweep_dirs:
        raise ValueError("no sweep directories given")
    profiles = {str(d): sorted({read_profile(run) for _, run in holdout_dirs(Path(d))})
                for d in sweep_dirs}
    names = {name for found in profiles.values() for name in found}
    if len(names) > 1:
        raise ValueError(f"training sweeps disagree on the hold-out profile: {profiles}")
    return names.pop() if names else 'standard'


def collect_training_rows(sweep_dirs: Sequence[Path], salt: str = labels.DEFAULT_SALT,
                          test_fraction: float = labels.DEFAULT_TEST_FRACTION,
                          drop_donor_negatives: bool = False) -> Tuple[pa.Table, dict]:
    """Labelled, deduplicated, motion-gated pairs of several sweeps.

    Parameters
    ----------
    sweep_dirs : sequence of Path
        Sweep directories (``bridge_gaps.py sweep --holdout``), e.g. one per seed.
    salt, test_fraction
        Fold assignment (:func:`gap_bridging.labels.fold_of`).
    drop_donor_negatives : bool
        Drop negatives whose START half is a decoy donor.

    Returns
    -------
    table : pa.Table
        :data:`gap_bridging.labels.LABELLED_SCHEMA` plus a string column
        ``sweep`` (the sweep directory); labels 0 and 1 only.
    feature_cfg : dict
        The feature keys of the hold-out configs
        (:func:`gap_bridging.features.feature_config`), equal across every
        sweep and horizon.

    Raises
    ------
    FileNotFoundError
        If a sweep has no completed hold-out or a hold-out has no pair features.
    ValueError
        If pair features are stale (:func:`read_pair_features`), two
        hold-outs disagree on a feature key (the message lists the keys) or
        the sweeps disagree on the hold-out profile (:func:`training_profile`).
    """
    if not sweep_dirs:
        raise ValueError("no sweep directories given")
    training_profile(sweep_dirs)
    parts = []
    feature_cfg: Optional[dict] = None
    feature_cfg_source: Optional[Path] = None
    for sweep_dir in sweep_dirs:
        runs = holdout_dirs(Path(sweep_dir))
        if not runs:
            raise FileNotFoundError(f"no completed holdout_<h>/ directory in {sweep_dir}; "
                                    f"{SWEEP_HINT}")
        tables, horizons = [], []
        for horizon, run_dir in runs:
            run_cfg = read_feature_config(run_dir)
            if feature_cfg is None:
                feature_cfg, feature_cfg_source = run_cfg, run_dir
            elif run_cfg != feature_cfg:
                differ = [k for k in features.FEATURE_CONFIG_KEYS if run_cfg[k] != feature_cfg[k]]
                raise ValueError(f"hold-out configs disagree on feature keys {differ}: "
                                 f"{feature_cfg_source} vs {run_dir}")
            pairs = read_pair_features(run_dir)
            truth = pq.read_table(run_dir / holdout.TRUTH_NAME)
            tables.append(labels.label_pairs(pairs, truth, salt, test_fraction))
            horizons.append(horizon)
        table = labels.dedupe_pairs(tables, horizons)
        keep = pc.and_(pc.is_in(table['gate'], value_set=pa.array(TRAIN_GATES)),
                       pc.not_equal(table['label'], labels.UNLABELLED))
        if drop_donor_negatives:
            donor_negative = pc.and_(pc.equal(table['label'], labels.NEGATIVE),
                                     pc.equal(pc.fill_null(table['role_b'], ''), DONOR_ROLE))
            keep = pc.and_(keep, pc.invert(donor_negative))
        table = table.filter(pc.fill_null(keep, False))
        table = table.append_column('sweep', pa.array([str(sweep_dir)] * table.num_rows,
                                                      pa.string()))
        parts.append(table)
    assert feature_cfg is not None
    return pa.concat_tables(parts).combine_chunks(), feature_cfg


def _groups(table: pa.Table) -> np.ndarray:
    """CV group per row: the original of the planted side.

    ``original_a`` when the END is planted (``role_a`` not null), else
    ``original_b``. Since ``labels.label_pairs`` fills both originals and
    labels a pair only when they share a fold, the group lies in the row's
    fold either way; using the planted side keeps every pair of one plant,
    decoy or flicker in one CV group.
    """
    planted_a = pc.is_valid(table['role_a'])
    groups = pc.if_else(planted_a, table['original_a'], table['original_b'])
    assert groups.null_count == 0, "every labelled pair must touch a planted half"
    return np.asarray(groups.to_pylist(), dtype=object)


def split_folds(table: pa.Table, names: Sequence[str] = features.FEATURE_NAMES
                ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``fold -> (X, y, groups)`` for ``'train'``, ``'test'`` and any other fold present.

    ``X`` is ``(n, len(names))`` from :func:`gap_bridging.features.feature_matrix`
    (columns ``names``, an ordered subset of ``FEATURE_NAMES``), ``y`` the int
    labels and ``groups`` the CV groups (:func:`_groups`: the original of the
    planted side). A fold with no rows gives empty arrays.
    """
    names = tuple(names)
    features.check_feature_subset(names)
    present = set(table['fold'].to_pylist()) if table.num_rows else set()
    fold_names = ['train', 'test'] + sorted(f for f in present if f not in ('train', 'test'))
    folds = {}
    for name in fold_names:
        part = table.filter(pc.equal(table['fold'], name)) if table.num_rows else table
        X = (features.feature_matrix(part, names) if part.num_rows
             else np.zeros((0, len(names))))
        y = part['label'].to_numpy().astype(np.int64) if part.num_rows else np.zeros(0, np.int64)
        groups = _groups(part) if part.num_rows else np.zeros(0, dtype=object)
        assert X.shape == (part.num_rows, len(names))
        folds[name] = (np.asarray(X, dtype=float), y, groups)
    return folds


# ============================================================
# MODEL
# ============================================================


def effective_splits(y: np.ndarray, groups: np.ndarray, n_splits: int) -> int:
    """Number of CV splits actually used: ``min(n_splits, groups per class)``.

    ``StratifiedGroupKFold`` needs every class in at least ``n_splits``
    groups for every fold to see both classes; with fewer, the split count
    falls back to the smallest number of groups holding a class.

    Raises
    ------
    ValueError
        If ``y`` has one class or some class lies in fewer than 2 groups.
    """
    y = np.asarray(y)
    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError(f"training needs both labels, got {classes.tolist()}")
    per_class = min(len(set(np.asarray(groups)[y == c].tolist())) for c in classes)
    if per_class < 2:
        raise ValueError(f"each label must occur in >= 2 groups for cross-validation, got {per_class}")
    return int(min(n_splits, per_class))


def _classifier(params: TrainParams, names: Sequence[str] = features.FEATURE_NAMES) -> Any:
    """Unfitted monotonic gradient-boosted classifier on the columns ``names``."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_depth=params.max_depth, max_iter=params.max_iter, learning_rate=params.learning_rate,
        min_samples_leaf=params.min_samples_leaf, l2_regularization=params.l2_regularization,
        monotonic_cst=list(features.monotonic_for(names)),
        early_stopping=False, random_state=params.seed)


def fit_scorer(X: np.ndarray, y: np.ndarray, groups: np.ndarray, params: TrainParams,
               names: Sequence[str] = features.FEATURE_NAMES) -> Tuple[Any, Any, np.ndarray]:
    """Fit classifier and calibrator; return ``(hgb, calibrator, p_oof_calibrated)``.

    Parameters
    ----------
    X : np.ndarray
        ``(n, len(names))`` features in ``names`` order (NaN allowed).
    y : np.ndarray
        ``(n,)`` labels 0/1.
    groups : np.ndarray
        ``(n,)`` group ids (planted original tracklet); never split across folds.
    params : TrainParams
    names : sequence of str
        The columns of ``X``, an ordered subset of ``features.FEATURE_NAMES``
        (default all); sets the monotonic constraints.

    Returns
    -------
    hgb
        Classifier refit on all rows.
    calibrator
        ``IsotonicRegression`` or :class:`gap_bridging.scorer.SigmoidCalibrator`
        fitted on the raw OOF probabilities.
    p_oof : np.ndarray
        ``(n,)`` calibrated OOF probabilities.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import StratifiedGroupKFold

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(np.int64)
    groups = np.asarray(groups)
    names = tuple(names)
    features.check_feature_subset(names)
    assert X.ndim == 2 and X.shape[1] == len(names), f"bad X shape {X.shape} for {len(names)} names"
    assert X.shape[0] == y.shape[0] == groups.shape[0], "X, y, groups lengths differ"
    assert set(np.unique(y).tolist()) <= {0, 1}, "labels must be 0/1"
    k = effective_splits(y, groups, params.n_splits)
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=params.seed)
    p_raw = np.full(y.shape[0], np.nan)
    for fit_idx, oof_idx in cv.split(X, y, groups):
        if np.unique(y[fit_idx]).size < 2:          # degenerate fold: constant prior
            p_raw[oof_idx] = float(np.mean(y[fit_idx]))
            continue
        model = _classifier(params, names).fit(X[fit_idx], y[fit_idx])
        p_raw[oof_idx] = model.predict_proba(X[oof_idx])[:, 1]
    assert np.all(np.isfinite(p_raw)), "every row must get an out-of-fold prediction"
    if params.calibration == 'isotonic':
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds='clip')
        calibrator.fit(p_raw, y)
    else:
        calibrator = SigmoidCalibrator.fit(p_raw, y, seed=params.seed)
    hgb = _classifier(params, names).fit(X, y)
    p_oof = np.clip(np.asarray(calibrator.predict(p_raw), dtype=float), 0.0, 1.0)
    return hgb, calibrator, p_oof


# ============================================================
# THRESHOLD AND METRICS
# ============================================================


def choose_threshold(p: np.ndarray, y: np.ndarray, target_precision: float) -> float:
    """Lowest probability threshold whose precision reaches the target.

    The smallest ``t`` among the unique values of ``p`` with
    ``precision({p >= t}) >= target_precision`` (so recall is maximal under
    the constraint). If no ``t`` qualifies: just above the highest negative
    (``nextafter(max p[y == 0], inf)``); with no negatives at all, the
    smallest positive probability. The fallback selects no pair (or, once
    clipped to 1, admits a negative at ``p = 1``); :func:`train_command`
    refuses it through :func:`target_reached`.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(np.int64)
    if p.size == 0 or p.shape != y.shape:
        raise ValueError("p and y must be non-empty and of equal shape")
    if not np.any(y == 0):
        return float(np.min(p[y == 1]))
    p_sorted = np.sort(p)
    p_pos = np.sort(p[y == 1])
    candidates = np.unique(p)
    n_at_least = p.size - np.searchsorted(p_sorted, candidates, side='left')
    tp_at_least = p_pos.size - np.searchsorted(p_pos, candidates, side='left')
    ok = tp_at_least >= target_precision * n_at_least - 1e-9 * n_at_least
    ok &= n_at_least > 0
    ok &= tp_at_least > 0
    if np.any(ok):
        return float(candidates[np.argmax(ok)])
    return float(np.nextafter(np.max(p[y == 0]), np.inf))


def _ratio(num: float, den: float) -> float:
    """``num / den`` as a float; NaN when ``den`` is 0 (e.g. precision with no predictions)."""
    return float(num) / float(den) if den else math.nan


def target_reached(p: np.ndarray, y: np.ndarray, threshold: float,
                   target_precision: float) -> bool:
    """Whether ``{p >= threshold}`` holds a positive and has precision ``>= target``.

    False when nothing is predicted positive (the :func:`choose_threshold`
    fallback) or when clipping the threshold to 1 admitted negatives.
    """
    precision = metrics(p, y, threshold)['precision']
    return bool(math.isfinite(precision) and precision >= target_precision - 1e-9)


def _roc_auc(p: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney ROC AUC (ties count one half); NaN with one class."""
    from scipy.stats import rankdata

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = rankdata(p)
    return float((np.sum(ranks[y == 1]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def metrics(p: np.ndarray, y: np.ndarray, threshold: float) -> Dict[str, float]:
    """Pair-level metrics of probabilities ``p`` against labels ``y``.

    Keys: ``n``, ``n_pos``, ``threshold``, ``precision``, ``recall``, ``f1``
    (predicted positive when ``p >= threshold``; NaN for 0/0), ``log_loss``
    (``p`` clipped to ``[1e-12, 1 - 1e-12]``), ``brier`` and ``roc_auc``
    (NaN with one class).
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(np.int64)
    assert p.shape == y.shape, "p and y must have equal shape"
    predicted = p >= threshold
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    if p.size:
        q = np.clip(p, 1e-12, 1 - 1e-12)
        log_loss = float(-np.mean(y * np.log(q) + (1 - y) * np.log1p(-q)))
        brier = float(np.mean((p - y) ** 2))
    else:
        log_loss = brier = math.nan
    return {'n': int(p.size), 'n_pos': int(np.sum(y == 1)), 'threshold': float(threshold),
            'precision': precision, 'recall': recall, 'f1': f1, 'log_loss': log_loss,
            'brier': brier, 'roc_auc': _roc_auc(p, y)}


def per_gap_metrics(table: pa.Table, p: np.ndarray, threshold: float) -> List[dict]:
    """Metrics per planted gap length (``gap_frames``), ascending.

    Each dict: ``gap_frames``, ``n``, ``n_pos``, ``precision``, ``recall``
    (at ``threshold``), ``mean_p_pos``, ``mean_p_neg`` (NaN when empty).
    """
    p = np.asarray(p, dtype=float)
    assert p.shape == (table.num_rows,), "one probability per row"
    gaps = table['gap_frames'].to_numpy() if table.num_rows else np.zeros(0, np.int64)
    y = table['label'].to_numpy().astype(np.int64) if table.num_rows else np.zeros(0, np.int64)
    rows = []
    for gap in np.unique(gaps):
        sel = gaps == gap
        m = metrics(p[sel], y[sel], threshold)
        pos, neg = p[sel & (y == 1)], p[sel & (y == 0)]
        rows.append({'gap_frames': int(gap), 'n': m['n'], 'n_pos': m['n_pos'],
                     'precision': m['precision'], 'recall': m['recall'],
                     'mean_p_pos': float(np.mean(pos)) if pos.size else math.nan,
                     'mean_p_neg': float(np.mean(neg)) if neg.size else math.nan})
    return rows


# ============================================================
# PROVENANCE
# ============================================================


def table_sha256(table: pa.Table) -> str:
    """Content hash of a table: column names, types, null masks and values in row order.

    Independent of chunking and Parquet encoding.
    """
    digest = hashlib.sha256()
    for name in table.column_names:
        column = table[name].combine_chunks() if table.num_rows else table[name]
        digest.update(f'{name}\x00{column.type}\x00{table.num_rows}\x00'.encode())
        mask = np.asarray(pc.is_null(column).to_numpy(zero_copy_only=False), dtype=np.uint8)
        digest.update(mask.tobytes())
        if pa.types.is_floating(column.type) or pa.types.is_integer(column.type):
            values = pc.fill_null(column, 0).to_numpy(zero_copy_only=False)
            digest.update(np.ascontiguousarray(values).tobytes())
        else:
            values = ['' if v is None else str(v) for v in column.to_pylist()]
            digest.update('\x1f'.join(values).encode())
    return digest.hexdigest()


def git_commit() -> str:
    """``git rev-parse HEAD`` of the repository, or ``''``."""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                                text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return ''
    return result.stdout.strip()


def metrics_csv_path(model_path: Path) -> Path:
    """``<model stem>_metrics.csv`` beside the model."""
    model_path = Path(model_path)
    return model_path.with_name(model_path.stem + '_metrics.csv')


def _write_metrics_csv(path: Path, fold: str, rows: List[dict]) -> None:
    """Write :func:`per_gap_metrics` rows to ``path`` (:data:`METRICS_FIELDS`, ``fold`` column set)."""
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(METRICS_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({'fold': fold, **row})


def write_overlay(config_path: Path, model_path: Path, threshold: float, created: str,
                  unmatched_probability: Optional[float] = None) -> None:
    """Create or update a config overlay to use the scorer at ``model_path``.

    Sets ``cost_model: learned``, ``scorer_path`` (relative to the YAML's
    directory, POSIX) and ``learned_min_probability``, plus
    ``learned_unmatched_probability`` when ``unmatched_probability`` is given
    (the pipeline rule ties it to the threshold; ``None`` leaves the key
    untouched, so a ``pair_oof`` overlay is written as before v006); other
    keys are kept. Comments of an existing file are replaced by one header
    line.
    """
    config_path = Path(config_path)
    values: dict = {}
    if config_path.is_file():
        values = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(values, dict):
            raise ValueError(f"{config_path} is not a YAML mapping")
    rel = os.path.relpath(Path(model_path).resolve(), config_path.resolve().parent)
    values.update({'cost_model': 'learned', 'scorer_path': Path(rel).as_posix(),
                   'learned_min_probability': float(threshold)})
    if unmatched_probability is not None:
        values['learned_unmatched_probability'] = float(unmatched_probability)
    header = (f"# Written by `bridge_gaps.py train` on {created}: learned link scorer overlay; "
              f"scorer_path is relative to this file.\n")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(header + yaml.safe_dump(values, sort_keys=False))


# ============================================================
# PIPELINE THRESHOLD RULE (v006)
# ============================================================

PIPELINE_NOTE = ("calibration uses train-fold out-of-fold predictions; the threshold is the "
                 "pipeline rule's choice on train-fold plants (gap_bridging.threshold); "
                 "promotion uses the test fold through the full pipeline")


def _curve_run(sweep_dir: Path, horizon: int) -> Path:
    """The completed ``holdout_<horizon>`` run of a sweep (FileNotFoundError naming the sweep)."""
    run = Path(sweep_dir) / f'holdout_{horizon}'
    if not (run / f'holdout_{horizon}.json').is_file():
        raise FileNotFoundError(f"{sweep_dir} has no completed holdout_{horizon} run for the "
                                f"pipeline threshold curve (--curve-horizon); {SWEEP_HINT}")
    return run


def _probe_manifest(names: Sequence[str], params: TrainParams, threshold: float
                    ) -> ScorerManifest:
    """A minimal in-memory manifest so the fitted model can score the threshold curve."""
    return ScorerManifest(
        sklearn_version='', feature_names=tuple(names),
        monotonic_cst=features.monotonic_for(names), hyperparameters=asdict(params),
        calibration=params.calibration, threshold=float(threshold), training_data_sha256='',
        n_train=0, n_test=0, n_positive_train=0, n_positive_test=0, oof_metrics={},
        test_metrics={}, salt='', test_fraction=0.0, sweeps=(), created='',
        feature_set=params.features)


# ============================================================
# COMMAND
# ============================================================


def train_command(sweeps: Sequence[Path], out: Path, params: TrainParams = TrainParams(),
                  salt: str = labels.DEFAULT_SALT,
                  test_fraction: float = labels.DEFAULT_TEST_FRACTION,
                  write_config: Optional[Path] = None, force: bool = False) -> ScorerManifest:
    """Collect, fit, threshold, evaluate and save a scorer.

    Parameters
    ----------
    sweeps : sequence of Path
        Sweep directories with hold-out runs and pair features.
    out : Path
        Model file (``.joblib``); the manifest and ``<stem>_metrics.csv`` go beside it.
    params : TrainParams
        Includes ``features``, the feature subset the model is trained on.
    salt, test_fraction
        Fold assignment.
    write_config : Path, optional
        Config overlay to create or update (:func:`write_overlay`).
    force : bool
        Overwrite an existing model, manifest and metrics CSV.

    Returns
    -------
    ScorerManifest
        The saved manifest. The threshold is clipped into ``[1e-12, 1]``
        (the range ``learned_min_probability`` accepts). Under
        ``params.threshold_rule == 'pipeline'`` it is the pipeline curve's
        choice (:func:`gap_bridging.threshold.choose_pipeline_threshold`
        over the train-fold plants of each sweep's
        ``holdout_<curve_horizon>`` run), the pair-level value is kept as
        ``pair_threshold``, the curve is written beside the model
        (:func:`gap_bridging.threshold.write_curve`) and summarised in
        ``threshold_summary``; ``feature_config``
        holds the hold-outs' feature keys, ``feature_names`` and
        ``monotonic_cst`` the resolved subset, ``feature_set`` the spec,
        ``training_profile`` the sweeps' hold-out profile
        (:func:`training_profile`), and ``hyperparameters`` records
        ``target_reached``.

    Raises
    ------
    FileExistsError
        If an output exists and ``force`` is false.
    FileNotFoundError
        Under the pipeline rule, if a sweep has no completed
        ``holdout_<curve_horizon>`` run.
    ValueError
        If the train fold is empty, the data are inconsistent
        (:func:`collect_training_rows`), or, under the ``pair_oof`` rule and
        unless ``params.allow_degenerate_threshold``, the clipped threshold
        is ``>= 1`` (every accepted pair would cost 0) or the target
        precision is not reached on the OOF predictions; under the
        ``pipeline`` rule, if no candidate qualifies
        (:func:`gap_bridging.threshold.choose_pipeline_threshold`). Nothing
        is written then.
    """
    import sklearn

    out = Path(out)
    names = params.feature_names
    csv_path = metrics_csv_path(out)
    outputs = [out, manifest_path(out), csv_path]
    pipeline_rule = params.threshold_rule == 'pipeline'
    if pipeline_rule:
        from gap_bridging import threshold as threshold_mod
        outputs.extend(threshold_mod.curve_paths(out))
    existing = [str(p) for p in outputs if p.exists()]
    if existing and not force:
        raise FileExistsError(f"training outputs already exist (use --force): {existing}")
    table, feature_cfg = collect_training_rows(sweeps, salt, test_fraction,
                                               params.drop_donor_negatives)
    folds = split_folds(table, names)
    X_train, y_train, g_train = folds['train']
    X_test, y_test, _ = folds['test']
    if y_train.size == 0:
        raise ValueError("the train fold is empty")
    print(f"[train] {table.num_rows} pairs from {len(sweeps)} sweep(s): train {y_train.size} "
          f"({int(y_train.sum())} positive), test {y_test.size} ({int(y_test.sum())} positive); "
          f"features {params.features!r} ({len(names)})", flush=True)
    hgb, calibrator, p_oof = fit_scorer(X_train, y_train, g_train, params, names)
    raw_threshold = choose_threshold(p_oof, y_train, params.target_precision)
    threshold = float(min(max(raw_threshold, THRESHOLD_FLOOR), 1.0))
    if threshold != raw_threshold:
        print(f"[train] threshold {raw_threshold!r} clipped to {threshold!r}", flush=True)
    reached = target_reached(p_oof, y_train, threshold, params.target_precision)
    pair_threshold = threshold
    refuse = not (params.allow_degenerate_threshold or pipeline_rule)
    if threshold >= 1.0 and refuse:
        raise ValueError(f"degenerate threshold {threshold!r}: every accepted pair would cost 0; "
                         f"{DEGENERATE_HINT} (or --allow-degenerate-threshold)")
    if not reached and refuse:
        raise ValueError(f"target precision {params.target_precision} is unreachable on the "
                         f"out-of-fold predictions (threshold {threshold!r}); lower "
                         f"--target-precision (or --allow-degenerate-threshold)")
    if (not reached or threshold >= 1.0) and not pipeline_rule:
        print(f"[train] WARNING: degenerate threshold {threshold!r} (target reached: {reached}) "
              f"kept by allow_degenerate_threshold", flush=True)
    threshold_summary: dict = {}
    curve = pooled = None
    if pipeline_rule:
        probe = LinkScorer(hgb, calibrator, _probe_manifest(names, params, pair_threshold))
        scenes = [threshold_mod.load_scene(_curve_run(Path(sweep), params.curve_horizon),
                                           salt, test_fraction) for sweep in sweeps]
        candidates = threshold_mod.candidate_thresholds(p_oof, y_train, params.max_candidates)
        n_plants = sum(int(pc.sum(pc.is_in(sc.truth_train['role'],
                                           value_set=pa.array(PLANT_ROLES))).as_py() or 0)
                       for sc in scenes)
        print(f"[train] pipeline threshold curve: {candidates.size} candidates x "
              f"{len(scenes)} scene(s) at horizon {params.curve_horizon}, train-fold plants "
              f"{n_plants}", flush=True)
        curve = threshold_mod.pipeline_curve(scenes, probe, candidates,
                                             progress=lambda msg: print(msg, flush=True))
        pooled = threshold_mod.pool_curve(curve)
        threshold = threshold_mod.choose_pipeline_threshold(pooled, params.through_gap)
        threshold_summary = threshold_mod.summary(pooled, threshold, params.through_gap,
                                                  pair_threshold)
        print(f"[train] pipeline rule chose threshold {threshold!r} (pair-level rule: "
              f"{pair_threshold!r}); through gap {params.through_gap}: "
              f"{threshold_summary['at_chosen']}", flush=True)
    oof_metrics = metrics(p_oof, y_train, threshold)
    if y_test.size:
        p_test = calibrated_probability(hgb, calibrator, X_test)
        test_metrics = metrics(p_test, y_test, threshold)
        gap_fold, gap_table, gap_p = 'test', table.filter(pc.equal(table['fold'], 'test')), p_test
    else:
        test_metrics = {}
        gap_fold, gap_table, gap_p = 'oof', table.filter(pc.equal(table['fold'], 'train')), p_oof
    hyper = asdict(params)
    hyper.update({'n_splits_used': effective_splits(y_train, g_train, params.n_splits),
                  'target_reached': reached, 'note': OOF_NOTE})
    if pipeline_rule:
        hyper['note'] = PIPELINE_NOTE
    manifest = ScorerManifest(
        sklearn_version=sklearn.__version__, feature_names=tuple(names),
        monotonic_cst=features.monotonic_for(names),
        hyperparameters=hyper, calibration=params.calibration, threshold=threshold,
        training_data_sha256=table_sha256(table), n_train=int(y_train.size),
        n_test=int(y_test.size), n_positive_train=int(y_train.sum()),
        n_positive_test=int(y_test.sum()), oof_metrics=oof_metrics, test_metrics=test_metrics,
        salt=salt, test_fraction=float(test_fraction), sweeps=tuple(str(s) for s in sweeps),
        created=datetime.now(timezone.utc).isoformat(timespec='seconds'), git_commit=git_commit(),
        feature_config=dict(feature_cfg), feature_set=params.features,
        training_profile=training_profile(sweeps), threshold_rule=params.threshold_rule,
        pair_threshold=float(pair_threshold), threshold_summary=threshold_summary)
    save_scorer(LinkScorer(hgb, calibrator, manifest), out, force=force)
    _write_metrics_csv(csv_path, gap_fold, per_gap_metrics(gap_table, gap_p, threshold))
    if pipeline_rule:
        assert curve is not None and pooled is not None
        written = threshold_mod.write_curve(curve, pooled, out)
        print(f"[train] wrote threshold curve {', '.join(str(w) for w in written)}", flush=True)
    print(f"[train] threshold {threshold:.6g}; OOF precision {oof_metrics['precision']:.4f} "
          f"recall {oof_metrics['recall']:.4f}; test {test_metrics or 'none'}", flush=True)
    print(f"[train] wrote {out}, {manifest_path(out)}, {csv_path}", flush=True)
    if write_config is not None:
        write_overlay(write_config, out, threshold, manifest.created,
                      unmatched_probability=threshold if pipeline_rule else None)
        print(f"[train] wrote config overlay {write_config}", flush=True)
    return manifest


# ============================================================
# COMMAND LINE (for bridge_gaps.py)
# ============================================================


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the ``train`` subcommand's options to ``parser``."""
    d = TrainParams()
    parser.add_argument('--sweeps', type=Path, nargs='+', required=True,
                        help='sweep directories with holdout_<h>/bridged/pair_features.parquet')
    parser.add_argument('--out', type=Path, required=True, help='model file (.joblib)')
    parser.add_argument('--write-config', type=Path, default=None,
                        help='config overlay to create or update with the scorer')
    parser.add_argument('--salt', default=labels.DEFAULT_SALT)
    parser.add_argument('--test-fraction', type=float, default=labels.DEFAULT_TEST_FRACTION)
    parser.add_argument('--seed', type=int, default=d.seed)
    parser.add_argument('--max-depth', type=int, default=d.max_depth)
    parser.add_argument('--max-iter', type=int, default=d.max_iter)
    parser.add_argument('--learning-rate', type=float, default=d.learning_rate)
    parser.add_argument('--min-samples-leaf', type=int, default=d.min_samples_leaf)
    parser.add_argument('--l2-regularization', type=float, default=d.l2_regularization)
    parser.add_argument('--n-splits', type=int, default=d.n_splits)
    parser.add_argument('--calibration', choices=CALIBRATIONS, default=d.calibration,
                        help=f'probability calibration (default {d.calibration}; isotonic tends '
                             'to put the threshold at p = 1)')
    parser.add_argument('--target-precision', type=float, default=d.target_precision)
    parser.add_argument('--drop-donor-negatives', action='store_true')
    parser.add_argument('--features', default=d.features,
                        help=f'feature subset to train on: one of {sorted(features.FEATURE_SETS)} '
                             f'(default {d.features}) or a comma-separated list of feature names')
    parser.add_argument('--allow-degenerate-threshold', action='store_true',
                        help='save a model whose threshold is 1 or whose target precision is '
                             'unreachable on the out-of-fold predictions (refused otherwise)')
    parser.add_argument('--threshold-rule', choices=THRESHOLD_RULES, default=d.threshold_rule,
                        help=f'how the threshold is chosen (default {d.threshold_rule}): '
                             'pair_oof = smallest OOF probability with pair precision at the '
                             'target; pipeline = smallest candidate above which the full linker '
                             'makes no false link on the train-fold plants of each sweep\'s '
                             'holdout_<curve-horizon> run at every gap through --through-gap '
                             '(dummy probability tied to the threshold)')
    parser.add_argument('--curve-horizon', type=int, default=d.curve_horizon,
                        help=f'hold-out horizon whose runs feed the pipeline curve '
                             f'(default {d.curve_horizon})')
    parser.add_argument('--through-gap', type=int, default=d.through_gap,
                        help=f'largest planted gap the pipeline rule must hold through '
                             f'(default {d.through_gap})')
    parser.add_argument('--max-candidates', type=int, default=d.max_candidates,
                        help=f'cap on candidate thresholds of the pipeline curve '
                             f'(default {d.max_candidates})')
    parser.add_argument('--force', action='store_true', help='overwrite existing outputs')


def run_from_args(args: argparse.Namespace) -> ScorerManifest:
    """Run :func:`train_command` from parsed :func:`add_arguments` options."""
    params = replace(TrainParams(), seed=args.seed, max_depth=args.max_depth,
                     max_iter=args.max_iter, learning_rate=args.learning_rate,
                     min_samples_leaf=args.min_samples_leaf,
                     l2_regularization=args.l2_regularization, n_splits=args.n_splits,
                     calibration=args.calibration, target_precision=args.target_precision,
                     drop_donor_negatives=args.drop_donor_negatives,
                     allow_degenerate_threshold=args.allow_degenerate_threshold,
                     features=args.features, threshold_rule=args.threshold_rule,
                     curve_horizon=args.curve_horizon, through_gap=args.through_gap,
                     max_candidates=args.max_candidates)
    return train_command(args.sweeps, args.out, params, args.salt, args.test_fraction,
                         args.write_config, args.force)
