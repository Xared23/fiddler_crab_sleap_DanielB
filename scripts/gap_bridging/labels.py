#!/usr/bin/env python3

"""
Labels and train/test folds for the learned link scorer (v004).

Labels come from the frame-deletion hold-out (:mod:`gap_bridging.holdout`):
a candidate pair is a positive when it rejoins the two halves of a planted
gap, a negative when it touches a planted half but is not that half's true
partner, and unlabelled otherwise. Folds are a deterministic function of the
ORIGINAL (pre-plant) tracklet id, so that every plant, decoy and flicker cut
from one tracklet, under every seed, lands in the same fold.

Planted sides. The END side (``tracklet_a``) of a pair is planted when it is
a truth ``left_id`` (any role) or a flicker ``middle_id``; the START side
(``tracklet_b``) when it is a truth ``right_id`` (any role) or a flicker
``middle_id``. Labels:

* ``1``: (left, right) of a ``plant`` or ``decoy_target``; (left, middle) and
  (middle, right) of a ``flicker``.
* ``-1``: neither side planted; the (left, right) of a ``flicker`` (the middle
  lies between them) and of a ``decoy_donor`` (its right half was translated,
  so the geometry is fake); both sides planted from originals in different
  folds.
* ``0``: every other pair with a planted side, among them the decoy hard
  negative (target left, donor right) and any END into a donor's moved right
  half (``role_b == 'decoy_donor'``; a trainer may drop those).

Sam
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

# ----- numerics -----
import numpy as np
import pyarrow as pa

# ----- local modules -----
from gap_bridging.features import FEATURE_NAMES, PAIR_FEATURES_SCHEMA, feature_matrix

# ============================================================
# CONSTANTS
# ============================================================

FOLDS = ('train', 'test', 'unlabelled')
DEFAULT_SALT = 'gap_bridging_v004'
DEFAULT_TEST_FRACTION = 0.4

#: Label codes in the ``label`` column.
POSITIVE, NEGATIVE, UNLABELLED = 1, 0, -1

LABELLED_SCHEMA = pa.schema(
    list(PAIR_FEATURES_SCHEMA)
    + [('label', pa.int8()), ('fold', pa.string()),
       ('role_a', pa.string()), ('role_b', pa.string()),
       ('original_a', pa.string()), ('original_b', pa.string())])


#: Hold-out truth roles (``holdout.ROLES``; repeated here so that holdout may
#: import this module without a cycle).
_ROLES = ('plant', 'decoy_target', 'decoy_donor', 'flicker')
_PAIR_SEP = '\x00'

# ============================================================
# FOLDS
# ============================================================


def fold_of(original_id: str, salt: str = DEFAULT_SALT,
            test_fraction: float = DEFAULT_TEST_FRACTION) -> str:
    """
    'test' or 'train' from a salted hash of the original tracklet id.

    Parameters
    ----------
    original_id : str
        Pre-plant tracklet id (truth ``original_id``).
    salt : str
        Hash salt; a different salt gives an independent split.
    test_fraction : float
        Probability of 'test', in ``[0, 1]``.

    Returns
    -------
    str
        'test' iff ``u < test_fraction`` with ``u`` the first 8 bytes of
        ``sha256(salt + NUL + original_id)`` as a big-endian integer over
        ``2**64``.

    Raises
    ------
    ValueError
        If ``test_fraction`` is outside ``[0, 1]``.
    """
    if not 0.0 <= float(test_fraction) <= 1.0:
        raise ValueError(f"test_fraction must be in [0, 1], got {test_fraction}")
    digest = hashlib.sha256(f"{salt}\x00{original_id}".encode()).digest()
    u = int.from_bytes(digest[:8], 'big') / 2 ** 64
    assert 0.0 <= u < 1.0
    return 'test' if u < float(test_fraction) else 'train'


def _truth_columns(truth: pa.Table) -> Dict[str, list]:
    """Truth columns as lists; ``role`` defaults to 'plant', ``middle_id`` to null."""
    for name in ('left_id', 'right_id', 'original_id'):
        if name not in truth.schema.names:
            raise ValueError(f"truth table has no column {name!r}")
    n = truth.num_rows
    columns = {name: truth[name].to_pylist() for name in ('left_id', 'right_id', 'original_id')}
    columns['role'] = truth['role'].to_pylist() if 'role' in truth.schema.names else ['plant'] * n
    columns['middle_id'] = (truth['middle_id'].to_pylist() if 'middle_id' in truth.schema.names
                            else [None] * n)
    for role in set(columns['role']):
        if role not in _ROLES:
            raise ValueError(f"unknown truth role {role!r}")
    for k in range(n):
        if None in (columns['left_id'][k], columns['right_id'][k], columns['original_id'][k]):
            raise ValueError(f"truth row {k} has a null left_id, right_id or original_id")
        if columns['role'][k] == 'flicker' and columns['middle_id'][k] is None:
            raise ValueError(f"flicker truth row {k} has no middle_id")
    return columns


def truth_folds(truth: pa.Table, salt: str = DEFAULT_SALT,
                test_fraction: float = DEFAULT_TEST_FRACTION) -> pa.Table:
    """
    The hold-out truth table plus a ``fold`` column from ``original_id``.

    Parameters
    ----------
    truth : pa.Table
        Truth table of :func:`gap_bridging.holdout.plant_gaps` (a table
        without ``role`` is all plants).
    salt, test_fraction
        As :func:`fold_of`.

    Returns
    -------
    pa.Table
        ``truth`` with a string column ``fold`` ('train' or 'test') appended.

    Raises
    ------
    ValueError
        On a missing or null ``original_id``, an unknown role or an invalid
        ``test_fraction``.
    """
    columns = _truth_columns(truth)
    folds = [fold_of(o, salt, test_fraction) for o in columns['original_id']]
    if 'fold' in truth.schema.names:
        truth = truth.drop_columns(['fold'])
    return truth.append_column('fold', pa.array(folds, type=pa.string()))


# ============================================================
# LABELS
# ============================================================


Side = Tuple[str, str, str]       # (role, original_id, fold) of a planted side


def _planted_sides(columns: Dict[str, list], folds: List[str]
                   ) -> Tuple[Dict[str, Side], Dict[str, Side], set, set]:
    """
    Planted END and START sides, positive pairs and explicitly unlabelled pairs.

    Returns
    -------
    (end_side, start_side, positive, unlabelled) : tuple
        ``tracklet_id -> (role, original, fold)`` for the END and START sides,
        and two sets of ``(tracklet_a, tracklet_b)``.

    Raises
    ------
    ValueError
        If a tracklet is planted twice on the same side.
    """
    end_side: Dict[str, Side] = {}
    start_side: Dict[str, Side] = {}
    positive: set = set()
    unlabelled: set = set()

    def put(table: Dict[str, Side], tracklet: str, value: Side, which: str) -> None:
        if tracklet in table:
            raise ValueError(f"tracklet {tracklet!r} is planted twice as a {which} side")
        table[tracklet] = value

    for k, (left, right, original, role, middle) in enumerate(zip(
            columns['left_id'], columns['right_id'], columns['original_id'],
            columns['role'], columns['middle_id'])):
        value = (role, original, folds[k])
        put(end_side, left, value, 'END')
        put(start_side, right, value, 'START')
        if role == 'flicker':
            put(end_side, middle, value, 'END')
            put(start_side, middle, value, 'START')
            positive.update({(left, middle), (middle, right)})
            unlabelled.add((left, right))
        elif role == 'decoy_donor':
            unlabelled.add((left, right))
        else:
            positive.add((left, right))
    assert not positive & unlabelled, "a pair cannot be both positive and unlabelled"
    return end_side, start_side, positive, unlabelled


def label_pairs(features: pa.Table, truth: pa.Table, salt: str = DEFAULT_SALT,
                test_fraction: float = DEFAULT_TEST_FRACTION) -> pa.Table:
    """
    Label the candidate pairs of one planted video against its hold-out truth.

    Parameters
    ----------
    features : pa.Table
        :data:`PAIR_FEATURES_SCHEMA` rows of the planted observations
        (:func:`gap_bridging.features.pair_feature_table`).
    truth : pa.Table
        The matching :func:`gap_bridging.holdout.plant_gaps` truth table.
    salt, test_fraction
        As :func:`fold_of`.

    Returns
    -------
    pa.Table
        :data:`LABELLED_SCHEMA`: the features rows (same order) plus
        ``label`` (1, 0 or -1, see the module docstring); ``fold`` ('train' or
        'test' from the planted side's original for labels 0 and 1,
        'unlabelled' for -1); ``role_a``/``original_a`` of the truth row that
        planted ``tracklet_a`` as an END side and ``role_b``/``original_b``
        of the row that planted ``tracklet_b`` as a START side (null when not
        planted).

    Raises
    ------
    ValueError
        On a features table without the key columns, an unknown truth role,
        a tracklet planted twice on one side or an invalid ``test_fraction``.
    """
    for name in PAIR_FEATURES_SCHEMA.names:
        if name not in features.schema.names:
            raise ValueError(f"features table has no column {name!r}")
    features = features.select(PAIR_FEATURES_SCHEMA.names).cast(PAIR_FEATURES_SCHEMA)
    columns = _truth_columns(truth)
    folds = [fold_of(o, salt, test_fraction) for o in columns['original_id']]
    end_side, start_side, positive, unlabelled = _planted_sides(columns, folds)

    a_ids = features['tracklet_a'].to_pylist()
    b_ids = features['tracklet_b'].to_pylist()
    n = features.num_rows
    label = np.full(n, UNLABELLED, dtype=np.int8)
    fold: List[str] = ['unlabelled'] * n
    role_a: List[Optional[str]] = [None] * n
    role_b: List[Optional[str]] = [None] * n
    original_a: List[Optional[str]] = [None] * n
    original_b: List[Optional[str]] = [None] * n
    for k, (a, b) in enumerate(zip(a_ids, b_ids)):
        side_a, side_b = end_side.get(a), start_side.get(b)
        if side_a is not None:
            role_a[k], original_a[k] = side_a[0], side_a[1]
        if side_b is not None:
            role_b[k], original_b[k] = side_b[0], side_b[1]
        if side_a is None and side_b is None:
            continue
        if (a, b) in unlabelled:
            continue
        if side_a is not None and side_b is not None and side_a[2] != side_b[2]:
            assert (a, b) not in positive, "a true pair shares its original"
            continue
        label[k] = POSITIVE if (a, b) in positive else NEGATIVE
        fold[k] = (side_a if side_a is not None else side_b)[2]
    assert all((f == 'unlabelled') == (lab == UNLABELLED) for f, lab in zip(fold, label))
    out = features
    for name, values, kind in (('label', label, pa.int8()), ('fold', fold, pa.string()),
                               ('role_a', role_a, pa.string()), ('role_b', role_b, pa.string()),
                               ('original_a', original_a, pa.string()),
                               ('original_b', original_b, pa.string())):
        out = out.append_column(name, pa.array(values, type=kind))
    out = out.cast(LABELLED_SCHEMA)
    assert out.num_rows == n
    return out


# ============================================================
# DEDUPLICATION
# ============================================================


def _same_features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-wise equality of two feature matrices, NaN equal to NaN."""
    return np.all((x == y) | (np.isnan(x) & np.isnan(y)), axis=1)


def dedupe_pairs(tables: Sequence[pa.Table], horizons: Sequence[int]) -> pa.Table:
    """
    One row per ``(tracklet_a, tracklet_b)`` across the horizons of one seed.

    Parameters
    ----------
    tables : sequence of pa.Table
        Labelled (or feature) tables with one schema, built from the same
        planted observations at different ``max_gap`` horizons.
    horizons : sequence of int
        The horizon of each table; distinct.

    Returns
    -------
    pa.Table
        The input schema; the rows of the largest horizon (in its order),
        then the pairs first seen at the next largest horizon, and so on, so
        every pair appears once, taken from the largest horizon that
        enumerates it.

    Raises
    ------
    ValueError
        On mismatched lengths, repeated horizons, differing schemas, a pair
        repeated within one table, or a recurring pair whose
        :data:`~gap_bridging.features.FEATURE_NAMES` values differ (NaN equal
        to NaN) between horizons.
    """
    if len(tables) != len(horizons) or not tables:
        raise ValueError("need one horizon per table and at least one table")
    if len(set(int(h) for h in horizons)) != len(horizons):
        raise ValueError(f"horizons must be distinct, got {list(horizons)}")
    schema = tables[0].schema
    for table in tables[1:]:
        if not table.schema.equals(schema):
            raise ValueError("tables to deduplicate must share one schema")
    order = sorted(range(len(tables)), key=lambda i: -int(horizons[i]))
    seen: Dict[Tuple[str, str], Tuple[int, int]] = {}
    pieces: List[pa.Table] = []
    matrices: Dict[int, np.ndarray] = {}
    for i in order:
        table = tables[i]
        matrices[i] = feature_matrix(table, FEATURE_NAMES)
        keys = list(zip(table['tracklet_a'].to_pylist(), table['tracklet_b'].to_pylist()))
        if len(set(keys)) != len(keys):
            raise ValueError(f"horizon {horizons[i]} repeats a (tracklet_a, tracklet_b) pair")
        keep = np.zeros(len(keys), dtype=bool)
        earlier_rows: List[int] = []
        earlier_refs: List[Tuple[int, int]] = []
        for k, key in enumerate(keys):
            if key in seen:
                earlier_rows.append(k)
                earlier_refs.append(seen[key])
            else:
                seen[key] = (i, k)
                keep[k] = True
        if earlier_rows:
            here = matrices[i][np.asarray(earlier_rows)]
            there = np.stack([matrices[j][r] for j, r in earlier_refs])
            same = _same_features(here, there)
            if not same.all():
                bad = [keys[earlier_rows[q]] for q in np.flatnonzero(~same)[:5]]
                raise ValueError(f"features differ between horizons for pairs {bad} "
                                 f"(horizon {horizons[i]})")
        pieces.append(table.filter(pa.array(keep)))
    out = pa.concat_tables(pieces).combine_chunks() if pieces else schema.empty_table()
    assert out.num_rows == len(seen)
    assert out.schema.equals(schema)
    return out
