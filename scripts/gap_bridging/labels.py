"""Labels and train/test folds for the learned link scorer (v004).

Labels come from the frame-deletion hold-out (:mod:`gap_bridging.holdout`):
a candidate pair is a positive when it rejoins the two halves of a planted
gap, a negative when it touches a planted half but is not that half's true
partner, and unlabelled otherwise. Folds are a deterministic function of the
ORIGINAL (pre-plant) tracklet id, so that every plant, decoy and flicker cut
from one tracklet, under every seed, lands in the same fold.

Interface stub written by the orchestrator; WP-F fills the functions.
"""
from __future__ import annotations

from typing import Sequence

import pyarrow as pa

from gap_bridging.features import PAIR_FEATURES_SCHEMA

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


# ============================================================
# FUNCTIONS (WP-F)
# ============================================================


def fold_of(original_id: str, salt: str = DEFAULT_SALT,
            test_fraction: float = DEFAULT_TEST_FRACTION) -> str:
    """'test' or 'train' from ``sha256(salt + NUL + original_id)``: the first
    8 bytes as a big-endian integer over 2**64, compared with ``test_fraction``."""
    raise NotImplementedError("WP-F")


def truth_folds(truth: pa.Table, salt: str = DEFAULT_SALT,
                test_fraction: float = DEFAULT_TEST_FRACTION) -> pa.Table:
    """The hold-out truth table plus a ``fold`` column from ``original_id``."""
    raise NotImplementedError("WP-F")


def label_pairs(features: pa.Table, truth: pa.Table, salt: str = DEFAULT_SALT,
                test_fraction: float = DEFAULT_TEST_FRACTION) -> pa.Table:
    """Features plus ``label``, ``fold``, ``role_a``/``role_b`` and
    ``original_a``/``original_b`` (:data:`LABELLED_SCHEMA`); see module doc."""
    raise NotImplementedError("WP-F")


def dedupe_pairs(tables: Sequence[pa.Table], horizons: Sequence[int]) -> pa.Table:
    """One row per ``(tracklet_a, tracklet_b)`` across horizons (the row of the
    largest horizon); raises if a recurring pair's features differ."""
    raise NotImplementedError("WP-F")
