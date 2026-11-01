"""Per-pair feature export for the learned link scorer (v004).

One row per enumerated candidate pair, in the enumeration order of
:func:`gap_bridging.candidates.link_tracklets` (ENDs sorted by (frame,
tracklet_id), then STARTs likewise), with the same R_max prefilter. The
features are the quantities the hand cost is built from, kept separate and
unweighted, plus tracklet-level context (lengths, cue sample sizes, rival
counts). Rival counts use the pair's own ``dt`` as the window, so every value
is identical at every horizon that enumerates the pair; this is what lets the
trainer deduplicate pairs across horizons.

Interface stub written by the orchestrator; WP-F fills the functions.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pyarrow as pa

from gap_bridging.config import BridgeConfig
from gap_bridging.types import Endpoint

# ============================================================
# CONSTANTS
# ============================================================

#: Feature columns in matrix order. All float64; NaN where unknown.
FEATURE_NAMES: Tuple[str, ...] = (
    'gap',                 # dt - 1 missing frames
    'dist',                # |pos_b - pos_a| px
    'err',                 # prediction error (both-sided, _pair_geometry) px
    'radius',              # extended radius R = r(dt) + speed * min(dt, h) px
    'dist_over_radius',    # dist / R
    'err_over_radius',     # err / R
    'speed_a', 'speed_b',  # |vel| px/frame (0 when unknown)
    'vel_known_a', 'vel_known_b',   # 1.0 / 0.0
    'shoulder',            # min(|du|, cap) / cap, shoulder-vector term in [0, 1]
    'heading_change',      # |wrap(h_b - h_a)| rad; NaN if a heading is NaN
    'size_term',           # hand size term; NaN if a side has size_n == 0
    'size_diff',           # |L_a - L_b| px; NaN likewise
    'side_product',        # claw_side_a * claw_side_b; NaN if a side has claw_side_n == 0
    'side_term',           # max(0, -side_product); NaN likewise
    'abs_side_a', 'abs_side_b',     # |claw_side|; NaN when unknown
    'size_n_a', 'size_n_b',
    'claw_side_n_a', 'claw_side_n_b',
    'n_frames_a', 'n_frames_b',
    'n_rival_starts',      # other motion-gated pairs from the same END with dt' <= dt
    'n_rival_ends',        # other motion-gated pairs into the same START with dt' <= dt
    'axis_flip',           # 1.0 if _axis_flip (Shoulder axis reversed, claw fixed), else 0.0
    'hand_cost',           # the sqrt hand cost (finite when both motion gates pass)
)

#: Monotonic constraint per feature for the scorer: -1 means P(true link) is
#: non-increasing in the feature; 0 means unconstrained.
MONOTONIC: dict = {name: -1 for name in ('gap', 'dist', 'err', 'dist_over_radius',
                                          'err_over_radius', 'shoulder', 'size_term',
                                          'side_term', 'hand_cost')}

#: Key columns identifying the pair (prefix of :data:`PAIR_FEATURES_SCHEMA`).
KEY_NAMES: Tuple[str, ...] = ('video_id', 'tracklet_a', 'tracklet_b', 'gap_frames', 'gate')

PAIR_FEATURES_SCHEMA = pa.schema(
    [('video_id', pa.string()), ('tracklet_a', pa.string()), ('tracklet_b', pa.string()),
     ('gap_frames', pa.int64()), ('gate', pa.string())]
    + [(name, pa.float64()) for name in FEATURE_NAMES])

PAIR_FEATURES_NAME = 'pair_features.parquet'


# ============================================================
# FUNCTIONS (WP-F)
# ============================================================


def pair_features(ends: List[Endpoint], starts: List[Endpoint], rows: np.ndarray,
                  cols: np.ndarray, cfg: BridgeConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Feature matrix and first failing motion gate for candidate pairs.

    Parameters
    ----------
    ends, starts : list of Endpoint
        Sorted as in ``link_tracklets`` (``candidates._sort_key``).
    rows, cols : np.ndarray
        Pair indices into ``ends`` and ``starts`` (``candidates._enumerate_pairs``).
    cfg : BridgeConfig
        Gate and cue parameters; evaluated with ``cost_model='hand'``.

    Returns
    -------
    X : np.ndarray
        ``(n, len(FEATURE_NAMES))`` float64 in :data:`FEATURE_NAMES` order.
    gate : np.ndarray
        ``(n,)`` object: first failing gate per pair ('' when none fails;
        'displacement', 'prediction_error', 'claw_side' or 'max_cost').
    """
    raise NotImplementedError("WP-F")


def pair_feature_table(ends: List[Endpoint], starts: List[Endpoint], cfg: BridgeConfig,
                       video_id: str) -> pa.Table:
    """Enumerate the video's candidate pairs exactly as ``link_tracklets`` and
    return one :data:`PAIR_FEATURES_SCHEMA` row per pair, in enumeration order."""
    raise NotImplementedError("WP-F")


def feature_matrix(table: pa.Table, names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
    """``(n, len(names))`` float64 matrix of the named columns of a features table."""
    raise NotImplementedError("WP-F")
