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
Since v005 a planting profile (:class:`HalfProfile`) sets how much of each
side is kept: ``'standard'`` (the default, bit-identical to v004) keeps
everything from the tracklet start to the gap and from the gap to the
tracklet end, with at least ``margin`` frames per side; ``'short'`` draws a
category per plant (``both_short``, ``one_short``, ``short_mid``, ``mid``,
``both_long``, :data:`CATEGORY_PROBABILITIES`) and trims each half to the
drawn length (1 to 3 frames, 4 to 9 frames, or everything), deleting the
trimmed frames, so the
planted halves match the side lengths of natural links (DEFECTS.md D-009).
``link_side_lengths`` and ``link_mix`` measure that mix on a link table.
``score_links`` compares a ``links.parquet`` audit table with the planted
truth: per gap length, how many planted pairs were re-linked (recovered), how
many planted halves were linked to something else (false_link) and how many
were left alone (missed), plus the decoy and flicker outcomes.
``run_holdout`` plants, runs :func:`gap_bridging.pipeline.run` and writes the
scores. ``pool_sweeps`` sums the per-gap counts of several sweeps (seeds) and
``apply_rule`` picks the highest horizon whose pooled precision meets a floor
at every planted gap (``bridge_gaps.py pool``).

All table work is pyarrow; rows and columns not touched by planting are kept
byte-identical.

Sam
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ----- numerics -----
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from scipy.stats import beta

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
    ('profile', pa.string()),       # v005: planting profile name (one of PROFILES)
    ('category', pa.string()),      # v005: short-profile category, else null
    ('left_frames', pa.int64()),    # v005: accepted frames kept in left_id
    ('right_frames', pa.int64()),   # v005: accepted frames kept in right_id (flicker: outer part)
])
TRUTH_V004_NAMES = tuple(TRUTH_SCHEMA.names[:9])     # the columns written before v005
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
POOL_COUNTS = ('planted', 'recovered', 'false_link', 'missed', 'decoy_planted',
               'decoy_recovered', 'decoy_false_link', 'flicker_planted', 'flicker_recovered',
               'flicker_spurious')
POOL_LEVEL = 0.95             # one-sided confidence of the exact precision lower bound

# ----- v005 planting profiles (DEFECTS.md D-009) -----
PROFILES = ('standard', 'short')
CATEGORIES = ('both_short', 'one_short', 'short_mid', 'mid', 'both_long')
SHORT_LENGTHS = (1, 2, 3)
SHORT_LENGTH_WEIGHTS = (0.58, 0.27, 0.15)      # demo links, horizon 30: 572 / 264 / 151 sides
MID_LENGTHS = (4, 5, 6, 7, 8, 9)               # drawn uniformly
CATEGORY_PROBABILITIES = (0.40, 0.18, 0.21, 0.13, 0.08)   # demo links, horizon 30 (link_mix):
# 0.401 / 0.177 / 0.214 / 0.127 / 0.081 of 827 links
PROBABILITY_TOL = 1e-9

# ============================================================
# PLANTING PROFILES
# ============================================================


def _check_lengths(name: str, lengths: tuple) -> None:
    """Raise ValueError unless ``lengths`` are integers >= 1, strictly ascending."""
    if not lengths or any(int(v) != v or v < 1 for v in lengths) \
            or any(b <= a for a, b in zip(lengths, lengths[1:])):
        raise ValueError(f"{name} must be integers >= 1, strictly ascending; got {lengths}")


def _check_distribution(name: str, values: tuple, size: int) -> None:
    """Raise ValueError unless ``values`` are ``size`` finite non-negative numbers summing to 1."""
    if len(values) != size or any(not math.isfinite(v) or v < 0 for v in values) \
            or abs(math.fsum(values) - 1.0) > PROBABILITY_TOL:
        raise ValueError(f"{name} must be {size} non-negative numbers summing to 1; got {values}")


@dataclass(frozen=True)
class HalfProfile:
    """How much of each side of a planted gap is kept (v005, DEFECTS.md D-009).

    Under ``'standard'`` the v004 draw runs unchanged and every other field is
    ignored by :func:`plant_gaps`. Under ``'short'`` each plant draws a
    category (:func:`draw_half_lengths`) and keeps exactly the drawn number of
    frames on a fixed side, or everything on a keep-all side; the frames
    beyond a fixed side are deleted (:func:`_plant_window`).

    Attributes
    ----------
    name : str
        One of :data:`PROFILES`.
    min_tracklet_len : int
        Minimum accepted frames of an eligible tracklet (standard 40, short
        10); under ``'short'`` it replaces ``plant_gaps``'s argument.
    probabilities : tuple of float
        Probability of each of :data:`CATEGORIES`.
    short_lengths, short_weights : tuple
        Lengths of a short side and their probabilities.
    mid_lengths : tuple of int
        Lengths of the fixed side of a ``mid`` plant and of the longer side
        of a ``short_mid`` plant (drawn uniformly); all greater than every
        short length.
    long_min : int
        Minimum frames of a keep-all side of ``both_long`` and of the long
        side of ``one_short``.
    mid_min : int
        Minimum frames of the keep-all side of a ``mid`` plant.
    trim_decoys : bool
        Draw decoy targets like plants (trimmed halves); False keeps the
        standard ``margin`` window for them. Donors are never trimmed.
    """

    name: str
    min_tracklet_len: int
    probabilities: tuple = CATEGORY_PROBABILITIES
    short_lengths: tuple = SHORT_LENGTHS
    short_weights: tuple = SHORT_LENGTH_WEIGHTS
    mid_lengths: tuple = MID_LENGTHS
    long_min: int = 10
    mid_min: int = 4
    trim_decoys: bool = True

    def __post_init__(self):
        for field in ('probabilities', 'short_lengths', 'short_weights', 'mid_lengths'):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if self.name not in PROFILES:
            raise ValueError(f"unknown profile {self.name!r}; expected one of {PROFILES}")
        if int(self.min_tracklet_len) != self.min_tracklet_len or self.min_tracklet_len < 1:
            raise ValueError(f"min_tracklet_len must be an integer >= 1, got "
                             f"{self.min_tracklet_len}")
        _check_distribution('probabilities', self.probabilities, len(CATEGORIES))
        _check_lengths('short_lengths', self.short_lengths)
        _check_lengths('mid_lengths', self.mid_lengths)
        _check_distribution('short_weights', self.short_weights, len(self.short_lengths))
        if self.short_lengths[-1] >= self.mid_lengths[0]:
            raise ValueError("every mid length must exceed every short length")
        if not (int(self.long_min) == self.long_min and int(self.mid_min) == self.mid_min
                and self.long_min >= self.mid_min >= 1):
            raise ValueError(f"need integers long_min >= mid_min >= 1, got {self.long_min}, "
                             f"{self.mid_min}")

    def to_dict(self) -> dict:
        """JSON-ready dictionary (tuples as lists)."""
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_name(cls, name: str) -> 'HalfProfile':
        """:data:`STANDARD_PROFILE` or :data:`SHORT_PROFILE` by name; ValueError otherwise."""
        profiles = {'standard': STANDARD_PROFILE, 'short': SHORT_PROFILE}
        if name not in profiles:
            raise ValueError(f"unknown profile {name!r}; expected one of {PROFILES}")
        return profiles[name]


STANDARD_PROFILE = HalfProfile('standard', min_tracklet_len=40)
SHORT_PROFILE = HalfProfile('short', min_tracklet_len=10)


def _as_profile(profile: 'HalfProfile | str') -> HalfProfile:
    """A :class:`HalfProfile` from a profile or its name."""
    if isinstance(profile, HalfProfile):
        return profile
    if isinstance(profile, str):
        return HalfProfile.from_name(profile)
    raise TypeError(f"profile must be a HalfProfile or a name, got {type(profile).__name__}")


def draw_half_lengths(rng: np.random.Generator,
                      profile: HalfProfile) -> Tuple[str, Optional[int], Optional[int]]:
    """Draw a category and the frames to keep on each side of one planted gap.

    Parameters
    ----------
    rng : np.random.Generator
        Generator; one category draw, then the category's own draws.
    profile : HalfProfile
        Category probabilities and side lengths.

    Returns
    -------
    category : str
        One of :data:`CATEGORIES`.
    kept_left, kept_right : int or None
        Frames to keep before / after the gap; None keeps every frame on that
        side (at least ``profile.long_min``, or ``profile.mid_min`` for the
        keep-all side of ``mid``). ``both_short``: two short draws (left,
        right); ``one_short``: one short draw on side ``rng.integers(2)`` (0:
        left); ``short_mid``: one short draw and one uniform mid draw, the
        short one on side ``rng.integers(2)``, both sides fixed (no
        keep-all side); ``mid``: one uniform mid draw on side
        ``rng.integers(2)``; ``both_long``: ``(None, None)``.
    """
    category = CATEGORIES[int(rng.choice(len(CATEGORIES), p=np.asarray(profile.probabilities)))]
    weights = np.asarray(profile.short_weights)

    def short() -> int:
        return int(profile.short_lengths[int(rng.choice(len(profile.short_lengths), p=weights))])

    if category == 'both_short':
        return category, short(), short()
    if category == 'both_long':
        return category, None, None
    if category == 'short_mid':
        kept_short = short()
        kept_mid = int(profile.mid_lengths[int(rng.integers(len(profile.mid_lengths)))])
        return (category, kept_short, kept_mid) if int(rng.integers(2)) == 0 \
            else (category, kept_mid, kept_short)
    if category == 'one_short':
        kept = short()
    else:
        kept = int(profile.mid_lengths[int(rng.integers(len(profile.mid_lengths)))])
    return (category, kept, None) if int(rng.integers(2)) == 0 else (category, None, kept)


def _keep_all_min(other: Optional[int], profile: HalfProfile) -> int:
    """Minimum frames of a keep-all side whose other side keeps ``other`` frames.

    ``mid_min`` opposite a mid length, else ``long_min`` (opposite a short
    length or another keep-all side).
    """
    return profile.mid_min if other is not None and other in profile.mid_lengths \
        else profile.long_min


def _side_needs(kept_left: Optional[int], kept_right: Optional[int],
                profile: HalfProfile) -> Tuple[int, int]:
    """Minimum frames on the left and right of the gap for one draw."""
    return (kept_left if kept_left is not None else _keep_all_min(kept_right, profile),
            kept_right if kept_right is not None else _keep_all_min(kept_left, profile))


def _window_need(gap: int, kept_left: Optional[int], kept_right: Optional[int],
                 profile: HalfProfile) -> int:
    """Accepted frames a tracklet needs for one draw under a trimming profile."""
    left, right = _side_needs(kept_left, kept_right, profile)
    return max(profile.min_tracklet_len, gap + left + right)


def _plant_window(first: int, n: int, gap: int, kept_left: Optional[int],
                  kept_right: Optional[int], profile: HalfProfile,
                  rng: np.random.Generator) -> Tuple[int, int, int]:
    """Place a gap of ``gap`` frames in a tracklet and trim its halves.

    Parameters
    ----------
    first, n : int
        First frame and number of (consecutive) frames of the tracklet.
    gap : int
        Deleted run length.
    kept_left, kept_right : int or None
        Frames kept before / after the run (None: every frame on that side).
    profile : HalfProfile
        Keep-all minima (:func:`_keep_all_min`).
    rng : np.random.Generator
        One ``integers`` draw.

    Returns
    -------
    first_del : int
        First deleted frame of the run, ``first + o`` with ``o`` uniform over
        the admissible offsets ``[L, n - gap - R]``, where ``L`` / ``R`` is the
        fixed number of kept frames of that side or its keep-all minimum.
    trim_before : int
        Frames deleted at the tracklet start (``o - kept_left``; 0 when the
        left side keeps all).
    trim_after : int
        Frames deleted at the tracklet end (``n - o - gap - kept_right``; 0
        when the right side keeps all).

    Raises
    ------
    ValueError
        If the tracklet is too short for the draw.
    """
    left, right = _side_needs(kept_left, kept_right, profile)
    if n - gap - right < left:
        raise ValueError(f"a {n}-frame tracklet cannot hold gap {gap} with {left} + {right} "
                         "kept frames")
    o = int(rng.integers(left, n - gap - right + 1))
    trim_before = o - kept_left if kept_left is not None else 0
    trim_after = n - o - gap - kept_right if kept_right is not None else 0
    if min(trim_before, trim_after) < 0:
        raise RuntimeError("negative trim")
    return first + o, int(trim_before), int(trim_after)


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
Operation = Tuple[str, int, int, List[Tuple[int, int]], int, Optional[str], Tuple[int, int]]
# (role, code, gap, deleted (first, last) runs, partner operation index or -1,
#  short-profile category or None, (trim_before, trim_after) frames deleted at the
#  tracklet start / end)
Window = Callable[[int, int], Tuple[int, int, int]]
# (first, n) -> (first_del, trim_before, trim_after) of a decoy target


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
            cues: _Cues | None = None, cfg: BridgeConfig | None = None,
            profile: HalfProfile = STANDARD_PROFILE) -> List[Operation]:
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

    Under ``profile.name == 'standard'`` the v004 draw runs unchanged
    (category None, trims ``(0, 0)``). Under ``'short'`` see
    :func:`_choose_short`.
    """
    if profile.name != 'standard':
        return _choose_short(spans, ids, gap_lengths, per_gap, margin, seed, decoys=decoys,
                             flicker=flicker, rows_of=rows_of, frames=frames, videos=videos,
                             cues=cues, cfg=cfg, profile=profile)
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
            ops.append(('plant', code, int(gap), [(first_del, first_del + gap - 1)], -1,
                        None, (0, 0)))
        taken = set(picked)
        remaining = [code for code in remaining if code not in taken]

        for _ in range(decoys):
            pair = _choose_decoy(eligible(need), remaining, spans, ids, gap, margin, rng,
                                 rows_of, frames, videos, cues, cfg)
            if pair is None:      # best effort: fewer decoys for this gap (see truth/scores)
                break
            target, donor, first_del, trims = pair
            run = [(first_del, first_del + gap - 1)]
            ops.append(('decoy_target', target, int(gap), run, len(ops) + 1, None, trims))
            ops.append(('decoy_donor', donor, int(gap), run, len(ops) - 1, None, (0, 0)))
            remaining = [code for code in remaining if code not in (target, donor)]

        if flicker:
            remaining = _choose_flicker(ops, remaining, eligible, offset, rng, gap, flicker,
                                        min_tracklet_len, margin)
    return _check_unique(ops)


def _check_unique(ops: List[Operation]) -> List[Operation]:
    """``ops``, after checking that no tracklet is planted twice."""
    codes = [op[1] for op in ops]
    if len(set(codes)) != len(codes):
        raise RuntimeError("a tracklet was planted twice")
    return ops


def _choose_flicker(ops: List[Operation], remaining: List[int],
                    eligible: Callable[[int], List[int]], offset: Callable[[int, int], int],
                    rng: np.random.Generator, gap: int, flicker: int, min_tracklet_len: int,
                    margin: int) -> List[int]:
    """Append up to ``flicker`` flicker singletons of gap ``gap`` to ``ops``; new ``remaining``.

    The v004 draw (same generator calls) under both profiles; flicker parts
    are never trimmed.
    """
    span = 2 * gap + 1
    need_f = max(min_tracklet_len, span + 2 * margin)
    pool = eligible(need_f)       # best effort: fewer when the pool is short
    picked = [pool[i] for i in rng.choice(len(pool), size=min(flicker, len(pool)),
                                          replace=False)]
    for code in picked:
        first_del = offset(code, span)
        kept = first_del + gap
        ops.append(('flicker', code, int(gap),
                    [(first_del, kept - 1), (kept + 1, kept + gap)], -1, None, (0, 0)))
    taken = set(picked)
    return [code for code in remaining if code not in taken]


def _choose_short(spans: Dict[int, Tuple[int, int]], ids: List[str], gap_lengths: Sequence[int],
                  per_gap: int, margin: int, seed: int, *, decoys: int, flicker: int,
                  rows_of: Dict[int, np.ndarray] | None, frames: np.ndarray | None,
                  videos: Dict[int, str] | None, cues: _Cues | None,
                  cfg: BridgeConfig | None, profile: HalfProfile) -> List[Operation]:
    """The draw of :func:`_choose` under a trimming profile (``'short'``).

    Per gap length (in the given order) and per plant: draw ``(category,
    kept_left, kept_right)`` (:func:`draw_half_lengths`), take the unused
    tracklets with at least :func:`_window_need` frames (ValueError when
    none: plants are mandatory), pick one with ``rng.integers`` and place the
    window (:func:`_plant_window`). Then ``decoys`` attempts, each drawing its
    own ``(category, kept_left, kept_right)`` before scanning targets when
    ``profile.trim_decoys`` (else the standard ``margin`` window over
    tracklets of at least ``max(profile.min_tracklet_len, g + 2 margin)``
    frames), then the standard flicker draw with ``profile.min_tracklet_len``.
    Eligible tracklets are sorted by id; the shorter gap lengths never depend
    on the longer ones.
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
        for _ in range(per_gap):
            category, kept_left, kept_right = draw_half_lengths(rng, profile)
            need = _window_need(gap, kept_left, kept_right, profile)
            pool = eligible(need)
            if not pool:
                raise ValueError(f"no eligible tracklet (>= {need} frames, unused) for a "
                                 f"{category} plant of gap {gap} under profile "
                                 f"{profile.name!r}")
            code = pool[int(rng.integers(len(pool)))]
            first_del, trim_before, trim_after = _plant_window(
                *spans[code], gap, kept_left, kept_right, profile, rng)
            ops.append(('plant', code, int(gap), [(first_del, first_del + gap - 1)], -1,
                        category, (trim_before, trim_after)))
            remaining = [c for c in remaining if c != code]

        for _ in range(decoys):
            if profile.trim_decoys:
                category, kept_left, kept_right = draw_half_lengths(rng, profile)
                need = _window_need(gap, kept_left, kept_right, profile)

                def window(first: int, n: int, _kl=kept_left, _kr=kept_right,
                           _gap=gap) -> Tuple[int, int, int]:
                    return _plant_window(first, n, _gap, _kl, _kr, profile, rng)
            else:
                category, window = None, None
                need = max(profile.min_tracklet_len, gap + 2 * margin)
            pair = _choose_decoy(eligible(need), remaining, spans, ids, gap, margin, rng,
                                 rows_of, frames, videos, cues, cfg, window=window)
            if pair is None:      # best effort: fewer decoys for this gap (see truth/scores)
                break
            target, donor, first_del, trims = pair
            run = [(first_del, first_del + gap - 1)]
            ops.append(('decoy_target', target, int(gap), run, len(ops) + 1, category, trims))
            ops.append(('decoy_donor', donor, int(gap), run, len(ops) - 1, None, (0, 0)))
            remaining = [code for code in remaining if code not in (target, donor)]

        if flicker:
            remaining = _choose_flicker(ops, remaining, eligible, offset, rng, gap, flicker,
                                        profile.min_tracklet_len, margin)
    return _check_unique(ops)


def _choose_decoy(targets: List[int], remaining: List[int], spans: Dict[int, Tuple[int, int]],
                  ids: List[str], gap: int, margin: int, rng: np.random.Generator,
                  rows_of: Dict[int, np.ndarray], frames: np.ndarray, videos: Dict[int, str],
                  cues: _Cues, cfg: BridgeConfig, window: Window | None = None
                  ) -> Tuple[int, int, int, Tuple[int, int]] | None:
    """One ``(target, donor, deleted_first_frame, (trim_before, trim_after))``, or None.

    Targets are tried in a random order. For a target C the deleted run
    starts at a uniform offset as for a plant (``window(first, n)`` when
    given, which also trims C's halves; C's left cue then covers the trimmed
    left half only; else the v004 ``margin`` offset); donors are the unused
    tracklets of the same video covering ``margin`` frames on each side of
    the run, whose right half's cues differ (:func:`cues_differ`) from C's
    left half; one is drawn uniformly, from the donors that are not eligible
    targets themselves when there are any.
    """
    eligible_set = set(targets)
    for t in rng.permutation(len(targets)):
        target = targets[int(t)]
        first, n = spans[target]
        if window is None:
            first_del = first + int(rng.integers(margin, n - gap - margin + 1))
            trims = (0, 0)
        else:
            first_del, *trims = window(first, n)
            trims = tuple(trims)
        lo, hi = first_del - margin, first_del + gap - 1 + margin
        rows_c = rows_of[target]
        left_c = cues.of(rows_c[(frames[rows_c] < first_del)
                                & (frames[rows_c] >= first + trims[0])])
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
            return target, donor, first_del, trims
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
               cue_config: BridgeConfig | None = None,
               profile: 'HalfProfile | str' = 'standard') -> Tuple[pa.Table, pa.Table]:
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
        its accepted frames must be consecutive and unique. Ignored under a
        ``'short'`` profile, whose ``min_tracklet_len`` applies to plants,
        decoy targets and flicker singletons instead.
    margin : int
        Minimum number of frames kept on each side of the deleted run (>= 1);
        the run starts at a uniform offset in ``[margin, len - g - margin]``.
        Under ``'short'`` it only sets the donor coverage of decoys, the
        flicker window and the window of untrimmed decoy targets
        (``trim_decoys=False``).
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
    profile : HalfProfile or str
        Planting profile (:class:`HalfProfile`, or ``'standard'`` /
        ``'short'``). ``'standard'`` reproduces v004 exactly (every other
        profile field is ignored). Under ``'short'`` each plant draws a
        category and side lengths (:func:`draw_half_lengths`), takes one
        unused tracklet with enough frames uniformly and places the run at a
        uniform admissible offset (:func:`_plant_window`); the frames beyond
        a fixed-length side are deleted, not kept as tracklets. Decoy targets
        are drawn the same way when ``trim_decoys`` (donors never are);
        flicker singletons keep the standard window.

    Returns
    -------
    planted : pa.Table
        ``obs`` without the deleted rows. The first part of a split tracklet
        keeps its id; every later part (the n-th new tracklet, n from 1, in
        truth order) gets ``tracklet_id = f"{old}/holdout{n:03d}"``,
        ``tracklet_local_id = max(tracklet_local_id) + n`` and its first row
        ``incoming_reason = 'no_candidate'``; so does the first kept row of a
        left half trimmed at its start (``'short'``). Decoy donor right halves
        are translated as above. Every other value is unchanged.
    truth : pa.Table
        One row per planted tracklet (:data:`TRUTH_SCHEMA`); ``left_id ==
        original_id``; ``role`` is ``plant``, ``decoy_target``,
        ``decoy_donor`` or ``flicker``. For a flicker ``right_id`` is the
        outer right part, ``middle_id`` the kept frame and the deleted frames
        are ``deleted_first_frame .. deleted_last_frame`` minus the middle.
        ``deleted_*_frame`` always bound the gap run (never the trims).
        ``profile`` is the profile name, ``category`` the drawn category of a
        ``'short'`` plant or trimmed decoy target (null otherwise), and
        ``left_frames`` / ``right_frames`` the accepted frames kept in
        ``left_id`` / ``right_id`` (flicker: left part / outer right part),
        under both profiles.

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
    profile = _as_profile(profile)
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
                  margin, seed, decoys=decoys, flicker=flicker, profile=profile, **extra)
    local_max = pc.max(obs['tracklet_local_id']).as_py() or 0
    keep = np.ones(obs.num_rows, dtype=bool)
    new_id = np.full(obs.num_rows, -1, dtype=np.int64)    # new-tracklet number per row
    heads: List[int] = []
    new_names: List[str] = []
    parts: List[List[str]] = []        # per operation: ids of its parts in frame order
    rows_by_op: List[np.ndarray] = []
    n_trimmed_heads = 0
    for role, code, gap, runs, _, _, (trim_before, trim_after) in ops:
        rows = np.flatnonzero(codes == code)
        old = ids[code]
        names = [old]
        first, n = spans[code]
        trims = [(first, first + trim_before - 1)] if trim_before else []
        if trim_after:
            trims.append((first + n - trim_after, first + n - 1))
        for first_del, last_del in list(runs) + trims:
            keep[rows[(frames[rows] >= first_del) & (frames[rows] <= last_del)]] = False
        if trim_before:      # the trimmed left half starts like a new tracklet
            heads.extend(rows[(frames[rows] == first + trim_before) & accepted[rows]].tolist())
            n_trimmed_heads += 1
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
    n_deleted = sum(last - first + 1 for op in ops for first, last in op[3]) \
        + sum(op[6][0] + op[6][1] for op in ops)
    if int((~keep).sum()) != n_deleted or len(heads) != len(new_names) + n_trimmed_heads:
        raise RuntimeError("planted rows do not match the requested gaps")

    truth: Dict[str, list] = {name: [] for name in TRUTH_SCHEMA.names}
    for (role, code, gap, runs, partner, category, trims), names in zip(ops, parts):
        first, n = spans[code]
        left_frames = runs[0][0] - first - trims[0]
        right_frames = first + n - 1 - runs[-1][1] - trims[1]
        if min(left_frames, right_frames) < 1:
            raise RuntimeError("a planted half has no frame")
        if role == 'decoy_target':
            partner_id = parts[partner][-1]
        elif role == 'decoy_donor':
            partner_id = parts[partner][0]
        else:
            partner_id = None
        values = (names[0], names[-1], ids[code], gap, runs[0][0], runs[-1][1], role,
                  names[1] if role == 'flicker' else None, partner_id, profile.name,
                  category, int(left_frames), int(right_frames))
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
    for k, (role, _, gap, runs, partner, _, _) in enumerate(ops):
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
# NATURAL LINK MIX (v005)
# ============================================================


def link_side_lengths(obs: pa.Table, links: pa.Table) -> Tuple[np.ndarray, np.ndarray]:
    """Accepted frames of both sides of every linked pair.

    Parameters
    ----------
    obs : pa.Table
        Observation table holding the linked tracklets (the bridging input or
        its output) with ``tracklet_id`` and ``accepted_for_tracking``.
    links : pa.Table
        ``links.parquet`` (``tracklet_a``, ``tracklet_b``, ``decision``).

    Returns
    -------
    n_a, n_b : np.ndarray
        ``(n_linked,)`` int64 accepted frames of ``tracklet_a`` / ``tracklet_b``
        over the ``decision == 'linked'`` rows, in row order; 0 for a tracklet
        absent from ``obs``.
    """
    for name in ('tracklet_a', 'tracklet_b', 'decision'):
        if name not in links.schema.names:
            raise ValueError(f"links table has no column {name!r}")
    for name in ('tracklet_id', 'accepted_for_tracking'):
        if name not in obs.schema.names:
            raise ValueError(f"observation table has no column {name!r}")
    accepted = obs.filter(pc.and_(pc.fill_null(obs['accepted_for_tracking'], False),
                                  pc.is_valid(obs['tracklet_id'])))
    grouped = accepted.group_by('tracklet_id').aggregate([('tracklet_id', 'count')])
    counts = dict(zip(grouped['tracklet_id'].to_pylist(),
                      grouped['tracklet_id_count'].to_pylist()))
    linked = links.filter(pc.fill_null(pc.equal(links['decision'], 'linked'), False))
    sides = [np.asarray([counts.get(t, 0) for t in linked[name].to_pylist()], dtype=np.int64)
             for name in ('tracklet_a', 'tracklet_b')]
    return sides[0], sides[1]


def link_mix(n_a: np.ndarray, n_b: np.ndarray, short_max: int = 3,
             long_min: int = 10) -> dict:
    """Fractions of links per side-length category (the :data:`CATEGORIES` mix).

    With ``lo = min(n_a, n_b) >= 1`` and ``hi = max(n_a, n_b)`` per link:
    ``both_short`` ``hi <= short_max``; ``one_short`` ``lo <= short_max`` and
    ``hi >= long_min``; ``short_mid`` ``lo <= short_max`` and ``short_max <
    hi < long_min``; ``mid`` ``short_max < lo < long_min``; ``both_long``
    ``lo >= long_min``. ``other`` is the remainder, links with a side of 0
    frames (a tracklet absent from the observation table), so the five
    categories and ``other`` sum to 1. These are the bins behind
    :data:`CATEGORY_PROBABILITIES` and match the planted categories
    (:func:`draw_half_lengths`) for the default profile lengths.

    Parameters
    ----------
    n_a, n_b : array-like of int
        Side lengths (:func:`link_side_lengths`), equal length, >= 0.
    short_max, long_min : int
        ``1 <= short_max < long_min``.

    Returns
    -------
    dict
        ``both_short``, ``one_short``, ``short_mid``, ``mid``,
        ``both_long``, ``other`` (fractions of ``n``; NaN when ``n == 0``),
        ``short_weights`` (tuple: fraction of each length ``1 .. short_max``
        among all sides of ``1 .. short_max`` frames, NaN when none) and
        ``n`` (number of links).
    """
    a = np.asarray(n_a, dtype=np.int64)
    b = np.asarray(n_b, dtype=np.int64)
    if a.shape != b.shape or a.ndim != 1 or (a.size and min(a.min(), b.min()) < 0):
        raise ValueError("n_a and n_b must be equal-length 1-D arrays of counts >= 0")
    if not 1 <= short_max < long_min:
        raise ValueError(f"need 1 <= short_max < long_min, got {short_max}, {long_min}")
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    known = lo >= 1
    short = known & (lo <= short_max)
    masks = {'both_short': short & (hi <= short_max),
             'one_short': short & (hi >= long_min),
             'short_mid': short & (hi > short_max) & (hi < long_min),
             'mid': (lo > short_max) & (lo < long_min),
             'both_long': lo >= long_min,
             'other': ~known}
    n = int(a.size)
    if int(np.sum([m.astype(np.int64) for m in masks.values()], axis=0).max(initial=1)) != 1 \
            or sum(int(m.sum()) for m in masks.values()) != n:
        raise RuntimeError("link categories do not partition the links")
    sides = np.concatenate([a, b])
    short_sides = sides[(sides >= 1) & (sides <= short_max)]
    weights = tuple(_ratio(int((short_sides == k).sum()), int(short_sides.size))
                    for k in range(1, short_max + 1))
    out = {name: _ratio(int(masks[name].sum()), n) for name in CATEGORIES + ('other',)}
    out.update(short_weights=weights, n=n)
    return out


# ============================================================
# RUN
# ============================================================


def _json_safe(row: dict) -> dict:
    """Replace NaN floats by None so the JSON stays standard."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}


def run_holdout(observations_path: Path, out_dir: Path, cfg: BridgeConfig,
                gap_lengths: Sequence[int], per_gap: int, min_tracklet_len: int = 40,
                seed: int = 0, force: bool = False, margin: int = 3, decoys: int = 0,
                flicker: int = 0, decoy_offset_px: float = 3.0,
                export_features: bool = False,
                profile: 'HalfProfile | str' = 'standard') -> dict:
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
    profile : HalfProfile or str
        Planting profile, passed to :func:`plant_gaps` (v005; default
        ``'standard'``).
    force : bool
        Overwrite existing outputs (also passed to :func:`pipeline.run`).
    export_features : bool
        Also write ``bridged/pair_features.parquet`` (v004 training input)
        when ``cfg.gate_model == 'sqrt'``; ignored under the linear gate
        model (``features.pair_feature_table`` is sqrt-only). The ``config``
        recorded in ``holdout_<max_gap>.json`` stays the caller's ``cfg``,
        while the ``bridged/`` marker records the configuration actually run
        (``export_pair_features: true`` when exporting).

    Returns
    -------
    dict
        Content of ``holdout_<max_gap>.json``: ``config``, ``seed``,
        ``gap_lengths``, ``per_gap``, ``min_tracklet_len``, ``margin``,
        ``decoys``, ``flicker``, ``decoy_offset_px``, ``profile``
        (:meth:`HalfProfile.to_dict`), ``input``, ``scores``
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
    profile = _as_profile(profile)
    obs = pipeline.read_observations(pipeline.input_files(observations_path))
    planted, truth = plant_gaps(obs, gap_lengths, per_gap, min_tracklet_len, margin, seed,
                                decoys=decoys, decoy_offset_px=decoy_offset_px,
                                flicker=flicker, cue_config=cfg, profile=profile)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(planted, out_dir / PLANTED_NAME)
    pq.write_table(truth, out_dir / TRUTH_NAME)
    export_features = bool(export_features) and cfg.gate_model == 'sqrt'
    run_cfg = replace(cfg, export_pair_features=True) if export_features else cfg
    pipeline.run(out_dir / PLANTED_NAME, out_dir / BRIDGED_DIR, run_cfg, force)
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
        'profile': profile.to_dict(),
        'input': {'path': str(observations_path), 'rows': obs.num_rows,
                  'planted_rows': planted.num_rows},
        'scores': rows,
        'overall': {'precision': rows[-1]['precision'], 'recall': rows[-1]['recall']},
        'n_unplanted_links': int(n_unplanted),
        'outputs': {'planted': PLANTED_NAME, 'truth': TRUTH_NAME, 'scores': SCORES_NAME,
                    'bridged': BRIDGED_DIR,
                    **({'pair_features': f'{BRIDGED_DIR}/pair_features.parquet'}
                       if export_features else {})},
    }
    if rows[-1]['gap_frames'] != ALL_GAPS or scores.num_rows < 2:
        raise RuntimeError("scores table lacks the per-gap or overall rows")
    (out_dir / json_name).write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    return summary


# ============================================================
# POOLING AND THE HORIZON RULE
# ============================================================


def precision_lower_bound(successes: int, trials: int, level: float = POOL_LEVEL) -> float:
    """One-sided exact (Clopper-Pearson) lower bound of a binomial proportion.

    ``successes`` of ``trials``; NaN when ``trials == 0``, 0 when
    ``successes == 0``. With ``successes == trials == n`` it equals
    ``(1 - level)^(1/n)``.
    """
    if not 0 <= successes <= trials or not 0.0 < level < 1.0:
        raise ValueError(f"need 0 <= successes <= trials and 0 < level < 1, got "
                         f"{successes}, {trials}, {level}")
    if trials == 0:
        return math.nan
    if successes == 0:
        return 0.0
    return float(beta.ppf(1.0 - level, successes, trials - successes + 1))


def read_by_gap(path: Path) -> List[dict]:
    """Rows of a ``holdout_by_gap.csv`` (``bridge_gaps.py sweep``), counts as int."""
    with open(path, newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        missing = [name for name in ('max_gap', 'gap') + POOL_COUNTS if name not in row]
        if missing:
            raise ValueError(f"{path}: columns {missing} missing")
    return [{name: int(row[name]) for name in ('max_gap', 'gap') + POOL_COUNTS} for row in rows]


def pool_sweeps(sweeps: Sequence[Sequence[dict]],
                level: float = POOL_LEVEL) -> Tuple[List[dict], List[dict]]:
    """
    Pool the per-gap hold-out counts of several sweeps (one per seed).

    Parameters
    ----------
    sweeps : sequence of sequence of dict
        Per sweep, the rows of its ``holdout_by_gap.csv`` (:func:`read_by_gap`):
        ``max_gap``, ``gap`` and the :data:`POOL_COUNTS`. Every sweep must hold
        the same ``(max_gap, gap)`` keys, each once.
    level : float
        Confidence of ``precision_lower`` (:func:`precision_lower_bound`).

    Returns
    -------
    by_gap : list of dict
        One row per ``(max_gap, gap)``, ascending: the summed counts, ``links
        = recovered + false_link``, ``recall = recovered / planted``,
        ``precision = recovered / links`` (NaN on a zero denominator) and
        ``precision_lower``. ``planted``, ``recovered`` and ``false_link``
        cover plants and decoy targets, so a decoy target linked to its donor
        (``decoy_false_link``) counts as a false link; a plant whose gap
        exceeds the horizon can only be ``missed`` or ``false_link``.
    by_horizon : list of dict
        One row per ``max_gap``, ascending: the counts summed over gaps, the
        same ratios and bound, and ``n_gaps``.

    Raises
    ------
    ValueError
        If no sweep is given, the sweeps hold different keys, a key repeats,
        or the counts are inconsistent (``recovered + false_link + missed !=
        planted`` or a decoy count exceeds its plant count).
    """
    if not sweeps:
        raise ValueError("no sweeps to pool")
    keys = None
    pooled: Dict[Tuple[int, int], Dict[str, int]] = {}
    for k, rows in enumerate(sweeps):
        own = [(int(r['max_gap']), int(r['gap'])) for r in rows]
        if len(set(own)) != len(own):
            raise ValueError(f"sweep {k} repeats a (max_gap, gap) row")
        if keys is None:
            keys = set(own)
        elif set(own) != keys:
            raise ValueError(f"sweep {k} has (max_gap, gap) rows {sorted(set(own) ^ keys)} "
                             "not shared by the first sweep")
        for key, row in zip(own, rows):
            acc = pooled.setdefault(key, dict.fromkeys(POOL_COUNTS, 0))
            for name in POOL_COUNTS:
                acc[name] += int(row[name])

    def ratios(acc: Dict[str, int]) -> dict:
        if acc['recovered'] + acc['false_link'] + acc['missed'] != acc['planted'] \
                or acc['decoy_false_link'] > acc['false_link'] \
                or acc['decoy_recovered'] > acc['recovered'] \
                or acc['decoy_planted'] > acc['planted']:
            raise ValueError(f"inconsistent hold-out counts {acc}")
        links = acc['recovered'] + acc['false_link']
        return {'links': links, 'recall': _ratio(acc['recovered'], acc['planted']),
                'precision': _ratio(acc['recovered'], links),
                'precision_lower': precision_lower_bound(acc['recovered'], links, level)}

    by_gap = [{'max_gap': h, 'gap': g, **acc, **ratios(acc)}
              for (h, g), acc in sorted(pooled.items())]
    by_horizon = []
    for horizon in sorted({h for h, _ in pooled}):
        rows = [r for r in by_gap if r['max_gap'] == horizon]
        total = {name: sum(r[name] for r in rows) for name in POOL_COUNTS}
        by_horizon.append({'max_gap': horizon, **total, **ratios(total), 'n_gaps': len(rows)})
    return by_gap, by_horizon


def apply_rule(by_gap: Sequence[dict], by_horizon: Sequence[dict],
               rule_precision: float = 1.0) -> Tuple[Optional[int], List[dict]]:
    """
    The highest horizon whose pooled precision is ``>= rule_precision`` at every planted gap.

    A gap with no link (precision NaN) passes; a gap with false links and no
    recovery has precision 0. Plants longer than the horizon count: their
    misses lower recall, and any false link from or to their halves fails the
    gap. The rule therefore needs every horizon to have been scored on the
    same planted gaps (the sweep plants every gap at every horizon).

    Parameters
    ----------
    by_gap, by_horizon : sequence of dict
        Output of :func:`pool_sweeps`.
    rule_precision : float
        Precision floor in ``[0, 1]``.

    Returns
    -------
    chosen : int or None
        The chosen horizon; None when no horizon passes.
    table : list of dict
        ``by_horizon`` rows with ``min_gap_precision`` (NaN when no gap has a
        link), ``rule_pass`` and ``failing_gaps`` (``"gap:precision"`` joined
        by spaces).

    Raises
    ------
    ValueError
        If ``rule_precision`` is outside ``[0, 1]`` or the horizons were scored
        on different planted gaps.
    """
    if not 0.0 <= rule_precision <= 1.0:
        raise ValueError(f"rule_precision must be in [0, 1], got {rule_precision}")
    gaps_of: Dict[int, List[dict]] = {}
    for row in by_gap:
        gaps_of.setdefault(int(row['max_gap']), []).append(row)
    gap_sets = {h: tuple(sorted(int(r['gap']) for r in rows)) for h, rows in gaps_of.items()}
    if len(set(gap_sets.values())) > 1:
        raise ValueError(f"horizons were scored on different planted gaps {gap_sets}; "
                         "re-run the sweep so every gap is planted at every horizon")
    table = []
    for row in by_horizon:
        rows = gaps_of.get(int(row['max_gap']), [])
        if not rows:
            raise ValueError(f"horizon {row['max_gap']} has no per-gap rows")
        scored = [r for r in rows if not math.isnan(r['precision'])]
        failing = [r for r in scored if r['precision'] < rule_precision]
        table.append({**row,
                      'min_gap_precision': min((r['precision'] for r in scored),
                                               default=math.nan),
                      'rule_pass': not failing,
                      'failing_gaps': ' '.join(f"{r['gap']}:{r['precision']:.3f}"
                                               for r in failing)})
    passing = [int(r['max_gap']) for r in table if r['rule_pass']]
    return (max(passing) if passing else None), table
