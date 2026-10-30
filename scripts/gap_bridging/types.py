"""Contract types shared by the gap-bridging modules.

Units: pixels, px/frame, int64 frames, float64 arrays. For a link from
tracklet A (ending at frame e) to tracklet B (starting at frame s),
``dt = s - e`` and the number of missing frames is ``gap = dt - 1``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NODES = ('shell', 'shoulder', 'elbow', 'claw', 'arm')
SHELL, SHOULDER = 0, 1

DECISIONS = ('linked', 'refused_margin', 'refused_gate', 'refused_assignment')
SOURCES = ('observed', 'recovered', 'interpolated')


@dataclass(frozen=True)
class Endpoint:
    """One tracklet START or END, with the local motion state at that frame."""
    tracklet_id: str
    video_id: str
    frame: int
    n_frames: int
    pos: np.ndarray        # (2,) shell position at `frame`
    vel: np.ndarray        # (2,) px/frame, speed clipped; zeros when n_frames < 2
    vel_known: bool        # n_frames >= 2
    heading: float         # atan2(shoulder - shell) at the endpoint (nearest valid frame in window); NaN if none
    offsets: np.ndarray    # (5,2) body-frame node offsets rotated by -heading; NaN rows unknown; shell row zeros


@dataclass(frozen=True)
class LinkDecision:
    """One row per enumerated candidate pair (same video, 1 <= gap <= max_gap)."""
    tracklet_a: str
    tracklet_b: str
    gap_frames: int
    cost: float            # NaN when a gate failed
    runner_up_cost: float  # NaN when no competitor
    decision: str          # one of DECISIONS
    reason: str
    n_chain_ignored: int = 0  # chain-reachable competitors excluded from the runner-up (linked/refused_margin)


@dataclass(frozen=True)
class Bridge:
    """The filled frames between a linked END (a) and START (b)."""
    video_id: str
    a: Endpoint
    b: Endpoint
    trajectory_id: str
    bridge_id: str
    frames: np.ndarray     # (k,) int64 missing frames e+1 .. s-1
    positions: np.ndarray  # (k,2) shell path
    headings: np.ndarray   # (k,) radians or NaN
    nodes: np.ndarray      # (k,5,2) node positions, NaN where unknown; nodes[:,0] == positions
    fill_confidence: str | None = None  # 'high'/'medium'/'low' from len(frames); set by fill_bridge
