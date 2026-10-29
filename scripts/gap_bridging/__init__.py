"""Post-processing that bridges short gaps between body-first tracklets.

Links tracklet pairs separated by 1..max_gap missing frames with a gated,
one-to-one assignment, then fills the missing frames with a previously
rejected real detection ("recovered") or a synthesized pose ("interpolated").
Inputs are never modified; every linking decision is written to an audit table.
"""
from __future__ import annotations

from gap_bridging.config import BridgeConfig, load_config
from gap_bridging.types import NODES, Bridge, Endpoint, LinkDecision

__all__ = ["BridgeConfig", "load_config", "NODES", "Bridge", "Endpoint", "LinkDecision"]
