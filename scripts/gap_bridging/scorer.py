"""Learned link scorer: model file, manifest and the ``Scorer`` protocol (v004).

Interface stub written by the orchestrator; WP-T fills the module.
"""
from __future__ import annotations

from typing import Protocol, Tuple

import numpy as np


class Scorer(Protocol):
    """What ``candidates.link_tracklets`` needs from a scorer."""

    @property
    def threshold(self) -> float:
        """Probability below which a pair is refused (``max_cost`` gate)."""

    @property
    def feature_names(self) -> Tuple[str, ...]:
        """Feature order the model expects (must equal ``features.FEATURE_NAMES``)."""

    def cost(self, X: np.ndarray) -> np.ndarray:
        """``-log p`` per row of ``X`` (``(n, F)`` -> ``(n,)``), finite."""
