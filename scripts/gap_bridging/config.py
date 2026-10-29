"""Frozen configuration for gap bridging. Distances are native-image pixels."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

import yaml

DEFAULT_RECOVERABLE = ('low_body_score', 'missing_position', 'outside_image',
                       'invisible', 'duplicate_candidate')


@dataclass(frozen=True)
class BridgeConfig:
    """Parameters for linking and filling. See docs/GAP_BRIDGING.md."""
    max_gap: int = 5                          # link when 1 <= gap <= max_gap; 0 disables linking
    velocity_window: int = 3                  # steps used for endpoint velocity and node offsets
    max_velocity_px_per_frame: float = 6.0    # clip endpoint speeds (tracker value)
    max_body_displacement_px: float = 12.0    # gate: dist <= this * dt
    max_prediction_error_px: float = 18.0     # gate: err <= this * dt
    motion_weight: float = 0.3
    shoulder_vector_weight: float = 0.15
    shoulder_vector_cost_cap_px: float = 12.0
    max_assignment_cost: float = 12.0         # per-frame-equivalent cost
    unmatched_cost: float = 7.0
    minimum_cost_margin: float = 4.0          # veto a link when the runner-up is within this
    recovery_radius_px: float = 8.0
    recoverable_reasons: tuple[str, ...] = DEFAULT_RECOVERABLE
    fps: float = 30000 / 1001

    def __post_init__(self) -> None:
        if self.max_gap < 0 or self.velocity_window < 1:
            raise ValueError("max_gap must be >= 0 and velocity_window >= 1")
        for name in ("max_velocity_px_per_frame", "max_body_displacement_px", "max_prediction_error_px",
                     "shoulder_vector_cost_cap_px", "max_assignment_cost", "unmatched_cost",
                     "minimum_cost_margin", "recovery_radius_px", "fps"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0 <= self.motion_weight <= 1 or self.shoulder_vector_weight < 0:
            raise ValueError("motion_weight must be in [0, 1] and shoulder_vector_weight >= 0")

    def to_dict(self) -> dict:
        return {f.name: (list(getattr(self, f.name)) if isinstance(getattr(self, f.name), tuple)
                         else getattr(self, f.name)) for f in fields(self)}


def load_config(path: Path | None = None, **overrides) -> BridgeConfig:
    """Load a YAML config (all keys optional) and apply keyword overrides."""
    values: dict = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(loaded) - {f.name for f in fields(BridgeConfig)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        values.update(loaded)
    values.update({k: v for k, v in overrides.items() if v is not None})
    if "recoverable_reasons" in values:
        values["recoverable_reasons"] = tuple(values["recoverable_reasons"])
    return replace(BridgeConfig(), **values)
