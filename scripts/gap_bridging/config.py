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
    # --- gate model (v002) ---------------------------------------------------
    # "sqrt": r(dt) = max(radius_min_px, radius_safety * (radius_r0_px + radius_k_px * sqrt(dt)))
    #         with dt = frames between the linked END and START. Gates and the dimensionless
    #         cost use the extended radius R = r(dt) + max(|v_a|, |v_b|) * min(dt, horizon);
    #         cost *= (1 + gap_penalty) ** (gap - 1), gap = dt - 1.
    # "linear": the v001 behaviour (12 * dt displacement gate, per-frame px costs).
    gate_model: str = "sqrt"
    radius_r0_px: float = 0.00                   # demo fit (nnls, p99 net displacement, dt 1..60)
    radius_k_px: float = 3.73
    radius_safety: float = 1.25               # multiplies the fitted radius
    radius_min_px: float = 6.0                # floor so adjacent-frame jitter is always inside
    prediction_error_ratio: float = 1.5       # prediction gate = ratio * r(dt)
    prediction_horizon_frames: int = 3        # velocity extrapolated for min(dt, horizon) frames; also damps long fills
    gap_penalty: float = 0.06                 # cost *= (1 + gap_penalty) ** (gap - 1); at equal pixel
                                              # distance a 1-frame gap beats a 30-frame gap (5.4 > r(31)/r(2))
    # Thresholds on the sqrt model's dimensionless cost (fractions of the extended radius).
    max_assignment_cost_sqrt: float = 1.0
    unmatched_cost_sqrt: float = 0.6
    minimum_cost_margin_sqrt: float = 0.25
    fill_confidence_gaps: tuple[int, int] = (5, 15)   # gap <= 5 high, <= 15 medium, else low

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
        if self.gate_model not in ("sqrt", "linear"):
            raise ValueError("gate_model must be 'sqrt' or 'linear'")
        if (self.radius_r0_px < 0 or self.radius_k_px < 0 or self.radius_safety <= 0
                or self.radius_min_px < 0 or self.prediction_error_ratio <= 0
                or self.prediction_horizon_frames < 1 or self.gap_penalty < 0
                or self.max_assignment_cost_sqrt < 0 or self.unmatched_cost_sqrt < 0
                or self.minimum_cost_margin_sqrt < 0):
            raise ValueError("invalid gate-model parameters")
        if not (0 < self.fill_confidence_gaps[0] <= self.fill_confidence_gaps[1]):
            raise ValueError("fill_confidence_gaps must be increasing and positive")

    def gate_radius(self, dt) -> "np.ndarray | float":
        """Search radius r(dt) in pixels for the configured gate model (array-safe)."""
        import numpy as np
        dt = np.asarray(dt, dtype=float)
        if self.gate_model == "linear":
            return self.max_body_displacement_px * dt
        return np.maximum(self.radius_min_px,
                          self.radius_safety * (self.radius_r0_px + self.radius_k_px * np.sqrt(dt)))

    def cost_thresholds(self) -> dict:
        """Assignment thresholds on the active gate model's cost scale.

        Linear: the px/frame values (max_assignment_cost, unmatched_cost,
        minimum_cost_margin). Sqrt: the dimensionless *_sqrt values.
        """
        if self.gate_model == "linear":
            return {"max_assignment_cost": self.max_assignment_cost, "unmatched_cost": self.unmatched_cost,
                    "minimum_cost_margin": self.minimum_cost_margin}
        return {"max_assignment_cost": self.max_assignment_cost_sqrt,
                "unmatched_cost": self.unmatched_cost_sqrt,
                "minimum_cost_margin": self.minimum_cost_margin_sqrt}

    cost_scale_defaults = cost_thresholds  # name used by the WP-A brief

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
    for key in ("recoverable_reasons", "fill_confidence_gaps"):
        if key in values:
            values[key] = tuple(values[key])
    return replace(BridgeConfig(), **values)
