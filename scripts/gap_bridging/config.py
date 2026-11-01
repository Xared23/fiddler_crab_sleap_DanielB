"""Frozen configuration for gap bridging. Distances are native-image pixels."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
from pathlib import Path

import yaml

DEFAULT_RECOVERABLE = ('low_body_score', 'missing_position', 'outside_image',
                       'invisible', 'duplicate_candidate')


@dataclass(frozen=True)
class BridgeConfig:
    """Parameters for linking and filling. See docs/GAP_BRIDGING.md."""
    max_gap: int = 30                         # link when 1 <= gap <= max_gap; 0 disables linking.
                                              # v003 rule: highest horizon with pooled hold-out precision
                                              # 1.0 at every planted gap (decoys counted, plants beyond
                                              # the horizon scored), demo seeds 0-4 (bridge_gaps.py pool)
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
    # --- v003: tail recovery (continuation-only relaxation of the Shell-score gate) ---
    tail_recovery: bool = True
    tail_radius_px: float = 8.0
    tail_max_frames: int = 15
    tail_reasons: tuple[str, ...] = ('low_body_score',)
    tail_exclusion_px: float = 5.0            # tracker's duplicate_shell_radius_px
    # --- v003: spurious-identity filter (flag only; nothing is deleted) ---
    min_identity_frames: int = 2              # real frames (observed/recovered/recovered_tail) below this ...
    min_identity_score: float = 0.6           # ... AND median instance_score below this -> 'spurious'
    # --- v003: appearance cues (sqrt model only) ---
    appearance_cues: bool = True
    size_scale_px: float = 3.0                # size term = |dL| / sqrt(scale^2 + noise^2/n_a + noise^2/n_b)
    size_noise_px: float = 1.0
    size_cost_cap: float = 3.0
    size_weight: float = 0.1
    claw_side_weight: float = 0.15            # side term = max(0, -side_a * side_b)
    claw_side_confidence: float = 0.9         # veto when both |side| >= this ...
    claw_side_min_frames: int = 5             # ... over at least this many valid frames, opposite signs
    # Staged linking (sqrt only): gaps (prev, stage] are linked stage by stage, shortest first;
    # stages above max_gap are dropped and max_gap is always the last stage.
    link_stages: tuple[int, ...] = (5, 15, 30, 45)
    # --- v004: cost model, assignment and feature export ---------------------
    # cost_model "hand": the sqrt/linear cost above. "learned": the probability p of a true
    # link from the scikit-learn scorer at scorer_path (features.FEATURE_NAMES), cost = -log p,
    # with the motion gates and the claw-side veto unchanged (sqrt gates only). Thresholds are
    # given as probabilities and converted by cost_thresholds().
    cost_model: str = "hand"
    scorer_path: str | None = None            # .joblib model with a .json manifest beside it
    learned_min_probability: float = 0.5      # max_cost gate: refuse when p < this
    learned_unmatched_probability: float = 0.5  # dummy diagonal = -log(this)
    learned_margin: float = 0.0               # runner-up margin in -log p units (log probability ratio)
    # assignment "staged": link_stages as above. "global": one assignment over gaps 1..max_gap
    # (Jaqaman et al. 2008 second stage). unmatched_cost_mode "adaptive": the dummy diagonal of
    # each connected component is adaptive_factor * its largest finite cost (Jaqaman 1.05 max C)
    # instead of the fixed unmatched cost.
    assignment: str = "staged"
    unmatched_cost_mode: str = "fixed"
    adaptive_factor: float = 1.05
    export_pair_features: bool = False        # also write pair_features.parquet (features.py) next to links.parquet

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
        if (self.tail_radius_px < 0 or self.tail_max_frames < 0 or self.tail_exclusion_px < 0
                or self.min_identity_frames < 1 or not 0 <= self.min_identity_score <= 1):
            raise ValueError("invalid tail-recovery or identity-filter parameters")
        if (self.size_scale_px <= 0 or self.size_noise_px < 0 or self.size_cost_cap < 0 or self.size_weight < 0
                or self.claw_side_weight < 0 or not 0 < self.claw_side_confidence <= 1 or self.claw_side_min_frames < 1):
            raise ValueError("invalid appearance-cue parameters")
        stages = tuple(self.link_stages)
        if not stages or any(int(s) != s or s < 1 for s in stages) or any(
                b <= a for a, b in zip(stages, stages[1:])):
            raise ValueError("link_stages must be positive integers, strictly increasing")
        if self.cost_model not in ("hand", "learned"):
            raise ValueError("cost_model must be 'hand' or 'learned'")
        if self.cost_model == "learned" and self.gate_model != "sqrt":
            raise ValueError("cost_model 'learned' requires gate_model 'sqrt'")
        if self.export_pair_features and self.gate_model != "sqrt":
            raise ValueError("export_pair_features requires gate_model 'sqrt' "
                             "(features.pair_feature_table is sqrt-only)")
        if (not 0 < self.learned_min_probability <= 1 or not 0 < self.learned_unmatched_probability <= 1
                or self.learned_margin < 0):
            raise ValueError("learned probabilities must be in (0, 1] and learned_margin >= 0")
        if self.assignment not in ("staged", "global"):
            raise ValueError("assignment must be 'staged' or 'global'")
        if self.unmatched_cost_mode not in ("fixed", "adaptive"):
            raise ValueError("unmatched_cost_mode must be 'fixed' or 'adaptive'")
        if self.adaptive_factor < 1:
            raise ValueError("adaptive_factor must be >= 1")

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
        minimum_cost_margin). Sqrt: the dimensionless *_sqrt values. Learned
        cost model: ``-log`` of ``learned_min_probability`` and
        ``learned_unmatched_probability`` and ``learned_margin`` (v004).
        """
        if self.cost_model == "learned":
            return {"max_assignment_cost": -math.log(self.learned_min_probability),
                    "unmatched_cost": -math.log(self.learned_unmatched_probability),
                    "minimum_cost_margin": self.learned_margin}
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
        scorer = values.get("scorer_path")
        if scorer is not None and not Path(scorer).is_absolute():
            values["scorer_path"] = str(Path(path).resolve().parent / scorer)
    values.update({k: v for k, v in overrides.items() if v is not None})
    if values.get("scorer_path") is not None:
        values["scorer_path"] = str(values["scorer_path"])
    for key in ("recoverable_reasons", "fill_confidence_gaps", "link_stages", "tail_reasons"):
        if key in values:
            values[key] = tuple(values[key])
    return replace(BridgeConfig(), **values)
