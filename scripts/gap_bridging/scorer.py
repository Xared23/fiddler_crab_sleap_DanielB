#!/usr/bin/env python3

"""
Learned link scorer: model file, manifest and the ``Scorer`` protocol (v004).

A scorer is a scikit-learn ``HistGradientBoostingClassifier`` on the pair
features of :mod:`gap_bridging.features` followed by a probability
calibrator (isotonic or Platt/sigmoid) fitted on out-of-fold predictions
(:mod:`gap_bridging.train`). ``LinkScorer.cost`` returns ``-log p`` for the
assignment in :func:`gap_bridging.candidates.link_tracklets`.

On disk a scorer is two files: ``<name>.joblib`` (``{'hgb', 'calibrator'}``)
and the manifest ``<name>.json`` (:class:`ScorerManifest`: scikit-learn
version, feature names, monotonic constraints, hyperparameters, threshold,
training-data hash, metrics, the feature-relevant config keys and the
feature-set spec). The feature names are an ordered subset of
:data:`gap_bridging.features.FEATURE_NAMES` (all 28 unless the scorer was
trained on a v005 ablation set); inference selects those columns by name.
``load_scorer`` refuses a model whose scikit-learn version differs from the
running code, since pickled estimators are not portable across versions, or
whose feature names are not an ordered subset of the running code's;
``resolve_scorer`` (and ``load_scorer`` given a config) also refuses a
config whose feature keys (:data:`gap_bridging.features.FEATURE_CONFIG_KEYS`)
differ from the ones the training features were computed with. A manifest
without ``feature_config`` (written before that check existed) is accepted
with a printed warning.

scikit-learn and joblib are imported lazily: importing this module (as the
pipeline does) does not import them until a model is fitted or loaded.
:mod:`gap_bridging.features` is imported lazily too, so that
``candidates`` may import this module without a cycle.

Sam
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from __future__ import annotations

from dataclasses import dataclass, field, fields
import functools
import json
import math
from pathlib import Path
from typing import Any, Optional, Protocol, Tuple

# ----- numerics -----
import numpy as np

# ----- local modules -----
from gap_bridging.config import BridgeConfig

# ============================================================
# CONSTANTS
# ============================================================

#: Probability floor before ``-log`` so that costs stay finite.
P_FLOOR = 1e-12
CALIBRATIONS = ('isotonic', 'sigmoid')
TRAIN_HINT = "run `bridge_gaps.py train` to fit one"
#: Tuple-valued manifest fields (lists in JSON).
_TUPLE_FIELDS = ('feature_names', 'monotonic_cst', 'sweeps')
#: Dict-valued manifest fields whose non-finite floats are written as null.
_METRIC_FIELDS = ('oof_metrics', 'test_metrics')

# ============================================================
# PROTOCOL
# ============================================================


class Scorer(Protocol):
    """What ``candidates.link_tracklets`` needs from a scorer."""

    @property
    def threshold(self) -> float:
        """Probability below which a pair is refused (``max_cost`` gate)."""

    @property
    def feature_names(self) -> Tuple[str, ...]:
        """Feature columns the model expects: an ordered subset of ``features.FEATURE_NAMES``."""

    def cost(self, X: np.ndarray) -> np.ndarray:
        """``-log p`` per row of ``X`` (``(n, len(feature_names))`` -> ``(n,)``), finite."""


# ============================================================
# ERRORS AND MANIFEST
# ============================================================


class ScorerVersionError(ValueError):
    """The model was pickled by a different scikit-learn version."""


def _finite_or_none(value: Any) -> Any:
    """``None`` for a non-finite float (strict JSON), else the value."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _none_to_nan(value: Any) -> Any:
    """Inverse of :func:`_finite_or_none` for metric values."""
    return math.nan if value is None else value


@dataclass(frozen=True)
class ScorerManifest:
    """Provenance and evaluation of a fitted scorer (``<model>.json``).

    ``threshold`` is the probability below which a pair is refused; it and
    the calibrator are both chosen on out-of-fold predictions of the train
    fold (``oof_metrics``), which is slightly optimistic. ``test_metrics``
    are pair-level metrics on the held-out test fold (``{}`` when there is
    none); the promotion decision uses the test fold through the full
    pipeline (hold-out sweep with ``cost_model: learned``), not these.
    ``feature_config`` is :func:`gap_bridging.features.feature_config` of
    the hold-out configs the training features came from (``{}`` for a
    manifest written before it was recorded). ``feature_set`` is the
    training ``--features`` spec (a :data:`gap_bridging.features.FEATURE_SETS`
    key or a comma-separated list; ``'all'`` for a manifest written before
    v005); ``feature_names`` is what it resolved to. ``training_profile`` is
    the hold-out profile of the training sweeps
    (:data:`gap_bridging.holdout.PROFILES`; ``'standard'`` for a manifest
    written before it was recorded).
    """
    sklearn_version: str
    feature_names: Tuple[str, ...]
    monotonic_cst: Tuple[int, ...]
    hyperparameters: dict
    calibration: str
    threshold: float
    training_data_sha256: str
    n_train: int
    n_test: int
    n_positive_train: int
    n_positive_test: int
    oof_metrics: dict
    test_metrics: dict
    salt: str
    test_fraction: float
    sweeps: Tuple[str, ...]
    created: str                 # ISO 8601, UTC
    git_commit: str = ''         # '' if unknown
    feature_config: dict = field(default_factory=dict)
    feature_set: str = 'all'
    training_profile: str = 'standard'

    def __post_init__(self) -> None:
        assert len(self.feature_names) == len(self.monotonic_cst), "one constraint per feature"
        assert self.calibration in CALIBRATIONS, f"calibration must be one of {CALIBRATIONS}"

    def to_json(self) -> dict:
        """JSON-ready dict: tuples as lists, non-finite metric values as ``None``."""
        out: dict = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in _TUPLE_FIELDS:
                value = list(value)
            elif f.name in _METRIC_FIELDS:
                value = {k: _finite_or_none(v) for k, v in value.items()}
            elif isinstance(value, dict):
                value = dict(value)
            out[f.name] = value
        return out

    @classmethod
    def from_json(cls, d: dict) -> 'ScorerManifest':
        """Inverse of :meth:`to_json` (lists back to tuples, ``None`` metrics to NaN).

        Optional fields (``git_commit``, ``feature_config``, ``feature_set``,
        ``training_profile``) may be absent.
        """
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown manifest keys: {sorted(unknown)}")
        values = dict(d)
        for name in _TUPLE_FIELDS:
            values[name] = tuple(values[name])
        values['monotonic_cst'] = tuple(int(v) for v in values['monotonic_cst'])
        for name in _METRIC_FIELDS:
            values[name] = {k: _none_to_nan(v) for k, v in values[name].items()}
        return cls(**values)


# ============================================================
# CALIBRATORS
# ============================================================


def _logit(p: np.ndarray) -> np.ndarray:
    """``log(p / (1 - p))`` with ``p`` clipped to ``[P_FLOOR, 1 - P_FLOOR]``."""
    p = np.clip(np.asarray(p, dtype=float), P_FLOOR, 1.0 - P_FLOOR)
    return np.log(p) - np.log1p(-p)


@dataclass
class SigmoidCalibrator:
    """Platt scaling on the logit of the raw probability.

    ``predict(p) = 1 / (1 + exp(-(coef * logit(p) + intercept)))``. Plain
    floats only, so the pickle does not depend on scikit-learn; ``coef > 0``
    keeps the calibrated probability non-decreasing in the raw one (and so
    the scorer's monotonic constraints intact).
    """
    coef: float = 1.0
    intercept: float = 0.0

    @classmethod
    def fit(cls, p_raw: np.ndarray, y: np.ndarray, seed: int = 0) -> 'SigmoidCalibrator':
        """Logistic regression of ``y`` on ``logit(p_raw)``.

        Raises
        ------
        ValueError
            If ``y`` has a single class or the fitted slope is not positive
            (the raw model ranks worse than chance; use isotonic).
        """
        from sklearn.linear_model import LogisticRegression

        y = np.asarray(y)
        if np.unique(y).size < 2:
            raise ValueError("sigmoid calibration needs both classes")
        model = LogisticRegression(C=1e6, random_state=seed)
        model.fit(_logit(p_raw)[:, None], y)
        coef = float(model.coef_[0, 0])
        if not coef > 0:
            raise ValueError(f"sigmoid calibration slope {coef} is not positive; use isotonic")
        return cls(coef=coef, intercept=float(model.intercept_[0]))

    def predict(self, p: np.ndarray) -> np.ndarray:
        """Calibrated probability per raw probability."""
        z = self.coef * _logit(p) + self.intercept
        return 0.5 * (1.0 + np.tanh(0.5 * z))       # stable logistic


def calibrated_probability(hgb: Any, calibrator: Any, X: np.ndarray) -> np.ndarray:
    """``clip(calibrator.predict(hgb.predict_proba(X)[:, 1]), 0, 1)``; ``(n,)``."""
    X = np.asarray(X, dtype=float)
    if X.shape[0] == 0:
        return np.zeros(0)
    p_raw = hgb.predict_proba(X)[:, 1]
    return np.clip(np.asarray(calibrator.predict(p_raw), dtype=float), 0.0, 1.0)


# ============================================================
# SCORER
# ============================================================


class LinkScorer:
    """Fitted classifier + calibrator + manifest; satisfies :class:`Scorer`."""

    def __init__(self, hgb: Any, calibrator: Any, manifest: ScorerManifest) -> None:
        self._hgb = hgb
        self._calibrator = calibrator
        self._manifest = manifest

    @property
    def manifest(self) -> ScorerManifest:
        return self._manifest

    @property
    def threshold(self) -> float:
        return float(self._manifest.threshold)

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return tuple(self._manifest.feature_names)

    @property
    def column_indices(self) -> Tuple[int, ...]:
        """Indices of :attr:`feature_names` in ``features.FEATURE_NAMES`` (increasing)."""
        from gap_bridging import features      # lazy: see check_feature_config

        return features.check_feature_subset(self.feature_names)

    @property
    def hgb(self) -> Any:
        return self._hgb

    @property
    def calibrator(self) -> Any:
        return self._calibrator

    def _check(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n_features = len(self.feature_names)
        if X.ndim != 2 or X.shape[1] != n_features:
            raise ValueError(f"expected X of shape (n, {n_features}), got {X.shape}")
        return X

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calibrated probability of a true link per row; NaN features allowed.

        Parameters
        ----------
        X : np.ndarray
            ``(n, F)`` in :attr:`feature_names` order.

        Returns
        -------
        np.ndarray
            ``(n,)`` in ``[0, 1]``.
        """
        return calibrated_probability(self._hgb, self._calibrator, self._check(X))

    def cost(self, X: np.ndarray) -> np.ndarray:
        """``-log(clip(p, 1e-12, 1))`` per row; finite and ``>= 0``."""
        p = self.predict_proba(X)
        return -np.log(np.clip(p, P_FLOOR, 1.0))

    def __repr__(self) -> str:
        m = self._manifest
        return (f"LinkScorer(calibration={m.calibration!r}, threshold={m.threshold:.4g}, "
                f"n_train={m.n_train}, created={m.created!r})")


# ============================================================
# PERSISTENCE
# ============================================================


def manifest_path(model_path: Path) -> Path:
    """The manifest JSON beside a model file (``model.joblib`` -> ``model.json``)."""
    return Path(model_path).with_suffix('.json')


def save_scorer(scorer: LinkScorer, path: Path, force: bool = False) -> None:
    """Write ``path`` (joblib) and its manifest JSON.

    Raises
    ------
    FileExistsError
        If either file exists and ``force`` is false.
    """
    import joblib

    path = Path(path)
    meta = manifest_path(path)
    assert meta != path, "the model file must not have a .json suffix"
    existing = [str(p) for p in (path, meta) if p.exists()]
    if existing and not force:
        raise FileExistsError(f"scorer files already exist (use --force): {existing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'hgb': scorer.hgb, 'calibrator': scorer.calibrator}, path)
    text = json.dumps(scorer.manifest.to_json(), indent=2, sort_keys=True, allow_nan=False)
    meta.write_text(text + '\n')


def check_feature_config(manifest: ScorerManifest, cfg: BridgeConfig, path: Path) -> None:
    """Refuse a config whose feature keys differ from the scorer's training config.

    Parameters
    ----------
    manifest : ScorerManifest
        The scorer's manifest.
    cfg : BridgeConfig
        The config the scorer is about to be used with.
    path : Path
        The model file (for messages).

    Raises
    ------
    ValueError
        If ``features.feature_config(cfg) != manifest.feature_config``; the
        message lists each differing key with both values. A manifest
        without ``feature_config`` only prints a warning (once per path).
    """
    from gap_bridging import features      # lazy: features may import candidates, which imports this

    if not manifest.feature_config:
        _warn_no_feature_config(str(path))
        return
    expected = dict(manifest.feature_config)
    actual = features.feature_config(cfg)
    if actual != expected:
        keys = sorted(set(actual) | set(expected))
        differ = {k: (expected.get(k), actual.get(k)) for k in keys
                  if expected.get(k) != actual.get(k)}
        raise ValueError(f"scorer {path} was trained on features computed with a different config; "
                         f"differing keys (trained, current): {differ}; retrain it or use the "
                         f"training config's values")


@functools.lru_cache(maxsize=None)
def _warn_no_feature_config(path: str) -> None:
    """Print, once per model path, that its manifest predates ``feature_config``."""
    print(f"[scorer] WARNING: {path} has no feature_config in its manifest (older model); "
          f"cannot check that the config computes the features it was trained on", flush=True)


def load_scorer(path: Path, cfg: Optional[BridgeConfig] = None) -> LinkScorer:
    """Read a scorer written by :func:`save_scorer`.

    Parameters
    ----------
    path : Path
        The model file.
    cfg : BridgeConfig, optional
        When given, checked with :func:`check_feature_config`.

    Raises
    ------
    FileNotFoundError
        If the model or its manifest is missing.
    ScorerVersionError
        If the installed scikit-learn differs from the manifest's.
    ValueError
        If the manifest's feature names are not an ordered subset of
        ``features.FEATURE_NAMES`` (unknown, repeated or permuted names), a
        ``feature_set`` naming a :data:`features.FEATURE_SETS` entry resolves
        to other names, ``monotonic_cst`` differs from
        ``features.monotonic_for(feature_names)``, the model's
        ``n_features_in_`` differs from the number of feature names, or
        ``cfg``'s feature keys differ from the manifest's.
    """
    path = Path(path)
    meta = manifest_path(path)
    for p, what in ((path, 'scorer model'), (meta, 'scorer manifest')):
        if not p.is_file():
            raise FileNotFoundError(f"{what} not found: {p}; {TRAIN_HINT}")
    manifest = ScorerManifest.from_json(json.loads(meta.read_text()))
    import sklearn

    from gap_bridging import features      # lazy: features may import candidates, which imports this

    if sklearn.__version__ != manifest.sklearn_version:
        raise ScorerVersionError(
            f"scorer {path} was trained with scikit-learn {manifest.sklearn_version} but "
            f"{sklearn.__version__} is installed; retrain it with `bridge_gaps.py train`")
    try:
        features.check_feature_subset(manifest.feature_names)
    except ValueError as exc:
        raise ValueError(f"scorer {path}: its manifest's feature_names are not an ordered "
                         f"subset of features.FEATURE_NAMES ({exc}); retrain it") from exc
    named = features.FEATURE_SETS.get(manifest.feature_set)
    if named is not None and tuple(named) != tuple(manifest.feature_names):
        raise ValueError(f"scorer {path}: feature_set {manifest.feature_set!r} resolves to "
                         f"{list(named)} but the manifest's feature_names are "
                         f"{list(manifest.feature_names)}; retrain it")
    expected_cst = features.monotonic_for(manifest.feature_names)
    if tuple(manifest.monotonic_cst) != expected_cst:
        raise ValueError(f"scorer {path}: monotonic_cst {list(manifest.monotonic_cst)} differs "
                         f"from features.monotonic_for(feature_names) {list(expected_cst)}; "
                         f"retrain it")
    if cfg is not None:
        check_feature_config(manifest, cfg, path)
    import joblib

    payload = joblib.load(path)
    assert set(payload) == {'hgb', 'calibrator'}, f"unexpected model payload keys {sorted(payload)}"
    n_in = getattr(payload['hgb'], 'n_features_in_', len(manifest.feature_names))
    if n_in != len(manifest.feature_names):
        raise ValueError(f"scorer {path}: the model takes {n_in} features but the manifest "
                         f"lists {len(manifest.feature_names)}; retrain it")
    return LinkScorer(payload['hgb'], payload['calibrator'], manifest)


@functools.lru_cache(maxsize=4)
def cached_scorer(path: str, mtime_ns: int) -> LinkScorer:
    """:func:`load_scorer` memoised on ``(path, mtime_ns)``; a re-saved file reloads."""
    del mtime_ns                    # part of the cache key only
    return load_scorer(Path(path))


def resolve_scorer(cfg: BridgeConfig) -> LinkScorer:
    """The scorer of a ``cost_model: learned`` config (cached by path and mtime).

    Raises
    ------
    ValueError
        If ``cfg.cost_model`` is not ``'learned'``, ``cfg.scorer_path`` is
        unset, or ``cfg``'s feature keys differ from the manifest's
        (:func:`check_feature_config`).
    FileNotFoundError
        If the model file is missing.
    """
    if cfg.cost_model != 'learned':
        raise ValueError(f"cost_model is {cfg.cost_model!r}, not 'learned'; no scorer to resolve")
    if cfg.scorer_path is None:
        raise ValueError(f"cost_model 'learned' needs scorer_path; {TRAIN_HINT}")
    path = Path(cfg.scorer_path)
    if not path.is_file():
        return load_scorer(path)    # raises FileNotFoundError with the hint
    scorer = cached_scorer(str(path), path.stat().st_mtime_ns)
    check_feature_config(scorer.manifest, cfg, path)
    return scorer
