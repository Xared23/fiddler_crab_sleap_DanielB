#!/usr/bin/env python3

"""
Fit of the sqrt gate radius to measured shell displacements (WP-D).

``displacement_quantiles`` pools, over every unbroken accepted tracklet, the
net shell displacement ``|p(f + dt) - p(f)|`` for dt = 1..max_dt and returns
its q-quantile per dt. ``fit_radius`` fits ``D(dt) = r0 + k * sqrt(dt)`` to
those quantiles by non-negative least squares (r0, k >= 0). The configured
gate radius is then ``max(radius_min_px, radius_safety * (r0 + k sqrt(dt)))``
(``BridgeConfig.gate_radius``). Units: pixels and frames.

Sam
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from __future__ import annotations

import math

# ----- numerics -----
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from scipy.optimize import nnls

# ============================================================
# CONSTANTS
# ============================================================

QUANTILE_SCHEMA = pa.schema([('dt', pa.int64()), ('quantile', pa.float64()), ('n', pa.int64())])

# ============================================================
# QUANTILES
# ============================================================


def _shell_column(table: pa.Table, axis: str) -> np.ndarray:
    """Shell coordinate (``shell_<axis>``, falling back to ``body_<axis>``) as float64, NaN for null."""
    names = table.schema.names
    column = None
    for name in (f'shell_{axis}', f'body_{axis}'):
        if name in names:
            values = pc.cast(table[name], pa.float64())
            column = values if column is None else pc.coalesce(column, values)
    if column is None:
        raise ValueError(f"observation table has neither shell_{axis} nor body_{axis}")
    return np.asarray(pc.fill_null(column, math.nan).to_numpy(), dtype=np.float64)


def unbroken_tracks(obs: pa.Table) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accepted rows of unbroken tracklets, sorted by (tracklet, frame).

    Parameters
    ----------
    obs : pa.Table
        Observation table with ``tracklet_id``, ``frame``,
        ``accepted_for_tracking`` and shell (or body) coordinates.

    Returns
    -------
    codes : np.ndarray
        ``(n,)`` int64 tracklet code of every kept row.
    frames : np.ndarray
        ``(n,)`` int64 frames.
    xy : np.ndarray
        ``(n, 2)`` shell positions (NaN where unknown).

    Notes
    -----
    A tracklet is kept when its accepted rows cover consecutive, unique
    frames, so ``row i + dt`` of the same code is exactly ``dt`` frames later.
    """
    for name in ('tracklet_id', 'frame', 'accepted_for_tracking'):
        if name not in obs.schema.names:
            raise ValueError(f"observation table has no column {name!r}")
    keep = pc.and_(pc.fill_null(obs['accepted_for_tracking'], False), pc.is_valid(obs['tracklet_id']))
    table = obs.filter(keep)
    encoded = pc.dictionary_encode(table['tracklet_id'].combine_chunks())
    codes = encoded.indices.to_numpy(zero_copy_only=False).astype(np.int64)
    frames = table['frame'].to_numpy().astype(np.int64)
    xy = np.stack([_shell_column(table, 'x'), _shell_column(table, 'y')], axis=1)
    order = np.lexsort((frames, codes))
    codes, frames, xy = codes[order], frames[order], xy[order]
    same = codes[1:] == codes[:-1]
    broken = np.unique(codes[1:][same & (np.diff(frames) != 1)])
    ok = ~np.isin(codes, broken)
    return codes[ok], frames[ok], xy[ok]


def displacement_quantiles(obs: pa.Table, max_dt: int = 60, q: float = 0.99,
                           min_samples: int = 200) -> pa.Table:
    """Quantile of the net shell displacement over dt frames, dt = 1..max_dt.

    Parameters
    ----------
    obs : pa.Table
        Observation table (see :func:`unbroken_tracks`).
    max_dt : int
        Largest frame separation (>= 1).
    q : float
        Quantile in (0, 1) (``numpy.quantile``, linear method).
    min_samples : int
        A dt with fewer finite displacement samples is left out.

    Returns
    -------
    pa.Table
        :data:`QUANTILE_SCHEMA` (dt, quantile in px, n samples), ascending dt.
        Samples are all pairs ``(f, f + dt)`` inside one unbroken tracklet,
        pooled across tracklets (overlapping windows, so not independent).
    """
    if max_dt < 1 or not 0 < q < 1 or min_samples < 1:
        raise ValueError("need max_dt >= 1, 0 < q < 1 and min_samples >= 1")
    codes, _, xy = unbroken_tracks(obs)
    rows: list[dict] = []
    for dt in range(1, int(max_dt) + 1):
        if codes.size <= dt:
            break
        same = codes[dt:] == codes[:-dt]
        step = np.hypot(*(xy[dt:] - xy[:-dt]).T)[same]
        step = step[np.isfinite(step)]
        if step.size >= min_samples:
            rows.append({'dt': dt, 'quantile': float(np.quantile(step, q)), 'n': int(step.size)})
    return pa.Table.from_pylist(rows, schema=QUANTILE_SCHEMA)


# ============================================================
# FIT
# ============================================================


def fit_radius(quantiles: pa.Table) -> dict:
    """Non-negative least squares of ``r0 + k * sqrt(dt)`` to the quantiles.

    Parameters
    ----------
    quantiles : pa.Table
        Output of :func:`displacement_quantiles` (at least two rows).

    Returns
    -------
    dict
        ``radius_r0_px``, ``radius_k_px`` (both >= 0), ``rmse`` (px) and
        ``n_dt`` (number of dt values fitted).
    """
    if quantiles.num_rows < 2:
        raise ValueError(f"need at least two dt values to fit, got {quantiles.num_rows}")
    dt = np.asarray(quantiles['dt'].to_numpy(), dtype=np.float64)
    value = np.asarray(quantiles['quantile'].to_numpy(), dtype=np.float64)
    design = np.column_stack([np.ones_like(dt), np.sqrt(dt)])
    (r0, k), _ = nnls(design, value)
    residual = design @ np.array([r0, k]) - value
    return {'radius_r0_px': float(r0), 'radius_k_px': float(k),
            'rmse': float(np.sqrt(np.mean(residual ** 2))), 'n_dt': int(dt.size)}
