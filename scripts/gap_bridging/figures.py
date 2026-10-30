#!/usr/bin/env python3

"""
Figures of the gap-bridging sweep and hold-out evaluation (WP-D).

``fig_sweep`` plots links and trajectory length against the horizon
``max_gap``; ``fig_holdout`` the hold-out precision and recall against the
planted gap length, one line per horizon; ``fig_radius_fit`` the measured
displacement quantiles with the fitted and configured gate radius; and
``fig_bridges`` renders long bridges as image crops of the source video
(cyan observed, yellow recovered, magenta interpolated), following
``logs/gap_bridging_report/make_figures.py``. Line art is written as PDF
(plus a 300-dpi PNG preview), image panels as 300-dpi PNG.

Sam
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

# ----- numerics -----
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ----- plotting -----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullLocator, ScalarFormatter  # noqa: E402

# ----- local modules -----
from gap_bridging.config import BridgeConfig  # noqa: E402

# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 14,
    "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": False, "ytick.right": False})
BLACK, BLUE, RED = "black", "#1f4e9c", "#c0392b"
SOURCE_COLORS = {"observed": "#19c3ff", "recovered": "#ffd400", "interpolated": "#ff2ad4"}
BRIDGE_COLUMNS = ("video_id", "frame", "trajectory_id", "source", "bridge_id", "gap_frames",
                  "fill_confidence", "body_x", "body_y", "shoulder_x", "shoulder_y")
MIN_CROP_PX = 220
CROP_PAD_PX = 100
FRAMES_PER_ROW = 7
ROWS_PER_PAGE = 5
LOG_TICKS = (1, 2, 5, 10, 20, 50)


def _save_line_art(fig: plt.Figure, out: Path) -> list[Path]:
    """Save ``out`` as PDF plus a 300-dpi PNG preview; return both paths."""
    out = Path(out).with_suffix(".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [out, out.with_suffix(".png")]


def _horizon_style(index: int) -> dict:
    """Colour, line style and marker of the index-th horizon (3 colours x 2 styles)."""
    return {"color": (BLACK, BLUE, RED)[index % 3], "ls": ("-", "--")[(index // 3) % 2],
            "marker": ("o", "s", "^", "D", "v", "x")[index % 6]}


# ============================================================
# SWEEP AND HOLD-OUT
# ============================================================


def fig_sweep(summary_csv: Path, out: Path) -> list[Path]:
    """Links (left) and mean and median trajectory length (right) vs ``max_gap``.

    Parameters
    ----------
    summary_csv : Path
        ``sweep_summary.csv`` with ``max_gap``, ``links``, ``mean_length``
        and ``median_length``.
    out : Path
        Output path; written as ``.pdf`` and ``.png``.

    Returns
    -------
    list of Path
        The written files.
    """
    s = pd.read_csv(summary_csv).sort_values("max_gap")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), sharex=True)
    axes[0].plot(s.max_gap, s.links, "o-", lw=2.0, color=BLACK)
    axes[0].set_ylabel(r"Links, $N_{\mathrm{link}}$")
    axes[1].plot(s.max_gap, s.mean_length, "o-", lw=2.0, color=BLACK, label="Mean")
    axes[1].plot(s.max_gap, s.median_length, "s--", lw=2.0, color=BLUE, label="Median")
    axes[1].set_ylabel(r"Trajectory length, $L$ (frames)")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xlabel(r"Horizon, $g_{\max}$ (frames)")
    fig.tight_layout()
    return _save_line_art(fig, out)


def fig_holdout(by_gap_csv: Path, out: Path) -> list[Path]:
    """Hold-out precision (left) and recall (right) vs planted gap, one line per horizon.

    Parameters
    ----------
    by_gap_csv : Path
        ``holdout_by_gap.csv`` (long form: ``max_gap``, ``gap``,
        ``precision``, ``recall``, ...).
    out : Path
        Output path; written as ``.pdf`` and ``.png``.

    Returns
    -------
    list of Path
        The written files.
    """
    h = pd.read_csv(by_gap_csv)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), sharex=True)
    for i, (horizon, rows) in enumerate(h.sort_values(["max_gap", "gap"]).groupby("max_gap")):
        style = _horizon_style(i)
        for ax, column in zip(axes, ("precision", "recall")):
            ax.plot(rows.gap, rows[column], lw=2.0, ms=6, mfc="none",
                    label=rf"$g_{{\max}} = {int(horizon)}$", **style)
    axes[0].set_ylabel(r"Precision, $P$")
    axes[1].set_ylabel(r"Recall, $R$")
    for ax in axes:
        ax.set_xlabel(r"Planted gap, $g$ (frames)")
        ax.set_xscale("log")
        ax.set_xticks(LOG_TICKS)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(-0.03, 1.03)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.tight_layout()
    return _save_line_art(fig, out)


def fig_radius_fit(quantiles: pa.Table, fit: dict, cfg: BridgeConfig, out: Path) -> list[Path]:
    """Measured displacement quantile vs dt with the fitted, configured and legacy radii.

    Parameters
    ----------
    quantiles : pa.Table
        :func:`gap_bridging.fit.displacement_quantiles` output.
    fit : dict
        :func:`gap_bridging.fit.fit_radius` output.
    cfg : BridgeConfig
        Supplies the configured sqrt radius (``radius_*``, safety and floor
        applied, regardless of ``gate_model``) and the legacy
        ``max_body_displacement_px * dt`` line.
    out : Path
        Output path; written as ``.pdf`` and ``.png``.

    Returns
    -------
    list of Path
        The written files.
    """
    dt = np.asarray(quantiles["dt"].to_numpy(), dtype=float)
    grid = np.linspace(1.0, dt.max(), 200)
    fitted = fit["radius_r0_px"] + fit["radius_k_px"] * np.sqrt(grid)
    configured = np.asarray(replace(cfg, gate_model="sqrt").gate_radius(grid), dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(dt, quantiles["quantile"].to_numpy(), "o", ms=4, mfc="none", color=BLACK,
            label="Measured 99th percentile")
    ax.plot(grid, fitted, lw=2.0, color=BLUE,
            label=rf"Fit ${fit['radius_r0_px']:.2f} + {fit['radius_k_px']:.2f}\sqrt{{\Delta t}}$")
    ax.plot(grid, configured, lw=2.0, color=RED, label=r"Gate radius $r(\Delta t)$")
    ax.plot(grid, cfg.max_body_displacement_px * grid, lw=2.0, ls="--", color=BLACK,
            label=rf"Legacy ${cfg.max_body_displacement_px:g}\,\Delta t$")
    ax.set_yscale("log")
    ax.set_xlabel(r"Frame separation, $\Delta t$ (frames)")
    ax.set_ylabel(r"Net displacement, $D$ (px)")
    ax.legend(frameon=False, fontsize=11)
    fig.tight_layout()
    return _save_line_art(fig, out)


# ============================================================
# RENDERED BRIDGES
# ============================================================


def _frame_offset(table: pa.Table) -> int:
    """``frame - video frame index``: from ``demo_frame`` when present, else 0."""
    if "demo_frame" not in table.schema.names:
        return 0
    t = table.select(["frame", "demo_frame"]).to_pandas().dropna()
    offsets = (t.frame - t.demo_frame).unique()
    if len(offsets) != 1:
        raise ValueError(f"frame - demo_frame is not constant: {offsets[:5]}")
    return int(offsets[0])


def _pick_bridges(obs: pd.DataFrame, n: int, min_gap: int, seed: int) -> pd.DataFrame:
    """Up to ``n`` bridges with ``gap_frames >= min_gap``, drawn with ``seed``, sorted by gap."""
    bridges = (obs[obs.bridge_id.notna()].groupby("bridge_id")
               .agg(gap=("gap_frames", "first"), trajectory_id=("trajectory_id", "first"),
                    confidence=("fill_confidence", "first"), first=("frame", "min"),
                    last=("frame", "max"))
               .reset_index().sort_values("bridge_id"))
    bridges = bridges[bridges.gap >= min_gap]
    if len(bridges) > n:
        rng = np.random.default_rng(seed)
        bridges = bridges.iloc[np.sort(rng.choice(len(bridges), size=n, replace=False))]
    return bridges.sort_values(["gap", "bridge_id"]).reset_index(drop=True)


def _read_crops(video_path: Path, requests: dict[int, list], offset: int) -> dict:
    """Decode the video once, in order, and cut the requested crops.

    ``requests`` maps a frame to ``[(key, x0, y0, size)]``; the result maps
    each key to an RGB crop.
    """
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {video_path}")
    crops: dict = {}
    try:
        first = min(requests) - offset
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        for index in range(first, max(requests) - offset + 1):
            ok, img = cap.read()
            if not ok:
                raise RuntimeError(f"cannot read video frame {index} of {video_path}")
            for key, x0, y0, size in requests.get(index + offset, ()):
                crops[key] = cv2.cvtColor(img[y0:y0 + size, x0:x0 + size], cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
    return crops


def fig_bridges(observations_parquet: Path, video_path: Path, out: Path, n: int = 20,
                min_gap: int = 10, seed: int = 0, video_id: str | None = None,
                frame_offset: int | None = None) -> list[Path]:
    """Render a sample of long bridges as image-crop strips, one bridge per row.

    Parameters
    ----------
    observations_parquet : Path
        ``observations.parquet`` of a bridging run (with ``gap_frames``).
    video_path : Path
        The video those observations were predicted on.
    out : Path
        Output stem; pages are written as ``<stem>_p<k>.png`` (300 dpi).
    n : int
        Number of bridges to draw (fewer when fewer qualify).
    min_gap : int
        Minimum ``gap_frames`` of a drawn bridge.
    seed : int
        Seed of the sample.
    video_id : str, optional
        Video to draw; required when the table holds several videos.
    frame_offset : int, optional
        ``frame - video frame index``; default from ``demo_frame`` when the
        table has it, else 0.

    Returns
    -------
    list of Path
        The written pages; empty when no bridge qualifies.

    Notes
    -----
    Each row shows up to seven frames evenly spaced from the last observed
    frame before the gap to the first observed frame after it; markers are
    the shell (circle) and the shell-shoulder segment of the trajectory.
    """
    table = pq.read_table(observations_parquet)
    offset = _frame_offset(table) if frame_offset is None else int(frame_offset)
    obs = table.select(list(BRIDGE_COLUMNS)).to_pandas()
    videos = sorted(obs.video_id.dropna().unique())
    if video_id is None:
        if len(videos) != 1:
            raise ValueError(f"several videos in {observations_parquet}; pass video_id")
        video_id = videos[0]
    obs = obs[obs.video_id == video_id]
    picks = _pick_bridges(obs, n, min_gap, seed)
    if picks.empty:
        return []
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        raise FileNotFoundError(f"cannot open video {video_path}")

    rows, requests = [], {}
    for r, pick in picks.iterrows():
        traj = obs[obs.trajectory_id == pick.trajectory_id]
        f0, f1 = int(pick["first"]) - 1, int(pick["last"]) + 1
        seg = traj[(traj.frame >= f0) & (traj.frame <= f1)]
        lo, hi = seg[["body_x", "body_y"]].min().to_numpy(), seg[["body_x", "body_y"]].max().to_numpy()
        size = int(min(max(MIN_CROP_PX, (hi - lo).max() + CROP_PAD_PX), width, height))
        c = (lo + hi) / 2
        x0 = int(np.clip(c[0] - size / 2, 0, width - size))
        y0 = int(np.clip(c[1] - size / 2, 0, height - size))
        show = [int(f) for f in np.unique(np.round(np.linspace(f0, f1, FRAMES_PER_ROW)))]
        for k, f in enumerate(show):
            requests.setdefault(f, []).append(((r, k), x0, y0, size))
        rows.append((pick, seg, show, x0, y0, size))
    crops = _read_crops(Path(video_path), requests, offset)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pages = []
    for p, start in enumerate(range(0, len(rows), ROWS_PER_PAGE), 1):
        chunk = rows[start:start + ROWS_PER_PAGE]
        fig, axes = plt.subplots(len(chunk), FRAMES_PER_ROW,
                                 figsize=(10.5, 1.65 * len(chunk) + 0.4), squeeze=False)
        for r, (pick, seg, show, x0, y0, size) in enumerate(chunk):
            for k, ax in enumerate(axes[r]):
                ax.set_xticks([])
                ax.set_yticks([])
                if k >= len(show):
                    ax.axis("off")
                    continue
                f = show[k]
                ax.imshow(crops[(start + r, k)])
                for _, q in seg[seg.frame == f].iterrows():
                    col = SOURCE_COLORS[q.source]
                    ax.plot(q.body_x - x0, q.body_y - y0, "o", mfc="none", mec=col, mew=1.5, ms=9)
                    if np.isfinite(q.shoulder_x):
                        ax.plot([q.body_x - x0, q.shoulder_x - x0],
                                [q.body_y - y0, q.shoulder_y - y0], "-", color=col, lw=1.5)
                ax.set_xlim(0, size)
                ax.set_ylim(size, 0)
                ax.set_title(f"$f={f}$", fontsize=10, pad=2)
                if k == 0:
                    ax.set_ylabel(f"$g={int(pick.gap)}$\n{pick.confidence}", fontsize=11)
        handles = [Line2D([], [], marker="o", mfc="none", mec=SOURCE_COLORS[k], mew=1.5, ms=9,
                          ls="", label=k) for k in SOURCE_COLORS]
        fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.02),
                   frameon=False)
        fig.tight_layout()
        page = out.with_name(f"{out.stem}_p{p}.png")
        fig.savefig(page, dpi=300, bbox_inches="tight")
        plt.close(fig)
        pages.append(page)
    return pages
