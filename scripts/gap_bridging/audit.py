#!/usr/bin/env python3

"""
Visual audit of the links a candidate linker run gained or lost (v005 WP-V).

The v005 promotion rule asks a human to look at the links on which a
candidate run disagrees with the current one. ``link_diff`` splits the
``linked`` pairs of two ``links.parquet`` tables into *gained* (linked by the
candidate only) and *lost* (linked by the current run only);
``sample_links`` draws up to ``n`` of each; ``link_rows`` tabulates every
drawn pair (costs, decisions and reasons of both runs, tracklet sizes and
endpoint frames); and ``fig_link_audit`` renders one strip of video crops per
link: the END tracklet's last observed frames, the gap midpoint and the START
tracklet's first observed frames, with both tracklets drawn.
``audit_command`` ties these together and writes
``<out>_gained_p<k>.png``, ``<out>_lost_p<k>.png`` and ``<out>.csv``.

Plotting and video decoding (matplotlib, cv2, :mod:`gap_bridging.figures`)
are imported inside the rendering function, so the tabular helpers need
neither.

Sam
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

# ----- numerics -----
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ============================================================
# CONSTANTS
# ============================================================

LINK_KEY = ('tracklet_a', 'tracklet_b')
AUDIT_CSV_COLUMNS = ('kind', 'tracklet_a', 'tracklet_b', 'gap_frames', 'end_frame_a',
                     'start_frame_b', 'n_frames_a', 'n_frames_b', 'cost_current',
                     'decision_current', 'reason_current', 'cost_candidate',
                     'decision_candidate', 'reason_candidate', 'page', 'row')
KINDS = ('gained', 'lost')
ABSENT = 'absent'
LINKS_NAME = 'links.parquet'
OBS_COLUMNS = ('video_id', 'frame', 'tracklet_id', 'accepted_for_tracking', 'body_x', 'body_y',
               'shoulder_x', 'shoulder_y', 'demo_frame')
END_COLOR, START_COLOR = '#19c3ff', '#ff2ad4'   # cyan END tracklet a, magenta START tracklet b
PANEL_IN, ROW_IN = 1.65, 2.15                  # panel width and row height (inches)

# ============================================================
# LINK DIFFERENCE
# ============================================================


def linked_pairs(links: pa.Table) -> set[tuple[str, str]]:
    """``(tracklet_a, tracklet_b)`` of the rows with ``decision == 'linked'``.

    Parameters
    ----------
    links : pa.Table
        A ``links.parquet`` table (:data:`gap_bridging.tables.LINKS_SCHEMA`).

    Returns
    -------
    set of tuple of str
        The linked pairs.
    """
    t = links.select(['decision', *LINK_KEY]).to_pydict()
    return {(a, b) for d, a, b in zip(t['decision'], t['tracklet_a'], t['tracklet_b'])
            if d == 'linked'}


def _gaps(links: pa.Table) -> dict[tuple[str, str], int]:
    """``gap_frames`` of every enumerated pair of ``links``."""
    t = links.select([*LINK_KEY, 'gap_frames']).to_pydict()
    return {(a, b): int(g) for a, b, g in zip(t['tracklet_a'], t['tracklet_b'], t['gap_frames'])}


def link_diff(current: pa.Table, candidate: pa.Table
              ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Pairs linked by the candidate only (gained) and by the current run only (lost).

    Parameters
    ----------
    current, candidate : pa.Table
        ``links.parquet`` tables of the two runs; refused rows are ignored.

    Returns
    -------
    gained, lost : list of tuple of str
        Pairs sorted by ``(gap_frames, tracklet_a, tracklet_b)``, the gap
        read from the table that links the pair.
    """
    now, new = linked_pairs(current), linked_pairs(candidate)
    gaps_now, gaps_new = _gaps(current), _gaps(candidate)
    gained = sorted(new - now, key=lambda p: (gaps_new[p], *p))
    lost = sorted(now - new, key=lambda p: (gaps_now[p], *p))
    return gained, lost


def sample_links(pairs: list, n: int, seed: int) -> list:
    """Up to ``n`` of ``pairs``: all when ``len(pairs) <= n``, else a seeded draw.

    Parameters
    ----------
    pairs : list
        Pairs in their audit order.
    n : int
        Sample size (``>= 0``).
    seed : int
        Seed of :func:`numpy.random.default_rng`.

    Returns
    -------
    list
        The kept pairs in their original order (draw without replacement).
    """
    if n < 0:
        raise ValueError(f"sample size must be >= 0, got {n}")
    pairs = list(pairs)
    if len(pairs) <= n:
        return pairs
    keep = np.sort(np.random.default_rng(seed).choice(len(pairs), size=n, replace=False))
    return [pairs[i] for i in keep]


# ============================================================
# AUDIT ROWS
# ============================================================


def _accepted(obs: pa.Table) -> pd.DataFrame:
    """Accepted tracklet rows of ``obs`` (shell, shoulder, frame, video) as a DataFrame."""
    names = [c for c in OBS_COLUMNS if c in obs.schema.names and c != 'demo_frame']
    t = obs.select(names).to_pandas()
    keep = t.tracklet_id.notna()
    if 'accepted_for_tracking' in t:
        keep &= t.accepted_for_tracking.fillna(False).astype(bool)
    return t[keep]


def _decisions(links: pa.Table) -> dict[tuple[str, str], tuple[float, str, str]]:
    """``(cost, decision, reason)`` of every enumerated pair of ``links``."""
    t = links.select([*LINK_KEY, 'cost', 'decision', 'reason']).to_pydict()
    return {(a, b): (math.nan if c is None else float(c), d, r)
            for a, b, c, d, r in zip(t['tracklet_a'], t['tracklet_b'], t['cost'], t['decision'],
                                     t['reason'])}


def link_rows(obs: pa.Table, links_current: pa.Table, links_candidate: pa.Table,
              pairs: list[tuple[str, str]], kind: str) -> list[dict]:
    """One :data:`AUDIT_CSV_COLUMNS` row per pair.

    Parameters
    ----------
    obs : pa.Table
        Input observations (``tracklet_id``, ``frame``,
        ``accepted_for_tracking``) of both runs.
    links_current, links_candidate : pa.Table
        ``links.parquet`` tables of the two runs.
    pairs : list of tuple of str
        ``(tracklet_a, tracklet_b)`` pairs, END tracklet first.
    kind : str
        ``'gained'`` or ``'lost'``.

    Returns
    -------
    list of dict
        ``cost_*``/``decision_*``/``reason_*`` from each table (NaN /
        ``'absent'`` / ``'absent'`` when that run did not enumerate the
        pair); ``gap_frames`` from the candidate table, else the current
        one; ``n_frames_*`` count accepted frames; ``end_frame_a`` is the last
        accepted frame of a, ``start_frame_b`` the first of b; ``page`` and
        ``row`` are None until :func:`fig_link_audit` fills them.

    Raises
    ------
    ValueError
        For an unknown ``kind`` or a tracklet absent from ``obs``.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    span = _accepted(obs).groupby('tracklet_id').frame.agg(['min', 'max', 'nunique'])
    now, new = _decisions(links_current), _decisions(links_candidate)
    gaps = {**_gaps(links_current), **_gaps(links_candidate)}
    rows = []
    for a, b in pairs:
        for tid in (a, b):
            if tid not in span.index:
                raise ValueError(f"tracklet {tid} has no accepted row in the observations")
        row = {'kind': kind, 'tracklet_a': a, 'tracklet_b': b,
               'gap_frames': gaps.get((a, b), int(span.at[b, 'min'] - span.at[a, 'max'] - 1)),
               'end_frame_a': int(span.at[a, 'max']), 'start_frame_b': int(span.at[b, 'min']),
               'n_frames_a': int(span.at[a, 'nunique']), 'n_frames_b': int(span.at[b, 'nunique'])}
        for suffix, table in (('current', now), ('candidate', new)):
            cost, decision, reason = table.get((a, b), (math.nan, ABSENT, ABSENT))
            row.update({f'cost_{suffix}': cost, f'decision_{suffix}': decision,
                        f'reason_{suffix}': reason})
        row.update(page=None, row=None)
        rows.append({c: row[c] for c in AUDIT_CSV_COLUMNS})
    return rows


# ============================================================
# RENDERED AUDIT
# ============================================================


def _cost_text(cost: float, decision: str) -> str:
    """Cost of one run for a row title: ``'absent'``, ``'nan'`` or three significant digits."""
    if decision == ABSENT:
        return ABSENT
    return 'nan' if not np.isfinite(cost) else f"{cost:.3g}"


def _page_stem(out: Path) -> str:
    """Stem of the page names: ``out``'s name without a ``.png``/``.pdf`` suffix."""
    out = Path(out)
    return out.stem if out.suffix.lower() in ('.png', '.pdf') else out.name


def _slots(frames: np.ndarray, k: int, end: bool) -> list[int | None]:
    """``k`` frames of a tracklet nearest the gap; None pads a short tracklet away from it."""
    shown = list(frames[-k:]) if end else list(frames[:k])
    pad = [None] * (k - len(shown))
    return pad + shown if end else shown + pad


def fig_link_audit(obs: pa.Table, video_path: Path, rows: list[dict], out: Path,
                   frame_offset: int | None = None, frames_per_row: int = 5,
                   rows_per_page: int = 5) -> list[Path]:
    """Render one strip of video crops per audited link; fill ``page`` and ``row``.

    Parameters
    ----------
    obs : pa.Table
        Input observations holding both tracklets of every row (shell
        ``body_x``/``body_y``, ``shoulder_x``/``shoulder_y``), all of one video.
    video_path : Path
        The video those observations were predicted on.
    rows : list of dict
        :func:`link_rows` output; ``page`` (file name of the page) and
        ``row`` (1-based row on that page) are set in place.
    out : Path
        Output stem; pages are written as ``<stem>_p<k>.png`` (300 dpi).
    frame_offset : int, optional
        ``frame - video frame index``; default
        :func:`gap_bridging.figures._frame_offset` (from ``demo_frame``, else 0).
    frames_per_row : int
        Panels per link, odd and ``>= 3``: ``(frames_per_row - 1) / 2``
        observed frames of each tracklet around one gap-midpoint panel.
    rows_per_page : int
        Links per page.

    Returns
    -------
    list of Path
        The written pages; empty when ``rows`` is empty.

    Notes
    -----
    Crops are centred on the midpoint of a's last and b's first shell
    position, with side ``max(MIN_CROP_PX, 2 d + CROP_PAD_PX)`` (``d`` their
    distance; clipped to the frame), fixed along the row. The END tracklet a
    is drawn cyan and the START tracklet b magenta (shell circle and
    shell-shoulder segment) on every panel where it has an accepted row; the
    midpoint panel also marks a's end and b's start with crosses joined by a
    dashed line. A tracklet with fewer observed frames than its slots (e.g. a
    one-frame tracklet) leaves the slots farthest from the gap as grey
    panels labelled with its frame count; no frame is repeated. The video is
    decoded once, sequentially, for every row.
    """
    if frames_per_row < 3 or frames_per_row % 2 == 0:
        raise ValueError(f"frames_per_row must be odd and >= 3, got {frames_per_row}")
    if rows_per_page < 1:
        raise ValueError(f"rows_per_page must be >= 1, got {rows_per_page}")
    if not rows:
        return []
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from gap_bridging import figures

    offset = figures._frame_offset(obs) if frame_offset is None else int(frame_offset)
    k = (frames_per_row - 1) // 2
    wanted = {r['tracklet_a'] for r in rows} | {r['tracklet_b'] for r in rows}
    acc = _accepted(obs)
    acc = acc[acc.tracklet_id.isin(wanted)].sort_values(['tracklet_id', 'frame'])
    acc = acc.drop_duplicates(['tracklet_id', 'frame'])
    missing = wanted - set(acc.tracklet_id)
    if missing:
        raise ValueError(f"tracklets without accepted rows: {sorted(missing)[:5]}")
    if 'video_id' in acc and acc.video_id.nunique() > 1:
        raise ValueError(f"audited tracklets span several videos: {sorted(acc.video_id.unique())}")
    tracks = {tid: t.set_index('frame') for tid, t in acc.groupby('tracklet_id')}

    cap = cv2.VideoCapture(str(video_path))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        raise FileNotFoundError(f"cannot open video {video_path}")

    layout, requests = [], {}
    for i, row in enumerate(rows):
        a, b = tracks[row['tracklet_a']], tracks[row['tracklet_b']]
        end = a.loc[a.index.max(), ['body_x', 'body_y']].to_numpy(float)
        start = b.loc[b.index.min(), ['body_x', 'body_y']].to_numpy(float)
        size = int(min(max(figures.MIN_CROP_PX,
                           2.0 * np.hypot(*(start - end)) + figures.CROP_PAD_PX), width, height))
        c = (end + start) / 2
        x0 = int(np.clip(c[0] - size / 2, 0, width - size))
        y0 = int(np.clip(c[1] - size / 2, 0, height - size))
        mid = (int(a.index.max()) + int(b.index.min())) // 2
        slots = ([('a', f) for f in _slots(a.index.to_numpy(), k, True)] + [('gap', mid)]
                 + [('b', f) for f in _slots(b.index.to_numpy(), k, False)])
        for j, (_, f) in enumerate(slots):
            if f is not None:
                requests.setdefault(int(f), []).append(((i, j), x0, y0, size))
        layout.append((slots, x0, y0, size, end, start))
    crops = figures._read_crops(Path(video_path), requests, offset)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = _page_stem(out)
    pages = []
    for p, first in enumerate(range(0, len(rows), rows_per_page), 1):
        chunk = range(first, min(first + rows_per_page, len(rows)))
        page = out.with_name(f"{stem}_p{p}.png")
        fig = plt.figure(figsize=(PANEL_IN * frames_per_row + 0.3, ROW_IN * len(chunk) + 0.5),
                         layout='constrained')
        subfigs = np.atleast_1d(fig.subfigures(len(chunk), 1))
        for sub, i in zip(subfigs, chunk):
            row = rows[i]
            slots, x0, y0, size, end, start = layout[i]
            sub.suptitle(f"{row['kind']}  $g={row['gap_frames']}$  a$={row['n_frames_a']}$f  "
                         f"b$={row['n_frames_b']}$f  cost "
                         f"{_cost_text(row['cost_current'], row['decision_current'])}/"
                         f"{_cost_text(row['cost_candidate'], row['decision_candidate'])}",
                         fontsize=11)
            sub.supxlabel(f"a: {row['tracklet_a']}   b: {row['tracklet_b']}\n"
                          f"current {row['decision_current']} ({row['reason_current']}); "
                          f"candidate {row['decision_candidate']} ({row['reason_candidate']})",
                          fontsize=6)
            axes = sub.subplots(1, frames_per_row)
            for j, (ax, (role, f)) in enumerate(zip(axes, slots)):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim(0, size)
                ax.set_ylim(size, 0)
                ax.set_aspect('equal')
                if f is None:
                    n_have = row['n_frames_a' if role == 'a' else 'n_frames_b']
                    ax.set_facecolor('0.85')
                    ax.text(0.5, 0.5, f"{role} has\n{n_have} frame{'s' * (n_have != 1)}",
                            transform=ax.transAxes, ha='center', va='center', fontsize=8)
                    ax.set_title(role, fontsize=9, pad=2)
                    continue
                ax.imshow(crops[(i, j)], extent=(0, size, size, 0))
                for track, color in ((tracks[row['tracklet_a']], END_COLOR),
                                     (tracks[row['tracklet_b']], START_COLOR)):
                    if f not in track.index:
                        continue
                    q = track.loc[f]
                    ax.plot(q.body_x - x0, q.body_y - y0, 'o', mfc='none', mec=color, mew=1.5,
                            ms=9)
                    if np.isfinite(q.shoulder_x) and np.isfinite(q.shoulder_y):
                        ax.plot([q.body_x - x0, q.shoulder_x - x0],
                                [q.body_y - y0, q.shoulder_y - y0], '-', color=color, lw=1.5)
                if role == 'gap':
                    ax.plot([end[0] - x0, start[0] - x0], [end[1] - y0, start[1] - y0], '--',
                            color='white', lw=0.8, alpha=0.8)
                    ax.plot(end[0] - x0, end[1] - y0, 'x', color=END_COLOR, mew=1.5, ms=7)
                    ax.plot(start[0] - x0, start[1] - y0, 'x', color=START_COLOR, mew=1.5, ms=7)
                ax.set_title(f"{role} $f={f}$", fontsize=9, pad=2)
            row['page'], row['row'] = page.name, i - first + 1
        handles = [Line2D([], [], marker='o', mfc='none', mec=END_COLOR, mew=1.5, ms=9,
                          color=END_COLOR, label='a (END tracklet)'),
                   Line2D([], [], marker='o', mfc='none', mec=START_COLOR, mew=1.5, ms=9,
                          color=START_COLOR, label='b (START tracklet)'),
                   Line2D([], [], marker='x', ls='--', color='0.4', mew=1.5,
                          label='a end / b start (gap panel)')]
        fig.legend(handles=handles, loc='outside lower center', ncol=3, frameon=False,
                   fontsize=10)
        fig.savefig(page, dpi=300, bbox_inches='tight')
        plt.close(fig)
        pages.append(page)
    return pages


# ============================================================
# COMMAND
# ============================================================


def _read_links(run: Path) -> pa.Table:
    """``<run>/links.parquet``; FileNotFoundError when absent."""
    path = Path(run) / LINKS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no {LINKS_NAME} in run directory {run}")
    return pq.read_table(path)


def _find_video(observations: Path) -> Path:
    """The single ``*.mp4`` next to an observation file; ValueError otherwise."""
    if not Path(observations).is_file():
        raise ValueError(f"{observations} is not a file; pass the video explicitly")
    found = sorted(Path(observations).parent.glob('*.mp4'))
    if len(found) != 1:
        raise ValueError(f"expected one *.mp4 next to {observations}, found {len(found)}; "
                         f"pass the video explicitly")
    return found[0]


def output_paths(out: Path) -> tuple[Path, list[Path]]:
    """The CSV and the existing pages of the audit stem ``out``."""
    out = Path(out)
    csv_path = out.with_name(f"{out.name}.csv")
    pages = [p for kind in KINDS for p in sorted(out.parent.glob(f"{out.name}_{kind}_p*.png"))]
    return csv_path, pages


def write_rows(path: Path, rows: list[dict]) -> None:
    """Write ``rows`` as a CSV with the :data:`AUDIT_CSV_COLUMNS` header."""
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def audit_command(current_run: Path, candidate_run: Path, observations: Path, out: Path,
                  video: Path | None = None, n: int = 20, seed: int = 0,
                  force: bool = False) -> dict:
    """Audit the links a candidate run gained and lost against the current run.

    Parameters
    ----------
    current_run, candidate_run : Path
        Bridging run directories, each with ``links.parquet``.
    observations : Path
        The input observation parquet both runs were made from (a file, or
        a directory of parts when ``video`` is given).
    out : Path
        Output stem: ``<out>_gained_p<k>.png``, ``<out>_lost_p<k>.png`` and
        ``<out>.csv`` (:data:`AUDIT_CSV_COLUMNS`, gained rows first).
    video : Path, optional
        The video; default the single ``*.mp4`` next to ``observations``.
    n : int
        Links rendered per kind (a seeded sample when more differ).
    seed : int
        Sample seed (used for both kinds).
    force : bool
        Overwrite an earlier audit with the same stem (its pages are
        removed first).

    Returns
    -------
    dict
        ``gained``/``lost`` (total differing links), ``rendered_gained``/
        ``rendered_lost``, ``pages`` (paths) and ``csv`` (path).

    Raises
    ------
    FileNotFoundError
        For a missing run, observation file or video.
    FileExistsError
        When outputs of the stem exist and ``force`` is False.
    ValueError
        When the video cannot be found next to ``observations``.
    """
    current, candidate = _read_links(current_run), _read_links(candidate_run)
    if not Path(observations).exists():
        raise FileNotFoundError(f"observations {observations} not found")
    video = _find_video(observations) if video is None else Path(video)
    if not video.is_file():
        raise FileNotFoundError(f"video {video} not found")
    out = Path(out)
    csv_path, old_pages = output_paths(out)
    existing = ([csv_path] if csv_path.exists() else []) + old_pages
    if existing and not force:
        raise FileExistsError(f"audit outputs exist ({existing[0]}, ...); pass force to overwrite")
    for path in old_pages:
        path.unlink()

    names = [c for c in OBS_COLUMNS if c in pq.read_schema(observations).names] \
        if Path(observations).is_file() else None
    obs = pq.read_table(observations, columns=names)
    gained, lost = link_diff(current, candidate)
    result = {'gained': len(gained), 'lost': len(lost), 'pages': [], 'csv': csv_path}
    rows = []
    for kind, pairs in (('gained', gained), ('lost', lost)):
        kind_rows = link_rows(obs, current, candidate, sample_links(pairs, n, seed), kind)
        result[f'rendered_{kind}'] = len(kind_rows)
        result['pages'] += fig_link_audit(obs, video, kind_rows, out.with_name(f"{out.name}_{kind}"))
        rows += kind_rows
    out.parent.mkdir(parents=True, exist_ok=True)
    write_rows(csv_path, rows)
    return result


def summary_line(result: dict) -> str:
    """One-line report of :func:`audit_command`'s result."""
    return (f"[audit] gained {result['gained']} (rendered {result['rendered_gained']}), "
            f"lost {result['lost']} (rendered {result['rendered_lost']}); "
            f"{len(result['pages'])} page(s); csv {result['csv']}")


# ============================================================
# CLI
# ============================================================


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the ``audit`` options to ``parser``."""
    parser.add_argument('--current', type=Path, required=True,
                        help='current bridging run directory (with links.parquet)')
    parser.add_argument('--candidate', type=Path, required=True,
                        help='candidate bridging run directory (with links.parquet)')
    parser.add_argument('--observations', type=Path, required=True,
                        help='input observation parquet both runs were made from')
    parser.add_argument('--out', type=Path, required=True,
                        help='output stem: <out>_gained_p<k>.png, <out>_lost_p<k>.png, <out>.csv')
    parser.add_argument('--video', type=Path, default=None,
                        help='source video (default: the single *.mp4 next to --observations)')
    parser.add_argument('--n', type=int, default=20, help='links rendered per kind (default 20)')
    parser.add_argument('--seed', type=int, default=0, help='sample seed (default 0)')
    parser.add_argument('--force', action='store_true', help='overwrite an earlier audit')


def run_from_args(args: argparse.Namespace) -> dict:
    """Run :func:`audit_command` from parsed :func:`add_arguments` options."""
    return audit_command(args.current, args.candidate, args.observations, args.out,
                         video=args.video, n=args.n, seed=args.seed, force=args.force)
