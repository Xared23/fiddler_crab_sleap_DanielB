#!/usr/bin/env python3

"""
Benchmark gap bridging on tiled copies of an observation table (v004).

``tile_observations`` places ``n`` copies of a clip side by side on a square
grid, far enough apart that no candidate pair crosses tiles, so the work grows
``n``-fold at the clip's own crowding. ``time_link`` times the linker alone
(endpoints plus per-video linking, as :func:`gap_bridging.pipeline.run` calls
them); ``time_pipeline`` times the whole :func:`gap_bridging.pipeline.run`
(reading, writing and every stage) and keeps the marker's per-stage
``timings`` when the pipeline records them::

    .venv/bin/python scripts/bench_gap_bridging.py --tiles 1,4,16 --repeats 5 \\
        --configs current=configs/tracking/gap_bridging_v001.yaml \\
                  global=configs/tracking/gap_bridging_v004_global.yaml

    # linker only, quick
    .venv/bin/python scripts/bench_gap_bridging.py --link-only --tiles 1,4 --repeats 3

Writes one CSV row per (config, tiles, stage) with the median, minimum and
maximum seconds over the repeats, the host, CPU count and library versions,
and the same table as Markdown.

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
import importlib.metadata as metadata
import math
import os
from pathlib import Path
import platform
import shlex
import socket
import sys
import tempfile
import time
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ----- numerics -----
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

# ----- local modules -----
from gap_bridging import geometry, holdout, pipeline  # noqa: E402
from gap_bridging.config import BridgeConfig, load_config  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_OBSERVATIONS = ROOT / 'demo/crab_cam2_2_middle_300/observations.parquet'
# v007: the pipeline default (learned S1 at p >= 0.5); pass --configs current=...v001.yaml for the hand cost
DEFAULT_CONFIG = ('default', ROOT / 'configs/tracking/gap_bridging_v007.yaml')
DEFAULT_OUT = ROOT / 'logs/bench/gap_bridging_v004.csv'
DEFAULT_MARKDOWN = ROOT / 'logs/bench/gap_bridging_v004.md'
TILE_MARGIN_PX = 400.0           # > R_max at max_gap 60 (< 300 px), so tiles never interact
PITCH_COLUMNS = ('shell_x', 'shell_y')
PITCH_FALLBACK = ('body_x', 'body_y')
# Unique string keys: suffixed f"/tile{k:02d}" on tile k >= 1 (nulls stay null).
SUFFIX_COLUMNS = ('tracklet_id', 'detection_id', 'prediction_detection_id')
# Integer keys offset by k * (max - min + 1) on tile k, so each tile's values lie in a
# disjoint range: instance_index (and its copy source_instance_index) keys a detection
# within (video_id, frame) and must stay unique and >= 0 there (synthetic rows use -1);
# raw_instance_id / source_instance_row / demo_instance_row are per-row keys;
# source_chunk_instance_row keys a row within its chunk file; tracklet_local_id is 1:1
# with tracklet_id (hold-out planting numbers new tracklets from its maximum).
OFFSET_COLUMNS = ('instance_index', 'source_instance_index', 'raw_instance_id',
                  'source_instance_row', 'demo_instance_row', 'source_chunk_instance_row',
                  'tracklet_local_id')
SORT_KEYS = ('video_id', 'frame', 'instance_index')
LINK_STAGE = 'link_only'          # geometry.endpoints + pipeline.link_videos
TOTAL_STAGE = 'total'             # wall clock of pipeline.run (read to write)
MARKER_TOTAL = 'stages_sum'       # the marker's own 'timings.total', renamed
VERSION_PACKAGES = (('numpy', 'numpy'), ('scipy', 'scipy'), ('pyarrow', 'pyarrow'),
                    ('sklearn', 'scikit-learn'))
SUMMARY_KEYS = ('config', 'tile', 'n_rows', 'n_tracklets', 'stage')
CSV_COLUMNS = (SUMMARY_KEYS + ('median_s', 'min_s', 'max_s', 'repeats', 'n_links',
                               'candidate_pairs', 'host', 'cpu_count', 'python')
               + tuple(name for name, _ in VERSION_PACKAGES))
MARKDOWN_COLUMNS = SUMMARY_KEYS + ('median_s', 'min_s', 'max_s', 'repeats', 'n_links',
                                   'candidate_pairs')

# ============================================================
# TILING
# ============================================================


def _extent(obs: pa.Table, column: str, fallback: str) -> float:
    """``max - min`` of a coordinate column (NaN and null skipped; 0 when empty)."""
    name = column if column in obs.schema.names else fallback
    values = obs[name].combine_chunks().fill_null(math.nan).to_numpy(zero_copy_only=False)
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.max() - finite.min()) if finite.size else 0.0


def default_pitch(obs: pa.Table) -> tuple[float, float]:
    """Per-axis Shell extent of ``obs`` plus :data:`TILE_MARGIN_PX`."""
    return tuple(_extent(obs, c, f) + TILE_MARGIN_PX
                 for c, f in zip(PITCH_COLUMNS, PITCH_FALLBACK))


def _shift(column: pa.ChunkedArray, delta: float) -> pa.Array:
    """Float column plus ``delta`` (nulls stay null)."""
    array = column.combine_chunks()
    return pc.add(array, pa.scalar(float(delta), array.type)) if delta else array


def _offset(column: pa.ChunkedArray, k: int) -> pa.Array:
    """Integer column plus ``k * (max - min + 1)``; overflow of the type is refused."""
    array = column.combine_chunks()
    lo, hi = pc.min_max(array).values()
    if k == 0 or not lo.is_valid:
        return array
    stride = int(hi.as_py()) - int(lo.as_py()) + 1
    top = int(hi.as_py()) + k * stride
    if top > np.iinfo(array.type.to_pandas_dtype()).max:
        raise OverflowError(f"tile offset {k * stride} overflows {array.type}")
    return pc.add(array, pa.scalar(k * stride, array.type))


def _suffix(column: pa.ChunkedArray, k: int) -> pa.Array:
    """String column plus ``f"/tile{k:02d}"`` (nulls stay null)."""
    array = column.combine_chunks()
    return pc.binary_join_element_wise(array, pa.scalar(f"/tile{k:02d}", array.type), '')


def tile_observations(obs: pa.Table, n: int,
                      pitch_px: tuple[float, float] | None = None) -> pa.Table:
    """``n`` non-interacting copies of an observation table on a square grid.

    Parameters
    ----------
    obs : pa.Table
        Observation table (published schema; extra columns copied).
    n : int
        Number of copies (>= 1). Copy ``k`` sits at grid cell
        ``(i, j) = (k mod s, k div s)``, ``s = ceil(sqrt(n))``.
    pitch_px : (float, float), optional
        Grid pitch in x and y; default the Shell extent of ``obs`` per axis
        plus :data:`TILE_MARGIN_PX`, beyond any gate radius.

    Returns
    -------
    pa.Table
        ``obs`` itself when ``n == 1``. Otherwise ``n * obs.num_rows`` rows,
        same schema, sorted by :data:`SORT_KEYS` (stable). In copy ``k``
        every coordinate column (:func:`holdout.coordinate_columns`) is
        shifted by ``(i * pitch_x, j * pitch_y)``; for ``k >= 1`` the
        :data:`SUFFIX_COLUMNS` get ``f"/tile{k:02d}"`` and the
        :data:`OFFSET_COLUMNS` ``k * (max - min + 1)``. Video, frame, time
        and every other column are copied unchanged, so all copies share each
        frame and video.
    """
    if n < 1 or int(n) != n:
        raise ValueError(f"n must be a positive integer, got {n}")
    if n == 1:
        return obs
    pitch = default_pitch(obs) if pitch_px is None else tuple(float(p) for p in pitch_px)
    if len(pitch) != 2 or not all(math.isfinite(p) for p in pitch):
        raise ValueError(f"pitch must be two finite numbers, got {pitch_px}")
    side = math.ceil(math.sqrt(n))
    coords = set(holdout.coordinate_columns(obs.schema))
    suffix = {c for c in SUFFIX_COLUMNS if c in obs.schema.names
              and pa.types.is_string(obs.schema.field(c).type)}
    offset = {c for c in OFFSET_COLUMNS if c in obs.schema.names
              and pa.types.is_integer(obs.schema.field(c).type)}
    copies = []
    for k in range(n):
        i, j = k % side, k // side
        arrays = []
        for field in obs.schema:
            column = obs[field.name]
            if field.name in coords:
                delta = i * pitch[0] if field.name.endswith('_x') else j * pitch[1]
                arrays.append(_shift(column, delta))
            elif k and field.name in suffix:
                arrays.append(_suffix(column, k))
            elif field.name in offset:
                arrays.append(_offset(column, k))
            else:
                arrays.append(column.combine_chunks())
        copies.append(pa.Table.from_arrays(arrays, schema=obs.schema))
    tiled = pa.concat_tables(copies)
    keys = [(c, 'ascending') for c in SORT_KEYS if c in obs.schema.names]
    if keys:
        tiled = tiled.take(pc.sort_indices(tiled, sort_keys=keys))
    assert tiled.num_rows == n * obs.num_rows
    return tiled


def n_tracklets(obs: pa.Table) -> int:
    """Distinct non-null ``tracklet_id`` values."""
    return len(pc.unique(obs['tracklet_id'].drop_null()))


# ============================================================
# TIMING
# ============================================================


def time_link(obs: pa.Table, cfg: BridgeConfig, repeats: int,
              stats: dict | None = None) -> list[float]:
    """Seconds of ``geometry.endpoints`` plus ``pipeline.link_videos``, per repeat.

    The calls are those of :func:`gap_bridging.pipeline.run` (videos sorted,
    nulls dropped). ``stats``, when given, is replaced by ``n_links`` and
    ``candidate_pairs`` (enumerated audit rows) of the last repeat.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    videos = sorted(set(obs['video_id'].unique().to_pylist()) - {None})
    seconds = []
    for _ in range(repeats):
        started = time.perf_counter()
        endpoints = geometry.endpoints(obs, cfg)
        pair_stats: dict = {}
        links, _ = pipeline.link_videos(endpoints, videos, cfg, stats=pair_stats)
        seconds.append(time.perf_counter() - started)
    if stats is not None:
        stats.clear()
        stats.update({'n_links': len(links),
                      'candidate_pairs': int(pair_stats.get('enumerated', 0))})
    return seconds


def _marker_timings(marker: dict) -> dict[str, float]:
    """Numeric ``marker['timings']`` entries (empty when absent); ``total`` renamed."""
    timings = marker.get('timings') or {}
    out = {}
    if isinstance(timings, dict):
        for stage, value in timings.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[MARKER_TOTAL if stage == TOTAL_STAGE else str(stage)] = float(value)
    return out


def time_pipeline(obs_path: Path, out: Path, cfg: BridgeConfig, repeats: int) -> list[dict]:
    """Time :func:`gap_bridging.pipeline.run` (``force=True``) per repeat.

    Returns
    -------
    list of dict
        Per repeat: ``total_s`` (perf_counter around the call), ``n_links``,
        ``candidate_pairs`` (``counts.candidate_pairs.enumerated``) and
        ``timings`` (the marker's per-stage seconds, its ``total`` renamed
        :data:`MARKER_TOTAL`; empty when the marker has no ``timings``).
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    rows = []
    for _ in range(repeats):
        started = time.perf_counter()
        marker = pipeline.run(obs_path, out, cfg, force=True)
        total = time.perf_counter() - started
        counts = marker.get('counts', {})
        rows.append({'total_s': total, 'n_links': int(counts.get('links', 0)),
                     'candidate_pairs': int(counts.get('candidate_pairs', {})
                                            .get('enumerated', 0)),
                     'timings': _marker_timings(marker)})
    return rows


# ============================================================
# SUMMARY AND OUTPUT
# ============================================================


def summarise(rows: Sequence[dict]) -> list[dict]:
    """Median, minimum and maximum seconds per ``(config, tile, stage)``.

    Parameters
    ----------
    rows : sequence of dict
        One per measurement: :data:`SUMMARY_KEYS`, ``seconds``, ``n_links``
        and ``candidate_pairs``.

    Returns
    -------
    list of dict
        One per group, in first-seen order: :data:`SUMMARY_KEYS`,
        ``median_s``, ``min_s``, ``max_s``, ``repeats`` and the last
        ``n_links`` / ``candidate_pairs`` of the group.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row['config'], row['tile'], row['stage']), []).append(row)
    out = []
    for members in groups.values():
        seconds = np.array([m['seconds'] for m in members], dtype=np.float64)
        last = members[-1]
        out.append({**{k: last[k] for k in SUMMARY_KEYS},
                    'median_s': float(np.median(seconds)), 'min_s': float(seconds.min()),
                    'max_s': float(seconds.max()), 'repeats': len(members),
                    'n_links': last['n_links'], 'candidate_pairs': last['candidate_pairs']})
    return out


def environment() -> dict:
    """Host, usable CPU count and the Python / library versions ('' if not installed)."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 0
    env = {'host': socket.gethostname(), 'cpu_count': cpus,
           'python': platform.python_version()}
    for column, dist in VERSION_PACKAGES:
        try:
            env[column] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            env[column] = ''
    return env


def write_csv(rows: Sequence[dict], path: Path) -> None:
    """Write summary rows (:data:`CSV_COLUMNS`; environment columns filled when missing)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    env = environment()
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction='raise')
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, env.get(c, '')) for c in CSV_COLUMNS})


def markdown_table(rows: Sequence[dict]) -> str:
    """Summary rows as a Markdown table (:data:`MARKDOWN_COLUMNS`; seconds to 4 places)."""
    lines = ['| ' + ' | '.join(MARKDOWN_COLUMNS) + ' |',
             '| ' + ' | '.join('---' if c in ('config', 'stage') else '---:'
                               for c in MARKDOWN_COLUMNS) + ' |']
    for row in rows:
        cells = [f"{row[c]:.4f}" if c.endswith('_s') else str(row[c]) for c in MARKDOWN_COLUMNS]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


# ============================================================
# CLI
# ============================================================


def _int_list(text: str) -> list[int]:
    """``"1,4,16"`` -> ``[1, 4, 16]`` (positive integers; argparse type)."""
    try:
        values = [int(v) for v in text.split(',') if v.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}") \
            from error
    if not values or any(v < 1 for v in values):
        raise argparse.ArgumentTypeError(f"expected positive integers, got {text!r}")
    return values


def _config_spec(text: str) -> tuple[str, Path]:
    """``name=path.yaml`` (argparse type); a relative path falls back to the repo root."""
    name, sep, path = text.partition('=')
    if not sep or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError(f"expected name=YAML, got {text!r}")
    resolved = Path(path)
    if not resolved.exists() and not resolved.is_absolute() and (ROOT / resolved).exists():
        resolved = ROOT / resolved
    return name.strip(), resolved


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip(),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--observations', type=Path, default=DEFAULT_OBSERVATIONS,
                        help='observation parquet file (default: the demo clip)')
    parser.add_argument('--tiles', type=_int_list, default=[1, 4, 16],
                        help='comma-separated tile counts (default 1,4,16)')
    parser.add_argument('--repeats', type=int, default=5, help='timed repeats per run')
    parser.add_argument('--max-gap', type=int, default=30,
                        help='max_gap set on every config (default 30)')
    parser.add_argument('--configs', type=_config_spec, nargs='+', default=None,
                        help='name=YAML ... (default current=configs/tracking/'
                             'gap_bridging_v001.yaml)')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT, help='summary CSV')
    parser.add_argument('--markdown', type=Path, default=DEFAULT_MARKDOWN,
                        help='summary Markdown table')
    parser.add_argument('--link-only', action='store_true',
                        help='time endpoints + linking only (no pipeline.run)')
    parser.add_argument('--workdir', type=Path, default=None,
                        help='tiled inputs and pipeline outputs (default: a temporary '
                             'directory, removed afterwards)')
    return parser


def _bench(args: argparse.Namespace, workdir: Path) -> list[dict]:
    """Every measurement of the run (rows for :func:`summarise`)."""
    configs = args.configs or [DEFAULT_CONFIG]
    names = [name for name, _ in configs]
    if len(set(names)) != len(names):
        raise ValueError(f"config names repeat: {names}")
    obs = pipeline.read_observations(pipeline.input_files(args.observations))
    measurements = []
    for tile in args.tiles:
        tiled = tile_observations(obs, tile)
        base = {'tile': tile, 'n_rows': tiled.num_rows, 'n_tracklets': n_tracklets(tiled)}
        path = workdir / f'tiles_{tile:02d}.parquet'
        if not args.link_only:
            pq.write_table(tiled, path)
        for name, yaml in configs:
            cfg = load_config(yaml, max_gap=args.max_gap)
            row = {**base, 'config': name}
            if args.link_only:
                stats: dict = {}
                seconds = time_link(tiled, cfg, args.repeats, stats)
                measurements += [{**row, **stats, 'stage': LINK_STAGE, 'seconds': s}
                                 for s in seconds]
                print(f"[bench] {name} x{tile}: {tiled.num_rows} rows, {stats['n_links']} "
                      f"links, link median {np.median(seconds):.3f}s", flush=True)
                continue
            runs = time_pipeline(path, workdir / f'run_{name}_{tile:02d}', cfg, args.repeats)
            for run in runs:
                extra = {**row, 'n_links': run['n_links'],
                         'candidate_pairs': run['candidate_pairs']}
                measurements.append({**extra, 'stage': TOTAL_STAGE, 'seconds': run['total_s']})
                measurements += [{**extra, 'stage': stage, 'seconds': value}
                                 for stage, value in run['timings'].items()]
            print(f"[bench] {name} x{tile}: {tiled.num_rows} rows, {runs[-1]['n_links']} "
                  f"links, pipeline median {np.median([r['total_s'] for r in runs]):.3f}s"
                  + ("" if runs[-1]['timings'] else " (no stage timings in marker)"),
                  flush=True)
    return measurements


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and write the CSV and Markdown summaries.

    Returns
    -------
    int
        0 on success, 1 on an input or configuration error.
    """
    args = build_parser().parse_args(argv)
    if args.repeats < 1:
        print("bench_gap_bridging: error: --repeats must be >= 1", file=sys.stderr)
        return 1
    try:
        if args.workdir is None:
            with tempfile.TemporaryDirectory(prefix='bench_gap_bridging_') as tmp:
                measurements = _bench(args, Path(tmp))
        else:
            args.workdir.mkdir(parents=True, exist_ok=True)
            measurements = _bench(args, args.workdir)
    except (FileNotFoundError, ValueError) as error:
        print(f"bench_gap_bridging: error: {error}", file=sys.stderr, flush=True)
        return 1
    env = environment()
    rows = [{**row, **env} for row in summarise(measurements)]
    write_csv(rows, args.out)
    command = shlex.join(['scripts/bench_gap_bridging.py']
                         + (sys.argv[1:] if argv is None else list(argv)))
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(
        f"Gap bridging benchmark: {env['host']}, {env['cpu_count']} CPUs, Python "
        f"{env['python']}, numpy {env['numpy']}, scipy {env['scipy']}, pyarrow "
        f"{env['pyarrow']}, scikit-learn {env['sklearn'] or 'n/a'}; input "
        f"`{args.observations}`, max_gap {args.max_gap}, "
        f"{'link only' if args.link_only else 'full pipeline'}.\n\n"
        f"```bash\n{command}\n```\n\n" + markdown_table(rows))
    print(f"[bench] wrote {args.out} and {args.markdown}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
