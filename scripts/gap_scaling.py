#!/usr/bin/env python3

"""
Scale gap bridging from the 300-frame demo to the 2,000-frame demo (v007).

The study cuts nested frame windows out of the 2,000-frame demo, all centred
on the 300-frame demo (whose frames are row-identical to the same frames of
the larger set), runs the hold-out sweeps and ``compare`` of
``scripts/bridge_gaps.py`` on each window, and collects how three linkers
behave as the data grow: the v003 hand cost, the v006 S1 learned scorer at
its pipeline threshold and the v005 S1 scorer at p >= 0.5, each with the
committed (frozen) model and with a model retrained on the window's own
sweeps. Subcommands::

    # nested windows plus their planting counts (constant plant density)
    gap_scaling.py windows --source demo/crab_cam2_2_middle_2000/observations.parquet \\
        --out logs/gap_scaling_v007/windows --sizes 300,600,1000,2000

    # learned-cost overlay for a scorer at one probability (threshold = dummy)
    gap_scaling.py overlay --scorer configs/tracking/link_scorer_v005_s1.joblib \\
        --probability 0.5 --out logs/gap_scaling_v007/configs/s1p50_frozen.yaml

    # natural-link outcomes of the max_gap runs of every window and variant
    gap_scaling.py natural --root logs/gap_scaling_v007 --sizes 300,600,1000,2000

    # wall time, per-stage timings and peak RSS of one bridging run, repeated
    gap_scaling.py measure --observations <window parquet> --config <yaml> \\
        --label w300:current --out logs/gap_scaling_v007/measure/w300_current.csv

    # tables, figures and the Markdown report
    gap_scaling.py report --root logs/gap_scaling_v007 --sizes 300,600,1000,2000 \\
        --docs docs/gap_bridging_v007_scaling.md --figures docs/figures

Sam
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ----- numerics and tables -----
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# ============================================================
# CONSTANTS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
ANCHOR_START = 14250          # first original frame of the 300-frame demo
ANCHOR_LEN = 300              # frames of the 300-frame demo
BASE_SIZE = 300               # window whose planting counts are the v005 recipe
BASE_PER_GAP = 8              # v005/v006 sweeps: --per-gap 8 --decoys 4 --flicker 2
BASE_DECOYS = 4
BASE_FLICKER = 2
HORIZON = 30                  # default max_gap and the promotion horizon
THROUGH_GAP = 30
HOLDOUT_GAPS: Tuple[int, ...] = (1, 2, 3, 5, 10, 15, 20, 30, 45, 60)
SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
VARIANTS: Tuple[Tuple[str, str], ...] = (
    ('current', 'v003 hand cost'),
    ('s1_frozen', 'v006 S1, frozen'),
    ('s1p50_frozen', 'v005 S1 p>=0.5, frozen'),
    ('s1_rt', 'v006 S1, retrained'),
    ('s1p50_rt', 'v005 S1 p>=0.5, retrained'),
)
SCHEMES: Tuple[str, ...] = ('current', 's1', 's1p50')
TIME_PATTERNS = {
    'wall_s': re.compile(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)'),
    'max_rss_kb': re.compile(r'Maximum resident set size \(kbytes\): (\d+)'),
    'user_s': re.compile(r'User time \(seconds\): ([\d.]+)'),
    'system_s': re.compile(r'System time \(seconds\): ([\d.]+)'),
}
MEASURE_FIELDS = ('label', 'repeat', 'wall_s', 'user_s', 'system_s', 'max_rss_kb',
                  'links', 'n_observed', 'tracklets', 'stage_read', 'stage_endpoints',
                  'stage_link', 'stage_chain', 'stage_tails', 'stage_fill',
                  'stage_assemble', 'stage_total')

# ============================================================
# WINDOWS
# ============================================================


def window_bounds(size: int, lo: int, hi: int, anchor_start: int = ANCHOR_START,
                  anchor_len: int = ANCHOR_LEN) -> Tuple[int, int]:
    """First and last original frame of a ``size``-frame window around the anchor.

    Parameters
    ----------
    size : int
        Window length in frames.
    lo, hi : int
        First and last frame available in the source (inclusive).
    anchor_start, anchor_len : int
        The anchor interval (the 300-frame demo) every window must contain.

    Returns
    -------
    Tuple[int, int]
        ``(start, last)`` inclusive. The window is centred on the anchor and
        shifted, never shrunk, to stay inside ``[lo, hi]``.
    """
    if hi < lo or anchor_len < 1:
        raise ValueError(f"bad source range [{lo}, {hi}] or anchor length {anchor_len}")
    if not anchor_len <= size <= hi - lo + 1:
        raise ValueError(f"window size {size} must lie in [{anchor_len}, {hi - lo + 1}]")
    if anchor_start < lo or anchor_start + anchor_len - 1 > hi:
        raise ValueError(f"anchor [{anchor_start}, +{anchor_len}) lies outside [{lo}, {hi}]")
    start = anchor_start + (anchor_len - size) // 2
    start = min(max(start, lo), hi - size + 1)
    last = start + size - 1
    assert start <= anchor_start and last >= anchor_start + anchor_len - 1
    return start, last


def planting(per_gap: int) -> Dict[str, int]:
    """Plants, decoys and flicker singletons per planted gap at the v005 ratios.

    Decoys and flicker keep the recipe's 4 : 8 and 2 : 8 ratios to the plants
    (rounded down, at least 1), so ``planting(8)`` is the v005/v006 recipe.
    """
    if per_gap < 1:
        raise ValueError(f"per_gap must be positive, got {per_gap}")
    out = {'per_gap': int(per_gap),
           'decoys': max(1, per_gap * BASE_DECOYS // BASE_PER_GAP),
           'flicker': max(1, per_gap * BASE_FLICKER // BASE_PER_GAP)}
    assert out['decoys'] <= out['per_gap'] and out['flicker'] <= out['per_gap']
    return out


def density_target(size: int, base_size: int = BASE_SIZE) -> int:
    """Plants per gap that keep the v005 plant density (8 per 300 frames)."""
    if size < 1 or base_size < 1:
        raise ValueError(f"sizes must be positive, got {size} and {base_size}")
    target = max(1, round(BASE_PER_GAP * size / base_size))
    assert target >= 1
    return target


def max_feasible_per_gap(observations: Path, target: int, seeds: Sequence[int],
                         gap_lengths: Sequence[int] = HOLDOUT_GAPS) -> int:
    """Largest plants-per-gap <= ``target`` that the short profile can plant for every seed.

    Parameters
    ----------
    observations : Path
        Observation parquet of one window.
    target : int
        Upper bound (the constant-density target).
    seeds : Sequence[int]
        Hold-out seeds that must all succeed.
    gap_lengths : Sequence[int]
        Planted gap lengths (the sweep defaults).

    Returns
    -------
    int
        The count, or 0 when even one plant per gap fails. Planting raises
        ValueError when it runs out of eligible tracklets; that is the probe.
    """
    if target < 1 or not seeds:
        raise ValueError("target must be positive and seeds non-empty")
    sys.path.insert(0, str(ROOT / 'scripts'))
    from gap_bridging import holdout, pipeline
    from gap_bridging.config import BridgeConfig
    obs = pipeline.read_observations(pipeline.input_files(Path(observations)))
    for per_gap in range(target, 0, -1):
        plan = planting(per_gap)
        try:
            for seed in seeds:
                holdout.plant_gaps(obs, gap_lengths, per_gap, seed=seed,
                                   decoys=plan['decoys'], flicker=plan['flicker'],
                                   cue_config=BridgeConfig(), profile='short')
        except ValueError:
            continue
        return per_gap
    return 0


def _sha256(path: Path) -> str:
    """Hex SHA-256 of a file, read in 16 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def cut_window(table: pa.Table, start: int, last: int) -> pa.Table:
    """Rows of ``table`` whose original ``frame`` lies in ``[start, last]``, in order."""
    if 'frame' not in table.column_names:
        raise ValueError("observation table has no 'frame' column")
    if last < start:
        raise ValueError(f"empty window [{start}, {last}]")
    mask = pc.and_(pc.greater_equal(table['frame'], start),
                   pc.less_equal(table['frame'], last))
    return table.filter(mask)


def make_windows(source: Path, out: Path, sizes: Sequence[int],
                 force: bool = False, probe: bool = True) -> List[Dict]:
    """Write ``w<size>/observations.parquet`` and ``window.json`` per window.

    Parameters
    ----------
    source : Path
        The 2,000-frame observation table.
    out : Path
        Directory receiving one ``w<size>`` folder per window.
    sizes : Sequence[int]
        Window lengths in frames.
    force : bool
        Overwrite existing windows.
    probe : bool
        Cap each window's plants at :func:`max_feasible_per_gap` (seeds
        :data:`SEEDS`); without it the constant-density target is recorded.

    Returns
    -------
    List[Dict]
        The ``window.json`` records, in ``sizes`` order.
    """
    if not sizes:
        raise ValueError("no window sizes given")
    table = pq.read_table(source)
    frames = table['frame']
    lo, hi = int(pc.min(frames).as_py()), int(pc.max(frames).as_py())
    records = []
    for size in sizes:
        start, last = window_bounds(int(size), lo, hi)
        folder = Path(out) / f"w{size}"
        if (folder / 'window.json').exists() and not force:
            raise FileExistsError(f"{folder} exists; pass --force to overwrite")
        folder.mkdir(parents=True, exist_ok=True)
        part = cut_window(table, start, last)
        pq.write_table(part, folder / 'observations.parquet')
        n_frames = len(set(part['frame'].to_pylist()))
        record = {'size': int(size), 'start': start, 'last': last, 'frames': n_frames,
                  'rows': part.num_rows,
                  'tracklets': int(pc.count_distinct(part['tracklet_id']).as_py()),
                  'source': str(source), 'source_sha256': _sha256(Path(source)),
                  'observations_sha256': _sha256(folder / 'observations.parquet'),
                  'density_target': density_target(int(size))}
        feasible = (max_feasible_per_gap(folder / 'observations.parquet',
                                         record['density_target'], SEEDS)
                    if probe else record['density_target'])
        if feasible < 1:
            raise ValueError(f"window {size}: no plant per gap is feasible")
        record.update({'feasible_probe': probe, **planting(feasible)})
        assert n_frames == size, f"window {size} holds {n_frames} distinct frames"
        (folder / 'window.json').write_text(json.dumps(record, indent=1) + '\n')
        records.append(record)
    return records


# ============================================================
# OVERLAYS
# ============================================================


def overlay_text(scorer: Path, probability: float, relative_to: Optional[Path] = None) -> str:
    """YAML learned-cost overlay with ``learned_min_probability`` = dummy = ``probability``.

    ``scorer_path`` is written relative to ``relative_to`` (the overlay's
    directory) when given and the scorer lies under it, else absolute.
    """
    if not 0.0 < probability < 1.0:
        raise ValueError(f"probability must lie in (0, 1), got {probability}")
    scorer = Path(scorer).resolve()
    if not scorer.is_file():
        raise FileNotFoundError(f"scorer {scorer} is missing")
    path = str(scorer)
    if relative_to is not None and scorer.parent == Path(relative_to).resolve():
        path = scorer.name
    return (f"# Written by gap_scaling.py overlay: learned link scorer at p >= {probability}"
            f" (unmatched dummy tied); scorer_path is relative to this file when bare.\n"
            f"cost_model: learned\nscorer_path: {path}\n"
            f"learned_min_probability: {probability!r}\n"
            f"learned_unmatched_probability: {probability!r}\n")


# ============================================================
# NATURAL LINKS
# ============================================================


def linked_pairs(links: pa.Table) -> set:
    """``(tracklet_a, tracklet_b)`` of the rows whose decision is ``linked``."""
    for name in ('tracklet_a', 'tracklet_b', 'decision'):
        if name not in links.column_names:
            raise ValueError(f"links table has no {name!r} column")
    kept = links.filter(pc.equal(links['decision'], 'linked'))
    return set(zip(kept['tracklet_a'].to_pylist(), kept['tracklet_b'].to_pylist()))


def agreement(a: set, b: set) -> Dict[str, float]:
    """Shared and exclusive link counts and the Jaccard index of two link sets."""
    both, only_a, only_b = len(a & b), len(a - b), len(b - a)
    union = both + only_a + only_b
    out = {'both': both, 'only_a': only_a, 'only_b': only_b,
           'jaccard': both / union if union else float('nan')}
    assert both + only_a == len(a) and both + only_b == len(b)
    return out


def length_summary(lengths: np.ndarray, window: int) -> Dict[str, float]:
    """Count, mean, quantiles and long-unit fractions of frame lengths.

    ``frac_frames_ge_<n>`` is the share of all unit-frames that sit in units
    of at least ``n`` frames; ``frac_full`` is the share of units spanning the
    whole window.
    """
    lengths = np.asarray(lengths, dtype=float)
    if lengths.ndim != 1 or window < 1:
        raise ValueError("lengths must be 1-D and the window positive")
    if lengths.size == 0:
        return {'n': 0, 'mean': math.nan, 'median': math.nan, 'p10': math.nan,
                'p90': math.nan, 'frac_frames_ge_30': math.nan,
                'frac_frames_ge_150': math.nan, 'frac_full': math.nan}
    if np.any(lengths < 1) or np.any(lengths > window):
        raise ValueError(f"lengths must lie in [1, {window}]")
    total = lengths.sum()
    return {'n': int(lengths.size), 'mean': float(lengths.mean()),
            'median': float(np.median(lengths)),
            'p10': float(np.quantile(lengths, 0.1)), 'p90': float(np.quantile(lengths, 0.9)),
            'frac_frames_ge_30': float(lengths[lengths >= 30].sum() / total),
            'frac_frames_ge_150': float(lengths[lengths >= 150].sum() / total),
            'frac_full': float(np.mean(lengths >= window))}


def tracklet_lengths(observations: pa.Table) -> np.ndarray:
    """Frame span (last - first + 1) of every accepted tracklet in a window."""
    acc = observations.filter(pc.and_(pc.is_valid(observations['tracklet_id']),
                                      pc.equal(observations['accepted_for_tracking'], True)))
    if acc.num_rows == 0:
        return np.zeros(0)
    grouped = acc.group_by('tracklet_id').aggregate([('frame', 'min'), ('frame', 'max')])
    spans = (grouped['frame_max'].to_numpy() - grouped['frame_min'].to_numpy() + 1)
    assert np.all(spans >= 1)
    return spans.astype(float)


def trajectory_lengths(trajectories: pa.Table) -> np.ndarray:
    """Frame span of every trajectory of a bridged run."""
    for name in ('start_frame', 'end_frame'):
        if name not in trajectories.column_names:
            raise ValueError(f"trajectories table has no {name!r} column")
    spans = trajectories['end_frame'].to_numpy() - trajectories['start_frame'].to_numpy() + 1
    assert np.all(spans >= 1)
    return spans.astype(float)


def _summary_row(sweep: Path, horizon: int) -> Dict[str, str]:
    """The ``sweep_summary.csv`` row of one horizon."""
    with open(Path(sweep) / 'sweep_summary.csv', newline='') as handle:
        for row in csv.DictReader(handle):
            if int(row['max_gap']) == horizon:
                return row
    raise ValueError(f"{sweep}/sweep_summary.csv has no row for max_gap {horizon}")


def _seed_dirs(root: Path, size: int, variant: str) -> List[Path]:
    """Seed sweep directories (``seed_<int>/``) of one window and variant, in seed order."""
    found = [p for p in (Path(root) / 'sweeps' / f"w{size}" / variant).glob('seed_*')
             if p.is_dir() and p.name[len('seed_'):].isdigit()]
    return sorted(found, key=lambda p: int(p.name[len('seed_'):]))


def natural_window(root: Path, size: int, horizon: int = HORIZON) -> Tuple[List[Dict], List[Dict],
                                                                        Dict[str, np.ndarray]]:
    """Natural-link rows, agreement-with-current rows and length arrays of one window.

    Parameters
    ----------
    root : Path
        Study root holding ``windows/`` and ``sweeps/``.
    size : int
        Window length.
    horizon : int
        The ``max_gap`` run to read (seed 0; the unplanted run does not
        depend on the seed, which is checked on the link counts).

    Returns
    -------
    Tuple[List[Dict], List[Dict], Dict[str, np.ndarray]]
        Per-variant summary rows, per-variant agreement rows against
        ``current`` and the trajectory-length arrays by variant (plus the
        input tracklet lengths under ``'tracklets'``).
    """
    obs = pq.read_table(Path(root) / 'windows' / f"w{size}" / 'observations.parquet')
    lengths = {'tracklets': tracklet_lengths(obs)}
    rows, agree, link_sets = [], [], {}
    for variant, _ in VARIANTS:
        seeds = _seed_dirs(root, size, variant)
        if not seeds:
            continue
        counts = {int(_summary_row(s, horizon)['links']) for s in seeds}
        if len(counts) != 1:
            raise ValueError(f"w{size} {variant}: unplanted link counts differ by seed {counts}")
        run = seeds[0] / f"max_gap_{horizon}"
        link_sets[variant] = linked_pairs(pq.read_table(run / 'links.parquet'))
        lengths[variant] = trajectory_lengths(pq.read_table(run / 'trajectories.parquet'))
        summary = _summary_row(seeds[0], horizon)
        row = {'size': size, 'variant': variant, 'links': counts.pop(),
               'links_per_kframe': 1000.0 * len(link_sets[variant]) / size,
               'identity_starts': int(summary['identity_starts']),
               'identity_starts_filtered': int(summary['identity_starts_filtered']),
               'spurious': int(summary['spurious'])}
        row.update({f"traj_{k}": v for k, v in length_summary(lengths[variant], size).items()})
        assert row['links'] == len(link_sets[variant]), f"w{size} {variant}: link count mismatch"
        rows.append(row)
    for variant in link_sets:
        if variant != 'current' and 'current' in link_sets:
            agree.append({'size': size, 'variant': variant,
                          **agreement(link_sets[variant], link_sets['current'])})
    return rows, agree, lengths


def _write_csv(rows: List[Dict], path: Path) -> None:
    """Write dict rows (union of keys, first-seen order) as CSV."""
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields: List[str] = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mk(path: Path) -> Path:
    """Create ``path`` (and parents) and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def natural_command(root: Path, sizes: Sequence[int], horizon: int = HORIZON) -> None:
    """Write ``natural/{summary,agreement,tracklets}.csv`` and the length arrays."""
    rows, agree, out = [], [], _mk(Path(root) / 'natural')
    tracklet_rows = []
    for size in sizes:
        r, a, lengths = natural_window(root, int(size), horizon)
        rows += r
        agree += a
        tracklet_rows.append({'size': size, **{f"tracklet_{k}": v for k, v in
                                               length_summary(lengths['tracklets'],
                                                              int(size)).items()}})
        np.savez(out / f"lengths_w{size}.npz", **lengths)
    _write_csv(rows, out / 'summary.csv')
    _write_csv(agree, out / 'agreement.csv')
    _write_csv(tracklet_rows, out / 'tracklets.csv')
    print(f"[natural] wrote {out}/summary.csv, agreement.csv, tracklets.csv", flush=True)


# ============================================================
# RUNTIME AND MEMORY
# ============================================================


def _clock_seconds(text: str) -> float:
    """Seconds of a ``[h:]mm:ss[.ss]`` clock string from GNU time."""
    parts = text.split(':')
    if not 2 <= len(parts) <= 3:
        raise ValueError(f"unrecognised clock {text!r}")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60.0 + float(part)
    assert seconds >= 0.0
    return seconds


def parse_gnu_time(text: str) -> Dict[str, float]:
    """Wall, user and system seconds and peak RSS (kB) from ``/usr/bin/time -v`` output."""
    out: Dict[str, float] = {}
    for key, pattern in TIME_PATTERNS.items():
        match = pattern.search(text)
        if match is None:
            raise ValueError(f"GNU time output lacks {key}")
        value = match.group(1)
        out[key] = _clock_seconds(value) if key == 'wall_s' else float(value)
    assert out['max_rss_kb'] > 0
    return out


def measure_run(observations: Path, config: Optional[Path], label: str, repeats: int,
                max_gap: int = HORIZON) -> List[Dict]:
    """Time ``bridge_gaps.py run`` ``repeats`` times under ``/usr/bin/time -v``.

    Parameters
    ----------
    observations : Path
        Observation parquet of one window.
    config : Path, optional
        YAML config (None: the built-in hand-cost defaults).
    label : str
        Row label (``w<size>:<scheme>``).
    repeats : int
        Number of runs; each writes into a fresh temporary directory.
    max_gap : int
        Horizon passed as ``--max-gap``.

    Returns
    -------
    List[Dict]
        One row per repeat with the ``MEASURE_FIELDS``.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be positive, got {repeats}")
    if not Path(observations).is_file():
        raise FileNotFoundError(observations)
    rows = []
    for repeat in range(repeats):
        with tempfile.TemporaryDirectory(prefix='gap_scaling_') as tmp:
            cmd = ['/usr/bin/time', '-v', sys.executable, str(ROOT / 'scripts/bridge_gaps.py'),
                   'run', '--observations', str(observations), '--out', f"{tmp}/run",
                   '--max-gap', str(max_gap)] + (['--config', str(config)] if config else [])
            done = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if done.returncode != 0:
                raise RuntimeError(f"{label} repeat {repeat} failed:\n{done.stderr[-2000:]}")
            marker = json.loads((Path(tmp) / 'run' / 'GAP_BRIDGING_COMPLETE.json').read_text())
        row = {'label': label, 'repeat': repeat, **parse_gnu_time(done.stderr)}
        row.update(_marker_fields(marker))
        rows.append(row)
    return rows


def _marker_fields(marker: Dict) -> Dict[str, float]:
    """Links, input size and per-stage seconds from a run's completion marker."""
    counts, timings = marker.get('counts', {}), marker.get('timings', {})
    if not timings:
        raise ValueError("completion marker has no timings")
    out = {'links': counts.get('links', counts.get('decision_linked')),
           'n_observed': counts.get('n_observed'),
           'tracklets': counts.get('tracklets')}
    out.update({f"stage_{k}": timings.get(k) for k in
                ('read', 'endpoints', 'link', 'chain', 'tails', 'fill', 'assemble', 'total')})
    return out


def scaling_exponent(sizes: Sequence[float], values: Sequence[float]) -> float:
    """Least-squares slope of log(value) against log(size) (the power-law exponent)."""
    xs, ys = np.asarray(sizes, float), np.asarray(values, float)
    if xs.size < 2 or xs.size != ys.size or not (np.all(xs > 0) and np.all(ys > 0)):
        raise ValueError("need at least two positive, paired points")
    x, y = np.log(xs), np.log(ys)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("sizes and values must be finite")
    if np.ptp(x) == 0.0:
        raise ValueError("sizes must not all be equal")
    slope = float(np.polyfit(x, y, 1)[0])
    assert math.isfinite(slope)
    return slope


# ============================================================
# AUDIT VIDEO
# ============================================================


def concat_video(bundle: Path, out: Path, verify: bool = True) -> Dict:
    """Stream-copy a chunked demo's MP4s into one video and check every decoded frame.

    Parameters
    ----------
    bundle : Path
        Demo folder holding ``chunks.json`` and ``frame_map.csv``.
    out : Path
        Output MP4 (frame index = ``demo_frame``).
    verify : bool
        Decode every frame and compare its BGR SHA-256 with ``frame_map.csv``.

    Returns
    -------
    Dict
        ``frames`` decoded, ``verified`` count and the output path.
    """
    import cv2
    chunks = json.loads((Path(bundle) / 'chunks.json').read_text())['chunks']
    movies = [Path(bundle) / c['video'] if 'video' in c else None for c in chunks]
    if not movies or any(m is None or not m.is_file() for m in movies):
        movies = sorted(Path(bundle).glob('chunk_*/*.mp4'))
    if not movies:
        raise FileNotFoundError(f"no chunk videos under {bundle}")
    out.parent.mkdir(parents=True, exist_ok=True)
    listing = out.with_suffix('.concat.txt')
    listing.write_text(''.join(f"file '{m.resolve()}'\n" for m in movies))
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-f',
                    'concat', '-safe', '0', '-i', str(listing), '-c', 'copy', str(out)],
                   check=True)
    with open(Path(bundle) / 'frame_map.csv', newline='') as handle:
        expected = [r['decoded_bgr_sha256'] for r in csv.DictReader(handle)]
    cap = cv2.VideoCapture(str(out), cv2.CAP_FFMPEG)
    count, verified = 0, 0
    for index in range(len(expected) + 1):
        ok, image = cap.read()
        if not ok:
            break
        count += 1
        if verify and index < len(expected):
            verified += hashlib.sha256(image.tobytes()).hexdigest() == expected[index]
    cap.release()
    if count != len(expected) or (verify and verified != len(expected)):
        raise ValueError(f"{out}: decoded {count} frames, {verified} matching of {len(expected)}")
    return {'video': str(out), 'frames': count, 'verified': verified, 'chunks': len(movies)}


# ============================================================
# REPORT
# ============================================================


def _promotion(root: Path, size: int, fold: str) -> Dict:
    """``promotion.json`` of one window's compare on ``fold``."""
    path = Path(root) / 'compare' / f"w{size}" / fold / 'promotion.json'
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run compare first")
    return json.loads(path.read_text())


def _by_csv(root: Path, size: int, fold: str, name: str) -> List[Dict[str, str]]:
    """Rows of one compare CSV (``comparison_by_gap`` or ``comparison_by_horizon``)."""
    with open(Path(root) / 'compare' / f"w{size}" / fold / f"{name}.csv", newline='') as handle:
        return list(csv.DictReader(handle))


def holdout_rows(root: Path, sizes: Sequence[int], horizon: int = HORIZON,
                 through_gap: int = THROUGH_GAP) -> List[Dict]:
    """Hold-out totals and the promotion rule at a fixed horizon, per window, variant and fold.

    ``compare`` checks the rule at the baseline's chosen horizon, which moves
    between windows; here :func:`gap_bridging.compare.promotion_check` is
    re-applied to compare's own pooled per-gap rows at ``horizon`` for every
    window, with compare's natural-link floor (horizon 30). Compare's chosen
    horizon and verdict are kept as ``compare_horizon`` / ``compare_ok``.
    Rows are written for ``max_gap`` ``horizon`` and 60.
    """
    sys.path.insert(0, str(ROOT / 'scripts'))
    from gap_bridging import compare
    rows = []
    for size in sizes:
        for fold in ('test', 'all'):
            promo = _promotion(root, size, fold)
            by_gap = _by_csv(root, size, fold, 'comparison_by_gap')
            base = [r for r in by_gap if r['variant'] == promo['baseline']]
            for rec in _by_csv(root, size, fold, 'comparison_by_horizon'):
                if int(rec['max_gap']) not in (horizon, 60):
                    continue
                native = promo['variants'][rec['variant']]['promotion']
                mine = compare.promotion_check(
                    base, [r for r in by_gap if r['variant'] == rec['variant']],
                    horizon, through_gap)
                demo_ok = native['demo_links'] >= native['demo_floor']
                failing = mine['failing'] + ([] if demo_ok else [compare.DEMO_FAIL])
                checked = int(rec['max_gap']) == horizon
                rows.append({'size': size, 'fold': fold, 'variant': rec['variant'],
                             'max_gap': int(rec['max_gap']), 'planted': int(rec['planted']),
                             'recovered': int(rec['recovered']),
                             'false_link': int(rec['false_link']),
                             'recall': float(rec['recall']), 'precision': float(rec['precision']),
                             'promotion_ok': (mine['ok'] and demo_ok) if checked else None,
                             'precision_ok': mine['precision_ok'] if checked else None,
                             'recall_ok': mine['recall_ok'] if checked else None,
                             'demo_links': native['demo_links'],
                             'demo_floor': native['demo_floor'],
                             'failing': ' '.join(failing) if checked else None,
                             'compare_horizon': promo['horizon'],
                             'chosen_horizon': promo['variants'][rec['variant']]['chosen'],
                             'compare_ok': native['ok']})
    assert rows, "no compare rows"
    return rows


def model_rows(root: Path, sizes: Sequence[int]) -> List[Dict]:
    """Retrained S1 manifest fields per window (thresholds, pair counts, test metrics)."""
    rows = []
    for size in sizes:
        path = Path(root) / 'models' / f"w{size}" / 'link_scorer_s1.json'
        man = json.loads(path.read_text())
        chosen = man.get('threshold_summary', {}).get('at_chosen', {})
        test = man.get('test_metrics', {})
        if man.get('threshold_rule') != 'pipeline':
            raise ValueError(f"{path}: expected the pipeline threshold rule")
        rows.append({'size': size, 'threshold': man['threshold'],
                     'pair_threshold': man.get('pair_threshold'),
                     'train_pairs': man.get('n_train'),
                     'train_positive': man.get('n_positive_train'),
                     'test_pairs': man.get('n_test'), 'test_positive': man.get('n_positive_test'),
                     'curve_recovered': chosen.get('recovered'),
                     'curve_planted': chosen.get('planted'),
                     'curve_false_link': chosen.get('false_link'),
                     'test_precision': test.get('precision'), 'test_recall': test.get('recall')})
    return rows


def role_scores(holdout_dir: Path, fold: str, roles: Sequence[str]) -> Dict[int, Dict]:
    """Per-gap hold-out counts of one ``holdout_<h>/`` run restricted to truth roles.

    Parameters
    ----------
    holdout_dir : Path
        A hold-out run directory (truth, ``bridged/links.parquet``).
    fold : str
        ``'test'``, ``'train'`` or ``'all'`` (as ``compare --fold``).
    roles : Sequence[str]
        Truth roles kept (``'plant'`` excludes decoy targets and flicker).

    Returns
    -------
    Dict[int, Dict]
        ``{gap: {'planted', 'recovered', 'false_link', 'missed'}}``; the same
        rule as :func:`gap_bridging.holdout.score_links` on the kept rows.
    """
    sys.path.insert(0, str(ROOT / 'scripts'))
    from gap_bridging import compare, holdout, labels
    if fold not in ('test', 'train', 'all') or not roles:
        raise ValueError(f"bad fold {fold!r} or empty roles")
    truth = pq.read_table(Path(holdout_dir) / holdout.TRUTH_NAME)
    if fold != 'all':
        truth = compare._keep_fold(labels.truth_folds(truth, compare.DEFAULT_SALT,
                                                      compare.DEFAULT_TEST_FRACTION), fold)
    truth = truth.filter(pc.is_in(truth['role'], pa.array(list(roles))))
    truth = truth.select([n for n in holdout.TRUTH_SCHEMA.names if n in truth.schema.names])
    links = pq.read_table(Path(holdout_dir) / holdout.BRIDGED_DIR / 'links.parquet',
                          columns=['tracklet_a', 'tracklet_b', 'decision'])
    keys = ('planted', 'recovered', 'false_link', 'missed')
    out = {int(r['gap_frames']): {k: int(r[k]) for k in keys}
           for r in holdout.score_links(links, truth, {}).to_pylist()
           if r['gap_frames'] != holdout.ALL_GAPS}
    assert all(v['recovered'] + v['false_link'] + v['missed'] == v['planted']
               for v in out.values())
    return out


def plants_only_rows(root: Path, sizes: Sequence[int], horizon: int = HORIZON,
                     through_gap: int = THROUGH_GAP) -> List[Dict]:
    """Plants-only (no decoy targets) hold-out totals and rule checks per window and variant.

    The rule mirrors ``compare``: precision 1.0 at every gap <= ``through_gap``
    with a plant, and recall at each such gap at least the hand cost's.
    """
    rows = []
    for size in sizes:
        for fold in ('test', 'all'):
            per: Dict[str, Dict[int, Dict]] = {}
            seeds_of = {v: [p.name for p in _seed_dirs(root, size, v)] for v, _ in VARIANTS}
            if not seeds_of['current'] or any(x != seeds_of['current'] for x in seeds_of.values()):
                raise ValueError(f"w{size}: variants must share a non-empty seed list, "
                                 f"got {seeds_of}")
            for variant, _ in VARIANTS:
                acc: Dict[int, Dict] = {}
                for seed in _seed_dirs(root, size, variant):
                    for gap, c in role_scores(seed / f"holdout_{horizon}", fold,
                                              ('plant',)).items():
                        tot = acc.setdefault(gap, dict.fromkeys(c, 0))
                        for k in c:
                            tot[k] += c[k]
                per[variant] = {g: c for g, c in acc.items() if g <= through_gap}
                if not per[variant]:
                    raise ValueError(f"w{size} {variant}: no planted gap <= {through_gap}")
            for variant, gaps in per.items():
                ref = per['current']
                rows.append({'size': size, 'fold': fold, 'variant': variant,
                             **{k: sum(c[k] for c in gaps.values())
                                for k in ('planted', 'recovered', 'false_link')},
                             'gaps_with_false': ' '.join(str(g) for g, c in sorted(gaps.items())
                                                         if c['false_link']) or '-',
                             'precision_ok': all(c['false_link'] == 0 for c in gaps.values()),
                             'recall_ok': all(c['recovered'] >= ref[g]['recovered']
                                              for g, c in gaps.items())})
    assert rows
    return rows


def _fmt(value, digits: int = 3) -> str:
    """Compact cell text for floats, ints, bools and None."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 'n/a'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, float):
        return f"{value:.{digits}g}" if abs(value) < 1000 else f"{value:.0f}"
    return str(value)


def markdown_table(header: Sequence[str], rows: Sequence[Sequence]) -> str:
    """A GitHub Markdown table; numeric-looking columns are right-aligned."""
    if not header or any(len(r) != len(header) for r in rows):
        raise ValueError("every row must have one cell per header column")
    cells = [[_fmt(c) for c in r] for r in rows]
    align = []
    for j in range(len(header)):
        numeric = all(re.fullmatch(r'[-+]?[\d.,/ ()e%]+|n/a', r[j] or 'n/a') for r in cells)
        align.append('---:' if numeric and cells else '---')
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(align) + '|']
    lines += ['| ' + ' | '.join(r) + ' |' for r in cells]
    return '\n'.join(lines)


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[1].strip())
    sub = parser.add_subparsers(dest='command', required=True)

    # ----- windows -----
    p = sub.add_parser('windows', help='cut nested windows and record planting counts')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--sizes', type=lambda s: [int(v) for v in s.split(',')], required=True)
    p.add_argument('--force', action='store_true')
    p.add_argument('--no-probe', action='store_true',
                   help='record the constant-density target without the feasibility probe')

    # ----- overlay -----
    p = sub.add_parser('overlay', help='learned overlay at one probability')
    p.add_argument('--scorer', type=Path, required=True)
    p.add_argument('--probability', type=float, required=True)
    p.add_argument('--out', type=Path, required=True)

    # ----- natural -----
    p = sub.add_parser('natural', help='natural-link outcomes per window and variant')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--sizes', type=lambda s: [int(v) for v in s.split(',')], required=True)
    p.add_argument('--horizon', type=int, default=HORIZON)

    # ----- measure -----
    p = sub.add_parser('measure', help='time and peak RSS of repeated bridging runs')
    p.add_argument('--observations', type=Path, required=True)
    p.add_argument('--config', type=Path, default=None)
    p.add_argument('--label', required=True)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--out', type=Path, required=True)

    # ----- concat-video -----
    p = sub.add_parser('concat-video', help='one verified video from the demo chunks')
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--no-verify', action='store_true')

    # ----- report -----
    p = sub.add_parser('report', help='tables, figures and the Markdown report')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--sizes', type=lambda s: [int(v) for v in s.split(',')], required=True)
    p.add_argument('--docs', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    p.add_argument('--dpi', type=int, default=600)

    args = parser.parse_args()

    # =========================================================
    # DISPATCH
    # =========================================================
    if args.command == 'windows':
        for rec in make_windows(args.source, args.out, args.sizes, args.force,
                                not args.no_probe):
            print(f"[windows] w{rec['size']}: frames {rec['start']}-{rec['last']}, "
                  f"{rec['rows']} rows, {rec['tracklets']} tracklets, per-gap {rec['per_gap']}, "
                  f"(target {rec['density_target']}), decoys {rec['decoys']}, "
                  f"flicker {rec['flicker']}", flush=True)
    elif args.command == 'overlay':
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(overlay_text(args.scorer, args.probability, args.out.parent))
        print(f"[overlay] wrote {args.out}", flush=True)
    elif args.command == 'natural':
        natural_command(args.root, args.sizes, args.horizon)
        _write_csv(plants_only_rows(args.root, args.sizes, args.horizon),
                   args.root / 'natural' / 'plants_only.csv')
        print(f"[natural] wrote {args.root}/natural/plants_only.csv", flush=True)
    elif args.command == 'measure':
        measured = measure_run(args.observations, args.config, args.label, args.repeats)
        _write_csv(measured, args.out)
        print(f"[measure] {args.label}: wall {[round(r['wall_s'], 2) for r in measured]} s, "
              f"peak RSS {max(r['max_rss_kb'] for r in measured) / 1024:.0f} MiB", flush=True)
    elif args.command == 'concat-video':
        done_video = concat_video(args.bundle, args.out, not args.no_verify)
        args.out.with_suffix('.json').write_text(json.dumps(done_video, indent=1) + '\n')
        print(f"[concat-video] {done_video}", flush=True)
    elif args.command == 'report':
        sys.path.insert(0, str(ROOT / 'scripts'))
        import gap_scaling_report  # noqa: E402  (figures need matplotlib only here)
        gap_scaling_report.write_report(args.root, args.sizes, args.docs, args.figures, args.dpi)
