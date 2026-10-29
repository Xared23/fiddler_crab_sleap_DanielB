#!/usr/bin/env python3

"""
End-to-end gap bridging of body-first tracklets (WP-D).

``run`` reads a published observation table, links tracklets across
``1..max_gap`` missing frames (per video), chains the links into
trajectories, fills every bridged frame with a recovered rejected detection
or an interpolated pose, checks the global invariants and writes
``observations.parquet``, ``trajectories.parquet``, ``links.parquet`` and the
``GAP_BRIDGING_COMPLETE.json`` marker. The input is never modified.

Sam
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.metadata as metadata
from pathlib import Path
import time

# ----- numerics -----
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# ----- local modules -----
from gap_bridging import candidates, chains, geometry, recover, tables
from gap_bridging.config import BridgeConfig
from gap_bridging.types import DECISIONS, Bridge, Endpoint
from pipeline_common import sha256

# ============================================================
# CONSTANTS
# ============================================================

VERSIONED_PACKAGES = ('numpy', 'scipy', 'pyarrow')

# ============================================================
# INPUT
# ============================================================


def _progress(stage: str, started: float, detail: str) -> None:
    """Print one progress line (flushed so Slurm logs stay current)."""
    print(f"[bridge_gaps] {time.monotonic() - started:7.1f}s {stage}: {detail}", flush=True)


def input_files(observations_path: Path) -> list[Path]:
    """Parquet files making up the observation input.

    Parameters
    ----------
    observations_path : Path
        A single parquet file, or a directory searched recursively for
        ``*.parquet`` (hidden files skipped), e.g. the published dataset's
        ``tables/observations/``.

    Returns
    -------
    list of Path
        Files in sorted path order.

    Raises
    ------
    FileNotFoundError
        If the path does not exist or a directory holds no parquet file.
    """
    path = Path(observations_path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"observations not found: {path}")
    files = sorted(p for p in path.rglob('*.parquet')
                   if not any(part.startswith('.') for part in p.relative_to(path).parts))
    if not files:
        raise FileNotFoundError(f"no parquet files under {path}")
    return files


def input_digest(observations_path: Path, files: list[Path]) -> tuple[str, list[dict] | None]:
    """SHA-256 of the input: the file hash, or for a directory the hash of its listing.

    Parameters
    ----------
    observations_path : Path
        Path passed to :func:`run`.
    files : list of Path
        Output of :func:`input_files`.

    Returns
    -------
    digest : str
        ``sha256(file)`` for a file input; for a directory, the SHA-256 of the
        lines ``"<relative path>\\t<file sha256>\\n"`` in sorted order.
    per_file : list of dict or None
        ``[{path, sha256}]`` for a directory input, else None.
    """
    root = Path(observations_path)
    if root.is_file():
        return sha256(root), None
    per_file = [{'path': str(f.relative_to(root)), 'sha256': sha256(f)} for f in files]
    listing = ''.join(f"{r['path']}\t{r['sha256']}\n" for r in per_file)
    return hashlib.sha256(listing.encode()).hexdigest(), per_file


def read_observations(files: list[Path]) -> pa.Table:
    """Read and concatenate the observation parquet files with pyarrow only."""
    parts = [pq.read_table(f) for f in files]
    if len(parts) == 1:
        return parts[0]
    schema = parts[0].schema
    for f, part in zip(files, parts):
        if not part.schema.equals(schema, check_metadata=False):
            raise ValueError(f"{f} has a different schema from {files[0]}")
    return pa.concat_tables(parts)


# ============================================================
# STAGES
# ============================================================


def link_videos(endpoints: dict[str, tuple[Endpoint, Endpoint]], videos: list[str],
                cfg: BridgeConfig) -> tuple[list[tuple[str, str]], pa.Table]:
    """Run the candidate linker per video.

    Parameters
    ----------
    endpoints : dict
        ``tracklet_id -> (start, end)`` from :func:`geometry.endpoints`.
    videos : list of str
        Every video of the input (videos without tracklets yield no rows).
    cfg : BridgeConfig
        Linking parameters.

    Returns
    -------
    links : list of (str, str)
        One-to-one links of all videos, video by video.
    links_table : pa.Table
        The audit table (:data:`tables.LINKS_SCHEMA`).
    """
    ends: dict[str, list[Endpoint]] = {v: [] for v in videos}
    starts: dict[str, list[Endpoint]] = {v: [] for v in videos}
    for start, end in endpoints.values():
        starts[start.video_id].append(start)
        ends[end.video_id].append(end)
    links: list[tuple[str, str]] = []
    parts = []
    for video in videos:
        video_links, decisions = candidates.link_tracklets(ends[video], starts[video], cfg)
        links.extend(video_links)
        parts.append(tables.links_table(decisions, video))
    return links, tables.concat_links(parts)


def fill_bridges(trajectories: dict[str, list[str]],
                 endpoints: dict[str, tuple[Endpoint, Endpoint]],
                 cfg: BridgeConfig) -> list[Bridge]:
    """One filled bridge per consecutive member pair of every trajectory."""
    bridges: list[Bridge] = []
    for traj, members in trajectories.items():
        pairs = zip(members[:-1], members[1:])
        for (a, b), bridge_id in zip(pairs, chains.bridge_ids(traj, members)):
            end_a, start_b = endpoints[a][1], endpoints[b][0]
            bridges.append(geometry.fill_bridge(end_a, start_b, traj, bridge_id, cfg))
    return bridges


# ============================================================
# INVARIANTS
# ============================================================


def check_invariants(tracklet_ids: list[str], trajectories: dict[str, list[str]],
                     n_links: int, n_input_rows: int, observations: pa.Table,
                     trajectory_table: pa.Table) -> None:
    """Raise ``RuntimeError`` if the assembled outputs violate a global invariant.

    Checks: every tracklet is in exactly one trajectory; ``n_trajectories ==
    n_tracklets - n_links``; output rows == input rows + interpolated rows;
    every trajectory's frames are contiguous and unique
    (``end - start + 1 == n_observed + n_recovered + n_interpolated`` and no
    (trajectory_id, frame) appears twice).
    """
    members = [m for ms in trajectories.values() for m in ms]
    if len(members) != len(set(members)) or set(members) != set(tracklet_ids):
        raise RuntimeError("tracklets are not partitioned into trajectories")
    if len(trajectories) != len(tracklet_ids) - n_links:
        raise RuntimeError(f"{len(trajectories)} trajectories != {len(tracklet_ids)} tracklets "
                           f"- {n_links} links")
    if trajectory_table.num_rows != len(trajectories):
        raise RuntimeError("trajectory table does not have one row per trajectory")
    n_interp = pc.sum(pc.equal(observations['source'], 'interpolated')).as_py() or 0
    if observations.num_rows != n_input_rows + n_interp:
        raise RuntimeError(f"{observations.num_rows} output rows != {n_input_rows} input rows "
                           f"+ {n_interp} interpolated")
    t = trajectory_table.to_pydict()
    span = np.asarray(t['end_frame'], np.int64) - np.asarray(t['start_frame'], np.int64) + 1
    filled = (np.asarray(t['n_observed'], np.int64) + np.asarray(t['n_recovered'], np.int64)
              + np.asarray(t['n_interpolated'], np.int64))
    bad = np.flatnonzero(span != filled)
    if bad.size:
        raise RuntimeError(f"{bad.size} trajectories are not contiguous, "
                           f"e.g. {t['trajectory_id'][bad[0]]}")
    assigned = observations.select(['trajectory_id', 'frame']).filter(
        pc.is_valid(observations['trajectory_id']))
    unique = assigned.group_by(['trajectory_id', 'frame']).aggregate([]).num_rows
    if unique != assigned.num_rows:
        raise RuntimeError(f"{assigned.num_rows - unique} duplicate (trajectory_id, frame) rows")


# ============================================================
# RUN
# ============================================================


def package_versions() -> dict[str, str]:
    """Installed versions of the numerical dependencies."""
    return {name: metadata.version(name) for name in VERSIONED_PACKAGES}


def run(observations_path: Path, out_dir: Path, cfg: BridgeConfig, force: bool = False) -> dict:
    """Bridge short gaps between tracklets and write the output tables.

    Parameters
    ----------
    observations_path : Path
        Published observation parquet file, or a directory of parquet parts.
    out_dir : Path
        Output directory; refuses existing outputs unless ``force``.
    cfg : BridgeConfig
        Linking, recovery and fill parameters.
    force : bool
        Overwrite existing outputs.

    Returns
    -------
    dict
        Content of ``GAP_BRIDGING_COMPLETE.json``: ``config``, ``input``,
        ``counts``, ``versions`` and ``outputs`` (sha256 and rows per file).

    Raises
    ------
    FileExistsError
        If outputs exist and ``force`` is False (checked before any work).
    RuntimeError
        If an output invariant is violated; nothing is written then.
    """
    started = time.monotonic()
    out_dir = Path(out_dir)
    existing = [n for n in tables.OUTPUT_NAMES + (tables.COMPLETE_NAME,) if (out_dir / n).exists()]
    if existing and not force:
        raise FileExistsError(f"outputs already exist in {out_dir}: {existing}; use --force")

    files = input_files(observations_path)
    digest, per_file = input_digest(observations_path, files)
    obs = read_observations(files)
    videos = sorted(set(obs['video_id'].unique().to_pylist()) - {None})
    _progress('read', started,
              f"{obs.num_rows} rows, {len(videos)} videos from {len(files)} file(s)")

    endpoints = geometry.endpoints(obs, cfg)
    tracklet_ids = sorted(endpoints)
    named = set(pc.unique(obs['tracklet_id'].drop_null()).to_pylist())
    if named != set(tracklet_ids):
        raise ValueError(f"{len(named ^ set(tracklet_ids))} tracklet ids have no accepted rows "
                         "or vice versa")
    _progress('endpoints', started, f"{len(tracklet_ids)} tracklets")

    links, links_table = link_videos(endpoints, videos, cfg)
    decision_counts = Counter(links_table['decision'].to_pylist())
    _progress('link', started, f"{len(links)} links from {links_table.num_rows} candidate pairs "
              f"{dict(sorted(decision_counts.items()))}")

    trajectories = chains.build_trajectories(tracklet_ids, links, endpoints)
    trajectory_of = chains.tracklet_to_trajectory(trajectories)
    _progress('chain', started, f"{len(trajectories)} trajectories")

    bridges = fill_bridges(trajectories, endpoints, cfg)
    recovered = recover.recover_detections(obs, bridges, cfg)
    bridge_of = recover.bridge_of_recovered(bridges, recovered)
    n_gap_frames = int(sum(len(b.frames) for b in bridges))
    _progress('fill', started, f"{len(bridges)} bridges over {n_gap_frames} frames, "
              f"{len(recovered)} recovered detections")

    fps = cfg.fps if 'time_s' in obs.schema.names else None
    pairs = [(b, geometry.synthetic_rows(b, obs.schema, fps=fps)) for b in bridges]
    synthetic = tables.synthetic_table(pairs, obs.schema, recovered)
    observations = tables.assemble_observations(obs, trajectory_of, recovered, bridge_of, synthetic)
    trajectory_table = tables.trajectory_table(observations, trajectories)
    check_invariants(tracklet_ids, trajectories, len(links), obs.num_rows, observations,
                     trajectory_table)
    _progress('assemble', started, f"{observations.num_rows} output rows "
              f"({synthetic.num_rows} interpolated); invariants hold")

    traj = trajectory_table.to_pydict()
    counts = {
        'videos': len(videos), 'tracklets': len(tracklet_ids), 'links': len(links),
        'trajectories': len(trajectories), 'bridges': len(bridges),
        'n_observed': int(sum(traj['n_observed'])), 'n_recovered': int(sum(traj['n_recovered'])),
        'n_interpolated': int(sum(traj['n_interpolated'])),
        'n_unassigned': int(observations['trajectory_id'].null_count),
        'decisions': {d: int(decision_counts.get(d, 0)) for d in DECISIONS},
    }
    if counts['n_recovered'] != len(recovered) or counts['n_interpolated'] != synthetic.num_rows:
        raise RuntimeError("trajectory counts disagree with recovered/interpolated rows")
    source = {'path': str(observations_path), 'sha256': digest, 'rows': obs.num_rows,
              'tracklets': len(tracklet_ids)}
    if per_file is not None:
        source['files'] = per_file
    summary = {'config': cfg.to_dict(), 'input': source, 'counts': counts,
               'versions': package_versions()}
    marker = tables.write_outputs(out_dir, observations, trajectory_table, links_table, summary,
                                  force)
    _progress('write', started, f"outputs in {out_dir}")
    return marker
