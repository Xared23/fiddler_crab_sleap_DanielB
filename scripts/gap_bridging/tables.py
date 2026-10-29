"""Output tables of the gap-bridging step and their atomic writer.

All table work is done in pyarrow: the input columns are carried through as the
original arrays (never round-tripped through pandas), so the original rows of
``observations.parquet`` are value- and type-identical to the input.
"""
from __future__ import annotations

from dataclasses import fields
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from gap_bridging.types import DECISIONS, SOURCES, Bridge, LinkDecision
from pipeline_common import atomic_json, sha256

RecoveryKey = tuple[str, int, str]

ADDED_FIELDS = (pa.field('trajectory_id', pa.string()), pa.field('source', pa.string()),
                pa.field('bridge_id', pa.string()))
TRAJECTORY_SCHEMA = pa.schema([
    ('trajectory_id', pa.string()), ('video_id', pa.string()),
    ('start_frame', pa.int64()), ('end_frame', pa.int64()),
    ('n_observed', pa.int64()), ('n_recovered', pa.int64()), ('n_interpolated', pa.int64()),
    ('n_tracklets', pa.int64()), ('member_tracklet_ids', pa.list_(pa.string())),
])
LINKS_SCHEMA = pa.schema([
    ('video_id', pa.string()), ('tracklet_a', pa.string()), ('tracklet_b', pa.string()),
    ('gap_frames', pa.int64()), ('cost', pa.float64()), ('runner_up_cost', pa.float64()),
    ('decision', pa.string()), ('reason', pa.string()),
])
OUTPUT_NAMES = ('observations.parquet', 'trajectories.parquet', 'links.parquet')
COMPLETE_NAME = 'GAP_BRIDGING_COMPLETE.json'


# --------------------------------------------------------------------------- #
# Observations
# --------------------------------------------------------------------------- #
def output_schema(input_schema: pa.Schema) -> pa.Schema:
    """Return the input schema plus ``trajectory_id``, ``source``, ``bridge_id``."""
    clash = {f.name for f in ADDED_FIELDS}.intersection(input_schema.names)
    if clash:
        raise ValueError(f"input already has output columns: {sorted(clash)}")
    schema = input_schema
    for field in ADDED_FIELDS:
        schema = schema.append(field)
    return schema


def synthetic_table(pairs: Sequence[tuple[Bridge, pa.Table]], input_schema: pa.Schema,
                    recovered: Mapping[RecoveryKey, str] | None = None) -> pa.Table:
    """Label synthetic bridge rows and drop the frames filled by a recovery.

    Parameters
    ----------
    pairs : sequence of (Bridge, pa.Table)
        Each bridge with its synthetic rows (one per ``bridge.frames`` entry, in
        that order, in the input template schema).
    input_schema : pa.Schema
        Schema of the input observation table.
    recovered : mapping, optional
        Recovery keys ``(video_id, frame, trajectory_id)``; those frames are
        filled by a real detection and get no synthetic row.

    Returns
    -------
    pa.Table
        Rows in :func:`output_schema` with ``source='interpolated'`` and the
        bridge's ``trajectory_id`` and ``bridge_id``; unsorted.
    """
    target = output_schema(input_schema)
    recovered = recovered or {}
    parts = []
    for bridge, table in pairs:
        frames = [int(f) for f in bridge.frames]
        if table.num_rows != len(frames) or table['frame'].to_pylist() != frames:
            raise ValueError(f"synthetic rows of {bridge.bridge_id} do not match bridge.frames")
        keep = [(bridge.video_id, f, bridge.trajectory_id) not in recovered for f in frames]
        table = table.select(input_schema.names).cast(input_schema).filter(pa.array(keep, pa.bool_()))
        n = table.num_rows
        table = (table.append_column(ADDED_FIELDS[0], pa.array([bridge.trajectory_id] * n, pa.string()))
                 .append_column(ADDED_FIELDS[1], pa.array(['interpolated'] * n, pa.string()))
                 .append_column(ADDED_FIELDS[2], pa.array([bridge.bridge_id] * n, pa.string())))
        parts.append(table.cast(target))
    if not parts:
        return target.empty_table()
    return pa.concat_tables(parts)


def assemble_observations(obs: pa.Table, trajectory_of_tracklet: dict[str, str],
                          recovered: dict[RecoveryKey, str], bridge_of_recovered: dict[RecoveryKey, str],
                          synthetic: pa.Table | None) -> pa.Table:
    """Build the output observation table.

    Parameters
    ----------
    obs : pa.Table
        Input observation table; its rows and columns are carried unchanged.
    trajectory_of_tracklet : dict
        ``tracklet_id -> trajectory_id`` covering every non-null tracklet_id.
    recovered : dict
        ``(video_id, frame, trajectory_id) -> detection_id`` of adopted rows.
    bridge_of_recovered : dict
        ``(video_id, frame, trajectory_id) -> bridge_id`` for the same keys.
    synthetic : pa.Table or None
        Interpolated rows in :func:`output_schema` (see :func:`synthetic_table`).

    Returns
    -------
    pa.Table
        Original rows in original order with ``trajectory_id`` (null for
        unassigned rows not recovered), ``source`` (observed / recovered) and
        ``bridge_id`` (null unless recovered) appended, then the synthetic rows
        sorted by (video_id, frame, trajectory_id).
    """
    target = output_schema(obs.schema)
    if set(recovered) != set(bridge_of_recovered):
        raise ValueError("recovered and bridge_of_recovered have different keys")
    for (video, frame, traj), det in recovered.items():
        if not det:
            raise ValueError(f"empty detection_id for recovery {(video, frame, traj)}")

    keys = list(trajectory_of_tracklet)
    tidx = pc.index_in(obs['tracklet_id'], value_set=pa.array(keys, pa.string()))
    unknown = pc.sum(pc.and_(pc.is_valid(obs['tracklet_id']), pc.is_null(tidx))).as_py() or 0
    if unknown:
        raise ValueError(f"{unknown} rows have a tracklet_id with no trajectory")
    traj = pc.take(pa.array([trajectory_of_tracklet[k] for k in keys], pa.string()), tidx)

    items = sorted(recovered.items())
    dets = [det for _, det in items]
    if len(set(dets)) != len(dets):
        raise ValueError("a detection is recovered more than once")
    ridx = pc.index_in(obs['detection_id'], value_set=pa.array(dets, pa.string()))
    is_rec = pc.is_valid(ridx)
    n_hit = pc.sum(is_rec).as_py() or 0
    if n_hit != len(dets):
        raise ValueError(f"{len(dets)} recovered detections matched {n_hit} input rows")
    if items:
        rec_rows = obs.filter(is_rec)
        rows = dict(zip(rec_rows['detection_id'].to_pylist(),
                        zip(rec_rows['video_id'].to_pylist(), rec_rows['frame'].to_pylist(),
                            rec_rows['accepted_for_tracking'].to_pylist(), rec_rows['tracklet_id'].to_pylist())))
        for (video, frame, _), det in items:
            r_video, r_frame, accepted, tracklet = rows[det]
            if (r_video, r_frame) != (video, frame) or accepted is not False or tracklet is not None:
                raise ValueError(f"detection {det!r} cannot be recovered at {video}/{frame}")
    rec_traj = pc.take(pa.array([key[2] for key, _ in items], pa.string()), ridx)
    rec_bridge = pc.take(pa.array([bridge_of_recovered[key] for key, _ in items], pa.string()), ridx)

    out = (obs.append_column(ADDED_FIELDS[0], pc.coalesce(traj, rec_traj))
           .append_column(ADDED_FIELDS[1], pc.if_else(is_rec, 'recovered', 'observed'))
           .append_column(ADDED_FIELDS[2], rec_bridge))
    out = out.cast(target)
    if synthetic is None or synthetic.num_rows == 0:
        return out
    synthetic = synthetic.select(target.names).cast(target)
    bad = set(synthetic['source'].to_pylist()) - {'interpolated'}
    if bad:
        raise ValueError(f"synthetic rows must have source='interpolated', got {sorted(map(str, bad))}")
    synthetic = synthetic.sort_by([('video_id', 'ascending'), ('frame', 'ascending'),
                                   ('trajectory_id', 'ascending')])
    return pa.concat_tables([out, synthetic])


# --------------------------------------------------------------------------- #
# Trajectories and links
# --------------------------------------------------------------------------- #
def trajectory_table(observations: pa.Table, trajectories: Mapping[str, Sequence[str]]) -> pa.Table:
    """Summarize each trajectory from the assembled observation table.

    Parameters
    ----------
    observations : pa.Table
        Output of :func:`assemble_observations`.
    trajectories : mapping
        ``trajectory_id -> ordered member tracklet ids``.

    Returns
    -------
    pa.Table
        :data:`TRAJECTORY_SCHEMA`, one row per trajectory, sorted by
        (video_id, trajectory_id).
    """
    has = pc.is_valid(observations['trajectory_id'])
    t = observations.select(['trajectory_id', 'video_id', 'frame', 'source']).filter(has)
    span = t.group_by(['trajectory_id', 'video_id']).aggregate(
        [('frame', 'min'), ('frame', 'max')]).to_pydict()
    counts = t.group_by(['trajectory_id', 'source']).aggregate([('frame', 'count')]).to_pydict()
    info: dict[str, dict] = {}
    for traj, video, lo, hi in zip(span['trajectory_id'], span['video_id'], span['frame_min'], span['frame_max']):
        if traj in info:
            raise ValueError(f"trajectory {traj!r} spans several videos")
        if traj not in trajectories:
            raise ValueError(f"observations reference unknown trajectory {traj!r}")
        info[traj] = dict(video_id=video, start_frame=lo, end_frame=hi,
                          **{f"n_{s}": 0 for s in SOURCES})
    for traj, source, n in zip(counts['trajectory_id'], counts['source'], counts['frame_count']):
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}")
        info[traj][f"n_{source}"] = n
    missing = set(trajectories) - set(info)
    if missing:
        raise ValueError(f"trajectories without observations: {sorted(missing)[:5]}")
    order = sorted(info, key=lambda k: (info[k]['video_id'], k))
    rows = [dict(trajectory_id=k, **info[k], n_tracklets=len(trajectories[k]),
                 member_tracklet_ids=list(trajectories[k])) for k in order]
    return pa.Table.from_pylist(rows, schema=TRAJECTORY_SCHEMA)


def links_table(decisions: list[LinkDecision], video_id: str) -> pa.Table:
    """Return the link audit table of one video (:data:`LINKS_SCHEMA`)."""
    names = [f.name for f in fields(LinkDecision)]
    columns: dict[str, list] = {'video_id': [video_id] * len(decisions)}
    for name in names:
        columns[name] = [getattr(d, name) for d in decisions]
    bad = set(columns['decision']) - set(DECISIONS)
    if bad:
        raise ValueError(f"unknown decisions: {sorted(bad)}")
    columns['gap_frames'] = [int(v) for v in columns['gap_frames']]
    for name in ('cost', 'runner_up_cost'):
        columns[name] = [float(v) for v in columns[name]]
    return pa.Table.from_pydict(columns, schema=LINKS_SCHEMA)


def concat_links(tables: Iterable[pa.Table]) -> pa.Table:
    """Concatenate per-video link tables (empty table when none)."""
    tables = [t.cast(LINKS_SCHEMA) for t in tables]
    return pa.concat_tables(tables) if tables else LINKS_SCHEMA.empty_table()


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def write_outputs(out_dir: Path, observations: pa.Table, trajectories: pa.Table, links: pa.Table,
                  summary: dict, force: bool = False) -> dict:
    """Write the three parquet outputs and the completion marker atomically.

    Each parquet is written (zstd) to ``.<name>.tmp`` in ``out_dir`` and moved
    into place with ``os.replace`` only once all three were written; the marker
    ``GAP_BRIDGING_COMPLETE.json`` is written last via ``atomic_json``. With
    ``force`` an existing marker is removed before any output is replaced.

    Parameters
    ----------
    out_dir : Path
        Output directory (created if needed).
    observations, trajectories, links : pa.Table
        Tables to write.
    summary : dict
        JSON-serializable run summary (config, input hash, counts, ...); must
        not contain the key ``outputs``.
    force : bool
        Overwrite existing outputs.

    Returns
    -------
    dict
        The marker content: ``summary`` plus ``outputs`` mapping each file name
        to its ``sha256`` and ``rows``.

    Raises
    ------
    FileExistsError
        If any output exists and ``force`` is False.
    """
    if 'outputs' in summary:
        raise ValueError("summary must not contain the key 'outputs'")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [n for n in OUTPUT_NAMES + (COMPLETE_NAME,) if (out_dir / n).exists()]
    if existing and not force:
        raise FileExistsError(f"outputs already exist in {out_dir}: {existing}; use --force")
    tables = dict(zip(OUTPUT_NAMES, (observations, trajectories, links)))
    temps = {name: out_dir / f".{name}.tmp" for name in OUTPUT_NAMES}
    outputs: dict[str, dict] = {}
    try:
        for name, table in tables.items():
            pq.write_table(table, temps[name], compression='zstd')
            outputs[name] = {'sha256': sha256(temps[name]), 'rows': table.num_rows}
        if (out_dir / COMPLETE_NAME).exists():
            (out_dir / COMPLETE_NAME).unlink()
        for name in OUTPUT_NAMES:
            os.replace(temps[name], out_dir / name)
    finally:
        for path in temps.values():
            if path.exists():
                path.unlink()
    marker = dict(summary, outputs=outputs)
    atomic_json(out_dir / COMPLETE_NAME, marker)
    return marker
