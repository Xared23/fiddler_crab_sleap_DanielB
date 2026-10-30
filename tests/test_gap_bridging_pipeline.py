"""Tests for gap-bridging WP-D: end-to-end pipeline, CLI and invariants."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from hypothesis import given, settings, strategies as st
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))

from gap_bridging import geometry, identity, pipeline, tables  # noqa: E402
from gap_bridging.config import BridgeConfig  # noqa: E402
from gap_bridging.types import DECISIONS, NODES  # noqa: E402
from gap_bridging_fixtures import (Crab, make_observations, node_positions,  # noqa: E402
                                   straight_crab, tracklet_count)
from pipeline_common import sha256  # noqa: E402

DEMO = ROOT / 'demo/crab_cam2_2_middle_300/observations.parquet'
PYTHON = ROOT / '.venv/bin/python'
CLI = ROOT / 'scripts/bridge_gaps.py'
CFG = BridgeConfig()
LINEAR = BridgeConfig(gate_model='linear')
OUTPUTS = tables.OUTPUT_NAMES
# v001 golden values of the linear model on the demo at max_gap 5 (D-003). The digests are
# those of ``interpolated_digest`` and ``links_digest``; both equal the pre-gate-model
# (3233ca6) outputs.
GOLDEN_INTERPOLATED_SHA256 = 'd459e647ba5a01b980f89cfd52b6504101f331fe6c6a9b80a7d61fabcfb866db'
GOLDEN_LINKS_SHA256 = '882625cadc8c56a806e4f27529750e97852c1d85f729db738aaf844af57d6f0c'


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def read_outputs(out_dir: Path) -> tuple[pa.Table, pa.Table, pa.Table, dict]:
    """Observations, trajectories, links and the marker of one run."""
    obs, traj, links = (pq.read_table(out_dir / name) for name in OUTPUTS)
    marker = json.loads((out_dir / tables.COMPLETE_NAME).read_text())
    return obs, traj, links, marker


def write_parquet(table: pa.Table, directory: Path, name: str = 'observations.parquet') -> Path:
    path = directory / name
    pq.write_table(table, path)
    return path


# Synthetic scene: three straight crabs far apart; gaps of 1..5 frames. The default
# scene walks at realistic speeds (<= 1 px/frame per component); the fast variant
# (2 to 4 px/frame, faster than the demo crabs sustain) is run under the legacy
# linear gate model only.
#   crab 1: gap 1 at frame 8, gap 2 at frames 18-19
#   crab 2: gap 4 at frames 10-13, gap 3 at frames 24-26 with frame 25 a planted
#           low_body_score detection (rejected, on the true path)
#   crab 3: gap 5 at frames 15-19
SCENE_DROPS = {1: [8, 18, 19], 2: [10, 11, 12, 13, 24, 26], 3: [15, 16, 17, 18, 19]}
SCENE_REJECT = {2: {25: 'low_body_score'}}
SCENE_GAPS = {1: [[8], [18, 19]], 2: [[10, 11, 12, 13], [24, 25, 26]], 3: [[15, 16, 17, 18, 19]]}


def scene_crabs(fast: bool = False) -> list[Crab]:
    if fast:
        return [straight_crab(1, 0, 40, (100.0, 100.0), (2.0, 1.0)),
                straight_crab(2, 0, 40, (900.0, 200.0), (-1.5, 2.0)),
                straight_crab(3, 0, 40, (500.0, 1200.0), (0.5, -3.0))]
    return [straight_crab(1, 0, 40, (100.0, 100.0), (1.0, 0.5)),
            straight_crab(2, 0, 40, (900.0, 200.0), (-0.75, 1.0)),
            straight_crab(3, 0, 40, (500.0, 1200.0), (0.25, -1.0))]


def scene_table(fast: bool = False) -> pa.Table:
    return make_observations(scene_crabs(fast), drop_frames=SCENE_DROPS, reject_frames=SCENE_REJECT)


def assert_gap_columns(case: unittest.TestCase, obs: pa.Table, cfg: BridgeConfig) -> int:
    """Bridged rows carry their bridge's length and class; observed rows carry nulls.

    Returns the number of bridged (recovered + interpolated) rows checked.
    """
    observed = obs.filter(pc.equal(obs['source'], 'observed'))
    case.assertEqual(observed['gap_frames'].null_count, observed.num_rows)
    case.assertEqual(observed['fill_confidence'].null_count, observed.num_rows)
    bridged = obs.filter(pc.not_equal(obs['source'], 'observed'))
    case.assertEqual(bridged['gap_frames'].null_count, 0)
    case.assertEqual(bridged['fill_confidence'].null_count, 0)
    rows: dict[str, list] = {}
    for row in bridged.select(['bridge_id', 'frame', 'gap_frames', 'fill_confidence']).to_pylist():
        rows.setdefault(row['bridge_id'], []).append(row)
    for bridge_id, members in rows.items():
        frames = sorted(r['frame'] for r in members)
        # Every frame of a bridge is filled (recovered or interpolated), contiguously.
        case.assertEqual(frames, list(range(frames[0], frames[-1] + 1)), bridge_id)
        gap = len(frames)
        case.assertEqual({r['gap_frames'] for r in members}, {gap}, bridge_id)
        case.assertEqual({r['fill_confidence'] for r in members},
                         {geometry.fill_confidence(gap, cfg)}, bridge_id)
    return bridged.num_rows


def assert_tails_outside_tracklets(case: unittest.TestCase, obs: pa.Table,
                                   traj: pa.Table) -> None:
    """Every recovered_tail row lies outside its trajectory's chained tracklets and bridges
    and inside the trajectory span; tail runs are contiguous with the span ends."""
    t = traj.to_pydict()
    span = {k: (s, e) for k, s, e in zip(t['trajectory_id'], t['start_frame'], t['end_frame'])}
    assigned = obs.filter(pc.is_valid(obs['trajectory_id'])).select(
        ['trajectory_id', 'frame', 'source']).to_pylist()
    core: dict[str, list[int]] = {}
    for row in assigned:
        if row['source'] != 'recovered_tail':
            core.setdefault(row['trajectory_id'], []).append(row['frame'])
    for row in assigned:
        if row['source'] != 'recovered_tail':
            continue
        lo, hi = min(core[row['trajectory_id']]), max(core[row['trajectory_id']])
        case.assertFalse(lo <= row['frame'] <= hi, row)
        start, end = span[row['trajectory_id']]
        case.assertTrue(start <= row['frame'] <= end, row)


def _fmt6(value) -> str:
    """Float rounded to 6 decimals (``nan`` for null or NaN)."""
    return 'nan' if value is None or math.isnan(value) else f"{value:.6f}"


def interpolated_digest(obs: pa.Table) -> str:
    """SHA-256 of the interpolated rows' (frame, trajectory_id, body_x, body_y), 6 decimals.

    Rows are sorted by (trajectory_id, frame); one line per row,
    ``"<frame>\t<trajectory_id>\t<body_x>\t<body_y>\n"``.
    """
    rows = obs.filter(pc.equal(obs['source'], 'interpolated')).select(
        ['frame', 'trajectory_id', 'body_x', 'body_y']).to_pylist()
    rows.sort(key=lambda r: (r['trajectory_id'], r['frame']))
    text = ''.join(f"{r['frame']}\t{r['trajectory_id']}\t"
                   f"{_fmt6(r['body_x'])}\t{_fmt6(r['body_y'])}\n" for r in rows)
    return hashlib.sha256(text.encode()).hexdigest()


def links_digest(links: pa.Table) -> str:
    """SHA-256 of the audit table without ``stage`` (floats to 6 decimals), in row order."""
    names = ('video_id', 'tracklet_a', 'tracklet_b', 'gap_frames', 'cost', 'runner_up_cost',
             'decision', 'reason', 'n_chain_ignored')
    lines = []
    for row in links.select(list(names)).to_pylist():
        lines.append('\t'.join(_fmt6(row[n]) if n in ('cost', 'runner_up_cost') else str(row[n])
                               for n in names))
    return hashlib.sha256(('\n'.join(lines) + '\n').encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Demo clip
# --------------------------------------------------------------------------- #
@unittest.skipUnless(DEMO.exists(), 'demo parquet not present')
class LinearGoldenTests(unittest.TestCase):
    """``gate_model='linear'`` reproduces v001 on the demo (D-002, D-003)."""

    def test_regression_D003_linear_demo_golden(self):
        cfg = BridgeConfig(gate_model='linear', max_gap=5)
        with tempfile.TemporaryDirectory() as tmp:
            marker = pipeline.run(DEMO, Path(tmp) / 'linear', cfg)
            obs, traj, links, _ = read_outputs(Path(tmp) / 'linear')
        counts = marker['counts']
        self.assertEqual((counts['links'], counts['trajectories']), (654, 527))
        self.assertEqual(counts['decisions'], {'linked': 654, 'refused_margin': 10,
                                               'refused_assignment': 146, 'refused_gate': 18494})
        self.assertEqual((counts['n_recovered'], counts['n_interpolated']), (15, 1323))
        self.assertEqual(counts['candidate_pairs'], {'enumerated': 19304, 'pruned_by_radius': 0})
        self.assertEqual(traj.num_rows, 527)
        self.assertEqual(interpolated_digest(obs), GOLDEN_INTERPOLATED_SHA256)
        self.assertEqual(links_digest(links), GOLDEN_LINKS_SHA256)


@unittest.skipUnless(DEMO.exists(), 'demo parquet not present')
class DemoRunTests(unittest.TestCase):
    """Full pipeline on the 300-frame demo clip."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls._tmp.name) / 'demo'
        cls.marker = pipeline.run(DEMO, cls.out, CFG)
        cls.input = pq.read_table(DEMO)
        cls.obs, cls.traj, cls.links, cls.disk_marker = read_outputs(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_outputs_exist_and_marker_matches_files(self):
        self.assertEqual(self.marker, self.disk_marker)
        for name in OUTPUTS:
            self.assertTrue((self.out / name).is_file())
            self.assertEqual(self.marker['outputs'][name]['sha256'], sha256(self.out / name))
        self.assertEqual(self.marker['outputs']['observations.parquet']['rows'], self.obs.num_rows)
        self.assertEqual(self.marker['outputs']['trajectories.parquet']['rows'], self.traj.num_rows)
        self.assertEqual(self.marker['outputs']['links.parquet']['rows'], self.links.num_rows)
        self.assertEqual(self.marker['input']['sha256'], sha256(DEMO))
        self.assertEqual(self.marker['input']['rows'], self.input.num_rows)
        self.assertEqual(self.marker['config'], CFG.to_dict())
        self.assertEqual(set(self.marker['versions']), {'numpy', 'scipy', 'pyarrow'})
        self.assertEqual(sorted(p.name for p in self.out.iterdir()),
                         sorted(OUTPUTS + (tables.COMPLETE_NAME,)))

    def test_original_rows_unchanged(self):
        original = self.obs.slice(0, self.input.num_rows).select(self.input.schema.names)
        self.assertTrue(original.equals(self.input))
        added = ['trajectory_id', 'source', 'bridge_id', 'gap_frames', 'fill_confidence',
                 'trajectory_status']
        self.assertEqual(self.obs.schema.names, self.input.schema.names + added)
        self.assertEqual(self.obs.schema, tables.final_output_schema(self.input.schema))

    def test_trajectory_status(self):
        self.assertEqual(self.traj.schema, tables.TRAJECTORY_SCHEMA)
        self.assertEqual(self.traj.schema.names[-1], 'trajectory_status')
        status = dict(zip(self.traj['trajectory_id'].to_pylist(),
                          self.traj['trajectory_status'].to_pylist()))
        self.assertEqual(set(status.values()), {'identity', 'spurious'})
        expected = identity.classify_trajectories(
            self.obs, min_frames=CFG.min_identity_frames, min_score=CFG.min_identity_score)
        self.assertEqual(status, expected)
        self.assertEqual(self.obs['trajectory_status'].null_count,
                         self.obs['trajectory_id'].null_count)
        counts = self.marker['counts']
        self.assertEqual(counts['spurious_trajectories'],
                         sum(v == 'spurious' for v in status.values()))
        starts = identity.identity_starts(self.traj, self.obs, status)
        self.assertEqual((counts['identity_starts'], counts['identity_starts_identity_only']),
                         (starts['all'], starts['identity']))
        self.assertLess(counts['identity_starts_identity_only'], counts['identity_starts'])
        t = self.traj.to_pydict()
        for k, traj in enumerate(t['trajectory_id']):     # only one-frame units are spurious
            if status[traj] == 'spurious':
                real = t['n_observed'][k] + t['n_recovered'][k] + t['n_recovered_tail'][k]
                self.assertLess(real, CFG.min_identity_frames)

    def test_recovered_tail_rows(self):
        tail = self.obs.filter(pc.equal(self.obs['source'], 'recovered_tail'))
        counts = self.marker['counts']
        self.assertEqual(tail.num_rows, counts['n_recovered_tail'])
        self.assertLessEqual(counts['n_recovered_tail'], 26)
        self.assertGreater(counts['n_recovered_tail'], 0)
        self.assertEqual(tail['trajectory_status'].null_count, 0)
        self.assertEqual(set(tail['rejection_reason'].to_pylist()), set(CFG.tail_reasons))
        assert_tails_outside_tracklets(self, self.obs, self.traj)

    def test_gap_columns(self):
        n = assert_gap_columns(self, self.obs, CFG)
        counts = self.marker['counts']
        self.assertEqual(n, counts['n_recovered'] + counts['n_interpolated']
                         + counts['n_recovered_tail'])
        self.assertGreater(counts['n_recovered'], 0)
        self.assertEqual(self.obs.schema.field('gap_frames').type, pa.int64())
        self.assertEqual(self.obs.schema.field('fill_confidence').type, pa.string())

    def test_marker_records_gate_model(self):
        self.assertEqual(self.marker['gate'], {'gate_model': CFG.gate_model, **CFG.cost_thresholds()})

    def test_counts_consistent(self):
        counts = self.marker['counts']
        tracklets = tracklet_count(self.input)
        self.assertEqual(counts['tracklets'], tracklets)
        self.assertEqual(self.marker['input']['tracklets'], tracklets)
        self.assertEqual(counts['trajectories'], self.traj.num_rows)
        self.assertEqual(counts['trajectories'], tracklets - counts['links'])
        decisions = pc.value_counts(self.links['decision']).to_pylist()
        self.assertEqual({d['values']: d['counts'] for d in decisions},
                         {k: v for k, v in counts['decisions'].items() if v})
        self.assertEqual(set(counts['decisions']), set(DECISIONS))
        self.assertEqual(counts['decisions']['linked'], counts['links'])
        self.assertEqual(counts['candidate_pairs']['enumerated'], self.links.num_rows)
        self.assertGreater(counts['candidate_pairs']['pruned_by_radius'], 0)   # sqrt prunes
        sources = {d['values']: d['counts']
                   for d in pc.value_counts(self.obs['source']).to_pylist()}
        self.assertEqual(sources.get('interpolated', 0), counts['n_interpolated'])
        self.assertEqual(sources.get('recovered', 0), counts['n_recovered'])
        self.assertEqual(sources.get('recovered_tail', 0), counts['n_recovered_tail'])
        self.assertEqual(self.obs.num_rows, self.input.num_rows + counts['n_interpolated'])
        self.assertLessEqual(counts['n_recovered'] + counts['n_recovered_tail'], 54)
        self.assertEqual(sum(self.traj['n_tracklets'].to_pylist()), tracklets)
        span = pc.add(pc.subtract(self.traj['end_frame'], self.traj['start_frame']), 1)
        filled = pc.add(pc.add(pc.add(self.traj['n_observed'], self.traj['n_recovered']),
                               self.traj['n_interpolated']), self.traj['n_recovered_tail'])
        self.assertTrue(pc.all(pc.equal(span, filled)).as_py())

    def test_chain_ignored_column(self):
        self.assertEqual(self.links.schema, tables.LINKS_SCHEMA)
        self.assertIn('n_chain_ignored', self.links.column_names)
        linked = self.links.filter(pc.equal(self.links['decision'], 'linked'))
        self.assertGreater(pc.sum(linked['n_chain_ignored']).as_py(), 0)

    def test_force_rerun_is_byte_identical(self):
        with self.assertRaises(FileExistsError):
            pipeline.run(DEMO, self.out, CFG)
        again = pipeline.run(DEMO, self.out, CFG, force=True)
        for name in OUTPUTS:
            expected = self.marker['outputs'][name]['sha256']
            self.assertEqual(again['outputs'][name]['sha256'], expected)
            self.assertEqual(sha256(self.out / name), self.marker['outputs'][name]['sha256'])


# --------------------------------------------------------------------------- #
# Synthetic end-to-end
# --------------------------------------------------------------------------- #
class SyntheticSceneTests(unittest.TestCase):
    """Three realistic-speed straight crabs, gaps 1..5, one planted rejected detection
    (default sqrt gate model)."""

    FAST = False
    CONFIG = replace(CFG, max_gap=5)   # the scene's gaps are 1..5; pinned against the default

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.input = scene_table(cls.FAST)
        cls.path = write_parquet(cls.input, tmp)
        cls.marker = pipeline.run(cls.path, tmp / 'out', cls.CONFIG)
        cls.obs, cls.traj, cls.links, _ = read_outputs(tmp / 'out')
        cls.crabs = {c.crab_id: c for c in scene_crabs(cls.FAST)}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def trajectory_of_crab(self, crab_id: int) -> str:
        """Trajectory of the crab's observed frame 0 row (rows sorted by frame, crab)."""
        rows = self.obs.filter(pc.and_(pc.equal(self.obs['frame'], 0),
                                       pc.equal(self.obs['source'], 'observed'))).to_pylist()
        return rows[crab_id - 1]['trajectory_id']

    def test_all_relinked(self):
        self.assertEqual(tracklet_count(self.input), 8)
        counts = self.marker['counts']
        self.assertEqual(counts['trajectories'], 3)
        self.assertEqual(counts['links'], 5)
        # Besides the true links only cross-crab pairs (hundreds of px apart) are in the time
        # window: linear audits them as displacement failures, sqrt prunes them beyond R_max.
        self.assertEqual(counts['decisions']['refused_margin'], 0)
        self.assertEqual(counts['decisions']['refused_assignment'], 0)
        linked = self.links.filter(pc.equal(self.links['decision'], 'linked'))
        self.assertEqual(sorted(linked['gap_frames'].to_pylist()), [1, 2, 3, 4, 5])
        refused = self.links.filter(pc.equal(self.links['decision'], 'refused_gate'))
        pairs = counts['candidate_pairs']
        self.assertEqual(pairs['enumerated'], self.links.num_rows)
        if self.CONFIG.gate_model == 'linear':
            self.assertEqual(set(refused['reason'].to_pylist()), {'displacement'})
            self.assertEqual(pairs['pruned_by_radius'], 0)
        else:
            self.assertEqual(refused.num_rows, 0)
            self.assertGreater(pairs['pruned_by_radius'], 0)
        for crab_id, crab in self.crabs.items():
            traj = self.traj.filter(pc.equal(self.traj['trajectory_id'],
                                             self.trajectory_of_crab(crab_id))).to_pylist()[0]
            self.assertEqual((traj['start_frame'], traj['end_frame']), (0, 39))
            self.assertEqual(traj['n_tracklets'], len(SCENE_GAPS[crab_id]) + 1)

    def test_planted_detection_recovered(self):
        rec = self.obs.filter(pc.equal(self.obs['source'], 'recovered'))
        self.assertEqual(rec.num_rows, 1)
        row = rec.to_pylist()[0]
        self.assertEqual(row['frame'], 25)
        self.assertEqual(row['rejection_reason'], 'low_body_score')
        traj = self.trajectory_of_crab(2)
        self.assertEqual(row['trajectory_id'], traj)
        self.assertEqual(row['bridge_id'], f'{traj}/b002')
        index = self.input['detection_id'].to_pylist().index(row['detection_id'])
        original = self.obs.slice(index, 1).select(self.input.schema.names)
        self.assertTrue(original.equals(self.input.slice(index, 1)))
        self.assertEqual(self.marker['counts']['n_recovered'], 1)

    def test_interpolated_rows_on_the_line(self):
        interp = self.obs.filter(pc.equal(self.obs['source'], 'interpolated')).to_pylist()
        expected = sum(len(f) for gaps in SCENE_GAPS.values() for f in gaps) - 1
        self.assertEqual(len(interp), expected)
        self.assertEqual(self.marker['counts']['n_interpolated'], expected)
        by_traj = {self.trajectory_of_crab(c): c for c in self.crabs}
        seen = set()
        for row in interp:
            crab = self.crabs[by_traj[row['trajectory_id']]]
            k = int(np.flatnonzero(crab.frames == row['frame'])[0])
            truth = node_positions(crab.positions[k], crab.headings[k], crab.offsets)
            for j, node in enumerate(NODES):
                self.assertAlmostEqual(row[f'{node}_x'], truth[j, 0], delta=1e-6)
                self.assertAlmostEqual(row[f'{node}_y'], truth[j, 1], delta=1e-6)
                self.assertIs(row[f'{node}_visible'], True)
                self.assertIsNone(row[f'{node}_score'])
            self.assertAlmostEqual(row['body_x'], truth[0, 0], delta=1e-6)
            self.assertAlmostEqual(row['body_y'], truth[0, 1], delta=1e-6)
            self.assertIsNone(row['body_score'])
            self.assertIsNone(row['instance_score'])
            self.assertIsNone(row['tracklet_id'])
            self.assertIs(row['accepted_for_tracking'], False)
            self.assertEqual(row['centroid_source'], 'interpolated')
            self.assertEqual(row['incoming_reason'], 'bridged')
            self.assertEqual((row['instance_index'], row['raw_instance_id']), (-1, -1))
            self.assertEqual(row['detection_id'],
                             f"synthetic/f{row['frame']:06d}/{row['bridge_id']}")
            self.assertAlmostEqual(row['time_s'], row['frame'] / self.CONFIG.fps, delta=1e-12)
            seen.add((by_traj[row['trajectory_id']], row['frame']))
        gap_frames = {(c, f) for c, gaps in SCENE_GAPS.items() for g in gaps for f in g}
        self.assertEqual(seen, gap_frames - {(2, 25)})

    def test_synthetic_rows_follow_originals_sorted(self):
        n = self.input.num_rows
        tail = self.obs.slice(n)
        self.assertEqual(set(tail['source'].to_pylist()), {'interpolated'})
        keys = list(zip(tail['video_id'].to_pylist(), tail['frame'].to_pylist(),
                        tail['trajectory_id'].to_pylist()))
        self.assertEqual(keys, sorted(keys))
        self.assertTrue(self.obs.slice(0, n).select(self.input.schema.names).equals(self.input))

    def test_gap_columns(self):
        n = assert_gap_columns(self, self.obs, self.CONFIG)
        self.assertEqual(n, sum(len(f) for gaps in SCENE_GAPS.values() for f in gaps))
        gaps = dict(zip(self.obs['bridge_id'].to_pylist(), self.obs['gap_frames'].to_pylist()))
        gaps.pop(None, None)
        self.assertEqual(sorted(gaps.values()), [1, 2, 3, 4, 5])
        self.assertEqual(self.marker['gate']['gate_model'], self.CONFIG.gate_model)


class TailSceneTests(unittest.TestCase):
    """The synthetic scene with rejected rows just outside the trajectories' ends."""

    REJECT = {1: {0: 'low_body_score', 39: 'low_body_score'},
              2: {25: 'low_body_score', 39: 'missing_position'},
              3: {0: 'low_body_score', 1: 'low_body_score', 38: 'low_body_score',
                  39: 'duplicate_candidate'}}

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.input = make_observations(scene_crabs(), drop_frames=SCENE_DROPS,
                                      reject_frames=cls.REJECT)
        cls.path = write_parquet(cls.input, tmp)
        cls.runs = {}
        for on in (True, False):
            marker = pipeline.run(cls.path, tmp / f'tails_{on}', replace(CFG, tail_recovery=on))
            cls.runs[on] = (marker, *read_outputs(tmp / f'tails_{on}')[:3])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_tails_adopted(self):
        marker, obs, traj, _ = self.runs[True]
        tail = obs.filter(pc.equal(obs['source'], 'recovered_tail')).to_pylist()
        # crab 1: frames 0 and 39; crab 3: frames 0, 1 and 38 (39 is not a tail reason).
        self.assertEqual(sorted(r['frame'] for r in tail), [0, 0, 1, 38, 39])
        self.assertEqual(marker['counts']['n_recovered_tail'], 5)
        self.assertEqual(set(r['rejection_reason'] for r in tail), {'low_body_score'})
        for row in tail:
            kind = 'tail_back' if row['frame'] < 20 else 'tail_fwd'
            self.assertEqual(row['bridge_id'], f"{row['trajectory_id']}/{kind}")
            self.assertEqual(row['trajectory_status'], 'identity')
        lengths = {r['bridge_id']: r['gap_frames'] for r in tail}
        self.assertEqual(sorted(lengths.values()), [1, 1, 1, 2])
        n = assert_gap_columns(self, obs, CFG)
        self.assertEqual(n, marker['counts']['n_recovered'] + marker['counts']['n_interpolated']
                         + marker['counts']['n_recovered_tail'])
        assert_tails_outside_tracklets(self, obs, traj)
        self.assertEqual(sum(traj['n_recovered_tail'].to_pylist()), 5)
        spans = {(r['start_frame'], r['end_frame']) for r in traj.to_pylist()}
        self.assertEqual(spans, {(0, 39), (0, 38)})       # crab 2 ends at 38 (39 has no position)

    def test_tails_never_change_links(self):
        on, off = self.runs[True], self.runs[False]
        self.assertEqual(on[3].num_rows, off[3].num_rows)
        self.assertEqual(on[0]['outputs']['links.parquet']['sha256'],
                         off[0]['outputs']['links.parquet']['sha256'])
        self.assertEqual(off[0]['counts']['n_recovered_tail'], 0)
        self.assertEqual(on[0]['counts']['n_unassigned'] + 5, off[0]['counts']['n_unassigned'])
        for name in ('links', 'trajectories', 'n_interpolated', 'n_recovered', 'decisions'):
            self.assertEqual(on[0]['counts'][name], off[0]['counts'][name], name)

    def test_invariant_violations_are_caught(self):
        marker, obs, traj, links = self.runs[True]
        tracklets = sorted(set(self.input['tracklet_id'].drop_null().to_pylist()))
        trajectories = {r['trajectory_id']: r['member_tracklet_ids'] for r in traj.to_pylist()}
        args = (tracklets, trajectories, marker['counts']['links'], self.input.num_rows)
        pipeline.check_invariants(*args, obs, traj)          # the written outputs pass
        source = obs['source'].to_pylist()
        # A tail row inside its trajectory's tracklet span.
        frames = obs['frame'].to_pylist()
        traj_ids = obs['trajectory_id'].to_pylist()
        inside = next(i for i, s in enumerate(source) if s == 'observed'
                      and traj_ids[i] is not None and 5 < frames[i] < 30)
        moved = obs.set_column(obs.schema.get_field_index('source'), obs.schema.field('source'),
                               pa.array([('recovered_tail' if i == inside else v)
                                         for i, v in enumerate(source)], pa.string()))
        n_obs = traj.to_pydict()
        row = n_obs['trajectory_id'].index(traj_ids[inside])
        fixed = traj.set_column(traj.schema.get_field_index('n_observed'), 'n_observed',
                                pa.array([v - (i == row) for i, v in enumerate(n_obs['n_observed'])],
                                         pa.int64()))
        fixed = fixed.set_column(fixed.schema.get_field_index('n_recovered_tail'),
                                 'n_recovered_tail',
                                 pa.array([v + (i == row) for i, v in
                                           enumerate(n_obs['n_recovered_tail'])], pa.int64()))
        with self.assertRaisesRegex(RuntimeError, 'inside'):
            pipeline.check_invariants(*args, moved, fixed.cast(traj.schema))
        # Contiguity counts the tails.
        no_tail = traj.set_column(traj.schema.get_field_index('n_recovered_tail'),
                                  'n_recovered_tail', pa.array([0] * traj.num_rows, pa.int64()))
        with self.assertRaisesRegex(RuntimeError, 'contiguous'):
            pipeline.check_invariants(*args, obs, no_tail.cast(traj.schema))
        # Status on an unassigned row, and disagreeing between the tables.
        status = obs['trajectory_status'].to_pylist()
        orphan = traj_ids.index(None)
        bad = obs.set_column(obs.schema.get_field_index('trajectory_status'), 'trajectory_status',
                             pa.array([('identity' if i == orphan else v)
                                       for i, v in enumerate(status)], pa.string()))
        with self.assertRaisesRegex(RuntimeError, 'assigned rows'):
            pipeline.check_invariants(*args, bad, traj)
        flipped = traj.set_column(traj.schema.get_field_index('trajectory_status'),
                                  'trajectory_status',
                                  pa.array(['spurious'] * traj.num_rows, pa.string()))
        with self.assertRaisesRegex(RuntimeError, 'differs'):
            pipeline.check_invariants(*args, obs, flipped.cast(traj.schema))


class LinearFastSceneTests(SyntheticSceneTests):
    """The same scene with crabs walking 2 to 4 px/frame under the legacy linear gate:
    the v001 path still re-links all five gaps."""

    FAST = True
    CONFIG = replace(LINEAR, max_gap=5)


class MaxGapZeroTests(unittest.TestCase):
    """max_gap=0 disables linking: the output is the identity relabelling."""

    def test_identity(self):
        table = scene_table()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_parquet(table, Path(tmp))
            marker = pipeline.run(path, Path(tmp) / 'out', BridgeConfig(max_gap=0))
            obs, traj, links, _ = read_outputs(Path(tmp) / 'out')
        n_tracklets = tracklet_count(table)
        self.assertEqual(traj.num_rows, n_tracklets)
        self.assertEqual(set(traj['n_tracklets'].to_pylist()), {1})
        self.assertEqual(links.num_rows, 0)
        self.assertEqual(links.schema, tables.LINKS_SCHEMA)
        self.assertEqual(obs.num_rows, table.num_rows)
        self.assertEqual(set(obs['source'].to_pylist()), {'observed'})
        self.assertEqual(obs['bridge_id'].null_count, obs.num_rows)
        self.assertEqual(marker['counts']['links'], 0)
        self.assertEqual(marker['counts']['n_interpolated'], 0)
        self.assertEqual(marker['counts']['n_recovered'], 0)
        # One trajectory per tracklet, and every row of a tracklet maps to it.
        pairs = obs.select(['tracklet_id', 'trajectory_id']).filter(pc.is_valid(obs['tracklet_id']))
        mapping = pairs.group_by(['tracklet_id', 'trajectory_id']).aggregate([])
        self.assertEqual(mapping.num_rows, n_tracklets)
        self.assertEqual(len(set(mapping['trajectory_id'].to_pylist())), n_tracklets)
        unassigned = obs.filter(pc.is_null(obs['tracklet_id']))
        self.assertEqual(unassigned['trajectory_id'].null_count, unassigned.num_rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
class CliTests(unittest.TestCase):
    """scripts/bridge_gaps.py as a subprocess."""

    def run_cli(self, *args: str) -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
        return subprocess.run([str(PYTHON), str(CLI), *args], capture_output=True, text=True,
                              cwd=str(ROOT), env=env, timeout=300)

    def test_smoke_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_parquet(scene_table(), Path(tmp))
            out = Path(tmp) / 'out'
            base = ['--observations', str(path), '--out', str(out),
                    '--config', str(ROOT / 'configs/tracking/gap_bridging_v001.yaml')]
            first = self.run_cli(*base)
            self.assertEqual(first.returncode, 0, first.stderr)
            marker = json.loads(first.stdout[first.stdout.index('\n{') + 1:])
            self.assertEqual(marker, json.loads((out / tables.COMPLETE_NAME).read_text()))
            self.assertEqual(marker['counts']['trajectories'], 3)
            self.assertIn('[bridge_gaps]', first.stdout)
            hashes = {n: sha256(out / n) for n in OUTPUTS}

            refused = self.run_cli(*base, '--max-gap', '0')
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn('already exist', refused.stderr)
            self.assertEqual({n: sha256(out / n) for n in OUTPUTS}, hashes)

            forced = self.run_cli(*base, '--max-gap', '0', '--margin', '2.5',
                                  '--recovery-radius', '3', '--force')
            self.assertEqual(forced.returncode, 0, forced.stderr)
            marker = json.loads((out / tables.COMPLETE_NAME).read_text())
            self.assertEqual(marker['config']['max_gap'], 0)
            self.assertEqual(marker['config']['minimum_cost_margin_sqrt'], 2.5)  # sqrt config
            self.assertEqual(marker['gate']['minimum_cost_margin'], 2.5)
            self.assertEqual(marker['config']['recovery_radius_px'], 3.0)
            self.assertEqual(marker['counts']['links'], 0)

    def test_regression_D005_margin_sets_active_model(self):
        """D-005: --margin sets the margin of the active gate model."""
        with tempfile.TemporaryDirectory() as tmp:
            path = write_parquet(scene_table(), Path(tmp))
            linear_yaml = Path(tmp) / 'linear.yaml'
            linear_yaml.write_text('gate_model: linear\n')
            cases = (('sqrt', ROOT / 'configs/tracking/gap_bridging_v001.yaml',
                      'minimum_cost_margin_sqrt', 'minimum_cost_margin', 0.25, 4.0),
                     ('linear', linear_yaml, 'minimum_cost_margin', 'minimum_cost_margin_sqrt',
                      4.0, 0.25))
            for model, config, active, other, default, other_default in cases:
                markers = {}
                for margin in (None, '0.5'):
                    out = Path(tmp) / f'{model}_{margin}'
                    extra = ['--margin', margin] if margin else []
                    result = self.run_cli('--observations', str(path), '--out', str(out),
                                          '--config', str(config), *extra)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    markers[margin] = json.loads((out / tables.COMPLETE_NAME).read_text())
                with self.subTest(model=model):
                    self.assertEqual(markers[None]['gate']['minimum_cost_margin'], default)
                    self.assertEqual(markers['0.5']['gate'],
                                     dict(markers[None]['gate'], minimum_cost_margin=0.5))
                    self.assertEqual(markers['0.5']['config'][active], 0.5)
                    self.assertEqual(markers['0.5']['config'][other], other_default)
            help_text = ' '.join(self.run_cli('run', '--help').stdout.split())
            self.assertIn("active gate model's cost scale", help_text)

    def test_missing_input_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_cli('--observations', str(Path(tmp) / 'absent.parquet'),
                                  '--out', str(Path(tmp) / 'out'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('not found', result.stderr)


class DirectoryInputTests(unittest.TestCase):
    """A directory of parquet parts is read in sorted order and hashed by listing."""

    def test_parts_equal_single_file(self):
        table = scene_table()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            single = write_parquet(table, tmp)
            parts = tmp / 'parts' / 'synthetic'
            parts.mkdir(parents=True)
            half = table.num_rows // 2
            write_parquet(table.slice(0, half), parts, 'part-000.parquet')
            write_parquet(table.slice(half), parts, 'part-001.parquet')
            a = pipeline.run(single, tmp / 'a', CFG)
            b = pipeline.run(tmp / 'parts', tmp / 'b', CFG)
            self.assertEqual([f['path'] for f in b['input']['files']],
                             ['synthetic/part-000.parquet', 'synthetic/part-001.parquet'])
            for name in OUTPUTS:
                self.assertEqual(sha256(tmp / 'a' / name), sha256(tmp / 'b' / name), name)
        self.assertEqual(a['counts'], b['counts'])
        self.assertNotIn('files', a['input'])


# --------------------------------------------------------------------------- #
# Hypothesis end-to-end
# --------------------------------------------------------------------------- #
# The random scenes walk at up to sqrt(2) px/frame; the sqrt gate admits only
# prediction_horizon_frames of directed motion, so they are bridged at a horizon of 5
# (pinned, independent of the shipped default).
MAX_GAP = 5
HYP_CFG = replace(CFG, max_gap=MAX_GAP)
N_FRAMES = 60


@st.composite
def crab_scenes(draw):
    """Piecewise-linear crabs on a 700-px grid with dropped-frame runs <= max_gap.

    Observed runs between drops may be as short as 1 frame (flicker
    tracklets and skip pairs, where an end also reaches the tracklet after
    next); the first and last frames are never dropped. Velocity components
    lie in [-1, 1] px/frame (realistic crab speeds; the sqrt gate is meant
    to refuse sustained faster walking), with up to two turns at random
    knots; every crab stays within 85 px of its start, i.e. > 500 px from
    every other crab.
    """
    n_crabs = draw(st.integers(1, 4))
    crabs, drops = [], {}
    for c in range(1, n_crabs + 1):
        knots = sorted(draw(st.sets(st.integers(5, N_FRAMES - 6), max_size=2)))
        bounds = [0] + knots + [N_FRAMES - 1]
        velocity = [draw(st.tuples(st.floats(-1.0, 1.0), st.floats(-1.0, 1.0)))
                    for _ in range(len(bounds) - 1)]
        positions = np.empty((N_FRAMES, 2))
        positions[0] = (300.0 + 700.0 * (c - 1), 300.0 + 350.0 * (c % 2))
        segment = 0
        for f in range(1, N_FRAMES):
            while f > bounds[segment + 1]:
                segment += 1
            positions[f] = positions[f - 1] + velocity[segment]
        heading = draw(st.floats(-math.pi, math.pi))
        crabs.append(Crab(c, np.arange(N_FRAMES, dtype=np.int64), positions,
                          np.full(N_FRAMES, heading)))
        dropped, frame = [], draw(st.integers(1, 2 * MAX_GAP + 2))
        while True:
            run = draw(st.integers(1, MAX_GAP))
            if frame + run > N_FRAMES - 1:
                break
            if draw(st.booleans()):
                dropped.extend(range(frame, frame + run))
            frame += run + draw(st.integers(1, 3 * MAX_GAP))
        drops[c] = dropped
    return crabs, drops


class HypothesisEndToEndTests(unittest.TestCase):
    """Random well-separated crabs with short gaps are fully re-linked."""

    @settings(deadline=None, max_examples=25)
    @given(crab_scenes())
    def test_relinked(self, scene):
        crabs, drops = scene
        table = make_observations(crabs, drop_frames=drops)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_parquet(table, Path(tmp))
            marker = pipeline.run(path, Path(tmp) / 'out', HYP_CFG)
            obs, traj, _, _ = read_outputs(Path(tmp) / 'out')
        self.assertEqual(marker['counts']['trajectories'], len(crabs))
        self.assertEqual(traj.num_rows, len(crabs))
        n_dropped = sum(len(v) for v in drops.values())
        self.assertEqual(marker['counts']['n_interpolated'], n_dropped)
        self.assertEqual(obs.num_rows, table.num_rows + n_dropped)
        for row in traj.to_pylist():
            self.assertEqual((row['start_frame'], row['end_frame']), (0, N_FRAMES - 1))
            rows = obs.filter(pc.equal(obs['trajectory_id'], row['trajectory_id']))
            frames = sorted(rows['frame'].to_pylist())
            self.assertEqual(frames, list(range(N_FRAMES)))
            observed = set(rows.filter(pc.equal(rows['source'], 'observed'))['frame'].to_pylist())
            interp = rows.filter(pc.equal(rows['source'], 'interpolated'))
            synthetic = set(interp['frame'].to_pylist())
            self.assertFalse(observed & synthetic)


class HypothesisTailTests(unittest.TestCase):
    """Random scenes with rejected rows around the trajectories: tail recovery never changes
    links.parquet, and adopted rows lie outside their tracklets' span."""

    @settings(deadline=None, max_examples=15)
    @given(crab_scenes(), st.data())
    def test_tails_never_change_links(self, scene, data):
        crabs, drops = scene
        reject = {}
        for crab in crabs:
            kept = [f for f in range(N_FRAMES) if f not in set(drops[crab.crab_id])]
            frames = data.draw(st.sets(st.sampled_from(kept[:3] + kept[-3:]), max_size=3))
            reasons = st.sampled_from(['low_body_score', 'low_body_score', 'missing_position'])
            reject[crab.crab_id] = {f: data.draw(reasons) for f in frames}
        table = make_observations(crabs, drop_frames=drops, reject_frames=reject)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_parquet(table, Path(tmp))
            markers = {on: pipeline.run(path, Path(tmp) / f'{on}', replace(HYP_CFG, tail_recovery=on))
                       for on in (True, False)}
            obs, traj, _, _ = read_outputs(Path(tmp) / 'True')
            hashes = {on: sha256(Path(tmp) / f'{on}' / 'links.parquet') for on in (True, False)}
        self.assertEqual(hashes[True], hashes[False])
        self.assertEqual(markers[False]['counts']['n_recovered_tail'], 0)
        tail = obs.filter(pc.equal(obs['source'], 'recovered_tail'))
        self.assertEqual(tail.num_rows, markers[True]['counts']['n_recovered_tail'])
        self.assertEqual(set(tail['rejection_reason'].to_pylist()) - {'low_body_score'}, set())
        self.assertEqual(tail['trajectory_status'].null_count, 0)
        assert_tails_outside_tracklets(self, obs, traj)


if __name__ == '__main__':
    unittest.main()
