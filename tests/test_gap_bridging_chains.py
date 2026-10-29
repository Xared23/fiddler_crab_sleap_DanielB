"""Tests for gap-bridging WP-C: chaining, detection recovery and output tables."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from hypothesis import given, settings, strategies as st
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))

from gap_bridging import chains, recover, tables  # noqa: E402
from gap_bridging.config import BridgeConfig  # noqa: E402
from gap_bridging.types import Bridge, Endpoint, LinkDecision  # noqa: E402
from gap_bridging_fixtures import make_observations, straight_crab  # noqa: E402
from pipeline_common import sha256  # noqa: E402

DEMO = ROOT / 'demo/crab_cam2_2_middle_300/observations.parquet'
CFG = BridgeConfig()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def endpoint(tid, video, frame, n=3, pos=(0.0, 0.0)):
    return Endpoint(tid, video, int(frame), n, np.asarray(pos, float), np.zeros(2), n >= 2, math.nan,
                    np.full((5, 2), np.nan))


def endpoints_for(spans, video='v'):
    """spans: {tracklet_id: (start_frame, end_frame)} -> endpoints dict."""
    return {t: (endpoint(t, video, s, e - s + 1), endpoint(t, video, e, e - s + 1)) for t, (s, e) in spans.items()}


def make_bridge(video, a_id, a_end, b_id, b_start, traj, bid, positions):
    frames = np.arange(a_end + 1, b_start, dtype=np.int64)
    positions = np.asarray(positions, float).reshape(len(frames), 2)
    nodes = np.full((len(frames), 5, 2), np.nan)
    nodes[:, 0] = positions
    return Bridge(video, endpoint(a_id, video, a_end), endpoint(b_id, video, b_start), traj, bid,
                  frames, positions, np.full(len(frames), np.nan), nodes)


def fake_synthetic(bridge, schema):
    """Stand-in for geometry.synthetic_rows: nulls except a few columns."""
    k = len(bridge.frames)
    cols = {}
    for f in schema:
        cols[f.name] = pa.nulls(k, f.type)
    cols['video_id'] = pa.array([bridge.video_id] * k, pa.string())
    cols['frame'] = pa.array(bridge.frames, pa.int64())
    cols['body_x'] = pa.array(bridge.positions[:, 0], pa.float64())
    cols['body_y'] = pa.array(bridge.positions[:, 1], pa.float64())
    cols['detection_id'] = pa.array([f"{bridge.video_id}/f{f:06d}/{bridge.bridge_id}" for f in bridge.frames])
    cols['accepted_for_tracking'] = pa.array([False] * k)
    return pa.table(cols, schema=schema)


def two_gap_scene():
    """One straight crab, frame 4 rejected (recoverable) and frame 8 missing; plus a decoy crab."""
    crab = straight_crab(1, 0, 12, (100.0, 100.0), (2.0, 0.0), video_id='vid')
    decoy = straight_crab(2, 0, 12, (400.0, 400.0), (0.0, 1.0), video_id='vid')
    obs = make_observations([crab, decoy], drop_frames={1: [8]}, reject_frames={1: {4: 'low_body_score'}})
    return crab, obs


def tracklet_spans(obs):
    acc = obs.filter(pc.is_valid(obs['tracklet_id'])).select(['tracklet_id', 'frame']).to_pydict()
    spans = {}
    for t, f in zip(acc['tracklet_id'], acc['frame']):
        lo, hi = spans.get(t, (f, f))
        spans[t] = (min(lo, f), max(hi, f))
    return spans


def run_scene():
    """Chain, recover and assemble the two-gap scene; return everything."""
    crab, obs = two_gap_scene()
    spans = tracklet_spans(obs)
    eps = endpoints_for(spans, 'vid')
    by_start = sorted(spans, key=lambda t: spans[t][0])
    crab_ids = [t for t in by_start if obs.filter(pc.equal(obs['tracklet_id'], t))['body_x'][0].as_py() < 300]
    links = list(zip(crab_ids[:-1], crab_ids[1:]))
    trajs = chains.build_trajectories(list(spans), links, eps)
    inverse = chains.tracklet_to_trajectory(trajs)
    traj = inverse[crab_ids[0]]
    bridges = []
    for (a, b), bid in zip(links, chains.bridge_ids(traj, trajs[traj])):
        frames = np.arange(spans[a][1] + 1, spans[b][0])
        positions = np.stack([crab.positions[f] for f in frames])
        bridges.append(make_bridge('vid', a, spans[a][1], b, spans[b][0], traj, bid, positions))
    recovered = recover.recover_detections(obs, bridges, CFG)
    bor = recover.bridge_of_recovered(bridges, recovered)
    synth = tables.synthetic_table([(b, fake_synthetic(b, obs.schema)) for b in bridges], obs.schema, recovered)
    out = tables.assemble_observations(obs, inverse, recovered, bor, synth)
    return dict(obs=obs, trajs=trajs, traj=traj, bridges=bridges, recovered=recovered, out=out, links=links)


# --------------------------------------------------------------------------- #
# Chains
# --------------------------------------------------------------------------- #
class ChainTests(unittest.TestCase):
    def test_three_tracklet_chain_and_singleton(self):
        eps = endpoints_for({'t1': (0, 3), 't2': (5, 8), 't3': (10, 12), 's': (2, 6)})
        trajs = chains.build_trajectories(['t3', 's', 't1', 't2'], [('t2', 't3'), ('t1', 't2')], eps)
        self.assertEqual(trajs, {'v/trajectory0000001': ['t1', 't2', 't3'], 'v/trajectory0000002': ['s']})
        self.assertEqual(chains.bridge_ids('v/trajectory0000001', trajs['v/trajectory0000001']),
                         ['v/trajectory0000001/b001', 'v/trajectory0000001/b002'])
        self.assertEqual(chains.bridge_ids('v/trajectory0000002', ['s']), [])

    def test_ordering_by_video_start_frame_then_tracklet(self):
        eps = {**endpoints_for({'b': (0, 2), 'a': (0, 1), 'c': (1, 4)}, 'v1'), **endpoints_for({'z': (0, 0)}, 'v0')}
        trajs = chains.build_trajectories(['c', 'b', 'a', 'z'], [], eps)
        self.assertEqual(list(trajs.items()), [('v0/trajectory0000001', ['z']), ('v1/trajectory0000001', ['a']),
                                               ('v1/trajectory0000002', ['b']), ('v1/trajectory0000003', ['c'])])

    def test_invalid_links_raise(self):
        eps = endpoints_for({'a': (0, 1), 'b': (3, 4), 'c': (6, 7)})
        cases = {'two successors': [('a', 'b'), ('a', 'c')], 'two predecessors': [('a', 'c'), ('b', 'c')],
                 'self loop': [('a', 'a')], 'unknown': [('a', 'x')], 'backwards': [('b', 'a')]}
        for name, links in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                chains.build_trajectories(['a', 'b', 'c'], links, eps)

    def test_cycle_raises(self):
        # Frame checks would also catch this, so bypass them with equal-frame fake endpoints.
        eps = endpoints_for({'a': (0, 1), 'b': (3, 4)})
        with mock.patch.object(chains, '_validate_links', return_value=({'a': 'b', 'b': 'a'}, {'a', 'b'})):
            with self.assertRaisesRegex(ValueError, 'cycle'):
                chains.build_trajectories(['a', 'b'], [], eps)

    def test_cross_video_link_raises(self):
        eps = {**endpoints_for({'a': (0, 1)}, 'v0'), **endpoints_for({'b': (3, 4)}, 'v1')}
        with self.assertRaisesRegex(ValueError, 'crosses videos'):
            chains.build_trajectories(['a', 'b'], [('a', 'b')], eps)


@st.composite
def link_scenes(draw):
    """Random tracklets in two videos plus a random one-to-one forward link set."""
    n = draw(st.integers(1, 25))
    spans, videos = {}, {}
    for i in range(n):
        start = draw(st.integers(0, 60))
        spans[f"t{i:03d}"] = (start, start + draw(st.integers(0, 5)))
        videos[f"t{i:03d}"] = draw(st.sampled_from(['va', 'vb']))
    ids = list(spans)
    order = draw(st.permutations(ids))
    links, has_succ, has_pred = [], set(), set()
    for a in order:
        for b in draw(st.permutations(ids))[:3]:
            if (a not in has_succ and b not in has_pred and a != b and videos[a] == videos[b]
                    and spans[a][1] < spans[b][0]):
                links.append((a, b))
                has_succ.add(a)
                has_pred.add(b)
    eps = {t: (endpoint(t, videos[t], s, e - s + 1), endpoint(t, videos[t], e, e - s + 1))
           for t, (s, e) in spans.items()}
    return ids, links, eps


class ChainPropertyTests(unittest.TestCase):
    @settings(deadline=None, max_examples=150)
    @given(link_scenes())
    def test_paths_cover_every_tracklet_once_in_frame_order(self, scene):
        ids, links, eps = scene
        trajs = chains.build_trajectories(ids, links, eps)
        members = [m for ms in trajs.values() for m in ms]
        self.assertEqual(sorted(members), sorted(ids))
        self.assertEqual(len(trajs), len(ids) - len(links))
        link_set = set(links)
        for traj, ms in trajs.items():
            video = eps[ms[0]][0].video_id
            self.assertTrue(traj.startswith(f"{video}/trajectory"))
            for a, b in zip(ms, ms[1:]):
                self.assertIn((a, b), link_set)
                self.assertLess(eps[a][1].frame, eps[b][0].frame)
            starts = [eps[m][0].frame for m in ms]
            self.assertEqual(starts, sorted(set(starts)))
        for video in {e[0].video_id for e in eps.values()}:
            ranks = [int(t.rsplit('trajectory', 1)[1]) for t in trajs if t.startswith(video + '/')]
            self.assertEqual(ranks, list(range(1, len(ranks) + 1)))

    @settings(deadline=None, max_examples=100)
    @given(link_scenes(), st.randoms(use_true_random=False))
    def test_invariant_to_input_permutation(self, scene, rnd):
        ids, links, eps = scene
        expected = chains.build_trajectories(ids, links, eps)
        ids2, links2 = list(ids), list(links)
        rnd.shuffle(ids2)
        rnd.shuffle(links2)
        eps2 = dict(rnd.sample(list(eps.items()), len(eps)))
        self.assertEqual(list(chains.build_trajectories(ids2, links2, eps2).items()), list(expected.items()))

    @settings(deadline=None, max_examples=100)
    @given(link_scenes(), st.randoms(use_true_random=False))
    def test_relabeling_invariance(self, scene, rnd):
        ids, links, eps = scene
        expected = chains.build_trajectories(ids, links, eps)
        # Arbitrary bijection: same partition into member lists.
        perm = dict(zip(ids, rnd.sample([f"x{i}" for i in range(len(ids))], len(ids))))
        def relabel(mapping):
            new_eps = {mapping[t]: tuple(endpoint(mapping[t], e.video_id, e.frame, e.n_frames) for e in pair)
                       for t, pair in eps.items()}
            return chains.build_trajectories([mapping[t] for t in ids],
                                             [(mapping[a], mapping[b]) for a, b in links], new_eps)
        got = relabel(perm)
        self.assertEqual(sorted(tuple(perm[m] for m in ms) for ms in expected.values()),
                         sorted(tuple(ms) for ms in got.values()))
        # Order-preserving bijection: identical trajectory ids too.
        mono = {t: f"y{i:04d}" for i, t in enumerate(sorted(ids))}
        got = relabel(mono)
        self.assertEqual({k: [mono[m] for m in v] for k, v in expected.items()}, got)


# --------------------------------------------------------------------------- #
# Recovery
# --------------------------------------------------------------------------- #
class RecoveryTests(unittest.TestCase):
    def scene(self):
        """Two crabs at frame 5 rejected/planted; bridges for two trajectories over frame 5."""
        a = straight_crab(1, 0, 10, (100.0, 100.0), (1.0, 0.0), video_id='v')
        b = straight_crab(2, 0, 10, (300.0, 300.0), (1.0, 0.0), video_id='v')
        c = straight_crab(3, 0, 10, (106.0, 100.0), (1.0, 0.0), video_id='v')   # accepted, 6 px from a's path
        obs = make_observations([a, b, c], reject_frames={1: {5: 'low_body_score'}, 2: {5: 'low_body_score'}})
        return a, b, obs

    def bridge(self, traj, pos, frames=(5,)):
        lo, hi = frames[0] - 1, frames[-1] + 1
        return make_bridge('v', traj + 'A', lo, traj + 'B', hi, traj, traj + '/b001', [pos] * len(frames))

    def det_at(self, obs, frame, x):
        rows = obs.filter(pc.and_(pc.equal(obs['frame'], frame), pc.equal(obs['body_x'], x)))
        return rows['detection_id'][0].as_py()

    def test_picks_nearest_within_radius_and_ignores_accepted(self):
        a, _, obs = self.scene()
        # Path point 2 px from a's rejected row (105,100); c's accepted row (111,100) is 4 px away but accepted.
        rec = recover.recover_detections(obs, [self.bridge('v/trajectory0000001', (107.0, 100.0))], CFG)
        self.assertEqual(rec, {('v', 5, 'v/trajectory0000001'): self.det_at(obs, 5, 105.0)})

    def test_outside_radius_is_ignored(self):
        _, _, obs = self.scene()
        rec = recover.recover_detections(obs, [self.bridge('v/trajectory0000001', (105.0, 108.5))], CFG)
        self.assertEqual(rec, {})
        rec = recover.recover_detections(obs, [self.bridge('v/trajectory0000001', (105.0, 108.0))], CFG)
        self.assertEqual(len(rec), 1)  # exactly at the radius counts

    def test_already_used_detection_is_skipped(self):
        _, _, obs = self.scene()
        first = self.bridge('v/trajectory0000001', (105.0, 101.0))
        second = self.bridge('v/trajectory0000002', (105.0, 100.0))   # nearer, but processed later
        rec = recover.recover_detections(obs, [second, first], CFG)
        self.assertEqual(rec, {('v', 5, 'v/trajectory0000001'): self.det_at(obs, 5, 105.0)})

    def test_non_recoverable_reason_and_wrong_frame_are_ignored(self):
        _, _, obs = self.scene()
        cfg = BridgeConfig(recoverable_reasons=('missing_position',))
        self.assertEqual(recover.recover_detections(obs, [self.bridge('v/trajectory0000001', (105.0, 100.0))], cfg), {})
        self.assertEqual(recover.recover_detections(obs, [self.bridge('v/trajectory0000001', (105.0, 100.0), (6,))],
                                                    CFG), {})

    def test_tie_break_by_detection_id(self):
        _, _, obs = self.scene()
        # Equidistant from (105,100) and (305,300).
        rec = recover.recover_detections(obs, [self.bridge('v/t', (205.0, 200.0))], BridgeConfig(recovery_radius_px=200))
        expected = min(self.det_at(obs, 5, 105.0), self.det_at(obs, 5, 305.0))
        self.assertEqual(rec, {('v', 5, 'v/t'): expected})

    def test_missing_position_rows_never_recovered(self):
        a = straight_crab(1, 0, 10, (100.0, 100.0), (1.0, 0.0), video_id='v')
        obs = make_observations([a], reject_frames={1: {5: 'missing_position'}})
        rec = recover.recover_detections(obs, [self.bridge('v/t', (105.0, 100.0))], CFG)
        self.assertEqual(rec, {})

    def test_bridge_of_recovered(self):
        b = self.bridge('v/t', (105.0, 100.0))
        self.assertEqual(recover.bridge_of_recovered([b], {('v', 5, 'v/t'): 'd'}), {('v', 5, 'v/t'): 'v/t/b001'})


@st.composite
def recovery_scenes(draw):
    n_det = draw(st.integers(0, 12))
    dets = [(f"v/f{draw(st.integers(1, 4)):06d}/i{i:04d}", draw(st.floats(0, 30)), draw(st.floats(0, 30)))
            for i in range(n_det)]
    n_br = draw(st.integers(1, 6))
    bridges = []
    for j in range(n_br):
        pos = [(draw(st.floats(0, 30)), draw(st.floats(0, 30))) for _ in range(4)]
        bridges.append(make_bridge('v', f"a{j}", 0, f"b{j}", 5, f"v/trajectory{j + 1:07d}",
                                   f"v/trajectory{j + 1:07d}/b001", pos))
    return dets, bridges


def detections_table(dets):
    rows = [dict(video_id='v', frame=int(d[3:9]), detection_id=d, accepted_for_tracking=False,
                 rejection_reason='low_body_score', tracklet_id=None, body_x=x, body_y=y) for d, x, y in dets]
    schema = pa.schema([('video_id', pa.string()), ('frame', pa.int64()), ('detection_id', pa.string()),
                        ('accepted_for_tracking', pa.bool_()), ('rejection_reason', pa.string()),
                        ('tracklet_id', pa.string()), ('body_x', pa.float64()), ('body_y', pa.float64())])
    return pa.Table.from_pylist(rows, schema=schema)


class RecoveryPropertyTests(unittest.TestCase):
    @settings(deadline=None, max_examples=200)
    @given(recovery_scenes())
    def test_each_detection_recovered_at_most_once_and_within_radius(self, scene):
        dets, bridges = scene
        rec = recover.recover_detections(detections_table(dets), bridges, CFG)
        self.assertEqual(len(set(rec.values())), len(rec))
        where = {d: (int(d[3:9]), x, y) for d, x, y in dets}
        paths = {(b.trajectory_id, int(f)): p for b in bridges for f, p in zip(b.frames, b.positions)}
        for (video, frame, traj), det in rec.items():
            f, x, y = where[det]
            self.assertEqual((video, f), ('v', frame))
            px, py = paths[(traj, frame)]
            self.assertLessEqual(math.hypot(x - px, y - py), CFG.recovery_radius_px)
        # Greedy optimality: the first bridge in processing order takes its nearest in-radius detection.
        first = recover.bridge_order(bridges)[0]
        f0, (px, py) = int(first.frames[0]), first.positions[0]
        options = sorted((math.hypot(x - px, y - py), d) for d, x, y in dets
                         if int(d[3:9]) == f0 and math.hypot(x - px, y - py) <= CFG.recovery_radius_px)
        self.assertEqual(rec.get(('v', f0, first.trajectory_id)), options[0][1] if options else None)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
class TableTests(unittest.TestCase):
    def test_scene_assembly(self):
        s = run_scene()
        obs, out = s['obs'], s['out']
        self.assertEqual(len(s['trajs']), 2)  # crab (3 tracklets) + decoy
        self.assertEqual(len(s['recovered']), 1)
        self.assertEqual(out.schema, tables.output_schema(obs.schema))
        self.assertEqual(out.schema.names[-3:], ['trajectory_id', 'source', 'bridge_id'])
        self.assertTrue(out.slice(0, obs.num_rows).select(obs.schema.names).equals(obs))
        self.assertEqual(out.num_rows, obs.num_rows + 1)  # frame 8 interpolated; frame 4 recovered
        src = out['source'].to_pylist()
        self.assertEqual(src.count('recovered'), 1)
        self.assertEqual(src[-1], 'interpolated')
        rec_row = out.filter(pc.equal(out['source'], 'recovered')).to_pylist()[0]
        self.assertEqual((rec_row['frame'], rec_row['trajectory_id'], rec_row['bridge_id']),
                         (4, s['traj'], f"{s['traj']}/b001"))
        last = out.slice(out.num_rows - 1).to_pylist()[0]
        self.assertEqual((last['frame'], last['bridge_id']), (8, f"{s['traj']}/b002"))
        self.assertIsNone(last['tracklet_local_id'])
        traj_rows = out.filter(pc.equal(out['trajectory_id'], s['traj']))
        self.assertEqual(sorted(traj_rows['frame'].to_pylist()), list(range(12)))

    def test_unassigned_rows_and_observed_source(self):
        crab = straight_crab(1, 0, 6, (0.0, 0.0), (1.0, 0.0), video_id='v')
        obs = make_observations([crab], reject_frames={1: {3: 'invisible'}})
        spans = tracklet_spans(obs)
        trajs = chains.build_trajectories(list(spans), [], endpoints_for(spans))
        out = tables.assemble_observations(obs, chains.tracklet_to_trajectory(trajs), {}, {}, None)
        row = out.filter(pc.equal(out['frame'], 3)).to_pylist()[0]
        self.assertEqual((row['trajectory_id'], row['source'], row['bridge_id']), (None, 'observed', None))
        self.assertEqual(set(out['source'].to_pylist()), {'observed'})
        with self.assertRaisesRegex(ValueError, 'no trajectory'):
            tables.assemble_observations(obs, {}, {}, {}, None)

    def test_synthetic_rows_sorted(self):
        crab, obs = two_gap_scene()
        b2 = make_bridge('vid', 'x', 7, 'y', 9, 'vid/trajectory0000002', 'vid/trajectory0000002/b001', [(0, 0)])
        b1 = make_bridge('vid', 'p', 7, 'q', 10, 'vid/trajectory0000001', 'vid/trajectory0000001/b001',
                         [(0, 0), (1, 1)])
        synth = tables.synthetic_table([(b, fake_synthetic(b, obs.schema)) for b in (b2, b1)], obs.schema)
        inverse = {t: 'vid/trajectory0000003' for t in tracklet_spans(obs)}
        out = tables.assemble_observations(obs, inverse, {}, {}, synth)
        tail = out.slice(obs.num_rows).select(['frame', 'trajectory_id']).to_pylist()
        self.assertEqual([(r['frame'], r['trajectory_id'][-1]) for r in tail], [(8, '1'), (8, '2'), (9, '1')])

    def test_rejects_bad_recovery(self):
        crab, obs = two_gap_scene()
        inverse = {t: 'vid/t' for t in tracklet_spans(obs)}
        accepted_det = obs['detection_id'][0].as_py()
        key = ('vid', 0, 'vid/t')
        with self.assertRaises(ValueError):
            tables.assemble_observations(obs, inverse, {key: accepted_det}, {key: 'vid/t/b001'}, None)
        with self.assertRaises(ValueError):
            tables.assemble_observations(obs, inverse, {key: 'nope'}, {key: 'vid/t/b001'}, None)

    def test_demo_schema_preserved(self):
        if not DEMO.exists():
            self.skipTest('demo parquet absent')
        obs = pq.read_table(DEMO)
        spans = tracklet_spans(obs)
        trajs = chains.build_trajectories(list(spans), [], endpoints_for(spans, 'Crab_Cam_2_2'))
        tid = next(iter(trajs))
        start = spans[trajs[tid][0]][0]
        bridge = make_bridge('Crab_Cam_2_2', 'a', start - 2, 'b', start, tid, tid + '/b001', [(1.0, 2.0)])
        synth = tables.synthetic_table([(bridge, fake_synthetic(bridge, obs.schema))], obs.schema)
        out = tables.assemble_observations(obs, chains.tracklet_to_trajectory(trajs), {}, {}, synth)
        self.assertEqual(out.schema, pa.schema(list(obs.schema) + list(tables.ADDED_FIELDS)))
        self.assertEqual(len(out.schema), 57)
        self.assertTrue(out.slice(0, obs.num_rows).select(obs.schema.names).equals(obs))
        with tempfile.TemporaryDirectory() as tmp:
            traj_table = tables.trajectory_table(out, trajs)
            tables.write_outputs(Path(tmp), out, traj_table, tables.concat_links([]), {'k': 1})
            back = pq.read_table(Path(tmp) / 'observations.parquet')
        self.assertTrue(back.slice(0, obs.num_rows).select(obs.schema.names).equals(obs))
        self.assertEqual(back.slice(0, obs.num_rows)['water'].null_count, obs['water'].null_count)
        self.assertEqual(back.num_rows, obs.num_rows + 1)
        self.assertEqual(traj_table.num_rows, len(spans))

    def test_trajectory_table(self):
        s = run_scene()
        t = tables.trajectory_table(s['out'], s['trajs'])
        self.assertEqual(t.schema, tables.TRAJECTORY_SCHEMA)
        rows = {r['trajectory_id']: r for r in t.to_pylist()}
        main = rows[s['traj']]
        self.assertEqual((main['start_frame'], main['end_frame'], main['n_observed'], main['n_recovered'],
                          main['n_interpolated'], main['n_tracklets']), (0, 11, 10, 1, 1, 3))
        self.assertEqual(main['member_tracklet_ids'], s['trajs'][s['traj']])
        other = [r for k, r in rows.items() if k != s['traj']][0]
        self.assertEqual((other['n_observed'], other['n_tracklets'], other['video_id']), (12, 1, 'vid'))
        self.assertEqual(sum(t['n_observed'].to_pylist()) + sum(t['n_recovered'].to_pylist())
                         + sum(t['n_interpolated'].to_pylist()), pc.sum(pc.is_valid(s['out']['trajectory_id'])).as_py())

    def test_links_table(self):
        decisions = [LinkDecision('a', 'b', 1, 2.5, math.nan, 'linked', '', 2),
                     LinkDecision('c', 'd', 3, math.nan, math.nan, 'refused_gate', 'displacement', 0)]
        t = tables.links_table(decisions, 'v')
        self.assertEqual(t.schema, tables.LINKS_SCHEMA)
        self.assertEqual(t.column_names, ['video_id', 'tracklet_a', 'tracklet_b', 'gap_frames', 'cost',
                                          'runner_up_cost', 'decision', 'reason', 'n_chain_ignored'])
        self.assertEqual(t['n_chain_ignored'].to_pylist(), [2, 0])
        both = tables.concat_links([t, tables.links_table(decisions[:1], 'w'), tables.links_table([], 'x')])
        self.assertEqual(both['video_id'].to_pylist(), ['v', 'v', 'w'])
        self.assertEqual(tables.concat_links([]).num_rows, 0)
        with self.assertRaises(ValueError):
            tables.links_table([LinkDecision('a', 'b', 1, 1.0, 1.0, 'maybe', '', 0)], 'v')


class WriteTests(unittest.TestCase):
    def outputs(self):
        s = run_scene()
        decisions = [LinkDecision(a, b, 1, 1.0, math.nan, 'linked', '', 0) for a, b in s['links']]
        return s['out'], tables.trajectory_table(s['out'], s['trajs']), tables.links_table(decisions, 'vid')

    def test_summary_counts_and_hashes(self):
        obs, traj, links = self.outputs()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'nested'
            marker = tables.write_outputs(out, obs, traj, links, {'counts': {'n_links': links.num_rows}})
            on_disk = json.loads((out / 'GAP_BRIDGING_COMPLETE.json').read_text())
            self.assertEqual(on_disk, marker)
            self.assertEqual(sorted(p.name for p in out.iterdir()),
                             ['GAP_BRIDGING_COMPLETE.json', 'links.parquet', 'observations.parquet',
                              'trajectories.parquet'])
            for name, table in zip(tables.OUTPUT_NAMES, (obs, traj, links)):
                self.assertEqual(marker['outputs'][name]['rows'], table.num_rows)
                self.assertEqual(pq.read_metadata(out / name).num_rows, table.num_rows)
                self.assertEqual(marker['outputs'][name]['sha256'], sha256(out / name))
                self.assertEqual(pq.read_metadata(out / name).row_group(0).column(0).compression, 'ZSTD')
            self.assertEqual(marker['counts']['n_links'], 2)
            self.assertTrue(pq.read_table(out / 'trajectories.parquet').equals(traj))

    def test_force_semantics(self):
        obs, traj, links = self.outputs()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            first = tables.write_outputs(out, obs, traj, links, {'run': 1})
            with self.assertRaises(FileExistsError):
                tables.write_outputs(out, obs, traj, links, {'run': 2})
            self.assertEqual(json.loads((out / 'GAP_BRIDGING_COMPLETE.json').read_text())['run'], 1)
            second = tables.write_outputs(out, obs, traj, links, {'run': 2}, force=True)
            self.assertEqual(second['run'], 2)
            self.assertEqual(first['outputs'], second['outputs'])  # deterministic bytes
            (out / 'GAP_BRIDGING_COMPLETE.json').unlink()
            with self.assertRaises(FileExistsError):  # a lone parquet still blocks
                tables.write_outputs(out, obs, traj, links, {'run': 3})

    def test_failure_leaves_no_temp_file(self):
        obs, traj, links = self.outputs()
        real = pq.write_table
        calls = []

        def flaky(table, where, **kwargs):
            calls.append(where)
            if len(calls) == 2:
                Path(where).write_bytes(b'partial')
                raise OSError('disk full')
            return real(table, where, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with mock.patch.object(tables.pq, 'write_table', side_effect=flaky):
                with self.assertRaises(OSError):
                    tables.write_outputs(out, obs, traj, links, {})
            self.assertEqual(len(calls), 2)
            self.assertTrue(Path(calls[0]).name.startswith('.') and Path(calls[0]).name.endswith('.tmp'))
            self.assertEqual(list(out.iterdir()), [])

    def test_forced_failure_keeps_previous_outputs(self):
        obs, traj, links = self.outputs()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tables.write_outputs(out, obs, traj, links, {'run': 1})
            before = {p.name: p.read_bytes() for p in out.iterdir()}
            with mock.patch.object(tables.pq, 'write_table', side_effect=OSError('boom')):
                with self.assertRaises(OSError):
                    tables.write_outputs(out, obs, traj, links, {'run': 2}, force=True)
            self.assertEqual({p.name: p.read_bytes() for p in out.iterdir()}, before)

    def test_summary_must_not_shadow_outputs(self):
        obs, traj, links = self.outputs()
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            tables.write_outputs(Path(tmp), obs, traj, links, {'outputs': 1})


if __name__ == '__main__':
    unittest.main()
