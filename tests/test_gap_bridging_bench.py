#!/usr/bin/env python3

"""
Tests for the gap-bridging benchmark (tiling, timing and the CLI; v004).

Sam
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import csv
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

# ----- numerics -----
from hypothesis import given, settings, strategies as st
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))

# ----- local modules -----
import bench_gap_bridging as bench  # noqa: E402
from gap_bridging import holdout, pipeline  # noqa: E402
from gap_bridging.config import BridgeConfig  # noqa: E402
from gap_bridging_fixtures import make_observations, straight_crab  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

DEMO = ROOT / 'demo/crab_cam2_2_middle_300/observations.parquet'
EXPECTED_COLUMNS = ('config', 'tile', 'n_rows', 'n_tracklets', 'stage', 'median_s', 'min_s',
                    'max_s', 'repeats', 'n_links', 'candidate_pairs', 'host', 'cpu_count',
                    'python', 'numpy', 'scipy', 'pyarrow', 'sklearn')

# ============================================================
# HELPERS
# ============================================================


def scene() -> pa.Table:
    """Six crabs over 60 frames, three of them broken by missed frames, one rejected row."""
    crabs = [straight_crab(c, 0, 60, (300.0 + 500.0 * c, 400.0 + 250.0 * (c % 2)),
                           (0.4 * (-1) ** c, 0.3)) for c in range(1, 7)]
    return make_observations(crabs, drop_frames={1: [20, 21, 22], 3: [10], 5: [30, 31, 32, 33, 34]},
                             reject_frames={2: {40: 'low_body_score'}})


SCENE = scene()


def tile_of(detection_id: str) -> int:
    """Tile index from a tiled ``detection_id`` (0 when unsuffixed)."""
    head, sep, tail = detection_id.rpartition('/tile')
    return int(tail) if sep and tail.isdigit() else 0


def strip(value: str | None) -> str | None:
    return None if value is None else value.split('/tile')[0]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


# ============================================================
# TILING
# ============================================================


class TileTests(unittest.TestCase):
    """tile_observations: identity at n = 1, counts, ids and shifted geometry."""

    def test_one_tile_is_identity(self):
        self.assertTrue(bench.tile_observations(SCENE, 1).equals(SCENE))
        self.assertIs(bench.tile_observations(SCENE, 1), SCENE)

    def test_counts_and_unique_ids(self):
        n_tracklets = bench.n_tracklets(SCENE)
        for n in (2, 4, 9):
            tiled = bench.tile_observations(SCENE, n)
            self.assertEqual(tiled.schema, SCENE.schema)
            self.assertEqual(tiled.num_rows, n * SCENE.num_rows)
            self.assertEqual(bench.n_tracklets(tiled), n * n_tracklets)
            self.assertEqual(len(pc.unique(tiled['detection_id'])), tiled.num_rows)
            for name in ('raw_instance_id',):
                self.assertEqual(len(pc.unique(tiled[name])), tiled.num_rows)
            keys = set(zip(tiled['video_id'].to_pylist(), tiled['frame'].to_pylist(),
                           tiled['instance_index'].to_pylist()))
            self.assertEqual(len(keys), tiled.num_rows)
            self.assertEqual(sorted(set(tiled['frame'].to_pylist())),
                             sorted(set(SCENE['frame'].to_pylist())))
            self.assertEqual(tiled['frame'].to_pylist(), sorted(tiled['frame'].to_pylist()))
            self.assertEqual(set(tiled['video_id'].to_pylist()),
                             set(SCENE['video_id'].to_pylist()))
            self.assertEqual(tiled['tracklet_id'].null_count,
                             n * SCENE['tracklet_id'].null_count)
            local = {}
            for tid, lid in zip(tiled['tracklet_id'].to_pylist(),
                                tiled['tracklet_local_id'].to_pylist()):
                if tid is not None:
                    self.assertEqual(local.setdefault(tid, lid), lid)
            self.assertEqual(len(set(local.values())), len(local))

    def test_tile_zero_unchanged(self):
        tiled = bench.tile_observations(SCENE, 4)
        first = tiled.filter(pc.invert(pc.match_substring(tiled['detection_id'], '/tile')))
        self.assertTrue(first.equals(SCENE))

    def test_default_pitch_clears_the_scene(self):
        px, py = bench.default_pitch(SCENE)
        xs = SCENE['shell_x'].drop_null().to_numpy()
        ys = SCENE['shell_y'].drop_null().to_numpy()
        self.assertAlmostEqual(px, xs.max() - xs.min() + bench.TILE_MARGIN_PX)
        self.assertAlmostEqual(py, ys.max() - ys.min() + bench.TILE_MARGIN_PX)

    def test_invalid_n(self):
        for n in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                bench.tile_observations(SCENE, n)

    @settings(deadline=None, max_examples=20)
    @given(n=st.integers(2, 6), px=st.floats(500.0, 1e5), py=st.floats(500.0, 1e5))
    def test_property_shifted_copies_equal_original(self, n, px, py):
        tiled = bench.tile_observations(SCENE, n, (px, py))
        side = math.ceil(math.sqrt(n))
        coords = holdout.coordinate_columns(SCENE.schema)
        original = {d: i for i, d in enumerate(SCENE['detection_id'].to_pylist())}
        tiles = np.array([tile_of(d) for d in tiled['detection_id'].to_pylist()])
        self.assertEqual(sorted(set(tiles.tolist())), list(range(n)))
        for k in range(n):
            part = tiled.filter(pa.array(tiles == k))
            order = [original[strip(d)] for d in part['detection_id'].to_pylist()]
            self.assertEqual(sorted(order), list(range(SCENE.num_rows)))
            base = SCENE.take(pa.array(order))
            shift = {'_x': (k % side) * px, '_y': (k // side) * py}
            for name in coords:
                got = part[name].to_numpy(zero_copy_only=False).astype(float)
                want = base[name].to_numpy(zero_copy_only=False).astype(float)
                self.assertTrue(np.array_equal(np.isnan(got), np.isnan(want)), name)
                ok = ~np.isnan(want)
                np.testing.assert_allclose(got[ok] - shift[name[-2:]], want[ok],
                                           rtol=0, atol=1e-6)
            self.assertEqual([strip(t) for t in part['tracklet_id'].to_pylist()],
                             base['tracklet_id'].to_pylist())
            others = [c for c in SCENE.schema.names if c not in coords
                      and c not in bench.SUFFIX_COLUMNS and c not in bench.OFFSET_COLUMNS]
            self.assertTrue(part.select(others).equals(base.select(others)))

    @unittest.skipUnless(DEMO.exists(), 'demo parquet not present')
    def test_demo_tiles(self):
        obs = pq.read_table(DEMO)
        tiled = bench.tile_observations(obs, 4)
        self.assertEqual(tiled.num_rows, 4 * obs.num_rows)
        self.assertEqual(bench.n_tracklets(tiled), 4 * bench.n_tracklets(obs))
        for name in ('detection_id', 'prediction_detection_id', 'raw_instance_id',
                     'demo_instance_row'):
            self.assertEqual(len(pc.unique(tiled[name])), tiled.num_rows, name)


# ============================================================
# NO INTERACTION
# ============================================================


class NoInteractionTests(unittest.TestCase):
    """Bridging two tiles gives exactly twice the links and trajectories."""

    def test_two_tiles_double_everything(self):
        cfg = BridgeConfig(max_gap=60)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pq.write_table(SCENE, tmp / 'one.parquet')
            pq.write_table(bench.tile_observations(SCENE, 2), tmp / 'two.parquet')
            one = pipeline.run(tmp / 'one.parquet', tmp / 'out1', cfg)
            two = pipeline.run(tmp / 'two.parquet', tmp / 'out2', cfg)
            links1 = pq.read_table(tmp / 'out1/links.parquet')
            links2 = pq.read_table(tmp / 'out2/links.parquet')
        self.assertGreater(one['counts']['links'], 0)
        for key in ('links', 'trajectories', 'tracklets', 'n_interpolated', 'n_recovered',
                    'n_recovered_tail'):
            self.assertEqual(two['counts'][key], 2 * one['counts'][key], key)
        self.assertEqual(two['counts']['candidate_pairs']['enumerated'],
                         2 * one['counts']['candidate_pairs']['enumerated'])
        self.assertEqual(links2.num_rows, 2 * links1.num_rows)
        for a, b in zip(links2['tracklet_a'].to_pylist(), links2['tracklet_b'].to_pylist()):
            self.assertEqual('/tile' in a, '/tile' in b)          # never across tiles

    def test_time_link_counts_match_pipeline(self):
        cfg = BridgeConfig(max_gap=30)
        stats = {}
        seconds = bench.time_link(bench.tile_observations(SCENE, 2), cfg, 2, stats)
        self.assertEqual(len(seconds), 2)
        self.assertTrue(all(s > 0 for s in seconds))
        with tempfile.TemporaryDirectory() as tmp:
            pq.write_table(SCENE, Path(tmp) / 'one.parquet')
            rows = bench.time_pipeline(Path(tmp) / 'one.parquet', Path(tmp) / 'out', cfg, 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(stats['n_links'], 2 * rows[0]['n_links'])
        self.assertEqual(stats['candidate_pairs'], 2 * rows[0]['candidate_pairs'])
        self.assertTrue(all(r['total_s'] > 0 for r in rows))


# ============================================================
# SUMMARY AND CLI
# ============================================================


class SummaryTests(unittest.TestCase):
    """summarise groups by (config, tile, stage)."""

    def test_summarise(self):
        base = {'config': 'c', 'tile': 1, 'n_rows': 10, 'n_tracklets': 2, 'n_links': 1,
                'candidate_pairs': 3}
        rows = [{**base, 'stage': 'total', 'seconds': s} for s in (3.0, 1.0, 2.0)]
        rows += [{**base, 'stage': 'link', 'seconds': 0.5}]
        out = bench.summarise(rows)
        self.assertEqual([(r['stage'], r['repeats']) for r in out], [('total', 3), ('link', 1)])
        self.assertEqual((out[0]['median_s'], out[0]['min_s'], out[0]['max_s']), (2.0, 1.0, 3.0))
        text = bench.markdown_table(out)
        self.assertEqual(len(text.strip().splitlines()), 4)
        self.assertIn('| c | 1 | 10 | 2 | total | 2.0000 | 1.0000 | 3.0000 | 3 | 1 | 3 |', text)

    def test_csv_columns_constant(self):
        self.assertEqual(bench.CSV_COLUMNS, EXPECTED_COLUMNS)


class CliTests(unittest.TestCase):
    """main on the synthetic scene: link-only and full mode, with and without timings."""

    def run_main(self, tmp: Path, *extra: str) -> tuple[list[dict], str]:
        path = tmp / 'scene.parquet'
        pq.write_table(SCENE, path)
        out, md = tmp / 'bench.csv', tmp / 'bench.md'
        code = bench.main(['--observations', str(path), '--repeats', '1', '--tiles', '1,2',
                           '--out', str(out), '--markdown', str(md), *extra])
        self.assertEqual(code, 0)
        with open(out, newline='') as handle:
            self.assertEqual(tuple(csv.DictReader(handle).fieldnames), EXPECTED_COLUMNS)
        return read_csv(out), md.read_text()

    def test_link_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, md = self.run_main(Path(tmp), '--link-only')
        self.assertEqual([(r['config'], r['tile'], r['stage']) for r in rows],
                         [('default', '1', bench.LINK_STAGE), ('default', '2', bench.LINK_STAGE)])
        self.assertEqual(int(rows[1]['n_rows']), 2 * int(rows[0]['n_rows']))
        self.assertEqual(int(rows[1]['n_links']), 2 * int(rows[0]['n_links']))
        self.assertEqual(int(rows[1]['n_tracklets']), 2 * int(rows[0]['n_tracklets']))
        self.assertTrue(all(r['repeats'] == '1' and r['host'] and r['numpy'] for r in rows))
        self.assertIn('| config | tile |', md)
        self.assertEqual(sum(line.startswith('| default |') for line in md.splitlines()), 2)

    def test_full_mode_with_two_configs(self):
        yaml = ROOT / 'configs/tracking/gap_bridging_v001.yaml'
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / 'work'
            rows, md = self.run_main(Path(tmp), '--configs', f'a={yaml}', f'b={yaml}',
                                     '--workdir', str(work))
            self.assertTrue((work / 'tiles_02.parquet').is_file())
        totals = [r for r in rows if r['stage'] == bench.TOTAL_STAGE]
        self.assertEqual([(r['config'], r['tile']) for r in totals],
                         [('a', '1'), ('b', '1'), ('a', '2'), ('b', '2')])
        self.assertEqual(totals[0]['n_links'], totals[1]['n_links'])
        self.assertEqual(int(totals[2]['n_links']), 2 * int(totals[0]['n_links']))
        self.assertIn('full pipeline', md)

    def test_full_mode_without_marker_timings(self):
        real = pipeline.run

        def no_timings(*args, **kwargs):
            marker = dict(real(*args, **kwargs))
            marker.pop('timings', None)
            return marker

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(pipeline, 'run', side_effect=no_timings):
            rows, _ = self.run_main(Path(tmp))
        self.assertEqual({r['stage'] for r in rows}, {bench.TOTAL_STAGE})
        self.assertEqual(len(rows), 2)

    def test_marker_timings_become_stages(self):
        fake = {'counts': {'links': 3, 'candidate_pairs': {'enumerated': 7}},
                'timings': {'read': 0.1, 'link': 0.2, 'total': 0.3, 'note': 'x'}}
        with mock.patch.object(pipeline, 'run', return_value=fake):
            rows = bench.time_pipeline(Path('unused'), Path('unused'), BridgeConfig(), 2)
        self.assertEqual(rows[0]['timings'], {'read': 0.1, 'link': 0.2, bench.MARKER_TOTAL: 0.3})
        self.assertEqual((rows[0]['n_links'], rows[0]['candidate_pairs']), (3, 7))

    def test_bad_input_returns_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = bench.main(['--observations', str(Path(tmp) / 'missing.parquet'),
                               '--out', str(Path(tmp) / 'x.csv'),
                               '--markdown', str(Path(tmp) / 'x.md'), '--link-only'])
        self.assertEqual(code, 1)


if __name__ == '__main__':
    unittest.main()
