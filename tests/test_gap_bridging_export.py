"""Tests for gap-bridging v003 WP-B: weak-label export of interpolated frames."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from gap_bridging import export, pipeline, tables  # noqa: E402
from gap_bridging.config import BridgeConfig  # noqa: E402
from pipeline_common import sha256  # noqa: E402

DEMO = ROOT / 'demo/crab_cam2_2_middle_300/observations.parquet'
DEMO_FIRST_FRAME = 14250
ALL = ('high', 'medium', 'low')


class DemoExportTests(unittest.TestCase):
    """Export from one demo run at max_gap 15 (shared by every test)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls.tmp.name) / 'run'
        cls.marker = pipeline.run(DEMO, cls.run_dir, BridgeConfig(max_gap=15))
        cls.obs = pq.read_table(cls.run_dir / 'observations.parquet')
        cls.n_interpolated = cls.marker['counts']['n_interpolated']

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def weak(self, **kwargs) -> pa.Table:
        params = dict(frame_offset=None, max_gap_frames=None, confidences=ALL)
        params.update(kwargs)
        return export.export_weak_labels(self.obs, **params)

    def test_all_interpolated_rows(self):
        weak = self.weak()
        self.assertTrue(weak.schema.equals(export.WEAK_SCHEMA))
        self.assertEqual(weak.num_rows, self.n_interpolated)
        self.assertGreater(weak.num_rows, 0)
        self.assertEqual(set(weak['source'].to_pylist()), {'interpolated'})
        self.assertEqual(set(weak['label_status'].to_pylist()), {'weak_interpolated'})
        keys = list(zip(weak['trajectory_id'].to_pylist(), weak['frame'].to_pylist()))
        self.assertEqual(len(set(keys)), len(keys))

    def test_local_frame_from_demo_frame(self):
        self.assertIn('demo_frame', self.obs.schema.names)
        weak = self.weak()
        local = weak['local_frame'].to_numpy()
        self.assertTrue(((local >= 0) & (local <= 299)).all())
        np.testing.assert_array_equal(local, weak['frame'].to_numpy() - DEMO_FIRST_FRAME)
        explicit = self.weak(frame_offset=DEMO_FIRST_FRAME)
        np.testing.assert_array_equal(explicit['local_frame'].to_numpy(), local)
        without = export.export_weak_labels(self.obs.drop_columns(['demo_frame']),
                                            frame_offset=None, max_gap_frames=None,
                                            confidences=ALL)
        np.testing.assert_array_equal(without['local_frame'].to_numpy(), weak['frame'].to_numpy())
        with self.assertRaises(ValueError):
            self.weak(frame_offset=DEMO_FIRST_FRAME + 1000)

    def test_crops_inside_frame(self):
        for size in (64, 256, 1000):
            with self.subTest(crop_size=size):
                weak = self.weak(crop_size=size)
                x0, y0 = weak['crop_x0'].to_numpy(), weak['crop_y0'].to_numpy()
                self.assertTrue((x0 >= 0).all() and (x0 + size <= export.WIDTH).all())
                self.assertTrue((y0 >= 0).all() and (y0 + size <= export.HEIGHT).all())
                self.assertEqual(set(weak['crop_size'].to_pylist()), {size})
                # Unclipped windows are centered on the rounded Shell.
                cx = np.round(weak['shell_x'].to_numpy()).astype(np.int64) - size // 2
                inside = (cx >= 0) & (cx <= export.WIDTH - size)
                np.testing.assert_array_equal(x0[inside], cx[inside])
        small = self.weak(width=500, height=400, crop_size=128)
        self.assertTrue((small['crop_x0'].to_numpy() <= 500 - 128).all())
        self.assertTrue((small['crop_y0'].to_numpy() <= 400 - 128).all())

    def test_filters(self):
        weak = self.weak()
        conf = weak['fill_confidence'].to_pylist()
        high = self.weak(confidences=('high',))
        self.assertEqual(high.num_rows, conf.count('high'))
        self.assertLess(high.num_rows, weak.num_rows)
        default = export.export_weak_labels(self.obs, frame_offset=None, max_gap_frames=None)
        self.assertEqual(default.num_rows, conf.count('high') + conf.count('medium'))
        gaps = weak['gap_frames'].to_numpy()
        for limit in (1, 3, 15):
            with self.subTest(max_gap_frames=limit):
                self.assertEqual(self.weak(max_gap_frames=limit).num_rows,
                                 int((gaps <= limit).sum()))
        with self.assertRaises(ValueError):
            self.weak(confidences=('certain',))
        with self.assertRaises(ValueError):
            self.weak(crop_size=0)

    def test_write_and_round_trip(self):
        weak = self.weak()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'weak'
            marker = export.write_weak_labels(weak, out, self.marker, force=False)
            back = pq.read_table(out / export.PARQUET_NAME)
            self.assertTrue(back.equals(weak))
            readme = (out / export.README_NAME).read_text()
            self.assertIn('fine-tuning candidates, not labels', readme)
            self.assertEqual(marker['counts']['rows'], weak.num_rows)
            self.assertEqual(marker['outputs'][export.PARQUET_NAME]['sha256'],
                             sha256(out / export.PARQUET_NAME))
            on_disk = json.loads((out / export.COMPLETE_NAME).read_text())
            self.assertEqual(on_disk, marker)
            with self.assertRaises(FileExistsError):
                export.write_weak_labels(weak, out, self.marker, force=False)
            export.write_weak_labels(weak, out, self.marker, force=True)

    def test_main_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'weak'
            code = export.main(['--bridged', str(self.run_dir), '--out', str(out),
                                '--max-gap-frames', '5', '--confidences', 'high', 'medium',
                                'low', '--crop-size', '128'])
            self.assertEqual(code, 0)
            marker = json.loads((out / export.COMPLETE_NAME).read_text())
            gaps = self.obs.filter(pc.equal(self.obs['source'], 'interpolated'))['gap_frames']
            self.assertEqual(marker['counts']['rows'],
                             pc.sum(pc.less_equal(gaps, 5)).as_py())
            self.assertEqual(marker['provenance']['source_marker_sha256'],
                             sha256(self.run_dir / tables.COMPLETE_NAME))
            self.assertEqual(marker['provenance']['source_interpolated_rows'], self.n_interpolated)
            self.assertTrue((out / export.README_NAME).exists())
            self.assertEqual(set(pq.read_table(out / export.PARQUET_NAME)['crop_size']
                                 .to_pylist()), {128})


if __name__ == '__main__':
    unittest.main()
