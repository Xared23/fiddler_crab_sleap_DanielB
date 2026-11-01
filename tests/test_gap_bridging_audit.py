#!/usr/bin/env python3

"""
Tests for the visual link audit (gained / lost links of a candidate linker run).

Sam
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import contextlib
import csv
import io
import math
from pathlib import Path
import sys
import tempfile
import time
from typing import Dict, List, Tuple
import unittest

# ----- numerics -----
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))

# ----- local modules -----
from gap_bridging import audit, tables  # noqa: E402
from gap_bridging_fixtures import make_observations, straight_crab  # noqa: E402
import audit_gap_links  # noqa: E402

try:
    import cv2
except ImportError:  # pragma: no cover - the venv ships opencv
    cv2 = None

# ============================================================
# CONSTANTS
# ============================================================

DEMO_DIR = ROOT / 'demo/crab_cam2_2_middle_300'
DEMO_OBS = DEMO_DIR / 'observations.parquet'
DEMO_VIDEO = DEMO_DIR / 'Crab_Cam_2_2.frames_014250_014549.mp4'
SWEEP = ROOT / 'logs/gap_sweep_v004/current/seed_0'
RUN_5, RUN_30 = SWEEP / 'max_gap_5', SWEEP / 'max_gap_30'
DEMO_READY = all(p.is_file() for p in (DEMO_OBS, DEMO_VIDEO, RUN_5 / 'links.parquet',
                                       RUN_30 / 'links.parquet'))
OFFSET = 100          # first observation frame of the synthetic scene
N_FRAMES = 40
WIDTH, HEIGHT = 320, 240

# ============================================================
# HELPERS
# ============================================================

Link = Tuple[str, str, int, float, str, str]


def links(rows: List[Link]) -> pa.Table:
    """A :data:`tables.LINKS_SCHEMA` table from ``(a, b, gap, cost, decision, reason)``."""
    columns: Dict[str, list] = {name: [] for name in tables.LINKS_SCHEMA.names}
    for a, b, gap, cost, decision, reason in rows:
        for name, value in (('video_id', 'synthetic'), ('tracklet_a', a), ('tracklet_b', b),
                            ('gap_frames', gap), ('cost', cost), ('runner_up_cost', math.nan),
                            ('decision', decision), ('reason', reason), ('n_chain_ignored', 0),
                            ('stage', 1), ('size_term', math.nan), ('side_term', math.nan)):
            columns[name].append(value)
    return pa.Table.from_pydict(columns, schema=tables.LINKS_SCHEMA)


def nonempty(path: Path) -> bool:
    """True when ``path`` is a file with content."""
    return Path(path).is_file() and Path(path).stat().st_size > 0


def scene() -> pa.Table:
    """Three straight crabs over frames 100..139 with a ``demo_frame`` column.

    Crab 1 loses frame 120 (two tracklets, gap 1); crab 2 is intact; crab 3
    loses frames 110 and 112, leaving a one-frame tracklet at 111.
    """
    crabs = [straight_crab(1, OFFSET, N_FRAMES, (60.0, 60.0), (2.0, 0.0)),
             straight_crab(2, OFFSET, N_FRAMES, (250.0, 200.0), (0.0, -1.0)),
             straight_crab(3, OFFSET, N_FRAMES, (60.0, 180.0), (1.5, 0.0))]
    table = make_observations(crabs, drop_frames={1: [120], 3: [110, 112]})
    local = pc.subtract(table['frame'], pa.scalar(OFFSET, pa.int64()))
    return table.append_column('demo_frame', local)


def tracklet_at(table: pa.Table, frame: int, x: float) -> str:
    """Tracklet id of the accepted row at ``frame`` whose shell x is nearest ``x``."""
    t = table.to_pandas()
    t = t[(t.frame == frame) & t.accepted_for_tracking]
    return t.iloc[int(np.argmin(np.abs(t.body_x.to_numpy() - x)))].tracklet_id


def write_video(path: Path, table: pa.Table) -> bool:
    """Write a 40-frame 320x240 MP4 with a disc at every shell; False if cv2 cannot."""
    if cv2 is None:
        return False
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (WIDTH, HEIGHT))
    if not writer.isOpened():
        return False
    t = table.to_pandas()
    for index in range(N_FRAMES):
        img = np.full((HEIGHT, WIDTH, 3), 90, np.uint8)
        img[::20] = 60
        for _, row in t[t.frame == OFFSET + index].iterrows():
            cv2.circle(img, (int(row.body_x), int(row.body_y)), 8, (40, 160, 220), -1)
        writer.write(img)
    writer.release()
    return nonempty(path)


def read_csv(path: Path) -> List[dict]:
    """Rows of a CSV file as dictionaries."""
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


class SyntheticScene:
    """Tracklet ids and link tables of :func:`scene` shared by the test classes."""

    def __init__(self):
        self.table = scene()
        self.a1 = tracklet_at(self.table, 119, 60.0 + 2 * 19)
        self.b1 = tracklet_at(self.table, 121, 60.0 + 2 * 21)
        self.a3 = tracklet_at(self.table, 109, 60.0 + 1.5 * 9)
        self.s3 = tracklet_at(self.table, 111, 60.0 + 1.5 * 11)
        self.b3 = tracklet_at(self.table, 113, 60.0 + 1.5 * 13)
        # current links crab 1 only; candidate drops crab 1 and links crab 3 twice
        self.current = links([(self.a1, self.b1, 1, 0.1, 'linked', 'cost 0.1'),
                              (self.a3, self.s3, 1, 0.9, 'refused_gate', 'displacement')])
        self.candidate = links([(self.a1, self.b1, 1, 0.7, 'refused_margin', 'margin'),
                                (self.a3, self.s3, 1, 0.2, 'linked', 'cost 0.2'),
                                (self.s3, self.b3, 1, 0.3, 'linked', 'cost 0.3')])


# ============================================================
# LINK DIFFERENCE AND SAMPLING
# ============================================================


class LinkDiffTests(unittest.TestCase):
    """linked_pairs / link_diff: decision 'linked' only, sorted by (gap, a, b)."""

    def test_gained_and_lost_sorted_by_gap_then_ids(self):
        current = links([('a', 'b', 3, 0.1, 'linked', ''), ('c', 'd', 1, 0.1, 'linked', ''),
                         ('e', 'f', 2, 0.1, 'linked', ''), ('k', 'l', 4, 0.5, 'linked', '')])
        candidate = links([('a', 'b', 3, 0.1, 'linked', ''), ('g', 'h', 5, 0.2, 'linked', ''),
                           ('i', 'j', 2, 0.2, 'linked', ''), ('h', 'z', 2, 0.2, 'linked', ''),
                           ('k', 'l', 4, 0.9, 'refused_margin', '')])
        gained, lost = audit.link_diff(current, candidate)
        self.assertEqual(gained, [('h', 'z'), ('i', 'j'), ('g', 'h')])
        self.assertEqual(lost, [('c', 'd'), ('e', 'f'), ('k', 'l')])

    def test_identical_tables_give_no_difference(self):
        table = links([('a', 'b', 3, 0.1, 'linked', ''), ('c', 'd', 1, 0.4, 'refused_gate', '')])
        self.assertEqual(audit.link_diff(table, table), ([], []))

    def test_refused_rows_are_ignored(self):
        current = links([('a', 'b', 1, 0.1, 'refused_gate', ''),
                         ('c', 'd', 1, 0.1, 'refused_assignment', '')])
        candidate = links([('a', 'b', 1, 0.1, 'refused_margin', ''),
                           ('e', 'f', 1, 0.1, 'refused_assignment', '')])
        self.assertEqual(audit.linked_pairs(current), set())
        self.assertEqual(audit.link_diff(current, candidate), ([], []))

    def test_empty_tables(self):
        empty = links([])
        self.assertEqual(audit.link_diff(empty, empty), ([], []))


class SampleLinksTests(unittest.TestCase):
    """sample_links keeps every pair when few, else a seeded ordered subset."""

    def setUp(self):
        self.pairs = [(f'a{i}', f'b{i}') for i in range(30)]

    def test_all_kept_in_order_when_n_covers(self):
        self.assertEqual(audit.sample_links(self.pairs[:5], 5, seed=1), self.pairs[:5])
        self.assertEqual(audit.sample_links(self.pairs[:5], 20, seed=1), self.pairs[:5])

    def test_seeded_subset_is_deterministic_subsequence(self):
        first = audit.sample_links(self.pairs, 7, seed=3)
        self.assertEqual(first, audit.sample_links(self.pairs, 7, seed=3))
        self.assertEqual(len(first), 7)
        self.assertEqual(len(set(first)), 7)
        positions = [self.pairs.index(p) for p in first]
        self.assertEqual(positions, sorted(positions))
        self.assertNotEqual(first, audit.sample_links(self.pairs, 7, seed=4))

    def test_zero_and_negative(self):
        self.assertEqual(audit.sample_links(self.pairs, 0, seed=0), [])
        with self.assertRaises(ValueError):
            audit.sample_links(self.pairs, -1, seed=0)


# ============================================================
# AUDIT ROWS
# ============================================================


class LinkRowsTests(unittest.TestCase):
    """link_rows looks costs up in both tables and frames in the observations."""

    @classmethod
    def setUpClass(cls):
        cls.s = SyntheticScene()

    def test_both_tables_and_frames(self):
        s = self.s
        (row,) = audit.link_rows(s.table, s.current, s.candidate, [(s.a1, s.b1)], 'lost')
        self.assertEqual(tuple(row), audit.AUDIT_CSV_COLUMNS)
        self.assertEqual((row['kind'], row['tracklet_a'], row['tracklet_b']), ('lost', s.a1, s.b1))
        self.assertEqual(row['gap_frames'], 1)
        self.assertEqual((row['end_frame_a'], row['start_frame_b']), (119, 121))
        self.assertEqual((row['n_frames_a'], row['n_frames_b']), (20, 19))
        self.assertAlmostEqual(row['cost_current'], 0.1)
        self.assertAlmostEqual(row['cost_candidate'], 0.7)
        self.assertEqual((row['decision_current'], row['decision_candidate']),
                         ('linked', 'refused_margin'))
        self.assertEqual((row['reason_current'], row['reason_candidate']), ('cost 0.1', 'margin'))
        self.assertIsNone(row['page'])
        self.assertIsNone(row['row'])

    def test_absent_pair_and_single_frame_tracklet(self):
        s = self.s
        rows = audit.link_rows(s.table, s.current, s.candidate, [(s.a3, s.s3), (s.s3, s.b3)],
                               'gained')
        self.assertEqual(rows[0]['decision_current'], 'refused_gate')
        self.assertEqual((rows[0]['n_frames_a'], rows[0]['n_frames_b']), (10, 1))
        self.assertEqual((rows[1]['end_frame_a'], rows[1]['start_frame_b']), (111, 113))
        self.assertEqual(rows[1]['decision_current'], 'absent')
        self.assertEqual(rows[1]['reason_current'], 'absent')
        self.assertTrue(math.isnan(rows[1]['cost_current']))
        self.assertEqual(rows[1]['decision_candidate'], 'linked')
        self.assertEqual(rows[1]['gap_frames'], 1)

    def test_rejected_rows_not_counted(self):
        s = self.s
        t = s.table.to_pandas()
        mask = (t.tracklet_id == s.b1) & (t.frame == 130)
        t.loc[mask, 'accepted_for_tracking'] = False
        t.loc[mask, 'tracklet_id'] = None
        t.loc[(t.tracklet_id == s.b1) & (t.frame == 131), 'accepted_for_tracking'] = False
        table = pa.Table.from_pandas(t, schema=s.table.schema, preserve_index=False)
        (row,) = audit.link_rows(table, s.current, s.candidate, [(s.a1, s.b1)], 'lost')
        self.assertEqual(row['n_frames_b'], 17)

    def test_unknown_tracklet_raises(self):
        s = self.s
        with self.assertRaises(ValueError):
            audit.link_rows(s.table, s.current, s.candidate, [(s.a1, 'nope')], 'lost')
        with self.assertRaises(ValueError):
            audit.link_rows(s.table, s.current, s.candidate, [(s.a1, s.b1)], 'odd')


# ============================================================
# RENDERED AUDIT
# ============================================================


@unittest.skipIf(cv2 is None, 'cv2 unavailable')
class FigLinkAuditTests(unittest.TestCase):
    """fig_link_audit on a synthetic MP4 whose frame 0 is observation frame 100."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.s = SyntheticScene()
        cls.video = cls.tmp / 'input' / 'clip.mp4'
        cls.video.parent.mkdir()
        if not write_video(cls.video, cls.s.table):
            raise unittest.SkipTest('cv2 cannot write mp4v')

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def rows(self) -> List[dict]:
        s = self.s
        return (audit.link_rows(s.table, s.current, s.candidate, [(s.a3, s.s3), (s.s3, s.b3)],
                                'gained')
                + audit.link_rows(s.table, s.current, s.candidate, [(s.a1, s.b1)], 'lost'))

    def test_one_page_with_offset_and_single_frame_tracklet(self):
        rows = self.rows()
        pages = audit.fig_link_audit(self.s.table, self.video, rows, self.tmp / 'out' / 'audit',
                                     frame_offset=OFFSET)
        self.assertEqual([p.name for p in pages], ['audit_p1.png'])
        self.assertTrue(nonempty(pages[0]))
        self.assertEqual([(r['page'], r['row']) for r in rows],
                         [('audit_p1.png', 1), ('audit_p1.png', 2), ('audit_p1.png', 3)])

    def test_offset_default_from_demo_frame(self):
        pages = audit.fig_link_audit(self.s.table, self.video, self.rows()[:1],
                                     self.tmp / 'default' / 'audit')
        self.assertEqual(len(pages), 1)

    def test_wrong_offset_reads_past_the_video(self):
        table = self.s.table.drop_columns(['demo_frame'])
        with self.assertRaises(RuntimeError):
            audit.fig_link_audit(table, self.video, self.rows()[:1], self.tmp / 'bad' / 'audit')

    def test_pagination_and_empty(self):
        rows = self.rows()
        pages = audit.fig_link_audit(self.s.table, self.video, rows, self.tmp / 'pg' / 'audit',
                                     rows_per_page=2)
        self.assertEqual([p.name for p in pages], ['audit_p1.png', 'audit_p2.png'])
        self.assertEqual([r['row'] for r in rows], [1, 2, 1])
        self.assertEqual(audit.fig_link_audit(self.s.table, self.video, [],
                                              self.tmp / 'none' / 'audit'), [])

    def test_frames_per_row_must_be_odd(self):
        with self.assertRaises(ValueError):
            audit.fig_link_audit(self.s.table, self.video, self.rows(), self.tmp / 'x' / 'audit',
                                 frames_per_row=4)


@unittest.skipIf(cv2 is None, 'cv2 unavailable')
class AuditCommandSyntheticTests(unittest.TestCase):
    """audit_command and the CLI on synthetic runs (video found next to the observations)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = self.tmp = Path(self._tmp.name)
        s = SyntheticScene()
        self.obs = tmp / 'input' / 'observations.parquet'
        self.obs.parent.mkdir()
        pq.write_table(s.table, self.obs)
        if not write_video(tmp / 'input' / 'clip.mp4', s.table):
            raise unittest.SkipTest('cv2 cannot write mp4v')
        self.current, self.candidate = tmp / 'current', tmp / 'candidate'
        for run, table in ((self.current, s.current), (self.candidate, s.candidate)):
            run.mkdir()
            pq.write_table(table, run / 'links.parquet')
        self.out = tmp / 'audit' / 'cmp'

    def tearDown(self):
        self._tmp.cleanup()

    def test_outputs_counts_and_overwrite(self):
        result = audit.audit_command(self.current, self.candidate, self.obs, self.out, n=1)
        self.assertEqual((result['gained'], result['lost']), (2, 1))
        self.assertEqual((result['rendered_gained'], result['rendered_lost']), (1, 1))
        names = sorted(Path(p).name for p in result['pages'])
        self.assertEqual(names, ['cmp_gained_p1.png', 'cmp_lost_p1.png'])
        rows = read_csv(result['csv'])
        self.assertEqual(Path(result['csv']), self.out.with_name('cmp.csv'))
        self.assertEqual(tuple(rows[0]), audit.AUDIT_CSV_COLUMNS)
        self.assertEqual([r['kind'] for r in rows], ['gained', 'lost'])
        self.assertEqual([r['page'] for r in rows], ['cmp_gained_p1.png', 'cmp_lost_p1.png'])
        with self.assertRaises(FileExistsError):
            audit.audit_command(self.current, self.candidate, self.obs, self.out, n=1)
        again = audit.audit_command(self.current, self.candidate, self.obs, self.out, n=5,
                                    force=True)
        self.assertEqual(len(read_csv(again['csv'])), 3)

    def test_stale_pages_removed_on_force(self):
        audit.audit_command(self.current, self.candidate, self.obs, self.out, n=5)
        stale = self.out.parent / 'cmp_gained_p2.png'
        stale.write_bytes(b'stale page of an earlier, longer audit')
        with self.assertRaises(FileExistsError):
            audit.audit_command(self.current, self.candidate, self.obs, self.out, n=1)
        audit.audit_command(self.current, self.candidate, self.obs, self.out, n=1, force=True)
        self.assertFalse((self.out.parent / 'cmp_gained_p2.png').exists())

    def test_video_discovery_errors(self):
        (self.tmp / 'input' / 'other.mp4').write_bytes(b'x')
        with self.assertRaises(ValueError):
            audit.audit_command(self.current, self.candidate, self.obs, self.out)
        for p in (self.tmp / 'input').glob('*.mp4'):
            p.unlink()
        with self.assertRaises(ValueError):
            audit.audit_command(self.current, self.candidate, self.obs, self.out)

    def test_cli(self):
        argv = ['--current', str(self.current), '--candidate', str(self.candidate),
                '--observations', str(self.obs), '--out', str(self.out), '--n', '2']
        args = audit_gap_links.build_parser().parse_args(argv + ['--seed', '4', '--force'])
        self.assertEqual((args.n, args.seed, args.force, args.video), (2, 4, True, None))
        self.assertEqual(args.current, self.current)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(audit_gap_links.main(argv), 0)
        self.assertIn('gained 2', stdout.getvalue())
        self.assertTrue(self.out.with_name('cmp.csv').is_file())
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(audit_gap_links.main(argv), 1)
        self.assertIn('exist', stderr.getvalue())

    def test_cli_missing_run(self):
        argv = ['--current', str(self.tmp / 'missing'), '--candidate', str(self.candidate),
                '--observations', str(self.obs), '--out', str(self.out)]
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(audit_gap_links.main(argv), 1)
        self.assertIn('links.parquet', stderr.getvalue())
        self.assertFalse(self.out.with_name('cmp.csv').exists())


@unittest.skipUnless(DEMO_READY and cv2 is not None, 'demo video or v004 sweep runs absent')
class AuditCommandDemoTests(unittest.TestCase):
    """End to end on the demo: max_gap 5 (current) against max_gap 30 (candidate)."""

    def test_demo_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'audit' / 'gap5_vs_gap30'
            start = time.perf_counter()
            result = audit.audit_command(RUN_5, RUN_30, DEMO_OBS, out, n=3, seed=0)
            print(f"\n[demo audit] {time.perf_counter() - start:.1f} s", file=sys.stderr)
            gained, lost = audit.link_diff(pq.read_table(RUN_5 / 'links.parquet'),
                                           pq.read_table(RUN_30 / 'links.parquet'))
            self.assertEqual((result['gained'], result['lost']), (len(gained), len(lost)))
            self.assertEqual(result['rendered_gained'], min(3, len(gained)))
            self.assertEqual(result['rendered_lost'], min(3, len(lost)))
            rows = read_csv(result['csv'])
            self.assertEqual(len(rows), min(3, len(gained)) + min(3, len(lost)))
            self.assertGreaterEqual(len(result['pages']), 1)
            for page in result['pages']:
                self.assertTrue(nonempty(page))
            self.assertEqual({r['page'] for r in rows}, {Path(p).name for p in result['pages']})
            for r in rows:
                self.assertEqual(int(r['start_frame_b']) - int(r['end_frame_a']) - 1,
                                 int(r['gap_frames']))
            with self.assertRaises(FileExistsError):
                audit.audit_command(RUN_5, RUN_30, DEMO_OBS, out, n=3)
            again = audit.audit_command(RUN_5, RUN_30, DEMO_OBS, out, n=1, force=True)
            self.assertEqual(len(read_csv(again['csv'])), min(1, len(gained)) + min(1, len(lost)))


if __name__ == '__main__':
    unittest.main()
