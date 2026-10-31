"""Tests for gap-bridging v003 WP-B: re-inference scaffolding (dry run only; no GPU here)."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.venv/bin/python'
SCRIPT = ROOT / 'scripts/reinfer_low_threshold.py'
JOB = ROOT / 'jobs/reinfer_low_threshold.sh'
VIDEO = ROOT / 'demo/crab_cam2_2_middle_300/Crab_Cam_2_2.frames_014250_014549.mp4'


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(PYTHON), str(SCRIPT), *args], capture_output=True, text=True,
                          cwd=ROOT, timeout=120)


class ReinferCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / 'reinfer'

    def tearDown(self):
        self.tmp.cleanup()

    def dry(self, *extra: str) -> subprocess.CompletedProcess:
        return run('--video', str(VIDEO), '--start', '0', '--stop', '300', '--out', str(self.out),
                   '--dry-run', *extra)

    def test_help(self):
        result = run('--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ('--video', '--start', '--stop', '--out', '--centroid-threshold',
                     '--keypoint-threshold', '--baseline-config', '--dry-run'):
            self.assertIn(flag, result.stdout)
        # The frame-number caveat for derive_tracks is part of the help.
        self.assertIn('derive_tracks requires start == 0', ' '.join(result.stdout.split()))

    def test_dry_run_reports_missing_checkpoints(self):
        result = self.dry()
        self.assertEqual(result.returncode, 2, result.stderr)
        plan = json.loads(result.stdout)
        self.assertFalse(plan['ok'])
        self.assertTrue(plan['dry_run'])
        missing = [Path(p) for p in plan['missing']]
        self.assertTrue(missing)
        self.assertTrue(all(not p.exists() for p in missing))
        self.assertEqual({p.name for p in missing}, {'best.ckpt'})
        self.assertEqual(set(plan['models']), {'centroid', 'centered'})
        self.assertEqual(plan['frames'], {'start': 0, 'stop_exclusive': 300, 'count': 300})
        self.assertEqual(plan['thresholds'], {'centroid': 0.05, 'keypoint': 0.2})
        self.assertEqual(Path(plan['outputs']['slp']).name,
                         'Crab_Cam_2_2.frames_014250_014549.frames_000000_000299.predictions.slp')
        self.assertFalse(self.out.exists(), 'a dry run writes nothing')

    def test_dry_run_does_not_import_torch(self):
        code = ("import sys; sys.path.insert(0, 'scripts'); import reinfer_low_threshold as r; "
                f"c = r.main(['--video', {str(VIDEO)!r}, '--start', '0', '--stop', '300', "
                f"'--out', {str(self.out)!r}, '--dry-run']); "
                "print('MODULES', 'torch' in sys.modules, 'sleap_nn' in sys.modules, c)")
        result = subprocess.run([str(PYTHON), '-c', code], capture_output=True, text=True,
                                cwd=ROOT, timeout=120)
        self.assertIn('MODULES False False 2', result.stdout, result.stderr)

    def test_missing_video_named(self):
        result = run('--video', str(self.out / 'absent.mp4'), '--start', '0', '--stop', '10',
                     '--out', str(self.out), '--dry-run')
        self.assertEqual(result.returncode, 2)
        self.assertIn(str(self.out / 'absent.mp4'), json.loads(result.stdout)['missing'])

    def test_invalid_arguments_rejected(self):
        bad = [('--start', '5', '--stop', '5'), ('--start', '-1', '--stop', '5'),
               ('--start', '0', '--stop', '5', '--centroid-threshold', '0'),
               ('--start', '0', '--stop', '5', '--keypoint-threshold', '1.0')]
        for args in bad:
            with self.subTest(args=args):
                result = run('--video', str(VIDEO), '--out', str(self.out), '--dry-run', *args)
                self.assertEqual(result.returncode, 2)
                self.assertIn('error', result.stderr)
                self.assertEqual(result.stdout, '')

    def test_low_centroid_threshold_is_a_problem(self):
        result = self.dry('--centroid-threshold', '0.01')
        self.assertEqual(result.returncode, 2)
        self.assertTrue(any('0.05' in p for p in json.loads(result.stdout)['problems']))


class JobScriptTests(unittest.TestCase):
    def test_bash_syntax(self):
        result = subprocess.run(['bash', '-n', str(JOB)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_root_from_submit_dir_and_gpu_guard(self):
        text = JOB.read_text()
        self.assertIn('ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"', text)
        self.assertNotIn('/scratch/gpfs/', text)
        self.assertIn('nvidia-smi', text)
        self.assertIn('set -euo pipefail', text)
        for directive in ('--gres=gpu:1', '--mem=64G', '--time=01:00:00', '--no-requeue'):
            self.assertIn(f'#SBATCH {directive}', text)


if __name__ == '__main__':
    unittest.main()
