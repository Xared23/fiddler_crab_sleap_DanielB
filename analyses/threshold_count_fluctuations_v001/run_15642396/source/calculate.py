"""Adjacent-frame count fluctuations for the three existing 200-frame predictions."""
from pathlib import Path
import hashlib
import json
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
SOURCE = ROOT / 'predictions/test runs/threshold_261015_2000_15641165'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fluctuation_stats(frames, counts):
    frames = np.asarray(frames)
    counts = np.asarray(counts, dtype=float)
    if frames.ndim != 1 or counts.shape != frames.shape or len(frames) < 3:
        raise ValueError('Need at least three aligned frames for both variance conventions')
    if not np.isfinite(frames).all() or not np.all(np.diff(frames) == 1):
        raise ValueError('Frames must be consecutive; never bridge gaps')
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError('Counts must be nonnegative finite integers')
    delta = np.diff(counts)
    absolute = np.abs(delta)
    result = {
        'n_frames': len(counts), 'n_frame_pairs': len(delta), 'mean_count': float(counts.mean()),
        'mean_absolute_change': float(absolute.mean()),
        'variance_absolute_change_population': float(absolute.var(ddof=0)),
        'variance_absolute_change_sample': float(absolute.var(ddof=1)),
        'mean_signed_change': float(delta.mean()),
        'variance_signed_change_population': float(delta.var(ddof=0)),
        'variance_signed_change_sample': float(delta.var(ddof=1)),
        'sd_signed_change_population': float(delta.std(ddof=0)),
        'rms_change': float(np.sqrt(np.mean(delta**2))),
        'fraction_unchanged': float(np.mean(delta == 0)),
        'mean_absolute_change_percent_of_mean_count': float(100*absolute.mean()/counts.mean()) if counts.mean() else None,
    }
    assert np.isclose(delta.sum(), counts[-1]-counts[0])
    assert np.isclose(result['rms_change']**2,
                      result['variance_signed_change_population']+result['mean_signed_change']**2)
    assert np.isclose(result['rms_change']**2,
                      result['variance_absolute_change_population']+result['mean_absolute_change']**2)
    return result, delta


class KnownAnswerTests(unittest.TestCase):
    def test_constant(self):
        s,d = fluctuation_stats(np.arange(5), [7]*5)
        self.assertEqual(s['mean_absolute_change'],0)
        self.assertEqual(s['variance_absolute_change_population'],0)
        self.assertEqual(s['variance_signed_change_population'],0)
        self.assertEqual(s['fraction_unchanged'],1)

    def test_linear_increase(self):
        s,d = fluctuation_stats(np.arange(4),[1,4,7,10])
        self.assertEqual(s['mean_signed_change'],3)
        self.assertEqual(s['mean_absolute_change'],3)
        self.assertEqual(s['variance_absolute_change_population'],0)
        self.assertEqual(s['variance_signed_change_population'],0)

    def test_alternating(self):
        s,d = fluctuation_stats(np.arange(4),[0,2,0,2])
        self.assertEqual(s['mean_absolute_change'],2)
        self.assertEqual(s['variance_absolute_change_population'],0)
        self.assertAlmostEqual(s['mean_signed_change'],2/3)
        self.assertAlmostEqual(s['variance_signed_change_population'],32/9)

    def test_hand_calculated_variances(self):
        s,d = fluctuation_stats(np.arange(4),[10,12,11,15])
        np.testing.assert_array_equal(d,[2,-1,4])
        self.assertAlmostEqual(s['mean_absolute_change'],7/3)
        self.assertAlmostEqual(s['variance_absolute_change_population'],14/9)
        self.assertAlmostEqual(s['variance_absolute_change_sample'],7/3)
        self.assertAlmostEqual(s['mean_signed_change'],5/3)
        self.assertAlmostEqual(s['variance_signed_change_population'],38/9)
        self.assertAlmostEqual(s['variance_signed_change_sample'],19/3)

    def test_offset_invariance(self):
        _,d = fluctuation_stats(np.arange(4),[0,2,1,5])
        _,e = fluctuation_stats(np.arange(4),[100,102,101,105])
        np.testing.assert_array_equal(d,e)

    def test_gap_rejected(self):
        with self.assertRaises(ValueError): fluctuation_stats([0,1,3],[1,2,3])

    def test_invalid_counts_rejected(self):
        for counts in [[0,1,np.nan],[0,-1,2],[0,1.5,2]]:
            with self.assertRaises(ValueError): fluctuation_stats([0,1,2],counts)


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run only through Slurm'
    out = BASE / f"run_{os.environ['SLURM_JOB_ID']}"
    out.mkdir(exist_ok=False)
    with (out/'tests.txt').open('w') as stream:
        tests = unittest.TextTestRunner(stream=stream,verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(KnownAnswerTests))
    if not tests.wasSuccessful(): raise RuntimeError('Known-answer tests failed')
    complete = json.loads((SOURCE/'COMPLETE.json').read_text())
    assert complete['status'] == 'complete'
    pins = {a['path']:a['sha256'] for a in complete['artifacts']}
    for name in ['counts_per_frame.csv','summary.json']:
        assert sha(SOURCE/name) == pins[name]
    counts = pd.read_csv(SOURCE/'counts_per_frame.csv')
    source_summary = json.loads((SOURCE/'summary.json').read_text())
    assert len(counts)==600 and set(counts.centroid_threshold)=={.2,.1,.05}
    rows, differences = [], []
    for threshold in [.2,.1,.05]:
        d = counts[counts.centroid_threshold==threshold].sort_values('frame')
        np.testing.assert_array_equal(d.frame,np.arange(200))
        np.testing.assert_allclose(d.time_s,d.frame*1001/30000,atol=1e-12)
        np.testing.assert_array_equal(d.assigned+d.unassigned,d.raw)
        assert (d.assigned<=d.finite_in_image).all() and (d.finite_in_image<=d.raw).all()
        s = source_summary[f'c{round(threshold*1000):03d}']
        assert d.raw.sum()==s['raw_detections']
        assert d.assigned.sum()==s['assigned_observations']
        for population in ['raw','finite_in_image','assigned']:
            stats, delta = fluctuation_stats(d.frame,d[population])
            assert stats['n_frame_pairs']==199
            rows.append(dict(centroid_threshold=threshold,population=population,**stats))
            differences.extend(dict(centroid_threshold=threshold,population=population,
                                    frame_from=int(a),frame_to=int(b),delta_count=int(change),
                                    absolute_delta_count=int(abs(change)))
                               for a,b,change in zip(d.frame.to_numpy()[:-1],d.frame.to_numpy()[1:],delta))
    result = pd.DataFrame(rows)
    result.to_csv(out/'fluctuation_summary.csv',index=False)
    pd.DataFrame(differences).to_csv(out/'adjacent_frame_changes.csv',index=False)
    # Read back the saved results instead of trusting only in-memory statistics.
    saved = pd.read_csv(out/'fluctuation_summary.csv')
    np.testing.assert_allclose(saved.select_dtypes('number'),result.select_dtypes('number'),rtol=1e-12,atol=1e-12)
    primary = result[result.population=='raw']
    lines = ['# Adjacent-frame detection-count fluctuations','',
             'Source: Cam 2_2 frames 0-199, 199 consecutive-frame pairs per threshold.',
             'Frame interval: 1001/30000 s (~33.37 ms). No temporal binning or smoothing.',
             'Primary N is the count of all pose predictions, consistent with the threshold comparison.',
             '', 'Definitions: delta = N(t+1)-N(t); magnitude = abs(delta).',
             'Variances below use divisor 199 (descriptive population variance, ddof=0).',
             'Sample variances (ddof=1), finite-in-image counts and ID-assigned counts are also saved.',
             '', '| Threshold | Mean abs change | Var(abs change) | Mean signed change | Var(signed change) |',
             '|---|---:|---:|---:|---:|']
    for r in primary.to_dict('records'):
        lines.append(f"| {r['centroid_threshold']} | {r['mean_absolute_change']:.6f} | "
                     f"{r['variance_absolute_change_population']:.6f} | {r['mean_signed_change']:.6f} | "
                     f"{r['variance_signed_change_population']:.6f} |")
    lines += ['', 'These are NET count changes: one detection disappearing and another appearing can cancel.',
              'They do not measure pose-coordinate jitter, individual appearance/disappearance rates or ID switches.',
              'The 199 increments are not assumed independent. No inference about true recall/precision is made.']
    (out/'results.md').write_text('\n'.join(lines)+'\n')
    print(primary.to_string(index=False),flush=True)
    receipt = {'status':'complete','job_id':os.environ['SLURM_JOB_ID'],
               'tests_run':tests.testsRun,'tests_passed':True,
               'source_prediction_run':str(SOURCE.relative_to(ROOT)),
               'source_completion_sha256':sha(SOURCE/'COMPLETE.json'),
               'counts_sha256':pins['counts_per_frame.csv'],
               'code_sha256':sha(Path(__file__)),'script_sha256':sha(BASE/'run.sh'),
               'artifacts':[{'path':p.name,'sha256':sha(p)} for p in sorted(out.iterdir()) if p.is_file()]}
    (out/'COMPLETE.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('COMPLETE',out,flush=True)


if __name__=='__main__': main()
