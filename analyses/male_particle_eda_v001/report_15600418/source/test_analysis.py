"""Known-answer tests of the same estimators used in analysis.py. Slurm only."""
import os
assert os.environ.get('SLURM_JOB_ID'), 'Run tests in Slurm'
import unittest
import report  # Validate report imports/syntax before processing real data.
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, cdist
from analysis import (segments, pairs, acf, displacement_stats, radial_histogram,
                      rectangle_pair_histogram, bootstrap_mean, W, H, RBINS)


class AnalysisControls(unittest.TestCase):
    def test_stationary(self):
        stats = displacement_stats(np.zeros((100,2)))
        self.assertEqual(stats['msd'], 0)
        self.assertEqual(stats['overlap'], 1)
        self.assertTrue(np.isnan(stats['alpha2']))  # Undefined, never spuriously Gaussian.

    def test_ballistic(self):
        frame=np.arange(100); seg=np.zeros(100,int)
        xy=np.column_stack([3*frame,4*frame])
        for lag in [1,3,10,30]:
            i=pairs(seg,frame,lag,1)
            stats=displacement_stats(xy[i+lag]-xy[i])
            self.assertEqual(len(i),100-lag)
            self.assertEqual(stats['msd'],25*lag**2)
            self.assertAlmostEqual(stats['alpha2'],-.5)

    def test_gaussian_walk_diffusion(self):
        rng=np.random.default_rng(137)
        # Independent 2-D walks; each component variance per frame = 4.
        increments=rng.normal(0,2,(16000,40,2))
        xy=np.concatenate([np.zeros((16000,1,2)),increments.cumsum(axis=1)],axis=1)
        for lag in [1,5,20,40]:
            stats=displacement_stats(xy[:,lag]-xy[:,0])
            self.assertLess(abs(stats['msd']/(8*lag)-1),.04)
            self.assertLess(abs(stats['alpha2']),.05)

    def test_localization_noise_plateau(self):
        rng=np.random.default_rng(201)
        # Stationary true particle, independent localization sigma=2 px per axis.
        a=rng.normal(0,2,(100000,2)); b=rng.normal(0,2,(100000,2))
        stats=displacement_stats(b-a)
        self.assertLess(abs(stats['msd']/16-1),.02)
        self.assertLess(abs(stats['alpha2']),.03)

    def test_no_gap_or_identity_bridging(self):
        ids=np.array([1,1,1,1,2,2,2]); f=np.array([0,1,3,4,0,1,2])
        sg,ll=segments(ids,f)
        self.assertEqual(ll.tolist(),[2,2,3])
        self.assertEqual(pairs(sg,f,1,1).tolist(),[0,2,4,5])
        self.assertEqual(pairs(sg,f,2,1).tolist(),[4])
        self.assertEqual(len(pairs(sg,f,3,1)),0)

    def test_land_filter_splits_crossings(self):
        f=np.arange(8); ids=np.ones(8,int); land=np.array([1,1,0,1,1,1,0,1],bool)
        sg,ll=segments(ids[land],f[land])
        self.assertEqual(ll.tolist(),[2,3,1])
        i=pairs(sg,f[land],1,1)
        self.assertTrue(np.all(np.diff(f[land])[i]==1))
        self.assertEqual(len(pairs(sg,f[land],3,1)),0)

    def test_fixed_origin_cohort(self):
        f=np.arange(50); sg=np.zeros(50,int)
        fixed=pairs(sg,f,30,10)
        self.assertEqual(fixed.tolist(),[0,10])
        for lag in [1,10,30]:
            self.assertTrue(set(fixed).issubset(set(pairs(sg,f,lag,10))))

    def test_radial_pdf_with_atom_at_zero(self):
        r=np.array([0,0,.05,1,10,100,10000])
        h,p=radial_histogram(r)
        self.assertEqual(h.sum(),len(r))
        self.assertAlmostEqual(np.sum(p*np.diff(RBINS)),1)
        self.assertEqual(h[0],3)
        with self.assertRaises(AssertionError): radial_histogram(np.array([30000.]))

    def test_rayleigh_cdf(self):
        rng=np.random.default_rng(218)
        r=np.linalg.norm(rng.normal(size=(120000,2)),axis=1)
        h,p=radial_histogram(r)
        observed=np.cumsum(h)/len(r)
        exact=1-np.exp(-RBINS[1:]**2/2)
        self.assertLess(np.max(abs(observed-exact)),.008)

    def test_nearest_neighbor_and_pair_exclusion(self):
        xy=np.array([[0.,0.],[3.,4.],[6.,8.]])
        self.assertEqual(sorted(pdist(xy)),[5,5,10])
        np.testing.assert_allclose(cKDTree(xy).query(xy,k=2)[0][:,1],[5,5,5])
        h=rectangle_pair_histogram(xy,np.array([0,6,11]))
        expected=np.array([2*W*H/((W-3)*(H-4)),W*H/((W-6)*(H-8))])
        np.testing.assert_allclose(h,expected)

    def test_uniform_rectangle_g_and_time_mixed_ratio(self):
        rng=np.random.default_rng(17); bins=np.arange(100,2100,200)
        same=np.zeros(len(bins)-1); mixed=same.copy(); cor=same.copy()
        n=140
        for _ in range(160):
            a=rng.random((n,2))*[W,H]; b=rng.random((n,2))*[W,H]
            same+=np.histogram(pdist(a),bins)[0]
            mixed+=np.histogram(cdist(a,b),bins)[0]
            cor+=rectangle_pair_histogram(a,bins)
        g=cor/(160*n*(n-1)/2*np.pi*np.diff(bins**2)/(W*H))
        ratio=(same/(n*(n-1)/2))/(mixed/(n*n))
        self.assertLess(np.max(abs(g-1)),.12)
        self.assertLess(np.max(abs(ratio-1)),.12)

    def test_iid_and_ar1_autocorrelation(self):
        rng=np.random.default_rng(199); n=100000; noise=rng.normal(size=n)
        a=acf(noise,10)
        self.assertAlmostEqual(a[0],1)
        self.assertLess(np.max(abs(a[1:])),.02)
        x=np.empty(n); x[0]=noise[0]
        for i in range(1,n): x[i]=.8*x[i-1]+noise[i]
        a=acf(x,10)
        np.testing.assert_allclose(a,.8**np.arange(11),atol=.025)

    def test_cell_correlation_removes_static_occupancy(self):
        rng=np.random.default_rng(3); x=rng.normal(size=(10000,4))
        np.testing.assert_allclose(acf(x,10),acf(x+np.array([10,100,50,20]),10),atol=1e-12)

    def test_block_resampling_constant(self):
        frames=np.arange(28800)[::10]
        lo,hi=bootstrap_mean(np.full(len(frames),7.),frames,np.random.default_rng(3))
        self.assertEqual(lo,7); self.assertEqual(hi,7)

    def test_coordinate_rotation_invariance(self):
        rng=np.random.default_rng(5); d=rng.normal(size=(1000,2)); theta=.7
        rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        for key in ['msd','alpha2','overlap']:
            self.assertAlmostEqual(displacement_stats(d)[key],displacement_stats(d@rot)[key])

    def test_invalid_displacements_fail(self):
        with self.assertRaises(AssertionError): displacement_stats(np.array([[np.nan,0]]))
        with self.assertRaises(AssertionError): displacement_stats(np.empty((0,2)))


if __name__=='__main__':
    import json, sys
    from pathlib import Path
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AnalysisControls))
    receipt={'job_id':os.environ['SLURM_JOB_ID'],'tests_run':result.testsRun,
             'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful()}
    (Path(__file__).parent/f"unit_tests_{os.environ['SLURM_JOB_ID']}.json").write_text(json.dumps(receipt,indent=2)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
