"""Synthetic identity, score-stage, gap-boundary and whole-SLP controls."""
from pathlib import Path
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np
import sleap_io as sio
import torch
from sleap_nn.inference.outputs import Outputs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from body_first_tracklets import Tracker
from grace_tracklets import GraceTracker
from grace_prediction_tests import (apply_condition, clone_instance, unpack_output, infer, control)
from threshold_prediction_tests import NODES, save_final, measurement_snapshot, assert_same_measurements
from test_body_first_tracklets import frame, CONFIG


class GraceTrackerTests(unittest.TestCase):
    def tracker(self, birth=.2, grace=5):
        return GraceTracker(CONFIG, birth, .05, grace, 1000, 1000)

    def test_five_missing_frames_link_six_do_not(self):
        for misses, expected_link in [(0, True), (1, True), (5, True), (6, False)]:
            with self.subTest(misses=misses):
                t = self.tracker()
                original = t.process(frame(0, [30]), [.3])['ids'][0]
                for f in range(1, misses+1):
                    result = t.process(frame(f, []), [])
                    self.assertEqual(len(result['ids']), 0)
                result = t.process(frame(misses+1, [31]), [.3])
                self.assertEqual(result['ids'][0] == original, expected_link)
                tracks = t.finish()
                if expected_link:
                    self.assertEqual(tracks[0]['observations'], 2)
                    self.assertEqual(tracks[0]['missing_frames'], misses)

    def test_weak_continuation_cannot_start_a_track(self):
        t = self.tracker()
        first = t.process(frame(0, [30]), [.1])
        self.assertEqual(first['ids'][0], -1)
        self.assertFalse(first['keep'][0])
        tid = t.process(frame(1, [30]), [.2])['ids'][0]
        result = t.process(frame(2, [31]), [.05])
        self.assertEqual(result['ids'][0], tid)
        self.assertTrue(result['keep'][0])
        self.assertEqual(result['reason'][0], 'adjacent_link')

    def test_lower_threshold_has_independent_birth(self):
        high, low = self.tracker(.2), self.tracker(.1)
        self.assertEqual(high.process(frame(0, [30]), [.15])['ids'][0], -1)
        self.assertGreater(low.process(frame(0, [30]), [.15])['ids'][0], 0)
        self.assertEqual(high.process(frame(1, [30]), [.06])['ids'][0], -1)
        self.assertGreater(low.process(frame(1, [30]), [.06])['ids'][0], 0)

    def test_below_continuation_and_nan_cannot_link(self):
        for score in [.049, np.nan]:
            t = self.tracker()
            t.process(frame(0, [30]), [.3])
            result = t.process(frame(1, [30]), [score])
            self.assertEqual(result['ids'][0], -1)
            self.assertFalse(result['keep'][0])

    def test_centroid_and_shell_scores_are_distinct(self):
        t = self.tracker()
        result = t.process(frame(0, [30, 90], scores=[.9, .3]), [.1, .9])
        np.testing.assert_array_equal(result['ids'], [-1, -1])
        np.testing.assert_array_equal(result['keep'], [False, True])

    def test_expired_track_not_revived_by_weak_pose(self):
        t = self.tracker()
        t.process(frame(0, [30]), [.3])
        for f in range(1, 7): t.process(frame(f, []), [])
        result = t.process(frame(7, [30]), [.1])
        self.assertEqual(result['ids'][0], -1)
        self.assertFalse(t.active)
        self.assertEqual(result['ended'][0]['termination'], 'grace_expired')

    def test_variable_count_birth_exit_and_empty(self):
        t = self.tracker()
        a = t.process(frame(0, [30]), [.3])['ids'][0]
        result = t.process(frame(1, [31, 100, 200]), [.08, .3, .08])
        self.assertEqual(result['ids'][0], a)
        self.assertGreater(result['ids'][1], 0)
        self.assertEqual(result['ids'][2], -1)
        self.assertEqual(result['keep'].sum(), 2)
        for f in range(2, 9):
            self.assertEqual(len(t.process(frame(f, []), [])['ids']), 0)
        self.assertFalse(t.active)

    def test_ambiguous_crossing_no_forced_link_or_shadow_birth(self):
        t = self.tracker()
        t.process(frame(0, [30, 42]), [.3, .3])
        result = t.process(frame(1, [36]), [.3])
        self.assertEqual(result['ids'][0], -1)
        self.assertTrue(result['keep'][0])
        self.assertEqual(result['reason'][0], 'ambiguous_cost_margin')
        self.assertEqual(t.next_id, 3)

    def test_rejected_current_competitor_preserves_ambiguity(self):
        t = self.tracker()
        t.process(frame(0, [30]), [.3])
        result = t.process(frame(1, [24, 36], scores=[.9, .1]), [.1, .1])
        np.testing.assert_array_equal(result['ids'], [-1, -1])

    def test_duplicate_shells_no_ids_and_retained_high_poses(self):
        result = self.tracker().process(frame(0, [30, 33]), [.3, .3])
        np.testing.assert_array_equal(result['ids'], [-1, -1])
        self.assertTrue(result['keep'].all())

    def test_shuffled_order_and_missing_shoulder(self):
        t = self.tracker()
        ids = t.process(frame(0, [30, 90], shoulder=False), [.3, .3])['ids']
        result = t.process(frame(1, [91, 31], shoulder=False), [.06, .06])
        np.testing.assert_array_equal(result['ids'], ids[::-1])

    def test_motion_across_gap_uses_elapsed_time(self):
        t = self.tracker()
        tid = t.process(frame(0, [30]), [.3])['ids'][0]
        t.process(frame(1, [34]), [.1])  # estimated velocity 2 pixels/frame
        for f in range(2, 7): t.process(frame(f, []), [])
        result = t.process(frame(7, [46]), [.1])
        self.assertEqual(result['ids'][0], tid)
        self.assertEqual(result['previous_frame'][0], 1)

    def test_wide_gap_displacement_does_not_relax_prediction_residual(self):
        t = self.tracker()
        tid = t.process(frame(0, [30]), [.3])['ids'][0]
        for f in range(1, 6): t.process(frame(f, []), [])
        result = t.process(frame(6, [90]), [.3])
        self.assertNotEqual(result['ids'][0], tid)
        self.assertEqual(result['reason'][0], 'birth')

    def test_nonconsecutive_or_reversed_frames_fail(self):
        for index in [0, -1, 2]:
            t = self.tracker(); t.process(frame(0, [30]), [.3])
            with self.assertRaises(ValueError): t.process(frame(index, [30]), [.3])

    def test_invalid_parameters_and_score_shapes_fail(self):
        with self.assertRaises(ValueError): self.tracker(.01)
        with self.assertRaises(ValueError): self.tracker(grace=-1)
        with self.assertRaises(ValueError): self.tracker().process(frame(0, [30]), [])

    def test_no_grace_is_adjacent_only_on_isolated_known_tracks(self):
        t = self.tracker(grace=0)
        old = Tracker('test', 1000, 1000, CONFIG)
        for index, xs in enumerate([[30, 90], [31, 91], [], [32, 92]]):
            f = frame(index, xs)
            a = t.process(f, [.3]*len(xs))
            b = old.process(f)
            np.testing.assert_array_equal(a['ids'], b['ids'])

    def test_no_capacity_limit(self):
        t = GraceTracker(CONFIG, .2, width=10000, height=1000)
        xs = list(range(30, 9900, 30))
        ids = t.process(frame(0, xs), [.3]*len(xs))['ids']
        self.assertEqual(len(set(ids)), len(xs))
        self.assertTrue((ids > 0).all())

    def test_gap_memory_is_not_an_observation(self):
        t = self.tracker(); t.process(frame(0, [30]), [.3])
        result = t.process(frame(1, []), [])
        self.assertEqual(result['pending_tracks'], 1)
        self.assertEqual(result['keep'].sum(), 0)
        self.assertEqual(t.finish()[0]['observations'], 1)


class GracePackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR'))
        self.directory = Path(self.temp.name)
        self.video_path = self.directory/'control.mp4'
        writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*'mp4v'), 30., (320,240))
        self.assertTrue(writer.isOpened())
        for _ in range(10): writer.write(np.zeros((240,320,3),np.uint8))
        writer.release()
        self.video = sio.load_video(self.video_path)
        self.skeleton = sio.Skeleton(nodes=list(NODES))
        self.assertEqual(NODES, ['Shell','Shoulder','Elbow','Claw','Arm'])

    def tearDown(self):
        self.temp.cleanup()

    def output(self, index, xs, scores):
        xy = np.full((1,len(xs),5,2), np.nan, dtype=np.float32)
        for j,x in enumerate(xs):
            xy[0,j,0] = [x,100]; xy[0,j,1] = [x,120]
        node = np.zeros((1,len(xs),5), dtype=np.float32)
        node[:,:,:2] = [.9,.4]
        sc = torch.tensor([scores], dtype=torch.float32)
        return Outputs(pred_keypoints=torch.from_numpy(xy), pred_peak_values=torch.from_numpy(node),
            pred_centroids=torch.from_numpy(xy[:,:,0].copy()), pred_centroid_values=sc,
            instance_scores=sc.clone(), frame_indices=torch.tensor([index]))

    def raw(self):
        frames, scores, slots = [], [], []
        for index, (xs, sc) in enumerate([([30,100],[.3,.15]), ([],[]), ([31,101],[.06,.06])]):
            ii, ss, sl, _ = unpack_output(self.output(index,xs,sc), self.skeleton,index)
            frames.append(sio.LabeledFrame(video=self.video, frame_idx=index, instances=ii))
            scores.append(ss); slots.append(sl)
        return sio.Labels(labeled_frames=frames, videos=[self.video], skeletons=[self.skeleton]), scores, slots

    def test_true_centroid_score_preserved(self):
        inst, sc, slots, raw = unpack_output(self.output(0,[30],[.06]),self.skeleton,0)
        self.assertAlmostEqual(inst[0].score,.06)
        self.assertAlmostEqual(float(inst[0].points['score'][0]),.9)
        self.assertEqual(slots.tolist(),[0])
        np.testing.assert_array_equal(sc, raw['centroid_scores'])

    def test_score_stage_mismatch_fails(self):
        out = self.output(0,[30],[.06]); out.instance_scores[:] = .9
        with self.assertRaises(AssertionError): unpack_output(out,self.skeleton,0)

    def test_wrong_frame_fails(self):
        with self.assertRaises(AssertionError): unpack_output(self.output(1,[30],[.1]),self.skeleton,0)

    def test_all_nan_pose_retained_in_raw_audit_not_packaged(self):
        out = self.output(0,[30,100],[.1,.3]); out.pred_keypoints[0,0] = float('nan')
        instances, scores, slots, raw = unpack_output(out,self.skeleton,0)
        self.assertEqual(len(instances),1)
        self.assertEqual(slots.tolist(),[1])
        self.assertEqual(len(raw['centroid_scores']),2)

    def test_two_independent_conditions_and_missing_positions_roundtrip(self):
        raw, scores, slots = self.raw()
        before = measurement_snapshot(raw)
        cfg = dict(continuation_threshold=.05, grace_frames=5)
        for birth, expected in [(.2,[1,0,1]), (.1,[2,0,2])]:
            labels, counts, audit, tracks, gaps = apply_condition(raw,scores,slots,cfg,CONFIG,'test',birth)
            self.assertEqual([len(lf) for lf in labels],expected)
            self.assertEqual(len(gaps),expected[0])
            self.assertTrue((gaps.missed_frames == 1).all())
            target = self.directory/f'b{birth}.predictions.slp'
            check = save_final(labels,target,self.video_path,0,3)
            self.assertEqual([len(lf) for lf in check],expected)
            self.assertIs(check[0].instances[0].track,check[2].instances[0].track)
            self.assertEqual(check[0].instances[0].points['xy'][0,0],30)
            self.assertEqual(check[2].instances[0].points['xy'][0,0],31)
            self.assertEqual(audit.retained.sum(),sum(expected))
            self.assertEqual(tracks.observations.sum(),sum(expected))
            cc, ct = control(raw,scores,birth,CONFIG,'control')
            self.assertEqual(cc.raw.tolist(),[expected[0],0,0])
            assert_same_measurements(before,measurement_snapshot(raw))
            self.assertTrue(all(i.track is None for lf in raw for i in lf))
        self.assertEqual(len(list(self.directory.glob('*.slp'))),2)
        self.assertFalse(list(self.directory.glob('*.tmp.slp')))

    def test_score_coverage_mismatch_fails(self):
        raw, scores, slots = self.raw()
        with self.assertRaises(ValueError):
            apply_condition(raw, scores[:-1], slots, dict(continuation_threshold=.05, grace_frames=5), CONFIG, 'test', .2)

    def test_all_empty_frames_are_kept_and_no_tracks_created(self):
        raw = sio.Labels(labeled_frames=[sio.LabeledFrame(video=self.video, frame_idx=f, instances=[])
            for f in range(3)], videos=[self.video], skeletons=[self.skeleton])
        empty = [np.empty(0) for _ in range(3)]
        labels, counts, audit, tracks, gaps = apply_condition(raw, empty, empty,
            dict(continuation_threshold=.05, grace_frames=5), CONFIG, 'test', .2)
        self.assertEqual(counts.retained.sum(), 0)
        self.assertEqual(len(tracks), 0)
        check = save_final(labels, self.directory/'empty.predictions.slp', self.video_path, 0, 3)
        self.assertEqual(len(check), 3)
        self.assertEqual(len(check.tracks), 0)

    def test_clone_does_not_mutate_points_or_source(self):
        raw, _, _ = self.raw()
        i = raw[0].instances[0]
        j = clone_instance(i)
        j.points['xy'][0,0] += 1
        self.assertEqual(i.points['xy'][0,0],30)

    def test_inference_is_one_whole_selection_with_no_output_path(self):
        class Fake:
            def predict_streaming(self,source,**kw):
                self.kw = kw
                return 'stream'
        p = Fake()
        self.assertEqual(infer(p,self.video_path),'stream')
        self.assertEqual(p.kw['frames'],list(range(1000)))
        self.assertEqual(p.kw['centroid_threshold'],.05)
        self.assertEqual(p.kw['keypoint_threshold'],.2)
        self.assertIsNone(p.kw['max_instances'])
        self.assertNotIn('output_path',p.kw)


if __name__ == '__main__':
    assert os.environ.get('SLURM_JOB_ID'), 'Run tests under Slurm'
    unittest.main(verbosity=2)
