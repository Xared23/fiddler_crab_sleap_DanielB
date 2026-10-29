"""Known-answer controls for the bounded, no-chunk threshold experiment."""
import copy
from pathlib import Path
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np
import sleap_io as sio
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from threshold_prediction_tests import (frame_arrays, measurement_snapshot, assert_same_measurements,
    attach_existing_tracker, save_final, spatial_matches, inference, NODES)
from pipeline_common import validate_labels

TRACKER=yaml.safe_load((ROOT/'configs/tracking/body_first_v001.yaml').read_text())


class ThresholdPredictionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR'))
        self.directory=Path(self.temp.name)
        self.video_path=self.directory/'control.mp4'
        writer=cv2.VideoWriter(str(self.video_path),cv2.VideoWriter_fourcc(*'mp4v'),30.,(320,240))
        self.assertTrue(writer.isOpened())
        for _ in range(6): writer.write(np.zeros((240,320,3),np.uint8))
        writer.release()
        self.video=sio.load_video(self.video_path)
        # SLEAP converts the supplied list in place; never pass the shared names.
        self.skeleton=sio.Skeleton(nodes=list(NODES))
        self.assertEqual(NODES,['Shell','Shoulder','Elbow','Claw','Arm'])

    def tearDown(self):
        self.temp.cleanup()

    def labels(self, frames):
        result=[]
        for frame_index, bodies in enumerate(frames):
            instances=[]
            for x,score in bodies:
                xy=np.full((5,2),np.nan); xy[0]=[x,100]; xy[1]=[x,120]
                instances.append(sio.PredictedInstance.from_numpy(xy,skeleton=self.skeleton,
                    point_scores=np.array([score,.4,0,0,0]),score=.8))
            result.append(sio.LabeledFrame(video=self.video,frame_idx=frame_index,instances=instances))
        return sio.Labels(labeled_frames=result,videos=[self.video],skeletons=[self.skeleton])

    def track(self, labels):
        return attach_existing_tracker(labels,TRACKER,'test_run','control',320,240)

    def test_counts_vary_birth_empty_reappearance_no_synthesis(self):
        labels=self.labels([[(30,.9)],[(31,.9),(90,.9)],[],[(32,.9)]])
        before=measurement_snapshot(labels)
        records,tracks=self.track(labels)
        self.assertEqual([r['raw'] for r in records],[1,2,0,1])
        self.assertEqual([r['assigned'] for r in records],[1,2,0,1])
        self.assertEqual(len(tracks),3)
        self.assertIs(labels[0].instances[0].track,labels[1].instances[0].track)
        self.assertIsNot(labels[0].instances[0].track,labels[3].instances[0].track)
        assert_same_measurements(before,measurement_snapshot(labels))

    def test_low_score_and_duplicate_predictions_retained(self):
        labels=self.labels([[(30,.9),(100,.1),(200,.9),(203,.9)]])
        records,tracks=self.track(labels)
        self.assertEqual(len(labels[0]),4)
        self.assertEqual(records[0]['assigned'],1)
        self.assertEqual(records[0]['unassigned'],3)
        self.assertTrue(all(i.track is None for i in labels[0].instances[1:]))

    def test_all_frame_slp_roundtrip_including_empty_and_unassigned(self):
        labels=self.labels([[(30,.9),(100,.1)],[],[(32,.9)]])
        self.track(labels)
        output=self.directory/'final.predictions.slp'
        check=save_final(labels,output,self.video_path,0,3)
        self.assertEqual([len(f) for f in check],[2,0,1])
        self.assertIsNone(check[0].instances[1].track)
        self.assertEqual(len(list(self.directory.glob('*.slp'))),1)
        self.assertFalse(list(self.directory.glob('*.tmp.slp')))
        self.assertFalse(list(self.directory.glob('*.frames_*.slp')))

    def test_existing_final_is_never_overwritten(self):
        labels=self.labels([[(30,.9)]])
        self.track(labels)
        output=self.directory/'final.slp'
        save_final(labels,output,self.video_path,0,1)
        before=output.read_bytes()
        with self.assertRaises(FileExistsError): save_final(labels,output,self.video_path,0,1)
        self.assertEqual(before,output.read_bytes())

    def test_missing_frame_coverage_fails(self):
        labels=self.labels([[],[]]); labels[1].frame_idx=2
        with self.assertRaises(RuntimeError): validate_labels(labels,self.video_path,0,3,NODES)

    def test_corrupt_measurement_fails(self):
        labels=self.labels([[(30,.9)]])
        before=measurement_snapshot(labels)
        labels[0].instances[0].points['xy'][0,0]+=1
        with self.assertRaises(AssertionError): assert_same_measurements(before,measurement_snapshot(labels))

    def test_corrupt_node_score_fails(self):
        labels=self.labels([[(30,.9)]])
        before=measurement_snapshot(labels)
        labels[0].instances[0].points['score'][0]=.1
        with self.assertRaises(AssertionError): assert_same_measurements(before,measurement_snapshot(labels))

    def test_matching_one_to_one_and_unmatched(self):
        base=np.array([[0.,0.],[100.,0.]])
        new=np.array([[1.,0.],[2.,0.],[101.,0.],[200.,0.]])
        matches,missing,added=spatial_matches(base,new,5)
        self.assertEqual([(i,j) for i,j,d in matches],[(0,0),(1,2)])
        self.assertEqual(missing,[]); self.assertEqual(added,[1,3])

    def test_matching_missing_positions_and_empty(self):
        matches,missing,added=spatial_matches(np.array([[np.nan,0.]]),np.array([[30.,0.]]))
        self.assertEqual(matches,[]); self.assertEqual(missing,[]); self.assertEqual(added,[0])
        self.assertEqual(spatial_matches(np.empty((0,2)),np.empty((0,2))),([],[],[]))

    def test_matching_radius_boundary_and_reordered_detections(self):
        base=np.array([[0.,0.],[100.,0.]])
        new=np.array([[103.,4.],[0.,0.]])
        matches,missing,added=spatial_matches(base,new,5)
        self.assertEqual([(i,j) for i,j,d in matches],[(0,1),(1,0)])
        self.assertFalse(missing or added)

    def test_inference_dispatch_no_count_cap_no_files_landmarks_fixed(self):
        class Fake:
            def predict(self,source,**kwargs):
                self.source=source; self.kwargs=kwargs
                return 'labels'
        predictor=Fake()
        for threshold in (.1,.05):
            self.assertEqual(inference(predictor,self.video_path,threshold,0,200),'labels')
            kw=predictor.kwargs
            self.assertEqual(kw['frames'],list(range(200)))
            self.assertEqual(kw['centroid_threshold'],threshold)
            self.assertEqual(kw['keypoint_threshold'],.2)
            self.assertIsNone(kw['max_instances'])
            self.assertFalse(kw['clean_empty_frames'])
            self.assertNotIn('output_path',kw)
        with self.assertRaises(ValueError): inference(predictor,self.video_path,.2,0,200)

    def test_zero_detection_selection_serializes(self):
        labels=self.labels([[],[],[]]); records,tracks=self.track(labels)
        self.assertEqual(tracks,[])
        check=save_final(labels,self.directory/'empty.slp',self.video_path,0,3)
        self.assertEqual(len(check),3); self.assertEqual(len(check.tracks),0)
        self.assertEqual(sum(len(f) for f in check),0)


if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID'), 'Run tests under Slurm'
    unittest.main(verbosity=2)
