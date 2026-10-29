"""Bounded synthetic regression controls for index-only demo extraction."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import h5py
import numpy as np
import pyarrow as pa
import sleap_io as sio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from prepare_crab_demo import normalize_tables, slice_slp, add_demo_indices, START, STOP

FDT = np.dtype([('frame_id','<u8'),('video','<u4'),('frame_idx','<u8'),('instance_id_start','<u8'),('instance_id_end','<u8')])
IDT = np.dtype([('instance_id','<i8'),('instance_type','u1'),('frame_id','<u8'),('skeleton','<u4'),('track','<i4'),('from_predicted','<i8'),('score','<f4'),('point_id_start','<u8'),('point_id_end','<u8'),('tracking_score','<f4')])
PDT = np.dtype([('x','<f8'),('y','<f8'),('visible','?'),('complete','?'),('score','<f8')])


class DemoTests(unittest.TestCase):
    def fixture(self):
        # Nonzero source instance/point offsets, and an empty middle frame.
        f = np.array([(5,0,5,10,11),(6,0,6,11,11),(7,0,7,11,13)], dtype=FDT)
        i = np.array([(10,1,5,0,9,-1,.8,50,55,0),
                      (11,1,7,0,-1,-1,.9,55,60,0),(12,1,7,0,15,-1,.6,60,65,0)], dtype=IDT)
        p = np.zeros(15,dtype=PDT)
        p['x']=np.arange(15)*1.125; p['y']=3.5; p['score']=.6; p['x'][0]=np.nan
        return f,i,p

    def test_remap_and_empty_frame(self):
        f,i,p=self.fixture()
        f0,i0,p0=[x.copy() for x in (f,i,p)]
        out,ins,pts=normalize_tables(f,i,p,5,3)
        np.testing.assert_array_equal(out['frame_idx'],[0,1,2])
        np.testing.assert_array_equal(out['instance_id_start'],[0,1,1])
        np.testing.assert_array_equal(ins['point_id_start'],[0,5,10])
        np.testing.assert_array_equal(ins['frame_id'],[0,2,2])
        self.assertEqual(pts.tobytes(),p.tobytes())
        for a,b in [(f,f0),(i,i0),(p,p0)]: self.assertEqual(a.tobytes(),b.tobytes())
        for field in ['score','tracking_score','track']:
            self.assertEqual(ins[field].tobytes(),i[field].tobytes())

    def test_missing_frame_rejected(self):
        f,i,p=self.fixture(); f['frame_idx'][1]=9
        with self.assertRaises(AssertionError): normalize_tables(f,i,p,5,3)

    def test_instance_gap_rejected(self):
        f,i,p=self.fixture(); i['instance_id'][1]=100
        with self.assertRaises(AssertionError): normalize_tables(f,i,p,5,3)

    def test_point_offset_corruption_rejected(self):
        f,i,p=self.fixture(); i['point_id_start'][1]=56
        with self.assertRaises(AssertionError): normalize_tables(f,i,p,5,3)

    def test_all_empty(self):
        f=np.array([(5,0,5,10,10),(6,0,6,10,10)],dtype=FDT)
        out,i,p=normalize_tables(f,np.zeros(0,dtype=IDT),np.zeros(0,dtype=PDT),5,2)
        self.assertEqual(len(i),0); self.assertTrue(np.all(out['instance_id_end']==0))

    def test_dataframe_source_preservation(self):
        table=pa.table({'frame':[START,STOP-1],'tracklet_id':[None,'original_id'],'value':[None,1.25]})
        out=add_demo_indices(table)
        self.assertTrue(out.select(table.column_names).equals(table))
        self.assertEqual(out['demo_frame'].to_pylist(),[0,299])

    def test_out_of_range_dataframe_rejected(self):
        with self.assertRaises(AssertionError): add_demo_indices(pa.table({'frame':[STOP]}))

    def test_actual_slp_roundtrip_and_no_overwrite(self):
        # Real SLEAP serialization plus index-only HDF extraction; no video needed.
        scratch=ROOT/'checks/demo_preparation'
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            tmp=Path(tmp)
            video=sio.Video(filename='original_full_video.mp4')
            skeleton=sio.Skeleton(['Shell','Shoulder','Elbow','Claw','Arm'])
            track=sio.Track(name='unchanged_full_run_track_id')
            labels=[]
            for frame in range(8):
                instances=[]
                if frame!=6:
                    instance=sio.PredictedInstance.from_numpy(np.full((5,2),float(frame)),skeleton=skeleton,
                         point_scores=np.array([.9,.8,.7,.6,.5]),score=.9,track=track if frame%2 else None)
                    instances=[instance]
                labels.append(sio.LabeledFrame(video=video,frame_idx=frame,instances=instances))
            source=tmp/'source.slp'
            sio.save_slp(sio.Labels(labels,videos=[video],skeletons=[skeleton]),source)
            # Writer versions may omit an empty provenance dataset; our fixed
            # production source includes it, so make the fixture match that format.
            with h5py.File(source,'r+') as f:
                for key in ['sessions_json','suggestions_json','provenance_json']:
                    if key not in f:
                        f.create_dataset(key,data=np.bytes_('{}') if key=='provenance_json' else [],
                                         **({} if key=='provenance_json' else {'dtype':h5py.string_dtype('utf-8')}))
            before=source.read_bytes()
            result=slice_slp(source,tmp/'out.slp','demo.mp4',5,3)
            out=sio.load_slp(tmp/'out.slp',open_videos=False)
            self.assertEqual([lf.frame_idx for lf in out],[0,1,2])
            self.assertEqual([len(lf.instances) for lf in out],[1,0,1])
            self.assertEqual([t.name for t in out.tracks],['unchanged_full_run_track_id'])
            self.assertEqual(out.videos[0].filename,'demo.mp4')
            self.assertEqual(result['observations'],2)
            self.assertEqual(source.read_bytes(),before)
            with self.assertRaises(FileExistsError): slice_slp(source,tmp/'out.slp','demo.mp4',5,3)


if __name__=='__main__':
    unittest.main(verbosity=2)
