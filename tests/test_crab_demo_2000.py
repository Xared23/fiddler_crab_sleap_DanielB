"""Controls specific to the 2000-frame replacement; execute only under Slurm."""
from pathlib import Path
import csv
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import imageio_ffmpeg
import numpy as np
import pyarrow as pa
import prepare_crab_demo as demo
from replace_crab_demo_2000 import allowed_path
from test_crab_demo import FDT, IDT, PDT


class Demo2000Tests(unittest.TestCase):
    def setUp(self):
        self.previous_profile={name:getattr(demo,name) for name in
            ['START','COUNT','STOP','NAME','FINAL','MAXRATE','README_TEMPLATE','SOURCE_FILES']}
        demo.configure_demo_2000()

    def tearDown(self):
        for name,value in self.previous_profile.items(): setattr(demo,name,value)

    def test_centered_exact_bounds_and_native_geometry(self):
        self.assertEqual((demo.START,demo.STOP,demo.COUNT),(13400,15400,2000))
        self.assertEqual(demo.START,(demo.TOTAL-demo.COUNT)//2)
        self.assertEqual(demo.NAME,'Crab_Cam_2_2.frames_013400_015399')
        self.assertEqual((demo.WIDTH,demo.HEIGHT),(7680,4320))
        self.assertEqual(demo.MAXRATE,'10M')
        self.assertLess((demo.COUNT/demo.FPS+1)*10e6/8, 95*1024**2)

    def test_full_2000_row_mapping_no_off_by_one(self):
        original=pa.table({'frame':np.arange(demo.START,demo.STOP)})
        mapped=demo.add_demo_indices(original)
        self.assertEqual(mapped.num_rows,2000)
        np.testing.assert_array_equal(mapped['demo_frame'].to_numpy(),np.arange(2000))
        self.assertTrue(mapped.select(['frame']).equals(original))
        self.assertAlmostEqual(mapped['demo_time_s'][-1].as_py(),1999*1001/30000)

    def test_all_empty_2000_slp_tables(self):
        f=np.zeros(2000,dtype=FDT)
        f['frame_id']=f['frame_idx']=np.arange(13400,15400)
        f['instance_id_start']=f['instance_id_end']=100
        out,i,p=demo.normalize_tables(f,np.zeros(0,IDT),np.zeros(0,PDT),13400,2000)
        np.testing.assert_array_equal(out['frame_idx'],np.arange(2000))
        self.assertEqual(len(i),0)
        self.assertTrue((out['instance_id_start']==0).all())

    def test_git_scope_rejects_unrelated_sources_and_full_data(self):
        for p in ['datasets/full.parquet','crab_vids/Crab_Cam_2_2.mp4','models/a.ckpt',
                  'scripts/grace_tracklets.py','archive/demo/old.mp4','demo/other/video.mp4']:
            self.assertFalse(allowed_path(p),p)
        self.assertTrue(allowed_path('demo/crab_cam2_2_middle_2000/observations.parquet'))
        self.assertTrue(allowed_path('demo/crab_cam2_2_middle_300/README.md'))

    def test_template_matches_profile(self):
        text=demo.README_TEMPLATE.read_text()
        for term in [demo.NAME,'0–1999','13400','15399','66.733','10-Mbit/s',
                     'demo/crab_cam2_2_middle_2000','local_frame = 1000']:
            self.assertIn(term,text)
        self.assertNotIn('14250',text)

    def test_path_only_git_replacement_preserves_unrelated_staging(self):
        scratch=demo.ROOT/'checks/demo_preparation'; scratch.mkdir(exist_ok=True,parents=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            root=Path(tmp)
            def git(*args):
                return subprocess.run(['git','-c','user.name=Demo control','-c','user.email=demo-control@example.invalid',
                    *args],cwd=root,text=True,capture_output=True,check=True,timeout=30).stdout
            git('init','--initial-branch=my-code')
            old=root/'demo/old'; old.mkdir(parents=True)
            (old/'clip.mp4').write_text('old control payload')
            (root/'unrelated.txt').write_text('original')
            git('add','--','demo/old','unrelated.txt'); git('commit','-m','control baseline')
            archive=root/'archive'; archive.mkdir(); old.rename(archive/'old')
            new=root/'demo/new'; new.mkdir(); (new/'clip.mp4').write_text('new control payload')
            (root/'unrelated.txt').write_text('unrelated user edit')
            git('add','-A','--','demo/old','demo/new')
            git('add','--','unrelated.txt')
            git('commit','--only','-m','control replacement','--','demo/old','demo/new')
            self.assertEqual(git('show','HEAD:unrelated.txt'),'original')
            self.assertEqual(git('diff','--cached','--name-only').strip(),'unrelated.txt')
            tree=git('ls-tree','-r','--name-only','HEAD').splitlines()
            self.assertIn('demo/new/clip.mp4',tree)
            self.assertNotIn('demo/old/clip.mp4',tree)
            self.assertFalse(any(p.startswith('archive/') for p in tree))
            self.assertEqual((archive/'old/clip.mp4').read_text(),'old control payload')

    def test_actual_short_video_encode_decode_and_frame_mapping(self):
        # Small real encode/decode exercises the same video path without extra
        # real-data trials. Rational source FPS exactly matches the production FPS.
        scratch=demo.ROOT/'checks/demo_preparation'; scratch.mkdir(exist_ok=True,parents=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            root=Path(tmp); source=root/'source.mp4'; stage=root/'stage'; stage.mkdir()
            images=np.array([np.full((48,64,3),10+15*f,dtype=np.uint8) for f in range(12)])
            cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-nostdin','-n',
                 '-f','rawvideo','-pix_fmt','bgr24','-s','64x48','-r','30000/1001','-i','pipe:0',
                 '-an','-c:v','libx264','-crf','0','-threads','1',str(source)]
            subprocess.run(cmd,input=images.tobytes(),capture_output=True,check=True,timeout=30)
            with patch.multiple(demo,VIDEO=source,WIDTH=64,HEIGHT=48,TOTAL=12,START=3,COUNT=6,NAME='control'):
                info=demo.make_video(stage)
            self.assertEqual(info['decoded_frames'],6)
            self.assertIn('10M',info['command'])
            with (stage/'frame_map.csv').open() as stream: rows=list(csv.DictReader(stream))
            self.assertEqual([int(r['demo_frame']) for r in rows],list(range(6)))
            self.assertEqual([int(r['source_frame']) for r in rows],list(range(3,9)))
            cap=cv2.VideoCapture(str(stage/'control.mp4'))
            means=[]
            while True:
                ok,image=cap.read()
                if not ok: break
                means.append(float(image.mean()))
            cap.release()
            np.testing.assert_allclose(means,[10+15*f for f in range(3,9)],atol=4,rtol=0)


if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID'),'Slurm only'
    demo.configure_demo_2000()
    unittest.main(verbosity=2)
