"""Known-answer chunk boundaries, stream-copy pixels, alignment and scoped Git controls."""
from pathlib import Path
import os
import subprocess
import tempfile
import sys
import unittest
import cv2
import imageio_ffmpeg
import numpy as np
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from chunk_crab_demo import select_start,plan_chunks,validate_packet_sequence,copy_video,probe,validate_copy,add_indices
from replace_crab_demo_2000 import allowed_path


def packets(n=3000,step=30,size=100):
    return [dict(pts=i*1001,dts=i*1001,duration=1001,size=size,
        flags='K_' if i%step==0 else '__',data_hash='SHA256:'+str(i)) for i in range(n)]


class ChunkTests(unittest.TestCase):
    def test_nearest_keyframe_and_exact_2000_coverage(self):
        p=packets(28800)
        start=select_start(p,13400,2000)
        self.assertEqual(start,13410)
        chunks=plan_chunks(p,start,2000,budget=21000)
        self.assertEqual(chunks[0][0],13410); self.assertEqual(chunks[-1][1],15410)
        self.assertEqual([i for a,b in chunks for i in range(a,b)],list(range(13410,15410)))
        self.assertTrue(all(a%30==0 and (b-a)*100<=21000 for a,b in chunks))
        self.assertEqual(chunks[-1][1]-chunks[-1][0],110)

    def test_oversize_gop_refuses_without_encoding_fallback(self):
        with self.assertRaises(ValueError): plan_chunks(packets(),0,2000,budget=2999)

    def test_nonkey_start_and_invalid_count_fail(self):
        for start,count in [(1,2000),(0,0),(2000,2000)]:
            with self.assertRaises(ValueError): plan_chunks(packets(),start,count)

    def test_reordered_or_missing_packet_rejected(self):
        p=packets(5); p[2]['pts']+=1
        with self.assertRaises(AssertionError): validate_packet_sequence(p)
        with self.assertRaises(AssertionError): validate_packet_sequence(packets(5)[1:])

    def test_dataframe_original_columns_and_two_local_indices(self):
        old=pa.table({'frame':[13620,13621],'tracklet_id':[None,'original']})
        new=add_indices(old,13410,13620,'chunk_002')
        self.assertTrue(new.select(old.column_names).equals(old))
        self.assertEqual(new['demo_frame'].to_pylist(),[210,211])
        self.assertEqual(new['chunk_frame'].to_pylist(),[0,1])
        self.assertEqual(new['chunk_id'].to_pylist(),['chunk_002']*2)

    def test_publication_scope(self):
        self.assertTrue(allowed_path('demo/crab_cam2_2_middle_2000/chunk_001/clip.mp4'))
        for p in ['archive/demo/old.mp4','crab_vids/Crab_Cam_2_2.mp4','models/model.ckpt','scripts/grace_tracklets.py']:
            self.assertFalse(allowed_path(p))

    def test_real_packet_copy_including_partial_final_gop(self):
        scratch=ROOT/'checks/demo_preparation/unit_tmp'; scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            root=Path(tmp); source=root/'synthetic.mp4'
            images=np.random.default_rng(123).integers(0,256,(23,48,64,3),dtype=np.uint8)
            # Encoding is ONLY used to create this tiny synthetic test fixture.
            cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-nostdin','-n',
                '-f','rawvideo','-pix_fmt','bgr24','-s','64x48','-r','30000/1001','-i','pipe:0',
                '-c:v','libx264','-preset','ultrafast','-crf','0','-g','5','-keyint_min','5',
                '-sc_threshold','0','-bf','0','-threads','1',str(source)]
            subprocess.run(cmd,input=images.tobytes(),capture_output=True,check=True,timeout=30)
            original=probe(source); validate_packet_sequence(original['packets'])
            source_cap=cv2.VideoCapture(str(source)); self.assertTrue(source_cap.set(cv2.CAP_PROP_POS_FRAMES,5))
            for number,(a,b) in enumerate([(5,10),(10,15),(15,17)]):
                target=root/f'{number}.mp4'; command=copy_video(source,target,a,b-a)
                self.assertEqual(command[command.index('-c:v')+1],'copy')
                self.assertNotIn('-crf',command); self.assertNotIn('-vf',command)
                validate_copy(original,probe(target),a,b)
                cap=cv2.VideoCapture(str(target))
                self.assertEqual(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)),b-a)
                for _ in range(a,b):
                    ok,x=source_cap.read(); good,y=cap.read(); self.assertTrue(ok and good)
                    np.testing.assert_array_equal(x,y)
                self.assertFalse(cap.read()[0]); cap.release()
                with self.assertRaises(FileExistsError): copy_video(source,target,a,b-a)
            source_cap.release()
            corrupt=probe(root/'0.mp4'); corrupt['packets'][0]['data_hash']='SHA256:changed'
            with self.assertRaises(AssertionError): validate_copy(original,corrupt,5,10)

    def test_path_only_commit_preserves_unrelated_staging(self):
        scratch=ROOT/'checks/demo_preparation/unit_tmp'; scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            root=Path(tmp)
            def git(*args):
                return subprocess.run(['git','-c','user.name=Demo control','-c','user.email=demo-control@example.invalid',
                    *args],cwd=root,text=True,capture_output=True,check=True,timeout=30).stdout
            git('init','--initial-branch=my-code')
            old=root/'demo/old'; old.mkdir(parents=True); (old/'clip.mp4').write_text('old')
            (root/'unrelated.txt').write_text('original')
            git('add','--','demo/old','unrelated.txt'); git('commit','-m','baseline')
            archive=root/'archive'; archive.mkdir(); old.rename(archive/'old')
            new=root/'demo/new'; new.mkdir(); (new/'clip.mp4').write_text('new')
            (root/'unrelated.txt').write_text('user edit')
            git('add','-A','--','demo/old','demo/new'); git('add','--','unrelated.txt')
            git('commit','--only','-m','replacement','--','demo/old','demo/new')
            self.assertEqual(git('show','HEAD:unrelated.txt'),'original')
            self.assertEqual(git('diff','--cached','--name-only').strip(),'unrelated.txt')
            self.assertEqual((archive/'old/clip.mp4').read_text(),'old')


if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID'),'Slurm only'
    unittest.main(verbosity=2)
