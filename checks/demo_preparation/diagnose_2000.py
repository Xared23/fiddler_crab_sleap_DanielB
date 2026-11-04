"""Finite read-only diagnosis of the failed demo encoding and Git publication state."""
import os, sys, json, subprocess
from pathlib import Path
import cv2
import numpy as np
ROOT=Path('/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap')
sys.path.insert(0,str(ROOT/'scripts'))
from pipeline_common import atomic_json
assert os.environ.get('SLURM_JOB_ID')
out=ROOT/'logs/demo_preparation'/f'diagnose_{os.environ["SLURM_JOB_ID"]}'
out.mkdir(exist_ok=False)
source=ROOT/'crab_vids/Crab_Cam_2_2.mp4'
clip=ROOT/'demo/.build_crab_demo_16760391/Crab_Cam_2_2.frames_013400_015399.mp4'
result={'clip_bytes':clip.stat().st_size,'samples':[]}
a=cv2.VideoCapture(str(source),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2])
b=cv2.VideoCapture(str(clip),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2])
for f in [0,1,50,1000,1999]:
    assert a.set(cv2.CAP_PROP_POS_FRAMES,13400+f) and b.set(cv2.CAP_PROP_POS_FRAMES,f)
    ok,x=a.read(); good,y=b.read(); assert ok and good
    xx=x[::8,::8].astype(float); yy=y[::8,::8].astype(float)
    row={'frame':f,'mae':float(np.abs(xx-yy).mean()),'source_mean':float(xx.mean()),'encoded_mean':float(yy.mean())}
    result['samples'].append(row); print(row,flush=True)
a.release(); b.release()
env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=15')
for key,args in [('head',['rev-parse','HEAD']),('status',['status','--short']),('staged',['diff','--cached','--name-only']),
                 ('recent',['log','-3','--oneline']),('remote',['ls-remote','--exit-code','origin','refs/heads/my-code'])]:
    p=subprocess.run(['git',*args],cwd=ROOT,env=env,text=True,capture_output=True,timeout=75)
    result[key]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    print(key,result[key],flush=True)
atomic_json(out/'diagnosis.json',result)
