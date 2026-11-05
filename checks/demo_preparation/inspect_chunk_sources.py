"""Read-only metadata/Git checks for the authorized lossless chunk extraction."""
import os, sys, json, shutil, subprocess, importlib.util
from pathlib import Path
ROOT=Path('/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap')
sys.path.insert(0,str(ROOT/'scripts'))
from pipeline_common import atomic_json
import imageio_ffmpeg
assert os.environ.get('SLURM_JOB_ID')
out=ROOT/'logs/demo_preparation'/f'chunk_inspect_{os.environ["SLURM_JOB_ID"]}'; out.mkdir(exist_ok=False)
result={'ffmpeg':imageio_ffmpeg.get_ffmpeg_exe(),'ffprobe':shutil.which('ffprobe'),'av':importlib.util.find_spec('av') is not None}
ff=result['ffmpeg']
p=subprocess.run([ff,'-hide_banner','-i',str(ROOT/'crab_vids/Crab_Cam_2_2.mp4')],text=True,capture_output=True,timeout=30)
result['video_header']=p.stderr
# Read MP4 sample-table metadata, no decoding or modification.
import struct
path=ROOT/'crab_vids/Crab_Cam_2_2.mp4'
def boxes(f,start,end):
    pos=start
    while pos+8<=end:
        f.seek(pos); size,kind=struct.unpack('>I4s',f.read(8)); header=8
        if size==1: size=struct.unpack('>Q',f.read(8))[0]; header=16
        if size==0: size=end-pos
        assert size>=header and pos+size<=end
        yield kind,pos+header,pos+size
        pos+=size
with path.open('rb') as f:
    def walk(start,end,trail=''):
        for kind,a,b in boxes(f,start,end):
            k=kind.decode('ascii','replace'); here=trail+'/'+k
            if kind in [b'moov',b'trak',b'mdia',b'minf',b'stbl']:
                walk(a,b,here)
            elif kind in [b'stss',b'stsz',b'stts',b'ctts',b'mdhd',b'hdlr']:
                f.seek(a); data=f.read(b-a)
                if kind==b'stss':
                    n=struct.unpack('>I',data[4:8])[0]; vals=list(struct.unpack('>'+str(n)+'I',data[8:]))
                    result[here]={'count':n,'near_middle':[x-1 for x in vals if 12000<x<17000], 'first':vals[:10]}
                elif kind==b'stsz':
                    size,n=struct.unpack('>II',data[4:12]); result[here]={'sample_size':size,'count':n}
                elif kind in [b'stts',b'ctts']:
                    n=struct.unpack('>I',data[4:8])[0]; result[here]={'entries':n,'first_hex':data[:120].hex()}
                else: result[here]=data[:80].hex()
    walk(0,path.stat().st_size)
env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=15')
for key,args in [('head',['rev-parse','HEAD']),('status',['status','--short']),('staged',['diff','--cached','--name-only']),('remote',['ls-remote','--exit-code','origin','refs/heads/my-code'])]:
    p=subprocess.run(['git',*args],cwd=ROOT,env=env,text=True,capture_output=True,timeout=75)
    result[key]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
atomic_json(out/'inspection.json',result)
print(json.dumps(result,indent=2),flush=True)
