"""Finite read-only input/codec inventory for the requested demo/archive."""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import h5py
import pyarrow.parquet as pq
import imageio_ffmpeg
import cv2

ROOT = Path(__file__).resolve().parents[1]
assert os.environ.get('SLURM_JOB_ID')
os.chdir(ROOT)
report = {'job_id': os.environ['SLURM_JOB_ID'], 'ffmpeg': imageio_ffmpeg.get_ffmpeg_exe(),
          'versions': {p: importlib.metadata.version(p) for p in ['sleap-io','numpy','pyarrow','imageio-ffmpeg','opencv-python']}}
ff = subprocess.run([report['ffmpeg'], '-hide_banner', '-encoders'], capture_output=True, text=True, check=True)
report['encoders'] = [x for x in ff.stdout.splitlines() if any(s in x for s in ['libx264','ffv1','libx265'])]
video = ROOT / 'crab_vids/Crab_Cam_2_2.mp4'
cap = cv2.VideoCapture(str(video))
assert cap.isOpened()
report['video'] = {'bytes': video.stat().st_size, 'frames': cap.get(cv2.CAP_PROP_FRAME_COUNT),
                   'fps': cap.get(cv2.CAP_PROP_FPS), 'width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                   'height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}
cap.release()
for kind, rel in [('raw','raw/Crab_Cam_2_2.predictions.slp'), ('tracked','tracklets/Crab_Cam_2_2.tracked.slp')]:
    path = ROOT / 'predictions/v007_best_body_first_v001/Crab_Cam_2_2' / rel
    with h5py.File(path) as f:
        report[kind] = {'bytes': path.stat().st_size, 'keys': list(f.keys()),
                        'datasets': {k: {'shape': f[k].shape, 'dtype': str(f[k].dtype)} for k in ['frames','instances','pred_points','points','tracks_json']},
                        'videos': [json.loads(x) for x in f['videos_json'][:]],
                        'first_frames': str(f['frames'][14250:14253]),
                        'sample_tracks': [json.loads(x) for x in f['tracks_json'][:3]]}
base = ROOT / 'datasets/v007_best_body_first_dataframe_v001'
for name, path in [('observations', next((base/'tables/observations/Crab_Cam_2_2').glob('*.parquet'))), ('frames',base/'tables/frames.parquet')]:
    report[name] = {'schema': str(pq.read_schema(path)), 'metadata': str(pq.read_schema(path).metadata)}
archive = []
for pattern in ['models/dan_*', 'predictions/Crab_Cam_2_*', 'datasets/legacy_native_tracklets_v001',
                'checks/legacy_dataset', 'checks/body_first_v001', 'logs/legacy_dataset', 'logs/tracking',
                '*v004_logs', 'predict_logs', 'danlabels.v004.slp']:
    for p in sorted(ROOT.glob(pattern)):
        files = [x for x in p.rglob('*') if x.is_file()] if p.is_dir() else [p]
        archive.append({'path': str(p.relative_to(ROOT)), 'files': len(files), 'bytes': sum(x.stat().st_size for x in files),
                        'symlinks': [str(x.relative_to(ROOT)) for x in files if x.is_symlink()]})
report['archive_candidates'] = archive
out = ROOT / 'logs/demo_preparation' / f"inspection_{os.environ['SLURM_JOB_ID']}.json"
out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
