"""Archive only legacy payloads, preserving v007's shared ROI provenance paths.

Slurm only; no Git commands, inference, deletion, or pinned-file rewriting.
Whole-directory renames preserve source bytes/mtimes. A per-file hash inventory
and post-move verification document the relocation. Failures are not retried.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import yaml
from pipeline_common import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'archive/v004_legacy'
LEGACY = Path('datasets/legacy_native_tracklets_v001')


def candidates():
    return sorted([p.relative_to(ROOT) for p in (ROOT / 'models').glob('dan_*')]) + [
        Path('danlabels.v004.slp'), Path('predictions/Crab_Cam_2_2'), Path('predictions/Crab_Cam_2_3'),
        LEGACY, Path('checks/legacy_dataset'), Path('checks/body_first_v001'),
        Path('logs/legacy_dataset'), Path('logs/tracking'), Path('centered_v004_logs'),
        Path('centroid_v004_logs'), Path('remap_v004_logs'), Path('predict_logs')]


def all_files(path):
    return sorted(p for p in path.rglob('*') if p.is_file() or p.is_symlink()) if path.is_dir() else [path]


def state(path):
    s = path.stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns, 'inode': s.st_ino, 'device': s.st_dev}


def git_state():
    # Files only: do not invoke Git, refresh the index, or inspect credentials.
    paths = [ROOT / '.git/HEAD', ROOT / '.git/config', ROOT / '.git/index', ROOT / '.gitignore']
    paths += [p for p in (ROOT / '.git/refs').rglob('*') if p.is_file()]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths if p.is_file()}


def active_state():
    paths = [ROOT / 'models/v007_19f', ROOT / 'predictions/v007_best_body_first_v001',
             ROOT / 'predictions/test runs', ROOT / 'datasets/v007_best_body_first_dataframe_v001',
             ROOT / 'crab_vids', ROOT / 'clip_cam2_8k.mp4', ROOT / 'danlabels.v007.slp',
             ROOT / 'data/labels/v007_19f', ROOT / 'configs/v007_19f']
    return {str(p.relative_to(ROOT)): state(p) for base in paths for p in all_files(base)}


def run():
    assert os.environ.get('SLURM_JOB_ID'), 'Slurm only'
    assert not DEST.exists(), f'Refusing existing archive: {DEST}'
    complete = json.loads((ROOT / 'datasets/v007_best_body_first_dataframe_v001/DATASET_COMPLETE.json').read_text())
    assert complete['status'] == 'validated_complete'
    rels = candidates()
    assert len(set(rels)) == len(rels)
    for rel in rels:
        assert (ROOT / rel).exists() and not (ROOT / rel).is_symlink(), rel
    for rel in rels:
        if str(rel).startswith('models/'):
            path = ROOT / rel / 'initial_config.yaml'
            if not path.exists():
                path = ROOT / rel / 'training_config.yaml'
            cfg = yaml.safe_load(path.read_text())
            labels = cfg['data_config']['train_labels_path']
            if isinstance(labels, str):
                labels = [labels]
            assert labels and all('v004' in p for p in labels), (rel, labels)
    # Pin active inputs and repository metadata; neither may be touched.
    active_before, git_before = active_state(), git_state()
    shared_before = {p.name: sha256(p) for p in (ROOT / LEGACY / 'config').iterdir() if p.is_file()}
    inventory = []
    for rel in rels:
        for p in all_files(ROOT / rel):
            assert not p.is_symlink(), f'Unexpected legacy symlink: {p}'
            inventory.append({'original_path': str(p.relative_to(ROOT)), **state(p), 'sha256': sha256(p)})
    DEST.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'inventory_complete_not_archived', 'job_id': os.environ['SLURM_JOB_ID'],
               'mappings': {str(rel): str((DEST / rel).relative_to(ROOT)) for rel in rels},
               'files': inventory, 'git_before': git_before,
               'policy': 'No source bytes changed. Original paths in historical manifests resolve under archive/v004_legacy/. Only shared ROI configuration remains accessible via a compatibility symlink.'}
    atomic_json(DEST / 'inventory.json', receipt)
    for rel in rels:
        target = DEST / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        (ROOT / rel).rename(target)
        print('ARCHIVED', rel, '->', target.relative_to(ROOT), flush=True)
    # Preserve only metadata compatibility, NOT old tables/predictions/models.
    stub = ROOT / LEGACY
    stub.mkdir(exist_ok=False)
    (stub / 'config').symlink_to(Path('../../archive/v004_legacy') / LEGACY / 'config', target_is_directory=True)
    (stub / 'README.md').write_text('# Archived v004 dataset\n\nThe payload is now at `archive/v004_legacy/datasets/legacy_native_tracklets_v001/`.\nThis directory is NOT an active dataframe. Only `config/` is a compatibility\nsymlink to preserve the v007 run\'s approved water-region provenance.\nUse `datasets/v007_best_body_first_dataframe_v001/` for current observations.\n')
    for item in inventory:
        p = DEST / item['original_path']
        assert state(p) == {k: item[k] for k in ['bytes','mtime_ns','inode','device']}, p
        assert sha256(p) == item['sha256'], p
    assert {p.name: sha256(p) for p in (stub/'config').iterdir() if p.is_file()} == shared_before
    assert active_state() == active_before, 'Active data changed during archival; review required'
    assert git_state() == git_before, 'Git metadata/ignore rules changed during archival; review required'
    receipt['status'] = 'archived_verified'
    receipt['file_count'] = len(inventory)
    receipt['total_bytes'] = sum(x['bytes'] for x in inventory)
    receipt['verification'] = {'all_file_hashes_unchanged': True, 'all_file_mtimes_and_inodes_unchanged': True,
                               'active_inputs_unchanged': True, 'git_metadata_and_ignore_rules_unchanged': True,
                               'shared_config_hashes_unchanged': True}
    atomic_json(DEST / 'inventory.json', receipt)
    (DEST / 'README.md').write_text('# Archived v004 model family and derived data\n\nHistorical only; do not use for new work. Relative paths below this directory\nmirror the former project layout. `inventory.json` records every original path,\nsize, mtime, inode, SHA-256 and relocation; status must be `archived_verified`.\n\nIncludes the native centroid/centered model pair, earlier v004 attempts, v004\nlabels, both old prediction directories, the legacy dataframe, old-data\nbody-first pilot evidence and legacy logs. Source code/configuration and\nhistorical documentation remain in place for provenance. Raw videos, the\ntraining clip, v007 labels/models/predictions/dataframes and v007 analyses\nwere not moved or modified. Historical manifests were not rewritten: prepend\n`archive/v004_legacy/` to their old legacy paths. SLP raw-video links still work.\n\nThe original legacy dataset location retains ONLY a README and a `config/`\nsymlink for v007 water-region approvals. It contains no old dataframe payload.\n\nGit has NOT been changed. Before staging any work, add `/archive/` to the\nproject ignore rules; otherwise some relocated artifacts (including checkpoints)\nwould no longer match their previous path-specific ignore rules.\n')
    atomic_json(DEST / 'ARCHIVE_COMPLETE.json', {k: receipt[k] for k in ['status','job_id','file_count','total_bytes','mappings','verification']})
    print(json.dumps({k: receipt[k] for k in ['status','file_count','total_bytes','verification']}, indent=2), flush=True)


if __name__ == '__main__':
    run()
