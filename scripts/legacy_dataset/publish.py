#!/usr/bin/env python3
"""One-shot atomic-gated publication after explicit bounded visual review.

All published data is from a validated staging directory. Consumers must require
DATASET_COMPLETE.json; intermediate directory renames alone are not completion.
"""
import json
import os
from pathlib import Path
import sys

import pyarrow.parquet as pq

from export import ROOT, DEST, checksum, pin, atomic_json, OBS_SCHEMA, TRACK_SCHEMA, FRAME_SCHEMA


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Slurm only'
    assert len(sys.argv) == 2 and sys.argv[1].isdigit(), 'Pass exact validated build job ID'
    build_id = sys.argv[1]
    work = DEST / f'.build_job_{build_id}'
    final_marker = DEST / 'DATASET_COMPLETE.json'
    assert not final_marker.exists()
    targets = [DEST / 'tables', DEST / 'qc/postprocessing', DEST / 'manifest.json',
               DEST / 'config/water_regions_effective_v002.json']
    assert not any(p.exists() for p in targets), 'Refusing overwrite/ambiguous partial publication'
    assert all((DEST / name).is_dir() for name in ['qc', 'config']), 'Run the audit report before publishing'
    fd = os.open(DEST / '.conversion.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, os.environ['SLURM_JOB_ID'].encode())
    os.close(fd)
    try:
        build_receipt = json.loads((work / 'BUILD_VALIDATED.json').read_text())
        assert build_receipt['build_job_id'] == build_id
        assert checksum(work / 'manifest.json') == build_receipt['manifest_sha256']
        manifest = json.loads((work / 'manifest.json').read_text())
        review_path = DEST / 'config/quality_review_v001.json'
        review = json.loads(review_path.read_text())
        assert review['build_job_id'] == build_id
        assert review['publication_disposition'] == 'accept_with_documented_limits'
        assert review['manifest_sha256'] == build_receipt['manifest_sha256']
        examples = json.loads((work / 'qc/review_examples.json').read_text())
        assert sorted(review['reviewed_previews']) == sorted(x['preview'] for x in examples)
        # Fail if implementation/config changed between tested build and publish.
        for rec in manifest['input_pins']:
            assert checksum(ROOT / rec['path']) == rec['sha256'], rec['path']
        # Sources were fully hashed during the build; require same stat identity.
        for rec in manifest['source_pins']:
            stat = (ROOT / rec['path']).stat()
            assert (stat.st_size, stat.st_mtime_ns) == (rec['bytes'], rec['mtime_ns']), rec['path']
        output_counts = {'observations': 0, 'rejected_observations': 0, 'tracklets': 0, 'frames': 0}
        for artifact in manifest['artifacts']:
            path = work / artifact['path']
            assert path.stat().st_size == artifact['bytes']
            assert checksum(path) == artifact['sha256'], path
            if path.suffix == '.parquet':
                if artifact['path'].startswith('tables/observations/'):
                    kind, schema = 'observations', OBS_SCHEMA
                elif artifact['path'].startswith('tables/rejected_observations/'):
                    kind, schema = 'rejected_observations', OBS_SCHEMA
                elif path.name == 'tracklets.parquet':
                    kind, schema = 'tracklets', TRACK_SCHEMA
                elif path.name == 'frames.parquet':
                    kind, schema = 'frames', FRAME_SCHEMA
                else:
                    raise AssertionError(path)
                assert pq.read_schema(path) == schema
                output_counts[kind] += pq.read_metadata(path).num_rows
        summaries = manifest['summaries'].values()
        assert output_counts['observations'] == sum(x['accepted_observations'] for x in summaries)
        assert output_counts['rejected_observations'] == sum(x['rejected_observations'] for x in summaries)
        assert output_counts['tracklets'] == sum(x['tracklets'] for x in summaries)
        assert output_counts['frames'] == 57600
        # Translate staged artifact paths to final paths in the published manifest.
        for artifact in manifest['artifacts']:
            if artifact['path'].startswith('qc/'):
                artifact['path'] = 'qc/postprocessing/' + artifact['path'][3:]
            elif artifact['path'] == 'water_regions_effective.json':
                artifact['path'] = 'config/water_regions_effective_v002.json'
        manifest['publication_job_id'] = os.environ['SLURM_JOB_ID']
        manifest['quality_review'] = pin(review_path)
        (work / 'tables').rename(DEST / 'tables')
        (work / 'qc').rename(DEST / 'qc/postprocessing')
        (work / 'water_regions_effective.json').rename(DEST / 'config/water_regions_effective_v002.json')
        atomic_json(DEST / 'manifest.json', manifest)
        atomic_json(final_marker, {'dataset_id': manifest['dataset_id'], 'build_job_id': build_id,
                                  'publication_job_id': os.environ['SLURM_JOB_ID'], 'status': 'validated_complete',
                                  'manifest_sha256': checksum(DEST / 'manifest.json'),
                                  'quality_review_sha256': checksum(review_path),
                                  'counts': output_counts, 'centroid_source': 'shell_proxy',
                                  'water_status': manifest['water_status'],
                                  'scientific_identity_accuracy': 'not_ground_truth_validated_no_zero_switch_guarantee',
                                  'completion_means': 'Pinned inputs/config/code, synthetic tests, saved-table readback, count/key/continuity checks and bounded visual review passed.'})
        print(json.dumps(json.loads(final_marker.read_text()), indent=2))
    finally:
        (DEST / '.conversion.lock').unlink()


if __name__ == '__main__':
    main()
