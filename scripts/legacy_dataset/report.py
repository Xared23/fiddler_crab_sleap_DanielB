#!/usr/bin/env python3
"""Summarize completed audit and render only proposed, unconfirmed water ROIs.

No prediction/dataframe conversion, neural inference, or original-file writes.
"""
from __future__ import annotations
import collections
import hashlib
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'datasets/legacy_native_tracklets_v001'


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as fp:
        for block in iter(lambda: fp.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Slurm only'
    final = DEST / 'qc'
    assert not final.exists(), 'Refusing to overwrite prior QC publication'
    fd = os.open(DEST / '.report.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    work = DEST / f'.report_job_{os.environ["SLURM_JOB_ID"]}'
    work.mkdir(exist_ok=False)
    try:
        inventory_path = DEST / 'audit/inventory.json'
        marker = json.loads((DEST / 'audit/AUDIT_COMPLETE.json').read_text())
        assert checksum(inventory_path) == marker['inventory']['sha256']
        inventory = json.loads(inventory_path.read_text())
        regions = json.loads((DEST / 'config/water_regions.json').read_text())
        report = {'status': 'audit_complete_conversion_pending_user_semantics',
                  'audit_job_id': inventory['job_id'], 'report_job_id': os.environ['SLURM_JOB_ID'],
                  'inventory_sha256': checksum(inventory_path),
                  'report_code_sha256': checksum(Path(__file__)),
                  'roi_config_sha256': checksum(DEST / 'config/water_regions.json'),
                  'test_code_sha256': checksum(ROOT / 'tests/legacy_dataset/test_audit.py'),
                  'test_job_id': '15428784', 'test_result': '5 synthetic audit tests passed; not association tests',
                  'videos': {}, 'decisions_required': ['Does body mean Shell position?',
                                                     'Approve explicitly labeled Shell centroid proxy, or keep centroid coordinates missing?',
                                                     'Confirm/edit each water polygon and reference-point membership rule.'],
                  'limitations': ['No converted observations, assigned tracklets, or definitive water labels yet.',
                                  'Static polygon previews are proposals, not segmentation or water confirmation.',
                                  'No historical checkpoint hashes were serialized; present hashes pin inputs as found.',
                                  'No per-frame timestamp scan or exhaustive video decode was performed.',
                                  'No ground-truth association accuracy or zero-switch guarantee.']}
        models = [str(ROOT / 'models' / name) for name in ['dan_centroid_native', 'dan_centered_native']]
        for name, result in inventory['videos'].items():
            chunks = result['chunks']
            all_files = chunks + [result['merged']]
            tracks = collections.Counter()
            for rec in chunks:
                tracks.update({k: int(v) for k, v in rec.get('instance_track_counts', {}).items()})
            centroid_fields = {rec['path']: [field for field in rec.get('hdf5_datasets', {}) if any(term in field.lower() for term in ['centroid', 'crop', 'bbox', 'transform'])]
                               for rec in all_files}
            centroid_fields = {k: v for k, v in centroid_fields.items() if v}
            prov_bad = [rec['path'] for rec in chunks if rec.get('provenance', {}).get('model_paths') != models]
            settings = collections.Counter(json.dumps(rec.get('provenance', {}).get('inference_config', {}), sort_keys=True) for rec in chunks)
            skeleton_variants = {json.dumps(rec.get('metadata', {}).get('skeletons', []), sort_keys=True) for rec in all_files}
            node_variants = {json.dumps(rec.get('metadata', {}).get('nodes', []), sort_keys=True) for rec in all_files}
            serialized_schemas = {json.dumps({k: rec.get('hdf5_datasets', {}).get(k, {}).get('dtype') for k in ['instances', 'pred_points', 'frames']}, sort_keys=True) for rec in all_files}
            measurements = result['measurements']
            counts = measurements['counts']
            assert counts['instances'] == result['coverage']['chunk_instances'], 'Measurement/source reconciliation failed'
            assert sum(tracks.values()) == counts['instances'], 'Track accounting failed'
            section = {'disposition': result['disposition'], 'video': result['video'],
                       'chunk_count': len(chunks), 'coverage': result['coverage'],
                       'all_source_video_references_match_local': all(rec.get('reference_matches_local_absolute_path') for rec in all_files),
                       'all_chunks_pass_sleap_io_readback': all(rec.get('sleap_io_readback_no_video_open') for rec in chunks),
                       'all_chunks_name_original_model_pair': not prov_bad, 'provenance_model_mismatch_files': prov_bad,
                       'skeleton_graph_metadata_variants': len(skeleton_variants), 'node_metadata_variants': len(node_variants),
                       'serialized_frame_instance_point_schema_variants': len(serialized_schemas),
                       'inference_configs_and_chunk_counts': [{'config': json.loads(k), 'chunks': v} for k, v in settings.items()],
                       'serialized_source_track_counts': dict(tracks),
                       'nonempty_track_catalog_files': [rec['path'] for rec in all_files if rec.get('tracks_json')],
                       'centroid_crop_bbox_transform_datasets_found': centroid_fields,
                       'measurements': measurements,
                       'fractions': {'shoulder_missing_xy': counts['Shoulder_missing_xy'] / counts['instances'],
                                     'both_trusted_inbounds_score_ge_0.2': counts['trusted_both_finite_inbounds_score_ge_0.2'] / counts['instances']},
                       'review_frame_indices': [x['frame'] for x in result['preview_evidence']['frames'] if x['decoded']],
                       'decoded_sample_failures': [x['frame'] for x in result['preview_evidence']['frames'] if not x['decoded']]}
            report['videos'][name] = section
            roi = regions['videos'][name]
            assert roi['confirmation_status'] == 'proposed_unconfirmed'
            assert [result['video']['width'], result['video']['height']] == roi['dimensions_wh']
            polygon = roi['polygon_xy']
            assert len(polygon) >= 3
            shape = Polygon(polygon)
            assert shape.is_valid and shape.area > 0, 'Invalid proposed polygon'
            assert all(0 <= x < roi['dimensions_wh'][0] and 0 <= y < roi['dimensions_wh'][1] for x, y in polygon)
            sheets = []
            for frame in roi['registration_review_frames']:
                base = Image.open(DEST / 'audit' / f'{name}.frame_{frame:06d}.png').convert('RGBA')
                assert base.size == (1600, 964), 'Audit preview geometry changed; do not misregister polygon'
                overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                transformed = [(x / 4.8, y / 4.8 + 64) for x, y in polygon]
                draw.polygon(transformed, fill=(255, 90, 0, 60))
                draw.line(transformed + [transformed[0]], fill=(255, 120, 0, 255), width=3)
                for k, (x, y) in enumerate(transformed):
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill='orange')
                    if k % 3 == 0:
                        draw.text((x + 4, y), str(k), fill='orange')
                composed = Image.alpha_composite(base, overlay).convert('RGB')
                label = ImageDraw.Draw(composed)
                label.rectangle((0, 20, 1599, 46), fill='#202020')
                label.text((10, 27), 'ORANGE: PROPOSED WATER ROI ONLY - user confirmation required. Unknown flags remain NA.', fill='orange')
                target = work / f'{name}.water_proposal.frame_{frame:06d}.png'
                composed.save(target)
                sheets.append(composed.resize((800, 482)))
            sheet = Image.new('RGB', (800 * len(sheets), 482), 'black')
            for k, image in enumerate(sheets):
                sheet.paste(image, (800 * k, 0))
            sheet.save(work / f'{name}.water_proposal.early_middle_late.png')
        write_json(work / 'assessment.json', report)
        # Read-back: ensure publication is valid JSON with expected identities/counts.
        loaded = json.loads((work / 'assessment.json').read_text())
        assert loaded == report
        write_json(work / 'REPORT_COMPLETE.json', {'job_id': os.environ['SLURM_JOB_ID'],
                                                 'assessment_sha256': checksum(work / 'assessment.json'),
                                                 'is_dataset_completion': False})
        work.rename(final)
        print(json.dumps({name: {'disposition': value['disposition'],
                                'observations': value['coverage']['chunk_instances'],
                                'tracks': value['serialized_source_track_counts'],
                                'detector_fields': value['centroid_crop_bbox_transform_datasets_found']}
                          for name, value in report['videos'].items()}, indent=2))
        print(f'Published audit assessment and UNCONFIRMED ROI previews to {final}')
    finally:
        (DEST / '.report.lock').unlink()


if __name__ == '__main__':
    main()
