#!/usr/bin/env python3
"""Read-only counts and within-frame burrow spacing using the accepted right ROI.

Repeated frames are never pooled for spacing or silently merged into physical
burrow identities. Exact coordinate overlap and cross-frame proximity are audit
information, not a deduplication/re-identification rule.
"""
from __future__ import annotations
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.distance import cdist, pdist
import sleap_io as sio

from female_regional_validation import right_bank_geometry, region_membership
from pipeline_common import ROOT, atomic_json, read_json, sha256

SOURCE = ROOT / 'data/labels/burrows/BurrowLabels3.slp'
ROI = ROOT / 'checks/female_v002/validation/right_bank_v001/ROI.json'
OUTPUT = ROOT / 'checks/burrow_locations_v001'
EXPECTED_SOURCE_SHA256 = '1c8aa0bbda851459f470e49c2d1cc73cda93b4eb06e5bae083f5a848ef7f7903'


def distance_summary(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {'n': 0, 'mean_px': None, 'median_px': None, 'min_px': None, 'max_px': None,
                'p10_px': None, 'p90_px': None}
    assert np.isfinite(values).all() and (values >= 0).all()
    return {'n': len(values), 'mean_px': float(np.mean(values)), 'median_px': float(np.median(values)),
            'min_px': float(np.min(values)), 'max_px': float(np.max(values)),
            'p10_px': float(np.quantile(values, .1)), 'p90_px': float(np.quantile(values, .9))}


def spacing(xy):
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if not np.isfinite(xy).all():
        raise ValueError('Filter invalid coordinates before distance calculations')
    unique = np.unique(xy, axis=0)
    if len(unique) < 2:
        nearest = np.empty(0)
        pairs = np.empty(0)
    else:
        matrix = cdist(unique, unique)
        np.fill_diagonal(matrix, np.inf)
        nearest = matrix.min(axis=1)
        pairs = pdist(unique)
        assert len(pairs) == len(unique)*(len(unique)-1)//2
    return {'annotation_points': len(xy), 'distinct_coordinate_points': len(unique),
            'exact_duplicate_points_excluded_from_spacing': len(xy)-len(unique),
            'nearest_neighbor': distance_summary(nearest), 'all_unordered_pairs': distance_summary(pairs)}


def valid_points(xy, width, height):
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    return (np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) & (xy[:, 0] < width)
            & (xy[:, 1] >= 0) & (xy[:, 1] < height))


def frame_summary(xy, geometry):
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    valid = valid_points(xy, geometry['width'], geometry['height'])
    right = region_membership(xy, geometry)
    assert not (right & ~valid).any()
    # For ROI-edge context, also measure each right-side point's nearest
    # neighbor anywhere in the full frame (instead of forcing it to be right).
    unique = np.unique(xy[valid], axis=0)
    right_unique = region_membership(unique, geometry)
    if len(unique) > 1:
        d = cdist(unique, unique)
        np.fill_diagonal(d, np.inf)
        anchor_nearest = d.min(axis=1)[right_unique]
    else:
        anchor_nearest = np.empty(0)
    return {'annotated_instances': len(xy), 'valid_in_image_instances': int(valid.sum()),
            'invalid_or_outside_instances': int((~valid).sum()),
            'right_of_puddle_instances': int(right.sum()),
            'outside_right_region_instances': int((valid & ~right).sum()),
            'full_frame_spacing': spacing(xy[valid]),
            'right_region_spacing': spacing(xy[right]),
            'right_region_anchors_nearest_neighbor_anywhere_in_frame': distance_summary(anchor_nearest)}


def run():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    if OUTPUT.exists():
        raise FileExistsError('Existing completed burrow statistics are never overwritten')
    assert sha256(SOURCE) == EXPECTED_SOURCE_SHA256, 'Burrow upload changed since audit'
    roi = read_json(ROI)
    mask_path = ROOT / roi['source_mask']
    assert sha256(mask_path) == roi['source_mask_sha256']
    mask = read_json(mask_path)
    entry = mask['videos'][roi['source_video_key']]
    width, height = roi['dimensions_wh']
    geometry = right_bank_geometry(entry['polygon_xy'], width, height)
    assert geometry['roi_xy'].tolist() == roi['region_polygon_xy']
    files = [SOURCE, ROI, mask_path, ROOT / 'scripts/burrow_location_stats.py',
             ROOT / 'tests/test_burrow_location_stats.py', ROOT / 'scripts/female_regional_validation.py']
    before = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    labels = sio.load_slp(SOURCE, open_videos=False)
    assert len(labels.videos) == len(labels.skeletons) == 1
    assert labels.skeletons[0].node_names == ['Burrow']
    assert not any(lf.predicted_instances for lf in labels)
    shape = labels.videos[0].backend_metadata['shape']
    assert shape[1:3] == [height, width]
    assert Path(labels.videos[0].filename).name == 'clip_cam2_8k.mp4'
    indices = [int(lf.frame_idx) for lf in labels]
    assert len(indices) == len(set(indices))
    frame_reports, frame_points, point_rows = [], {}, []
    track_frames = {}
    for lf in sorted(labels, key=lambda frame: frame.frame_idx):
        if not lf.user_instances:
            continue
        frame = int(lf.frame_idx)
        xy = np.array([inst.numpy()[0] for inst in lf.user_instances], dtype=float).reshape(-1, 2)
        frame_reports.append(dict(frame_summary(xy, geometry), frame=frame))
        valid = valid_points(xy, width, height)
        right = region_membership(xy, geometry)
        frame_points[frame] = xy[valid]
        for i, (inst, (x, y), good, inside) in enumerate(zip(lf.user_instances, xy, valid, right)):
            track = None if inst.track is None else inst.track.name
            if track is not None:
                track_frames.setdefault(track, set()).add(frame)
            point_rows.append({'frame': frame, 'instance_index': i, 'Burrow_x_px': x,
                'Burrow_y_px': y, 'valid_in_image': bool(good), 'right_of_puddle': bool(inside),
                'source_track_name': track})
    assert sum(r['annotated_instances'] for r in frame_reports) == 564
    # Exact-coordinate union is explicitly NOT a count of unique physical
    # burrows: near-identical repeated locations need not have identical labels.
    all_points = np.concatenate(list(frame_points.values()))
    exact_union = np.unique(all_points, axis=0)
    across_frames = []
    keys = sorted(frame_points)
    for i, first in enumerate(keys):
        for second in keys[i+1:]:
            a, b = np.unique(frame_points[first], axis=0), np.unique(frame_points[second], axis=0)
            exact_overlap = len(set(map(tuple, a)) & set(map(tuple, b)))
            matrix = cdist(a, b)
            across_frames.append({'frames': [first, second], 'exact_coordinate_overlap': exact_overlap,
                'first_to_second_nearest_px': distance_summary(matrix.min(axis=1)) if len(a) and len(b) else distance_summary([]),
                'second_to_first_nearest_px': distance_summary(matrix.min(axis=0)) if len(a) and len(b) else distance_summary([]),
                'same_exact_coordinate_set': set(map(tuple, a)) == set(map(tuple, b)),
                'note': 'Cross-frame proximity is a diagnostic only. No distance threshold, cross-frame merging or physical identity inference applied.'})
    tests_path = ROOT / f'checks/burrow_location_tests_{os.environ["SLURM_JOB_ID"]}.json'
    tests = read_json(tests_path)
    assert tests['passed'] is True
    assert tests['test_source_sha256'] == sha256(ROOT / 'tests/test_burrow_location_stats.py')
    assert tests['analysis_source_sha256'] == sha256(ROOT / 'scripts/burrow_location_stats.py')
    report = {'status': 'validated_complete', 'job_id': os.environ['SLURM_JOB_ID'],
        'source': str(SOURCE.relative_to(ROOT)), 'source_hashes': before,
        'stored_frame_records': len(labels), 'annotated_frame_records': len(frame_reports),
        'stored_frames_without_human_instances': [int(lf.frame_idx) for lf in labels if not lf.user_instances],
        'total_annotation_instances_across_frames': len(point_rows),
        'total_right_region_annotation_instances_across_frames': sum(r['right_of_puddle_instances'] for r in frame_reports),
        'exact_distinct_coordinates_across_frames': len(exact_union),
        'exact_distinct_coordinates_in_right_region_across_frames': int(region_membership(exact_union, geometry).sum()),
        'per_frame': frame_reports, 'cross_frame_diagnostics': across_frames,
        'tracks': {'catalog_count': len(labels.tracks), 'instances_with_track': sum(r['source_track_name'] is not None for r in point_rows),
                   'track_names_observed_in_multiple_frames': sorted(k for k, v in track_frames.items() if len(v) > 1)},
        'units': 'Original 7680x4320 image pixels; Euclidean distances, no perspective/physical-length calibration.',
        'roi': {'reference': str(ROI.relative_to(ROOT)), 'reference_sha256': sha256(ROI),
                'point_reference_for_this_analysis': 'Burrow', 'definition': roi['definition'],
                'boundary_policy': roi['boundary_policy'], 'region_polygon_xy': roi['region_polygon_xy']},
        'spacing_definition': 'For each labeled frame separately: mean distance from each distinct valid annotated Burrow coordinate to its nearest other annotated coordinate. Right-region spacing uses BOTH points in the right ROI. All-pairs means count each unordered pair once.',
        'limitations': ['Annotations across different frames may repeat the same physical burrows; totals across frames are not certified unique-burrow counts.',
            'No cross-frame point merging or association threshold has been invented.',
            'Exact within-frame duplicate coordinates are removed ONLY for distance calculations and remain in annotation counts/audit CSV.',
            'Nearest neighbors are among annotated burrows, not necessarily all physical burrows; incomplete labeling can increase apparent spacing.',
            'No image-edge or ROI-edge statistical correction; right-anchor/full-frame-neighbor distances are also supplied for boundary context.',
            'Pixel spacing depends on perspective and is not a physical distance.'],
        'tests': tests, 'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'shapely', 'sleap-io')}}
    stage = Path(tempfile.mkdtemp(prefix='.burrow_locations_v001.', dir=OUTPUT.parent))
    with (stage / 'annotations.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(point_rows[0]))
        writer.writeheader()
        writer.writerows(point_rows)
    with (stage / 'annotations.csv').open() as stream:
        saved_rows = list(csv.DictReader(stream))
    assert len(saved_rows) == len(point_rows)
    assert sum(r['right_of_puddle'] == 'True' for r in saved_rows) == report['total_right_region_annotation_instances_across_frames']
    def value(v):
        return 'NA' if v is None else f'{v:.2f}'
    lines = ['# Burrow annotation counts and pixel spacing', '',
        f"Source: `{report['source']}`. {len(point_rows)} annotation instances over {len(frame_reports)} labeled frames; do not assume the sum is unique physical burrows.", '',
        'Same curved right-of-puddle region and vertical tip extensions as female conditional validation. Distances are within each frame, never across pooled repeated frames.', '',
        '| Frame | Annotated points | Right of puddle | Full-frame mean nearest neighbor (px) | Right-region mean nearest neighbor (px) | Full-frame mean all-pairs distance (px) | Right-region mean all-pairs distance (px) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in frame_reports:
        full, right = row['full_frame_spacing'], row['right_region_spacing']
        lines.append(f"| {row['frame']} | {row['annotated_instances']} | {row['right_of_puddle_instances']} | {value(full['nearest_neighbor']['mean_px'])} | {value(right['nearest_neighbor']['mean_px'])} | {value(full['all_unordered_pairs']['mean_px'])} | {value(right['all_unordered_pairs']['mean_px'])} |")
    lines += ['', f"Exact distinct coordinate pairs across frames: {len(exact_union)} overall, {report['exact_distinct_coordinates_in_right_region_across_frames']} in the right ROI. This is a coordinate audit, NOT a physical-burrow deduplication.", '',
        '## Limits', ''] + ['- '+text for text in report['limitations']]
    (stage / 'report.md').write_text('\n'.join(lines)+'\n')
    assert before == {str(p.relative_to(ROOT)): sha256(p) for p in files}, 'Source changed during analysis'
    report['artifacts'] = {p.name: sha256(p) for p in stage.iterdir() if p.is_file()}
    atomic_json(stage / 'BURROW_STATS_COMPLETE.json', report)
    stage.rename(OUTPUT)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    run()
