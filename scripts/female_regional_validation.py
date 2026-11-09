#!/usr/bin/env python3
"""Condition saved female validation on the right bank of the accepted puddle.

No model loading, training, inference, tracking, video decoding, or changes to
pinned production files. Extend the actual right shoreline vertically to the
image's top/bottom; do NOT replace its curved/overhanging bank with an x cutoff.
"""
from __future__ import annotations
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
import sleap_io as sio
from female_workflow import matched_distances, metrics
from pipeline_common import (ROOT, atomic_json, dataset_manifest, load_run,
    project_path, read_json, sha256)

MASK = ROOT / 'datasets/v007_best_body_first_dataframe_v001/config/water_regions.json'
OUTPUT = ROOT / 'checks/female_v002/validation/right_bank_v001'


def right_bank_geometry(polygon_xy, width, height):
    """Choose the shore path whose right-edge closure excludes the water area.

Selecting a real contour path preserves bank indentations even where a
horizontal line crosses the water more than once. The alternative contour path
would enclose the water and is explicitly rejected.
"""
    vertices = np.asarray(polygon_xy, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or not np.isfinite(vertices).all():
        raise ValueError('Invalid water coordinates')
    if np.array_equal(vertices[0], vertices[-1]):
        vertices = vertices[:-1]
    if len(vertices) < 3 or width <= 0 or height <= 0:
        raise ValueError('Invalid polygon/dimensions')
    if not ((vertices >= 0).all() and (vertices[:, 0] < width).all() and (vertices[:, 1] < height).all()):
        raise ValueError('Water outside image; right-edge closure would be ambiguous')
    water = Polygon(vertices)
    if not water.is_valid or water.area <= 0:
        raise ValueError('Invalid water polygon')
    top_indices = np.flatnonzero(vertices[:, 1] == vertices[:, 1].min())
    bottom_indices = np.flatnonzero(vertices[:, 1] == vertices[:, 1].max())
    top = int(top_indices[np.argmax(vertices[top_indices, 0])])
    bottom = int(bottom_indices[np.argmax(vertices[bottom_indices, 0])])
    if top == bottom:
        raise ValueError('Water must have vertical extent')
    candidates = []
    for direction in (1, -1):
        indices, index = [top], top
        for _ in range(len(vertices)):
            index = (index+direction) % len(vertices)
            indices.append(index)
            if index == bottom:
                break
        chain = vertices[indices]
        shore = np.vstack(([chain[0, 0], 0.], chain, [chain[-1, 0], float(height)]))
        shore = shore[np.r_[True, np.any(np.diff(shore, axis=0) != 0, axis=1)]]
        roi_xy = np.vstack((shore, [float(width), float(height)], [float(width), 0.]))
        roi = Polygon(roi_xy)
        if (roi.is_valid and roi.area > 0
                and roi.intersection(water).area <= 1e-7
                and roi.covers(LineString([(width, 0), (width, height)]))):
            candidates.append((roi, shore, roi_xy, indices))
    if len(candidates) != 1:
        raise ValueError(f'Expected one unambiguous right shoreline, found {len(candidates)}')
    roi, shore, roi_xy, indices = candidates[0]
    return {'water': water, 'roi': roi, 'shore': LineString(shore), 'shore_xy': shore,
            'roi_xy': roi_xy, 'source_vertex_indices': indices,
            'top_tip_xy': vertices[top].tolist(), 'bottom_tip_xy': vertices[bottom].tolist(),
            'width': width, 'height': height}


def region_membership(xy, geometry):
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    valid = (np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) & (xy[:, 0] < geometry['width'])
             & (xy[:, 1] >= 0) & (xy[:, 1] < geometry['height']))
    keep = np.zeros(len(xy), dtype=bool)
    ix = np.flatnonzero(valid)
    if len(ix):
        points = xy[ix]
        keep[ix] = (shapely.intersects_xy(geometry['roi'], points[:, 0], points[:, 1])
            & ~shapely.intersects_xy(geometry['water'], points[:, 0], points[:, 1])
            & ~shapely.intersects(geometry['shore'], shapely.points(points)))
    return keep


def summarize(truths, predictions, geometry, radii, restrict):
    distances = {str(r): [] for r in radii}
    per_frame = []
    nt = npred = 0
    for frame in sorted(truths):
        gt, pred = truths[frame], predictions[frame]
        if restrict:
            gt, pred = gt[region_membership(gt, geometry)], pred[region_membership(pred, geometry)]
        finite_predictions = pred[np.isfinite(pred).all(axis=1)]
        record = {'frame': frame, 'human_instances': len(gt), 'predicted_instances': len(pred), 'metrics': {}}
        for radius in radii:
            d = matched_distances(gt, finite_predictions, radius)
            record['metrics'][str(radius)] = metrics(len(gt), len(pred), d)
            distances[str(radius)].extend(d.tolist())
        nt, npred = nt+len(gt), npred+len(pred)
        per_frame.append(record)
    return {'human_instances': nt, 'predicted_instances': npred, 'per_frame': per_frame,
            'metrics_by_matching_radius_px': {r: metrics(nt, npred, d) for r, d in distances.items()}}


def read_points(labels, user):
    frames = [int(lf.frame_idx) for lf in labels]
    if len(frames) != len(set(frames)) or len(labels.videos) != 1:
        raise RuntimeError('Duplicate frame or multiple videos')
    if len(labels.skeletons) != 1 or labels.skeletons[0].node_names != ['FBody']:
        raise RuntimeError('Not the female one-node labels')
    return {int(lf.frame_idx): np.array([i.numpy()[0] for i in (lf.user_instances if user else lf.instances)],
                                       dtype=float).reshape(-1, 2) for lf in labels}


def run():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    if OUTPUT.exists():
        raise FileExistsError(f'No overwrite of existing regional validation: {OUTPUT}')
    config = ROOT / 'configs/female_v002/run.yaml'
    cfg = load_run(config)
    ds = dataset_manifest(cfg)['manifest']
    original_path = ROOT / 'checks/female_v002/validation/VALIDATION_COMPLETE.json'
    original = read_json(original_path)
    pred_path = original_path.parent / 'heldout.predictions.slp'
    truth_path = ROOT / 'data/labels/female_v002/val.slp'
    assert sha256(pred_path) == original['prediction_sha256']
    mask = read_json(MASK)
    assert mask['version'] == 'legacy_water_accepted_v002'
    assert mask['boundary_included'] is True and mask['uncertainty_band_px'] == 0
    entry = mask['videos']['Crab_Cam_2_2']
    assert entry['confirmation_status'] == 'accepted_for_initial_use'
    # Both accepted per-video outlines are identical for this fixed camera.
    assert entry['polygon_xy'] == mask['videos']['Crab_Cam_2_3']['polygon_xy']
    width, height = entry['dimensions_wh']
    assert ds['video']['shape'][1:3] == [height, width]
    geometry = right_bank_geometry(entry['polygon_xy'], width, height)
    files = [config, MASK, original_path, pred_path, truth_path,
             ROOT / 'data/labels/female_v002/manifest.json',
             ROOT / 'scripts/female_regional_validation.py', ROOT / 'tests/test_female_regional_validation.py',
             ROOT / 'scripts/female_workflow.py', ROOT / 'scripts/pipeline_common.py']
    before = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    truths_slp = sio.load_slp(truth_path, open_videos=False)
    predictions_slp = sio.load_slp(pred_path, open_videos=False)
    truths, predictions = read_points(truths_slp, True), read_points(predictions_slp, False)
    expected = original['frames']
    assert sorted(truths) == sorted(predictions) == expected == ds['splits']['val']['frames']
    assert Path(truths_slp.videos[0].filename).resolve() == Path(predictions_slp.videos[0].filename).resolve() == project_path(ds['video']['path'])
    assert not predictions_slp.tracks and not any(lf.user_instances for lf in predictions_slp)
    radii = cfg['validation']['matching_radii_px']
    baseline = summarize(truths, predictions, geometry, radii, False)
    assert baseline['human_instances'] == original['human_instances']
    assert baseline['predicted_instances'] == original['predicted_instances']
    assert baseline['metrics_by_matching_radius_px'] == original['metrics_by_matching_radius_px']
    for new, old in zip(baseline['per_frame'], original['per_frame']):
        assert new['frame'] == old['frame'] and new['metrics'] == old['metrics']
    regional = summarize(truths, predictions, geometry, radii, True)
    excluded = {key: baseline[key]-regional[key] for key in ('human_instances', 'predicted_instances')}
    stage = Path(tempfile.mkdtemp(prefix='.right_bank_v001.', dir=OUTPUT.parent))
    roi = {'source_mask': str(MASK.relative_to(ROOT)), 'source_mask_sha256': sha256(MASK),
        'mask_version': mask['version'], 'source_video_key': 'Crab_Cam_2_2',
        'applied_validation_video': ds['video']['path'], 'dimensions_wh': [width, height],
        'coordinate_convention': 'Native image pixels, x right, y down; no coordinate transform.',
        'definition': 'Actual right shoreline, closed to the right image edge; vertical extensions from top/bottom tips to image top/bottom.',
        'point_reference': 'FBody for BOTH human labels and predictions',
        'boundary_policy': 'Strictly right of shoreline/vertical extensions; water and boundary excluded. In-image top edge included away from shoreline.',
        'uncertainty_band_px': 0, 'top_tip_xy': geometry['top_tip_xy'],
        'bottom_tip_xy': geometry['bottom_tip_xy'],
        'shoreline_with_vertical_extensions_xy': geometry['shore_xy'].tolist(),
        'region_polygon_xy': geometry['roi_xy'].tolist(),
        'source_water_vertex_indices': geometry['source_vertex_indices'],
        'matching_policy': 'Filter human labels and predictions independently to this same ROI BEFORE one-to-one maximum-cardinality/minimum-distance matching.',
        'authorization': 'User selected curved right shoreline with vertical tip boundaries; same fixed outline as accepted project ROI.'}
    atomic_json(stage / 'ROI.json', roi)
    with (stage / 'roi_vertices_px.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['vertex', 'x_px', 'y_px'])
        for index, (x, y) in enumerate(geometry['roi_xy']):
            writer.writerow([index, x, y])
    with (stage / 'point_membership.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['frame', 'source', 'instance_index', 'FBody_x_px', 'FBody_y_px', 'in_right_bank_roi'])
        for frame in expected:
            for source, xy in [('human', truths[frame]), ('prediction', predictions[frame])]:
                flags = region_membership(xy, geometry)
                for i, ((x, y), inside) in enumerate(zip(xy, flags)):
                    writer.writerow([frame, source, i, x, y, bool(inside)])
    # CSV readback independently reconciles retained/excluded source counts.
    with (stage / 'point_membership.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == baseline['human_instances']+baseline['predicted_instances']
    for source, key in [('human', 'human_instances'), ('prediction', 'predicted_instances')]:
        assert sum(r['source'] == source and r['in_right_bank_roi'] == 'True' for r in rows) == regional[key]
    primary = str(cfg['validation']['primary_matching_radius_px'])
    def percent(value):
        return 'NA' if value is None else f'{100*value:.2f}%'
    lines = ['# Female validation conditioned on the right bank of the puddle', '',
        f"Held-out original clip frames: {expected}. Exact epoch-75 centroid and the same selected centered model; no new inference.", '',
        'The ROI follows the accepted right shoreline, extended vertically to image edges.',
        f"Top tip: {geometry['top_tip_xy']}; bottom tip: {geometry['bottom_tip_xy']}. Thus the lower vertical boundary is x=2590 from y=3170 to y=4320; the upper tip already meets y=0.",
        'Boundary/water points are excluded; both labels and predictions are filtered before matching.', '',
        f'## Primary comparison ({primary}-pixel one-to-one matching radius)', '',
        '| Selection | Labels | Predictions | TP | FP relative to labels | FN | Precision | Recall | F1 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, data in [('Full held-out frames', baseline), ('Right-bank ROI', regional)]:
        m = data['metrics_by_matching_radius_px'][primary]
        lines.append(f"| {name} | {data['human_instances']} | {data['predicted_instances']} | {m['true_positives']} | {m['false_positives_relative_to_labels']} | {m['false_negatives']} | {percent(m['precision'])} | {percent(m['recall'])} | {percent(m['f1'])} |")
    lines += ['', '## Regional results by frame', '', '| Frame | Labels | Predictions | TP | FP | FN | Precision | Recall | F1 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for frame in regional['per_frame']:
        m = frame['metrics'][primary]
        lines.append(f"| {frame['frame']} | {frame['human_instances']} | {frame['predicted_instances']} | {m['true_positives']} | {m['false_positives_relative_to_labels']} | {m['false_negatives']} | {percent(m['precision'])} | {percent(m['recall'])} | {percent(m['f1'])} |")
    lines += ['', '## Regional matching-radius checks', '', '| Radius (px) | Precision | Recall | F1 | Median matched error (px) |', '|---|---:|---:|---:|---:|']
    for radius in radii:
        m = regional['metrics_by_matching_radius_px'][str(radius)]
        error = m['matched_error_px']
        median = 'NA' if error is None else f"{error['median']:.4f}"
        lines.append(f"| {radius} | {percent(m['precision'])} | {percent(m['recall'])} | {percent(m['f1'])} | {median} |")
    limitations = list(original['limitations']) + [
        'This is a post-hoc spatial subset requested after reviewing the overall validation; not a new independent test.',
        'ROI is a fixed coarse hand-drawn outline from the same camera; no new image-based registration or shoreline refinement.',
        'Point errors across the ROI boundary can change inclusion. No uncertainty band or special boundary tolerance is added.',
        'Regional filtering does not alter training, active video predictions, or canonical all-detection outputs.']
    lines += ['', '## Limits', ''] + ['- '+text for text in limitations]
    (stage / 'report.md').write_text('\n'.join(lines)+'\n')
    assert before == {str(p.relative_to(ROOT)): sha256(p) for p in files}, 'Source changed during analysis'
    tests_path = ROOT / f'checks/female_v002/regional_tests_{os.environ["SLURM_JOB_ID"]}.json'
    tests = read_json(tests_path)
    assert tests['passed'] is True and tests['test_source_sha256'] == sha256(ROOT / 'tests/test_female_regional_validation.py')
    report = {'status': 'validated_complete', 'run_id': 'female_v002_right_bank_v001',
        'job_id': os.environ['SLURM_JOB_ID'], 'frames': expected, 'primary_matching_radius_px': int(primary),
        'source_validation_job_id': original['job_id'], 'source_models': original['selected_models'],
        'source_hashes': before, 'roi': roi, 'full_frame_reproduced': baseline,
        'regional': regional, 'excluded_by_region': excluded, 'limitations': limitations,
        'tests': tests, 'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'shapely', 'sleap-io')},
        'artifacts': {p.name: sha256(p) for p in sorted(stage.iterdir()) if p.is_file()}}
    atomic_json(stage / 'REGIONAL_VALIDATION_COMPLETE.json', report)
    stage.rename(OUTPUT)
    print(json.dumps({'output': str(OUTPUT), 'regional': regional, 'excluded_by_region': excluded}, indent=2), flush=True)


if __name__ == '__main__':
    run()
