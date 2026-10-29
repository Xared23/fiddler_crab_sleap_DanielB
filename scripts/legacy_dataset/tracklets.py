"""Deterministic adjacent-frame tracklets using Shell and Shoulder only.

No persistent identity, motion prediction, assignment fallback, interpolation,
water-based exclusion, or neural inference. Missing Shoulder yields a retained
body-only singleton, not an invented point or a body-only temporal link.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
import shapely
from shapely.geometry import Polygon


@dataclass
class Frame:
    index: int
    body: np.ndarray
    shoulder: np.ndarray
    body_score: np.ndarray
    shoulder_score: np.ndarray
    body_visible: np.ndarray
    shoulder_visible: np.ndarray
    observed: bool = True

    def __len__(self):
        return len(self.body)

    @classmethod
    def empty(cls, index, observed=True):
        return cls(index, np.empty((0, 2)), np.empty((0, 2)), np.empty(0),
                   np.empty(0), np.empty(0, bool), np.empty(0, bool), observed)


def in_image(xy, width, height):
    return (np.isfinite(xy).all(1) & (xy[:, 0] >= 0) & (xy[:, 0] < width)
            & (xy[:, 1] >= 0) & (xy[:, 1] < height))


def validity(frame, width, height, config):
    n = len(frame)
    finite = np.isfinite(frame.body).all(1)
    inside = in_image(frame.body, width, height)
    reasons = np.full(n, '', dtype=object)
    # First applicable reason has priority, with duplicates assessed among every
    # finite in-image body, even if the competing detection itself is rejected.
    for mask, reason in [
        (~finite, 'missing_position'),
        (finite & ~inside, 'outside_image'),
        (~frame.body_visible, 'invisible'),
        (~np.isfinite(frame.body_score) | (frame.body_score < config['body_score_min']), 'low_body_score')]:
        reasons[(reasons == '') & mask] = reason
    eligible = np.flatnonzero(inside)
    duplicate = np.zeros(n, bool)
    if len(eligible) > 1:
        pairs = cKDTree(frame.body[eligible]).query_pairs(config['duplicate_shell_radius_px'], output_type='ndarray')
        if pairs.size:
            duplicate[eligible[pairs.ravel()]] = True
    reasons[(reasons == '') & duplicate] = 'duplicate_candidate'
    accepted = reasons == ''
    shoulder_valid = (in_image(frame.shoulder, width, height) & frame.shoulder_visible
                      & np.isfinite(frame.shoulder_score)
                      & (frame.shoulder_score >= config['shoulder_score_min']))
    length = np.linalg.norm(frame.shoulder - frame.body, axis=1)
    geometry_valid = np.isfinite(length) & (length >= config['body_shoulder_length_min_px']) & (length <= config['body_shoulder_length_max_px'])
    linkable = accepted & shoulder_valid & geometry_valid
    return {'accepted': accepted, 'reasons': reasons, 'inside': inside,
            'duplicate': duplicate, 'shoulder_valid': shoulder_valid,
            'geometry_valid': geometry_valid, 'linkable': linkable}


def associate(previous, current, pv, cv, config):
    """Return mutual-isolated links and explicit reasons for non-links.

The ambiguity neighborhood is broader than the displacement gate. Any second
finite in-image body inside this neighborhood vetoes a match, including one
with low confidence or missing Shoulder. A shoulder gate cannot erase a nearby
body competitor to manufacture uniqueness.
"""
    links = {}
    incoming = np.full(len(current), 'no_candidate', dtype=object)
    outgoing = np.full(len(previous), 'no_candidate', dtype=object)
    left = np.flatnonzero(pv['inside'])
    right = np.flatnonzero(cv['inside'])
    near_prev = [[] for _ in range(len(current))]
    near_cur = [[] for _ in range(len(previous))]
    if len(left) and len(right):
        neighborhoods = cKDTree(previous.body[left]).query_ball_point(
            current.body[right], config['max_body_displacement_px'] + config['ambiguity_guard_px'])
        for j, neighbors in zip(right, neighborhoods):
            for k in neighbors:
                i = int(left[k])
                near_prev[j].append(i)
                near_cur[i].append(int(j))
    for j in range(len(current)):
        if not cv['accepted'][j]:
            incoming[j] = 'rejected'
        elif not cv['linkable'][j]:
            incoming[j] = 'shoulder_unreliable'
        elif len(near_prev[j]) > 1:
            incoming[j] = 'ambiguous_competitors'
        elif len(near_prev[j]) == 1:
            i = near_prev[j][0]
            if len(near_cur[i]) > 1:
                incoming[j] = 'ambiguous_competitors'
            elif not pv['accepted'][i]:
                incoming[j] = 'predecessor_rejected'
            elif not pv['linkable'][i]:
                incoming[j] = 'predecessor_shoulder_unreliable'
            elif np.linalg.norm(previous.body[i] - current.body[j]) > config['max_body_displacement_px']:
                incoming[j] = 'body_displacement_gate'
            elif (np.linalg.norm(previous.shoulder[i] - current.shoulder[j]) > config['max_shoulder_displacement_px']
                  or np.linalg.norm((previous.shoulder[i] - previous.body[i]) - (current.shoulder[j] - current.body[j])) > config['max_body_shoulder_vector_change_px']):
                incoming[j] = 'shoulder_displacement_or_vector_gate'
            else:
                links[j] = i
                incoming[j] = 'linked'
    inverse = {i: j for j, i in links.items()}
    for i in range(len(previous)):
        if not pv['accepted'][i]:
            outgoing[i] = 'rejected'
        elif i in inverse:
            outgoing[i] = 'linked'
        elif not pv['linkable'][i]:
            outgoing[i] = 'shoulder_unreliable'
        elif len(near_cur[i]) > 1:
            outgoing[i] = 'ambiguous_competitors'
        elif len(near_cur[i]) == 1:
            j = near_cur[i][0]
            if len(near_prev[j]) > 1:
                outgoing[i] = 'ambiguous_competitors'
            elif not cv['accepted'][j]:
                outgoing[i] = 'successor_rejected'
            elif not cv['linkable'][j]:
                outgoing[i] = 'successor_shoulder_unreliable'
            else:
                outgoing[i] = incoming[j]
    assert len(set(links.values())) == len(links), 'Non-one-to-one association'
    return links, incoming, outgoing


class Tracker:
    def __init__(self, video_id, width, height, config):
        self.video_id, self.width, self.height, self.config = video_id, width, height, config
        self.prefix = f"{config['dataset_id']}/{video_id}/t"
        self.next_id = 1
        self.previous = None
        self.pv = None
        self.previous_ids = None
        self.active = {}

    def name(self, local_id):
        return self.prefix + f'{local_id:09d}'

    def _close(self, local_id, reason):
        info = self.active.pop(int(local_id))
        return {'tracklet_id': self.name(int(local_id)), 'tracklet_local_id': int(local_id),
                'video_id': self.video_id, 'start_frame': info[0], 'end_frame': info[1],
                'length': info[2], 'start_reason': info[3], 'termination_reason': reason}

    def process(self, current):
        if self.previous is not None:
            assert current.index > self.previous.index, 'Frames must be strictly chronological'
        cv = validity(current, self.width, self.height, self.config)
        ended = []
        links = {}
        incoming = np.full(len(current), 'video_start', dtype=object)
        if self.previous is not None:
            if current.index != self.previous.index + 1 or not current.observed or not self.previous.observed:
                reason = 'missing_span'
                incoming[:] = reason
                outgoing = np.full(len(self.previous), reason, dtype=object)
            elif len(current) == 0:
                outgoing = np.full(len(self.previous), 'empty_frame', dtype=object)
            elif len(self.previous) == 0:
                incoming[:] = 'after_empty_frame'
                outgoing = np.empty(0, dtype=object)
            else:
                links, incoming, outgoing = associate(self.previous, current, self.pv, cv, self.config)
            linked_predecessors = set(links.values())
            for i in np.flatnonzero(self.pv['accepted']):
                if int(i) not in linked_predecessors:
                    ended.append(self._close(self.previous_ids[i], str(outgoing[i])))
        ids = np.full(len(current), -1, dtype=np.int64)
        for j in np.flatnonzero(cv['accepted']):
            if int(j) in links:
                local_id = int(self.previous_ids[links[int(j)]])
                assert local_id in self.active
                info = self.active[local_id]
                assert info[1] + 1 == current.index
                info[1] = current.index
                info[2] += 1
            else:
                local_id = self.next_id
                self.next_id += 1
                start_reason = str(incoming[j])
                if not cv['linkable'][j]:
                    start_reason = 'shoulder_unreliable'
                    incoming[j] = start_reason
                self.active[local_id] = [current.index, current.index, 1, start_reason]
            ids[j] = local_id
        assert len(np.unique(ids[ids > 0])) == np.count_nonzero(cv['accepted'])
        self.previous, self.pv, self.previous_ids = current, cv, ids
        return {'ids': ids, 'validity': cv, 'incoming_reason': incoming,
                'predecessor_index': np.array([links.get(j, -1) for j in range(len(current))], dtype=np.int64),
                'ended': ended}

    def finish(self):
        result = [self._close(i, 'video_end') for i in sorted(self.active)]
        self.previous = None
        return result


def water_membership(xy, width, height, polygon_xy, confirmed=True, uncertainty_px=0):
    """Return list[bool|None]; boundary included, unknown never defaults to land."""
    result = [None] * len(xy)
    if not confirmed or polygon_xy is None:
        return result
    polygon = Polygon(polygon_xy)
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError('Invalid water polygon')
    indices = np.flatnonzero(in_image(xy, width, height))
    if len(indices):
        points = xy[indices]
        inside = shapely.intersects_xy(polygon, points[:, 0], points[:, 1])
        uncertain = np.zeros(len(indices), bool)
        if uncertainty_px > 0:
            uncertain = shapely.distance(shapely.points(points), polygon.boundary) <= uncertainty_px
        for i, member, unknown in zip(indices, inside, uncertain):
            result[int(i)] = None if unknown else bool(member)
    return result
