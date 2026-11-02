"""Body-first, adjacent-frame tracklets with optional Shoulder and bounded motion.

Reuses the published tracker data/validity primitives without modifying them.
Hungarian birth/death assignment is followed by absolute and bidirectional
alternative-margin checks. No observation is synthesized; a missed frame ends ID.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from legacy_dataset.tracklets import Frame, validity as original_validity


def validity(frame, width, height, config):
    result = original_validity(frame, width, height, config)
    result['linkable'] = result['accepted'].copy()
    return result


def association(previous, current, pv, cv, velocity, config):
    m, n = len(previous), len(current)
    links, details = {}, {}
    incoming = np.full(n, 'no_candidate', dtype=object)
    outgoing = np.full(m, 'no_candidate', dtype=object)
    if not m or not n:
        return links, incoming, outgoing, details
    delta = current.body[None, :, :] - previous.body[:, None, :]
    distance = np.linalg.norm(delta, axis=2)
    error = np.linalg.norm(delta - velocity[:, None, :], axis=2)
    weight = config['motion_weight']
    costs = (1 - weight) * distance + weight * error
    shoulder_ok = ((pv['shoulder_valid'] & pv['geometry_valid'])[:, None]
                   & (cv['shoulder_valid'] & cv['geometry_valid'])[None, :])
    vectors = np.linalg.norm((current.shoulder - current.body)[None, :, :]
                            - (previous.shoulder - previous.body)[:, None, :], axis=2)
    costs += np.where(shoulder_ok,
        config['shoulder_vector_weight'] * np.minimum(np.nan_to_num(vectors, nan=0.0), config['shoulder_vector_cost_cap_px']), 0)
    feasible = (pv['inside'][:, None] & cv['inside'][None, :]
                & (distance <= config['max_body_displacement_px'])
                & (error <= config['max_prediction_error_px']) & np.isfinite(costs))
    # Retain the scores of rejected competitors for the ambiguity tests, but
    # never assign them an identity or use them as a track's predecessor.
    all_costs = np.where(feasible, costs, np.inf)
    assignable = (feasible & pv['accepted'][:, None] & cv['accepted'][None, :]
                  & (costs <= config['max_assignment_cost']))
    big = 1e9
    matrix = np.full((m + n, m + n), big, dtype=float)
    matrix[:m, :n] = np.where(assignable, costs, big)
    matrix[np.arange(m), n + np.arange(m)] = config['unmatched_cost']
    matrix[m + np.arange(n), np.arange(n)] = config['unmatched_cost']
    matrix[m:, n:] = 0
    rows, columns = linear_sum_assignment(matrix)
    incoming[~cv['accepted']] = 'rejected'
    outgoing[~pv['accepted']] = 'rejected'
    for i, j in zip(rows, columns):
        if i >= m or j >= n or not assignable[i, j]:
            continue
        row = all_costs[i].copy()
        column = all_costs[:, j].copy()
        row[j] = column[i] = np.inf
        alternative = min(float(row.min()), float(column.min()))
        margin = alternative - float(costs[i, j])
        if margin < config['minimum_alternative_margin']:
            incoming[j] = outgoing[i] = 'ambiguous_cost_margin'
            continue
        links[int(j)] = int(i)
        incoming[j] = outgoing[i] = 'linked'
        details[int(j)] = {'cost': float(costs[i, j]),
                           'margin': None if not np.isfinite(margin) else float(margin),
                           'body_step': float(distance[i, j]),
                           'prediction_error': float(error[i, j]),
                           'shoulder_used': bool(shoulder_ok[i, j])}
    assert len(set(links.values())) == len(links)
    return links, incoming, outgoing, details


class Tracker:
    def __init__(self, video_id, width, height, config):
        self.video_id, self.width, self.height, self.config = video_id, width, height, config
        self.prefix = f"{config['dataset_id']}/{video_id}/t"
        self.next_id = 1
        self.previous = self.pv = self.previous_ids = self.velocity = None
        self.active = {}

    def name(self, local_id):
        return self.prefix + f'{local_id:09d}'

    def close(self, local_id, reason):
        info = self.active.pop(int(local_id))
        return {'tracklet_id': self.name(int(local_id)), 'tracklet_local_id': int(local_id),
                'video_id': self.video_id, 'start_frame': info[0], 'end_frame': info[1],
                'length': info[2], 'start_reason': info[3], 'termination_reason': reason}

    def process(self, current):
        cv = validity(current, self.width, self.height, self.config)
        incoming = np.full(len(current), 'video_start', dtype=object)
        links, details, ended = {}, {}, []
        if self.previous is not None:
            if current.index <= self.previous.index:
                raise ValueError('Frames must be strictly chronological')
            if current.index != self.previous.index + 1 or not current.observed or not self.previous.observed:
                incoming[:] = 'missing_span'
                outgoing = np.full(len(self.previous), 'missing_span', dtype=object)
            elif not len(current):
                outgoing = np.full(len(self.previous), 'empty_frame', dtype=object)
            elif not len(self.previous):
                incoming[:] = 'after_empty_frame'
                outgoing = np.empty(0, dtype=object)
            else:
                links, incoming, outgoing, details = association(
                    self.previous, current, self.pv, cv, self.velocity, self.config)
            predecessors = set(links.values())
            for i in np.flatnonzero(self.pv['accepted']):
                if int(i) not in predecessors:
                    ended.append(self.close(self.previous_ids[i], str(outgoing[i])))
        ids = np.full(len(current), -1, dtype=np.int64)
        velocity = np.zeros((len(current), 2), dtype=float)
        for j in np.flatnonzero(cv['accepted']):
            if int(j) in links:
                i = links[int(j)]
                local_id = int(self.previous_ids[i])
                info = self.active[local_id]
                assert info[1] + 1 == current.index
                info[1], info[2] = current.index, info[2] + 1
                a = self.config['velocity_update_weight']
                velocity[j] = (1 - a) * self.velocity[i] + a * (current.body[j] - self.previous.body[i])
                speed = np.linalg.norm(velocity[j])
                if speed > self.config['max_velocity_px_per_frame']:
                    velocity[j] *= self.config['max_velocity_px_per_frame'] / speed
            else:
                local_id = self.next_id
                self.next_id += 1
                self.active[local_id] = [current.index, current.index, 1, str(incoming[j])]
            ids[j] = local_id
        self.previous, self.pv, self.previous_ids, self.velocity = current, cv, ids, velocity
        assert len(np.unique(ids[ids > 0])) == np.count_nonzero(cv['accepted'])
        return {'ids': ids, 'validity': cv, 'incoming_reason': incoming,
                'predecessor_index': np.array([links.get(j, -1) for j in range(len(current))], dtype=np.int64),
                'ended': ended, 'link_details': details}

    def finish(self):
        result = [self.close(i, 'video_end') for i in sorted(self.active)]
        self.previous = self.pv = self.previous_ids = self.velocity = None
        return result
