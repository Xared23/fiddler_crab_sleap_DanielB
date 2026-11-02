"""Experimental variable-count association with centroid hysteresis and short gaps.

No interpolation: active memory is not an observation. A last observation at f
can link at f+6 (five intervening misses), but not f+7. Uses unchanged body
validity gates; centroid confidence is a separate detection-stage score.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from body_first_tracklets import validity


@dataclass
class State:
    first: int
    last: int
    count: int
    body: np.ndarray
    shoulder_vector: np.ndarray
    shoulder_ok: bool
    velocity: np.ndarray


class GraceTracker:
    def __init__(self, config, birth_threshold, continuation_threshold=.05,
                 grace_frames=5, width=7680, height=4320):
        if not 0 <= continuation_threshold <= birth_threshold <= 1:
            raise ValueError('Invalid confidence thresholds')
        if not isinstance(grace_frames, int) or grace_frames < 0:
            raise ValueError('grace_frames must be a nonnegative integer')
        self.config = dict(config)
        self.birth_threshold = birth_threshold
        self.continuation_threshold = continuation_threshold
        self.grace_frames = grace_frames
        self.width, self.height = width, height
        self.active = {}
        self.next_id = 1
        self.last_frame = None

    def close(self, tid, reason):
        s = self.active.pop(tid)
        return dict(track_id=tid, start_frame=s.first, end_frame=s.last,
                    observations=s.count, span_frames=s.last-s.first+1,
                    missing_frames=s.last-s.first+1-s.count, termination=reason)

    def process(self, frame, centroid_scores):
        if self.last_frame is not None and frame.index != self.last_frame + 1:
            raise ValueError('Supply every frame in order, including empty frames')
        scores = np.asarray(centroid_scores, dtype=float)
        if scores.shape != (len(frame),):
            raise ValueError('One centroid score required per pose')
        self.last_frame = frame.index
        ended = [self.close(tid, 'grace_expired') for tid, s in list(self.active.items())
                 if frame.index-s.last-1 > self.grace_frames]
        cv = validity(frame, self.width, self.height, self.config)
        eligible = (cv['accepted'] & np.isfinite(scores)
                    & (scores >= self.continuation_threshold) & frame.observed)
        ids = np.full(len(frame), -1, dtype=np.int64)
        reason = np.where(eligible, 'below_birth_threshold', 'body_or_centroid_rejected').astype(object)
        previous_frame = np.full(len(frame), -1, dtype=np.int64)
        link_cost = np.full(len(frame), np.nan)
        links = {}
        tids = sorted(self.active)
        m, n = len(tids), len(frame)
        if m and n:
            cfg = self.config
            states = [self.active[tid] for tid in tids]
            dt = np.array([frame.index-s.last for s in states], dtype=float)
            body = np.array([s.body for s in states])
            velocity = np.array([s.velocity for s in states])
            delta = frame.body[None, :, :] - body[:, None, :]
            distance = np.linalg.norm(delta, axis=2)
            error = np.linalg.norm(delta-velocity[:, None, :]*dt[:, None, None], axis=2)
            costs = (1-cfg['motion_weight'])*distance/dt[:, None] + cfg['motion_weight']*error
            shoulder_ok = (np.array([s.shoulder_ok for s in states])[:, None]
                           & (cv['shoulder_valid'] & cv['geometry_valid'])[None, :])
            vector_error = np.linalg.norm((frame.shoulder-frame.body)[None, :, :]
                - np.array([s.shoulder_vector for s in states])[:, None, :], axis=2)
            costs += np.where(shoulder_ok, cfg['shoulder_vector_weight'] * np.minimum(
                np.nan_to_num(vector_error, nan=0), cfg['shoulder_vector_cost_cap_px']), 0)
            feasible = (cv['inside'][None, :] & np.isfinite(costs)
                        & (distance <= cfg['max_body_displacement_px']*dt[:, None])
                        & (error <= cfg['max_prediction_error_px']))
            # Rejected current bodies still compete; a weak score cannot erase ambiguity.
            all_costs = np.where(feasible, costs, np.inf)
            assignable = feasible & eligible[None, :] & (costs <= cfg['max_assignment_cost'])
            matrix = np.full((m+n, m+n), 1e9)
            matrix[:m, :n] = np.where(assignable, costs, 1e9)
            matrix[np.arange(m), n+np.arange(m)] = cfg['unmatched_cost']
            matrix[m+np.arange(n), np.arange(n)] = cfg['unmatched_cost']
            matrix[m:, n:] = 0
            rows, cols = linear_sum_assignment(matrix)
            # Do not spawn a shadow ID for a high-score detection competing for an
            # existing ID. Keep it unassigned until an unambiguous link or expiry.
            reason[eligible & assignable.any(axis=0)] = 'unresolved_existing_track'
            for i, j in zip(rows, cols):
                if i >= m or j >= n or not assignable[i, j]:
                    continue
                row, col = all_costs[i].copy(), all_costs[:, j].copy()
                row[j] = col[i] = np.inf
                margin = min(row.min(), col.min())-costs[i, j]
                if margin < cfg['minimum_alternative_margin']:
                    reason[j] = 'ambiguous_cost_margin'
                    continue
                links[int(j)] = tids[i]
                link_cost[j] = costs[i, j]
        for j in np.flatnonzero(eligible):
            shoulder_ok = bool(cv['shoulder_valid'][j] and cv['geometry_valid'][j])
            if int(j) in links:
                tid = links[int(j)]
                s = self.active[tid]
                previous_frame[j] = s.last
                dt = frame.index-s.last
                a = self.config['velocity_update_weight']
                velocity = (1-a)*s.velocity + a*(frame.body[j]-s.body)/dt
                speed = np.linalg.norm(velocity)
                if speed > self.config['max_velocity_px_per_frame']:
                    velocity *= self.config['max_velocity_px_per_frame']/speed
                s.last, s.count = frame.index, s.count+1
                s.body, s.shoulder_vector = frame.body[j].copy(), (frame.shoulder[j]-frame.body[j]).copy()
                s.shoulder_ok, s.velocity = shoulder_ok, velocity
                reason[j] = 'gap_link' if dt > 1 else 'adjacent_link'
            elif scores[j] >= self.birth_threshold and reason[j] == 'below_birth_threshold':
                tid = self.next_id
                self.next_id += 1
                self.active[tid] = State(frame.index, frame.index, 1, frame.body[j].copy(),
                    (frame.shoulder[j]-frame.body[j]).copy(), shoulder_ok, np.zeros(2))
                reason[j] = 'birth'
            else:
                continue
            ids[j] = tid
        # High-threshold detections remain visible even when no ID is safe.
        # Sub-threshold detections are admitted only as successful continuations.
        keep = (np.isfinite(scores) & (scores >= self.birth_threshold)) | (ids > 0)
        assert len(set(ids[ids > 0])) == int((ids > 0).sum())
        assert all(tid in self.active for tid in ids[ids > 0])
        return dict(ids=ids, keep=keep, reason=reason, previous_frame=previous_frame,
                    link_cost=link_cost, validity=cv, ended=ended,
                    pending_tracks=sum(s.last < frame.index for s in self.active.values()))

    def finish(self):
        return [self.close(tid, 'selection_end') for tid in sorted(self.active)]
