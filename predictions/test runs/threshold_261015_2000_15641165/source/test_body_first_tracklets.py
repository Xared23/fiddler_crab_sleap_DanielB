import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from body_first_tracklets import Frame, Tracker

CONFIG = yaml.safe_load((ROOT / 'configs/tracking/body_first_v001.yaml').read_text())


def frame(index, xs, shoulder=True, scores=None):
    body = np.array([[x, 100.] for x in xs], dtype=float).reshape(-1, 2)
    sh = body + [0, 20] if shoulder else np.full_like(body, np.nan)
    n = len(body)
    return Frame(index, body, sh, np.array(scores if scores is not None else [0.9] * n),
                 np.full(n, 0.3 if shoulder else 0.), np.ones(n, bool), np.full(n, shoulder))


class BodyFirstTests(unittest.TestCase):
    def tracker(self, video='test'):
        return Tracker(video, 1000, 1000, copy.deepcopy(CONFIG))

    def test_missing_shoulder_preserves_body_identity(self):
        t = self.tracker()
        first = t.process(frame(0, [30], False))['ids'][0]
        for k in range(1, 10):
            result = t.process(frame(k, [30 + .1 * k], k % 2 == 0))
            self.assertEqual(result['ids'][0], first)
        self.assertEqual(t.finish()[0]['length'], 10)

    def test_shuffled_detection_order(self):
        t = self.tracker()
        original = t.process(frame(0, [30, 70]))['ids']
        current = t.process(frame(1, [71, 31]))['ids']
        np.testing.assert_array_equal(current, original[::-1])

    def test_nearby_but_clearly_distinguishable_neighbors(self):
        t = self.tracker()
        a = t.process(frame(0, [30, 48]))['ids']
        b = t.process(frame(1, [31, 49]))['ids']
        np.testing.assert_array_equal(a, b)

    def test_equal_alternatives_are_not_forced(self):
        t = self.tracker()
        a = t.process(frame(0, [30, 42]))['ids']
        b = t.process(frame(1, [36]))
        self.assertNotIn(b['ids'][0], a)
        self.assertEqual(b['incoming_reason'][0], 'ambiguous_cost_margin')

    def test_rejected_competitor_still_counts(self):
        t = self.tracker()
        a = t.process(frame(0, [30, 42], scores=[.9, .1]))['ids'][0]
        self.assertNotEqual(t.process(frame(1, [36]))['ids'][0], a)

    def test_far_candidate_is_not_forced(self):
        t = self.tracker()
        a = t.process(frame(0, [30]))['ids'][0]
        self.assertNotEqual(t.process(frame(1, [80]))['ids'][0], a)

    def test_empty_frame_ends_identity(self):
        t = self.tracker()
        a = t.process(frame(0, [30]))['ids'][0]
        t.process(frame(1, []))
        self.assertNotEqual(t.process(frame(2, [30]))['ids'][0], a)

    def test_missing_frame_ends_identity(self):
        t = self.tracker()
        a = t.process(frame(39, [30]))['ids'][0]
        self.assertNotEqual(t.process(frame(41, [30]))['ids'][0], a)

    def test_chunk_boundary_is_not_a_gap(self):
        t = self.tracker()
        a = t.process(frame(39, [30]))['ids'][0]
        self.assertEqual(t.process(frame(40, [30]))['ids'][0], a)

    def test_low_confidence_body_not_assigned(self):
        result = self.tracker().process(frame(0, [30], scores=[.1]))
        self.assertEqual(result['ids'][0], -1)

    def test_invalid_body_not_assigned(self):
        result = self.tracker().process(frame(0, [np.nan, -10, 1001]))
        np.testing.assert_array_equal(result['ids'], [-1, -1, -1])

    def test_duplicates_are_not_arbitrarily_resolved(self):
        result = self.tracker().process(frame(0, [30, 33]))
        np.testing.assert_array_equal(result['ids'], [-1, -1])

    def test_unusable_shoulder_geometry_is_ignored(self):
        t = self.tracker()
        a = t.process(frame(0, [30]))['ids'][0]
        f = frame(1, [31])
        f.shoulder += [400, 400]
        self.assertEqual(t.process(f)['ids'][0], a)

    def test_motion_cannot_exceed_cap(self):
        t = self.tracker()
        ids = [t.process(frame(k, [30 + 10 * k]))['ids'][0] for k in range(12)]
        self.assertEqual(len(set(ids)), 1)
        self.assertLessEqual(float(np.linalg.norm(t.velocity[0])), CONFIG['max_velocity_px_per_frame'] + 1e-10)

    def test_raw_measurements_unchanged(self):
        t = self.tracker()
        f = frame(0, [30, 70], False)
        before = f.body.copy(), f.shoulder.copy()
        t.process(f)
        np.testing.assert_array_equal(f.body, before[0])
        np.testing.assert_array_equal(f.shoulder, before[1])

    def test_deterministic_ids(self):
        a, b = self.tracker(), self.tracker()
        for k in range(6):
            np.testing.assert_array_equal(a.process(frame(k, [30 + k, 70 - k]))['ids'],
                                          b.process(frame(k, [30 + k, 70 - k]))['ids'])

    def test_video_scoped_ids(self):
        self.assertNotEqual(self.tracker('a').name(1), self.tracker('b').name(1))

    def test_repeated_frame_rejected(self):
        t = self.tracker()
        t.process(frame(0, [30]))
        with self.assertRaises(ValueError):
            t.process(frame(0, [30]))

    def test_one_to_one_and_birth(self):
        t = self.tracker()
        a = t.process(frame(0, [30]))['ids'][0]
        result = t.process(frame(1, [31, 90]))
        self.assertEqual(result['ids'][0], a)
        self.assertNotEqual(result['ids'][1], a)
        self.assertEqual(result['predecessor_index'].tolist(), [0, -1])

    def test_closed_track_does_not_reappear(self):
        t = self.tracker()
        t.process(frame(0, [30]))
        old = t.process(frame(1, [31]))['ids'][0]
        t.process(frame(2, []))
        new = t.process(frame(3, [32]))['ids'][0]
        self.assertNotEqual(old, new)
        np.testing.assert_array_equal(t.velocity, [[0., 0.]])


if __name__ == '__main__':
    unittest.main()
