import unittest
from annolid.tracker.build_BoTSORT import build_BoT_SORT_tracker
from annolid.tracker.bot_sort import BoTSORT


class TestBuildTracker(unittest.TestCase):

    def test_botsort_tracker_type(self):
        tracker = build_BoT_SORT_tracker()
        self.assertTrue(isinstance(tracker, BoTSORT))

    def test_botsort_track_high_thresh(self):
        tracker = build_BoT_SORT_tracker()
        self.assertAlmostEqual(tracker.track_high_thresh, 0.33824964456239337)

    def test_botsort_new_track_thresh(self):
        tracker = build_BoT_SORT_tracker()
        self.assertAlmostEqual(tracker.new_track_thresh, 0.21144301345190655)

    def test_botsort_proximity_thresh(self):
        tracker = build_BoT_SORT_tracker()
        self.assertAlmostEqual(tracker.proximity_thresh, 0.5945380911899254)

    def test_botsort_appearance_thresh(self):
        tracker = build_BoT_SORT_tracker()
        self.assertAlmostEqual(tracker.appearance_thresh, 0.4818211117541298)

    def test_botsort_lambda(self):
        tracker = build_BoT_SORT_tracker()
        self.assertAlmostEqual(tracker.lambda_, 0.9896143462366406)


if __name__ == '__main__':
    unittest.main()
