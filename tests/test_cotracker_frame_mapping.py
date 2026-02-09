import unittest

import pytest
import torch

pytest.importorskip("shapely")

from annolid.tracker.cotracker.track import CoTrackerProcessor


class TestCoTrackerFrameMapping(unittest.TestCase):
    def _processor(self, start_frame=0, end_frame=-1):
        processor = CoTrackerProcessor.__new__(CoTrackerProcessor)
        processor.start_frame = start_frame
        processor.end_frame = end_frame
        saved_frames = []

        def _save(frame_number, _points, _description=""):
            saved_frames.append(frame_number)

        processor.save_frame_json = _save
        return processor, saved_frames

    def test_extract_frame_points_uses_chunk_start_frame(self):
        processor, saved = self._processor()
        tracks = torch.zeros((1, 3, 2, 2), dtype=torch.int64)
        visibility = torch.ones((1, 3, 2), dtype=torch.bool)

        processor.extract_frame_points(tracks, visibility, chunk_start_frame=10)
        self.assertEqual(saved, [10, 11, 12])

    def test_extract_frame_points_respects_local_indices(self):
        processor, saved = self._processor()
        tracks = torch.zeros((1, 3, 1, 2), dtype=torch.int64)

        processor.extract_frame_points(
            tracks, chunk_start_frame=5, local_frame_indices=[1, 2]
        )
        self.assertEqual(saved, [6, 7])

    def test_extract_frame_points_respects_start_end_frame(self):
        processor, saved = self._processor(start_frame=11, end_frame=11)
        tracks = torch.zeros((1, 3, 1, 2), dtype=torch.int64)

        processor.extract_frame_points(tracks, chunk_start_frame=10)
        self.assertEqual(saved, [11])


if __name__ == "__main__":
    unittest.main()
