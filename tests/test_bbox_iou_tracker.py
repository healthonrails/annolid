"""
Tests for the standalone BBoxIOUTracker.

No detectron2 required — uses SimpleInstances directly.
"""

from __future__ import annotations

import torch
import unittest

from annolid.tracker.simple_instances import SimpleInstances
from annolid.tracker.bbox_iou_tracker import BBoxIOUTracker


def _make_instances(boxes, scores=None, classes=None, image_size=(480, 640)):
    """Helper to create SimpleInstances for testing."""
    inst = SimpleInstances(image_size=image_size)
    inst.pred_boxes = torch.FloatTensor(boxes)
    n = len(boxes)
    inst.scores = torch.FloatTensor(scores or [0.9] * n)
    inst.pred_classes = torch.IntTensor(classes or [0] * n)
    return inst


class TestBBoxIOUTrackerImport(unittest.TestCase):
    def test_module_importable(self):
        """The tracker module should import cleanly (no D2)."""
        import annolid.tracker.bbox_iou_tracker as mod

        self.assertTrue(hasattr(mod, "BBoxIOUTracker"))


class TestBBoxIOUTracker(unittest.TestCase):
    def test_constructor_params(self):
        t = BBoxIOUTracker(video_height=480, video_width=640)
        self.assertEqual(t._video_height, 480)
        self.assertEqual(t._video_width, 640)
        self.assertEqual(t._max_num_instances, 200)
        self.assertEqual(t._max_lost_frame_count, 0)
        self.assertAlmostEqual(t._min_box_rel_dim, 0.02)
        self.assertEqual(t._min_instance_period, 1)
        self.assertAlmostEqual(t._track_iou_threshold, 0.5)

    def test_first_frame_assigns_sequential_ids(self):
        t = BBoxIOUTracker(video_height=100, video_width=100)
        inst = _make_instances(
            [[10, 10, 50, 50], [60, 60, 90, 90]],
            image_size=(100, 100),
        )
        result = t.update(inst)
        self.assertEqual(result.ID, [0, 1])
        self.assertEqual(result.ID_period, [1, 1])

    def test_second_frame_same_boxes_keeps_ids(self):
        t = BBoxIOUTracker(video_height=100, video_width=100, track_iou_threshold=0.3)
        frame1 = _make_instances(
            [[10, 10, 50, 50], [60, 60, 90, 90]],
            image_size=(100, 100),
        )
        t.update(frame1)

        # Same boxes in frame 2 — should get same IDs
        frame2 = _make_instances(
            [[10, 10, 50, 50], [60, 60, 90, 90]],
            image_size=(100, 100),
        )
        result = t.update(frame2)
        self.assertEqual(result.ID[0], 0)
        self.assertEqual(result.ID[1], 1)
        self.assertEqual(result.ID_period[0], 2)

    def test_new_detection_gets_new_id(self):
        t = BBoxIOUTracker(video_height=100, video_width=100, track_iou_threshold=0.3)
        frame1 = _make_instances(
            [[10, 10, 50, 50]],
            image_size=(100, 100),
        )
        t.update(frame1)

        # Completely different box in frame 2
        frame2 = _make_instances(
            [[70, 70, 95, 95]],
            image_size=(100, 100),
        )
        result = t.update(frame2)
        # Should have a new ID (not 0)
        self.assertNotEqual(result.ID[0], 0)

    def test_empty_frame(self):
        t = BBoxIOUTracker(video_height=100, video_width=100)
        inst = _make_instances([], image_size=(100, 100))
        result = t.update(inst)
        self.assertEqual(len(result), 0)


class TestSimpleInstances(unittest.TestCase):
    def test_has_and_set(self):
        inst = SimpleInstances(image_size=(100, 100))
        self.assertFalse(inst.has("ID"))
        inst.set("ID", [1, 2])
        self.assertTrue(inst.has("ID"))
        self.assertEqual(inst.ID, [1, 2])

    def test_len(self):
        inst = SimpleInstances(image_size=(100, 100))
        inst.pred_boxes = torch.FloatTensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        self.assertEqual(len(inst), 2)

    def test_to_device(self):
        inst = SimpleInstances(image_size=(100, 100))
        inst.pred_boxes = torch.FloatTensor([[1, 2, 3, 4]])
        inst.ID = [0]
        moved = inst.to("cpu")
        self.assertEqual(moved.pred_boxes.device.type, "cpu")
        self.assertEqual(moved.ID, [0])

    def test_cat(self):
        a = SimpleInstances(image_size=(100, 100))
        a.pred_boxes = torch.FloatTensor([[1, 2, 3, 4]])
        a.ID = [0]

        b = SimpleInstances(image_size=(100, 100))
        b.pred_boxes = torch.FloatTensor([[5, 6, 7, 8]])
        b.ID = [1]

        merged = SimpleInstances.cat([a, b])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged.ID, [0, 1])


if __name__ == "__main__":
    unittest.main()
