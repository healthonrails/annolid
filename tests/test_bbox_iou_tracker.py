"""
Tests for D2BBoxIOUTracker.

These tests do NOT require detectron2 to be installed.
The ``instantiate`` call is mocked so the logic can be verified in isolation.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest.mock import MagicMock


def _install_d2_stub():
    """
    Install a minimal detectron2 stub into sys.modules so that the tracker
    module can be imported without a real detectron2 installation.
    """
    d2 = types.ModuleType("detectron2")
    d2_config = types.ModuleType("detectron2.config")
    d2_config.CfgNode = MagicMock()
    d2_config.instantiate = MagicMock()
    d2_tracking = types.ModuleType("detectron2.tracking")
    d2_tracking_base = types.ModuleType("detectron2.tracking.base_tracker")
    d2_tracking_base.build_tracker_head = MagicMock()

    sys.modules.setdefault("detectron2", d2)
    sys.modules.setdefault("detectron2.config", d2_config)
    sys.modules.setdefault("detectron2.tracking", d2_tracking)
    sys.modules.setdefault("detectron2.tracking.base_tracker", d2_tracking_base)
    return d2_config


class TestD2BBoxIOUTrackerImport(unittest.TestCase):
    """Verify the module imports cleanly even without detectron2 installed."""

    def test_module_importable_without_d2(self):
        """The module should be importable even when detectron2 is absent."""
        # Temporarily hide detectron2
        saved = {k: v for k, v in sys.modules.items() if k.startswith("detectron2")}
        for k in list(saved):
            del sys.modules[k]

        tracker_mod_name = "annolid.tracker.bbox_iou_tracker"
        if tracker_mod_name in sys.modules:
            del sys.modules[tracker_mod_name]

        try:
            mod = importlib.import_module(tracker_mod_name)
            self.assertFalse(mod._D2_AVAILABLE)
        finally:
            # Restore
            sys.modules.update(saved)
            if tracker_mod_name in sys.modules:
                del sys.modules[tracker_mod_name]


class TestD2BBoxIOUTrackerWithStub(unittest.TestCase):
    """Unit tests for D2BBoxIOUTracker using a mocked detectron2."""

    def setUp(self):
        self.d2_config = _install_d2_stub()
        # Force re-import so the stub is picked up
        mod_name = "annolid.tracker.bbox_iou_tracker"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        import importlib as _il

        self.mod = _il.import_module(mod_name)
        self.mod._D2_AVAILABLE = True  # pretend D2 is there

    def _make_tracker(self, **kwargs):
        return self.mod.D2BBoxIOUTracker(height=480, width=640, **kwargs)

    def test_img_size_stored(self):
        t = self._make_tracker()
        self.assertEqual(t._img_size, (480, 640))

    def test_default_hyperparams(self):
        t = self._make_tracker()
        self.assertEqual(t._max_num_instances, 10)
        self.assertEqual(t._max_lost_frame_count, 3)
        self.assertAlmostEqual(t._min_box_rel_dim, 0.02)
        self.assertEqual(t._min_instance_period, 1)
        self.assertAlmostEqual(t._track_iou_threshold, 0.5)

    def test_get_tracker_calls_instantiate_with_correct_keys(self):
        t = self._make_tracker(
            max_num_instances=5,
            max_lost_frame_count=2,
            track_iou_threshold=0.4,
        )
        # Patch the instantiate name inside the already-imported tracker module
        mock_instantiate = MagicMock()
        with unittest.mock.patch.object(self.mod, "instantiate", mock_instantiate):
            t.get_tracker()
        call_args = mock_instantiate.call_args[0][0]
        self.assertIn("_target_", call_args)
        self.assertIn("IOUWeightedHungarianBBoxIOUTracker", call_args["_target_"])
        self.assertEqual(call_args["video_height"], 480)
        self.assertEqual(call_args["video_width"], 640)
        self.assertEqual(call_args["max_num_instances"], 5)
        self.assertEqual(call_args["max_lost_frame_count"], 2)
        self.assertAlmostEqual(call_args["track_iou_threshold"], 0.4)

    def test_tracker_from_config_emits_deprecation_warning(self):
        t = self._make_tracker()
        with self.assertWarns(DeprecationWarning):
            t.tracker_from_config()

    def test_raises_importerror_when_d2_unavailable(self):
        self.mod._D2_AVAILABLE = False
        with self.assertRaises(ImportError):
            self.mod.D2BBoxIOUTracker(height=480, width=640)
        self.mod._D2_AVAILABLE = True  # restore for other tests


if __name__ == "__main__":
    unittest.main()
