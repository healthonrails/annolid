import unittest
import numpy as np
from annolid.utils.shapes import extract_flow_points_in_mask  

class TestExtractFlowPointsInMask(unittest.TestCase):

    def test_empty_mask_returns_empty(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        flow = np.zeros((100, 100, 2), dtype=np.float32)
        result = extract_flow_points_in_mask(mask, flow, num_points=5)
        self.assertEqual(result.shape, (0, 2))

    def test_insufficient_points_returns_available(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2, 2] = 1
        mask[3, 3] = 1
        flow = np.ones((10, 10, 2), dtype=np.float32) * 2.0  # Ensure large enough motion
        result = extract_flow_points_in_mask(mask, flow, num_points=5, min_magnitude=0.1)
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.shape[1], 2)

    def test_exact_number_of_points_returned(self):
        mask = np.ones((20, 20), dtype=np.uint8)
        flow = np.random.rand(20, 20, 2).astype(np.float32) * 3
        result = extract_flow_points_in_mask(mask, flow, num_points=8)
        self.assertEqual(result.shape, (8, 2))

    def test_low_motion_triggers_fallback(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        flow = np.random.rand(10, 10, 2).astype(np.float32) * 0.01  # Almost no motion
        result = extract_flow_points_in_mask(mask, flow, num_points=4, min_magnitude=1.0)
        self.assertEqual(result.shape, (4, 2))

if __name__ == '__main__':
    unittest.main()
