import unittest
from annolid.utils.config import merge_configs, get_config


class TestConfigs(unittest.TestCase):
    def setUp(self):
        self.deep_sort_config = get_config("./annolid/configs/deep_sort.yaml")
        self.custom_dataset_config = get_config("./annolid/configs/custom_dataset.yaml")
        self.keypoints_config = get_config("./annolid/configs/keypoints.yaml")

    def test_load_dataset_config(self):
        self.assertIsNotNone(self.deep_sort_config) and self.assertIn(
            "DEEPSORT", self.deep_sort_config
        )

    def test_load_deep_sort_config(self):
        self.assertIsNotNone(self.custom_dataset_config) and self.assertIn(
            "DATASET", self.custom_dataset_config
        )

    def test_merge_configs(self):
        final_config = merge_configs(
            [self.deep_sort_config, self.custom_dataset_config]
        )
        self.assertIn("DEEPSORT", final_config) and self.assertIn(
            "DATASET", final_config
        )

    def test_load_keypoints_config(self):
        self.assertIsNotNone(self.keypoints_config) and self.assertIn(
            "HEAD", self.keypoints_config
        ) and self.assertIn("EVENTS", self.keypoints_config) and self.assertIn(
            "NAME", self.keypoints_config
        ) and self.assertIn("ZONES", self.keypoints_config)


if __name__ == "__main__":
    unittest.main()
