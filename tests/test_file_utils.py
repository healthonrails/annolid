import unittest
import os
from datetime import datetime, timedelta
from unittest.mock import patch
from annolid.utils.files import find_most_recent_file


class TestFindMostRecentFile(unittest.TestCase):
    def setUp(self):
        # Create a temporary folder and some files for testing
        self.temp_folder = "temp_folder"
        os.makedirs(self.temp_folder)

        # Create files with different modification times
        self.file_paths = []
        for i in range(1, 4):
            file_path = os.path.join(self.temp_folder, f"test_file_{i}.txt")
            with open(file_path, "w") as f:
                f.write(f"Test content {i}")

            # Modify file times to create different timestamps
            modification_time = datetime.now() - timedelta(days=i)
            os.utime(
                file_path,
                (modification_time.timestamp(), modification_time.timestamp()),
            )

            self.file_paths.append(file_path)

    def tearDown(self):
        # Remove the temporary folder and files after testing
        for file_path in self.file_paths:
            os.remove(file_path)
        os.rmdir(self.temp_folder)

    def test_find_most_recent_file(self):
        # Mock the current time for predictable results
        mock_current_time = datetime.now()
        with patch("annolid.utils.files") as mock_datetime:
            mock_datetime.now.return_value = mock_current_time

            # Call the function with the temporary folder
            result = find_most_recent_file(self.temp_folder, file_ext=".txt")

        # Expect the most recent file to be the first in the list (modified 1 day ago)
        expected_result = os.path.join(self.temp_folder, "test_file_1.txt")
        self.assertEqual(result, expected_result)

    def test_find_most_recent_file_no_files(self):
        # Call the function with an empty folder
        empty_folder = "empty_folder"
        os.makedirs(empty_folder)
        result = find_most_recent_file(empty_folder)

        # Expect the result to be None since there are no files
        self.assertIsNone(result)
        os.rmdir(empty_folder)


if __name__ == "__main__":
    unittest.main()
