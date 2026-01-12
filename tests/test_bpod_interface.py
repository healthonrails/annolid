import unittest
from unittest.mock import patch, MagicMock
from annolid.hardware.bpod.bpod_interface import BpodController
import logging


class TestBpodController(unittest.TestCase):
    """Unit tests for the BpodController class."""

    def setUp(self):
        """Set up the test environment by mocking serial connection."""
        self.mock_serial = MagicMock()
        self.bpod_controller = BpodController(port="/dev/ttyUSB0")

        # Patch the serial.Serial to simulate port opening behavior
        patch('annolid.hardware.bpod.bpod_interface.serial.Serial',
              self.mock_serial).start()

    def test_connect_invalid_port(self):
        """Test handling of an invalid port."""
        # Simulate a failure in opening the port
        self.mock_serial.side_effect = OSError(
            "Could not open port /dev/ttyUSB0")

        with self.assertRaises(OSError):
            self.bpod_controller.connect()

        # Ensure the error was logged
        self.mock_serial.assert_called_with("/dev/ttyUSB0")

    def test_send_event_when_not_connected(self):
        """Test sending an event when the connection is not open."""
        # Simulate a closed serial connection
        self.bpod_controller.serial_connection.is_open = False
        self.bpod_controller.send_event(1)

        # Ensure that no event is written when not connected
        self.mock_serial.write.assert_not_called()

    def tearDown(self):
        """Clean up any resources after each test."""
        patch.stopall()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
