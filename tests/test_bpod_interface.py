import unittest
from unittest.mock import patch, MagicMock
from annolid.hardware.bpod.bpod_interface import BpodController
import logging


class TestBpodController(unittest.TestCase):
    """Unit tests for the BpodController class."""

    def setUp(self):
        """Set up the test environment."""
        self.bpod_controller = BpodController(port="/dev/ttyUSB0")

    def test_connect_invalid_port(self):
        """Test handling of an invalid port."""
        with patch(
            "annolid.hardware.bpod.bpod_interface.serial.Serial"
        ) as mock_serial_class:
            # Setup the mock instance returned by the class
            mock_serial_instance = MagicMock()
            mock_serial_instance.side_effect = OSError(
                "Could not open port /dev/ttyUSB0"
            )
            mock_serial_class.return_value = mock_serial_instance
            mock_serial_class.side_effect = OSError("Could not open port /dev/ttyUSB0")

            with self.assertRaises(OSError):
                self.bpod_controller.connect()

            # Ensure the error was logged
            mock_serial_class.assert_called_with("/dev/ttyUSB0")

    def test_send_event_when_not_connected(self):
        """Test sending an event when the connection is not open."""
        with patch(
            "annolid.hardware.bpod.bpod_interface.serial.Serial"
        ) as mock_serial_class:
            # Simulate a closed serial connection
            self.bpod_controller.serial_connection.is_open = False
            self.bpod_controller.send_event(1)

            # Ensure that no event is written when not connected
            mock_serial_class.return_value.write.assert_not_called()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
