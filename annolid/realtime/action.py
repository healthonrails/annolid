import zmq
import serial
import time
import logging
import json
from typing import Optional, Dict, Callable

from annolid.utils.logger import logger


class ActionProcess:
    """
    A standalone process for controlling hardware (e.g., Bpod) via serial communication.

    This module listens for JSON commands on a ZeroMQ PULL socket and translates them
    into specific byte sequences sent over a serial port. It is designed to be simple,
    robust, and completely decoupled from the experimental logic.
    """

    def __init__(self, command_address: str, serial_port: str, baud_rate: int = 115200):
        """
        Initializes the ActionProcess and attempts to connect to the serial device.

        Args:
            command_address (str): The ZMQ address to bind and listen for commands on (e.g., "tcp://*:5556").
            serial_port (str): The name of the serial port (e.g., '/dev/tty.usbmodem123' or 'COM3').
            baud_rate (int): The baud rate for the serial connection.
        """
        self.command_address = command_address
        self.serial_port = serial_port
        self.baud_rate = baud_rate

        self.context: Optional[zmq.Context] = None
        self.command_receiver: Optional[zmq.Socket] = None
        self.hardware_device: Optional[serial.Serial] = None

        self.running = True

        # --- Command Mapping for Extensibility ---
        # Maps command strings to handler methods. Easy to add new actions.
        self.command_map: Dict[str, Callable] = {
            "trigger_reward_valve": self._handle_trigger_reward
        }

    def _setup(self):
        """Initializes ZMQ socket and connects to the serial device."""
        logger.info("Setting up ActionProcess...")

        # 1. Set up ZMQ PULL socket to receive commands
        self.context = zmq.Context()
        self.command_receiver = self.context.socket(zmq.PULL)
        self.command_receiver.bind(self.command_address)
        logger.info(f"ZMQ command receiver bound to {self.command_address}")

        # 2. Connect to the serial hardware device
        try:
            self.hardware_device = serial.Serial(
                port=self.serial_port, baudrate=self.baud_rate, timeout=1
            )
            # Wait a moment for the serial port to initialize
            time.sleep(2)
            logger.info(f"Successfully connected to hardware on {self.serial_port}")
        except serial.SerialException as e:
            logger.critical(
                f"FATAL: Could not connect to hardware on '{self.serial_port}'. "
                f"Please check the port and permissions. Error: {e}"
            )
            # Setting this to None will prevent the run loop from starting.
            self.hardware_device = None

    def run(self):
        """Main execution loop. Waits for commands and executes them."""
        self._setup()

        if not self.hardware_device:
            logger.error(
                "Cannot start run loop because hardware is not connected. Shutting down."
            )
            self._cleanup()
            return

        logger.info("Action process started. Waiting for commands...")
        while self.running:
            try:
                # Blocking wait for a command from the Decision module.
                # This is efficient as the process does nothing else.
                command_msg = self.command_receiver.recv_json()

                action_name = command_msg.get("action")
                handler = self.command_map.get(action_name)

                if handler:
                    handler(command_msg)
                else:
                    logger.warning(f"Received unknown command: '{action_name}'")

            except json.JSONDecodeError:
                logger.warning("Received invalid (non-JSON) message.")
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received. Shutting down...")
                self.running = False
            except Exception as e:
                logger.error(
                    f"An error occurred in the action loop: {e}", exc_info=True
                )

        self._cleanup()

    # --- Command Handler Methods ---

    def _handle_trigger_reward(self, command: dict):
        """
        Handles the 'trigger_reward_valve' command.

        This translates the abstract command into a specific byte that the Bpod
        state machine is programmed to understand.
        """
        # Example: The Bpod protocol is set up so that receiving the byte 'A' (ASCII 65)
        # triggers the state transition that opens the reward valve.
        reward_byte = b"A"
        logger.info(
            f"Sending command byte '{reward_byte.decode()}' to hardware for action: {command['action']}"
        )
        self.hardware_device.write(reward_byte)

    def _cleanup(self):
        """Closes all open resources."""
        logger.info("Cleaning up action process resources...")
        if self.hardware_device and self.hardware_device.is_open:
            self.hardware_device.close()
        if self.command_receiver and not self.command_receiver.closed:
            self.command_receiver.close()
        if self.context and not self.context.closed:
            self.context.term()
        logger.info("Cleanup complete.")


def main():
    """Example of how to launch the ActionProcess."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    )

    # This address must match the one the DecisionProcess pushes to.
    COMMAND_ADDRESS = "tcp://*:5556"

    # This must be the correct serial port for the Bpod device.
    # On macOS: /dev/tty.usbmodem...
    # On Linux: /dev/ttyACM0
    # On Windows: COM3
    SERIAL_PORT = "/dev/tty.usbmodemX00000000"

    action_process = ActionProcess(COMMAND_ADDRESS, SERIAL_PORT)

    try:
        action_process.run()
    except KeyboardInterrupt:
        logger.info("Main function caught KeyboardInterrupt.")
    finally:
        action_process.running = False


if __name__ == "__main__":
    main()
