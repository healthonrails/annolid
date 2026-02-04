from __future__ import annotations

import logging
import time
import struct
from types import SimpleNamespace

try:
    import serial  # type: ignore
except ModuleNotFoundError:  # pragma: no cover

    class _MissingPySerial:
        def __init__(self, *_args, **_kwargs):
            raise ModuleNotFoundError(
                "Optional dependency 'pyserial' is required for serial communication. "
                "Install it with `pip install pyserial`."
            )

    serial = SimpleNamespace(Serial=_MissingPySerial, SerialException=Exception)


class BpodStateMachine:
    """
    A Python class for communicating with a Bpod State Machine via its USB serial port.

    This class provides methods to send commands to the Bpod and receive responses,
    allowing control and data acquisition during experiments.

    Example usage:
        bpod = BpodStateMachine("/dev/tty.usbmodem1234567")
        bpod.connect()
        # ... use Bpod commands ...
        bpod.disconnect()
    """

    def __init__(self, port, baudrate=115200, timeout=1):
        """
        Initializes the BpodStateMachine object.

        Args:
            port (str): The serial port of the Bpod State Machine (e.g., "/dev/tty.usbmodem1234567").
            baudrate (int, optional): The baud rate for serial communication. Defaults to 115200.
            timeout (int, optional): The timeout for serial read operations (in seconds). Defaults to 1.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None  # Serial port object
        self.firmware_version = None
        self.machine_type = None
        self.hardware_config = None
        self.module_info = None
        self.timestamp_transmission_scheme = None
        self.n_modules = 0  # Store number of modules

    def connect(self):
        """
        Establishes a serial connection to the Bpod State Machine.
        """
        self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        time.sleep(0.15)  # Wait for discovery byte

    def disconnect(self):
        """
        Closes the serial connection to the Bpod State Machine.
        """
        if self.ser:
            self.send_command("Z")  # Disconnect command
            self.ser.close()
            self.ser = None

    def send_command(self, command, data=None):
        """
        Sends a command byte and optional data to the Bpod.

        Args:
            command (str): The command byte (single character string).
            data (bytes, optional): The data to send after the command byte. Defaults to None.

        Raises:
            Exception: If not connected to the Bpod.
        """
        if not self.ser:
            raise Exception("Not connected to Bpod State Machine.")

        self.ser.write(command.encode())  # Convert command to bytes

        if data:
            self.ser.write(data)

    def read_response(self, size=1):
        """
        Reads a specified number of bytes from the Bpod's serial port.

        Args:
            size (int, optional): The number of bytes to read. Defaults to 1.

        Returns:
            bytes: The received bytes.

        Raises:
            Exception: If not connected to the Bpod.
        """
        if not self.ser:
            raise Exception("Not connected to Bpod State Machine.")
        return self.ser.read(size)

    def handshake(self):
        """
        Performs a handshake with the Bpod to confirm a valid connection.

        Raises:
            Exception: If the handshake fails.
        """
        self.send_command("6")  # Handshake
        response = self.read_response()
        if response and response[0] != 53:  # ASCII '5'
            raise Exception("Handshake failed.")
        # Clear potential extra discovery byte:
        self.ser.reset_input_buffer()

    def get_firmware_version(self):
        """
        Retrieves the firmware version and machine type of the Bpod.

        Returns:
            tuple: A tuple containing the firmware version (int) and machine type (int).
        """
        self.send_command("F")
        firmware_bytes = self.read_response(2)
        machine_type_bytes = self.read_response(2)
        self.firmware_version = struct.unpack("<H", firmware_bytes)[0]
        self.machine_type = struct.unpack("<H", machine_type_bytes)[0]
        return self.firmware_version, self.machine_type

    def reset_session_clock(self):
        """Resets the session clock on the Bpod."""
        self.send_command("*")
        response = self.read_response()
        if response and response[0] != 1:
            raise Exception("Session clock reset failed")  # Add exception

    def get_timestamp_transmission_scheme(self):
        """
        Retrieves the timestamp transmission scheme from the Bpod.

        Returns:
            int: The timestamp transmission scheme (0 for post-trial, 1 for live).
        """
        self.send_command("G")
        scheme_byte = self.read_response()
        self.timestamp_transmission_scheme = scheme_byte[0]
        return self.timestamp_transmission_scheme

    def get_hardware_config(self):
        """
        Retrieves the Bpod's hardware configuration (excluding modules).

        Returns:
            dict: A dictionary containing the hardware configuration parameters.
        """
        self.send_command("H")
        max_states = struct.unpack("<H", self.read_response(2))[0]
        timer_period = struct.unpack("<H", self.read_response(2))[0]
        max_serial_events = self.read_response()[0]
        n_global_timers = self.read_response()[0]
        n_global_counters = self.read_response()[0]
        n_conditions = self.read_response()[0]
        n_inputs = self.read_response()[0]
        input_description_array = self.read_response(n_inputs)
        n_outputs = self.read_response()[0]
        output_description_array = self.read_response(n_outputs)

        self.hardware_config = {
            "MaxStates": max_states,
            "TimerPeriod": timer_period,
            "maxSerialEvents": max_serial_events,
            "nGlobalTimers": n_global_timers,
            "nGlobalCounters": n_global_counters,
            "nConditions": n_conditions,
            "nInputs": n_inputs,
            # Convert to appropriate type (e.g. list or np array)
            "inputDescriptionArray": input_description_array,
            "nOutputs": n_outputs,
            # Convert to appropriate type if needed
            "outputDescriptionArray": output_description_array,
        }

        # Calculate n_modules:  Count instances of 'U' in the outputDescriptionArray
        self.n_modules = output_description_array.count(
            b"U"
        )  # Assuming 'U' is represented as bytes

        return self.hardware_config

    def get_module_info(self):
        """
        Retrieves information about connected modules.

        Returns:
             list: A list of dictionaries, each containing information about a module.
        """
        self.send_command("M")

        module_info = []  # List to store info for each module.
        # Iterate through potential module connection points:
        for _ in range(self.n_modules):  # Iterate up to n_modules
            # Check each for a connected module.
            module_connected = self.read_response()[0]

            if module_connected == 1:
                module_data = {}
                module_data["moduleFirmwareVersion"] = struct.unpack(
                    "<I", self.read_response(4)
                )[0]
                module_name_length = self.read_response()[0]
                module_data["moduleName"] = self.read_response(
                    module_name_length
                ).decode()  # Decode bytes to string
                more_info_follows = self.read_response()[0]

                while more_info_follows == 1:
                    # Info type is a single ASCII character.
                    info_type = self.read_response().decode()
                    if info_type == "#":
                        # Requested number of events.
                        module_data["nEvents"] = self.read_response()[0]
                    elif info_type == "E":
                        n_event_names = self.read_response()[0]
                        module_data["eventNames"] = []
                        for _ in range(n_event_names):
                            event_name_length = self.read_response()[0]
                            # Decode to string.
                            event_name = self.read_response(event_name_length).decode()
                            module_data["eventNames"].append(event_name)
                    more_info_follows = self.read_response()[0]
                module_info.append(module_data)  # append to list
            else:
                # Placeholder for a potentially missing module:
                module_info.append(None)

        self.module_info = module_info  # Store it in the class attribute
        return module_info

    def set_module_event_allocation(self, event_allocation):
        """
        Sets the number of behavior events allocated to each module.

        Args:
            event_allocation (list): A list of integers, where each integer represents
                the number of events allocated to the corresponding module.

        Raises:
            Exception: If the number of allocations does not match the number of modules.
        """
        if len(event_allocation) != self.n_modules:
            raise Exception(
                "Number of event allocations must match the number of modules."
            )

        data = b"%" + bytes(event_allocation)  # % is the command code
        self.send_command("", data=data)

        response = self.read_response()  # Confirmation byte
        if response and response[0] != 1:
            raise Exception("Setting module event allocation failed.")

    def set_input_channel_states(self, channel_states):
        """
        Sets the enabled/disabled state of each input channel.

        Args:
            channel_states (list or bytes): A list/bytes of 0s and 1s, representing disabled or enabled,
            respectively, with each value corresponding to an input channel.
            The length must match the 'nInputs' value from the hardware config.

        Raises:
            Exception: If 'nInputs' is not available, or if the length of `channel_states` is incorrect.
        """

        if self.hardware_config is None or "nInputs" not in self.hardware_config:
            raise Exception(
                "Hardware configuration not retrieved. Call get_hardware_config() first."
            )

        n_inputs = self.hardware_config["nInputs"]
        if len(channel_states) != n_inputs:
            raise Exception(
                f"Length of channel_states ({len(channel_states)}) does not match nInputs ({n_inputs})"
            )

        data = b"E" + bytes(channel_states)  # Command code plus states
        self.send_command("", data=data)
        response = self.read_response()
        if response and response[0] != 1:
            raise Exception("Setting input channel states failed.")

    def set_module_relay(self, module_number, state):
        """
        Enables/Disables relay of incoming bytes from a module to the USB port.

        Args:
            module_number (int): The module number (0-indexed).
            state (int): The state of the module relay (0 = relay off, 1 = relay on).

        Raises:
            ValueError: If module_number or state is invalid.
        """
        if not 0 <= module_number < self.n_modules:
            raise ValueError("Invalid module number.")
        if state not in [0, 1]:
            raise ValueError("Invalid relay state. Must be 0 or 1.")

        data = b"J" + bytes([module_number, state])  # 'J' is the command code
        self.send_command("", data=data)

    def set_state_sync_channel(self, output_channel, sync_mode):
        """
        Sets a state synchronization channel.

        Args:
            output_channel (int): The digital output channel index for synchronization.
            sync_mode (int): The synchronization mode (0 = high on start, low on end; 1 = switches on transition).

        Raises:
            ValueError: If output_channel or sync_mode is invalid.
        """

        # Correct index check
        if (
            not 0
            <= output_channel
            < len(self.hardware_config["outputDescriptionArray"])
        ):
            raise ValueError(
                f"Invalid output channel. Must be between 0 and {len(self.hardware_config['outputDescriptionArray']) - 1}"
            )

        if sync_mode not in [0, 1]:
            raise ValueError("Invalid sync mode. Must be 0 or 1.")

        data = b"K" + bytes([output_channel, sync_mode])
        self.send_command("", data=data)
        response = self.read_response()
        if response and response[0] != 1:
            raise Exception("Setting state sync channel failed.")

    def override_digital_output(self, channel_index, new_state):
        """
        Overrides the state of a digital output line.

        Args:
            channel_index (int): The index of the digital output channel to override.
            new_state (int): The new state of the channel.

        Raises:
            ValueError: If the channel index is out of range.
        """
        if not 0 <= channel_index < len(self.hardware_config["outputDescriptionArray"]):
            raise ValueError("Invalid output channel index.")

        # 'O' is the command code
        data = b"O" + bytes([channel_index, new_state])
        self.send_command("", data=data)

    def read_digital_input(self, channel_index):
        """
        Reads the state of a digital input channel.

        Args:
            channel_index (int): The index of the digital input channel to read.

        Returns:
            int: The state of the channel (0 = low, 1 = high).

        Raises:
            ValueError: if channel_index is out of range.
        """
        if not 0 <= channel_index < len(self.hardware_config["inputDescriptionArray"]):
            raise ValueError("Invalid input channel index.")

        data = b"I" + bytes([channel_index])  # Command code and channel index
        self.send_command("", data=data)

        state = self.read_response()[0]
        return state

    def send_message_to_module(self, module_index, message):
        """
        Transmits a string of bytes to a connected module.

        Args:
            module_index (int):  The index of the target module.
            message (bytes): The message to transmit.

        Raises:
            ValueError: If the module index is invalid.
        """

        if not 0 <= module_index < self.n_modules:
            raise ValueError("Invalid module index.")
        n_bytes = len(message)
        data = (
            b"T" + bytes([module_index, n_bytes]) + message
        )  # 'T' is the command code
        self.send_command("", data=data)

    def store_serial_messages(self, module_index, messages):
        """
        Stores serial messages for a module, retrievable by index.

        Args:
            module_index (int): The module index.
            messages (list): A list of tuples, where each tuple contains:
                (message_index (int), message (bytes))

        Raises:
            ValueError: if message data is incorrect
        """

        if not 0 <= module_index < self.n_modules:
            raise ValueError("Invalid module index.")

        n_messages = len(messages)
        data = b"L" + bytes([module_index, n_messages])

        for message_index, message in messages:
            if not 1 <= message_index <= 255:
                raise ValueError("Message index must be between 1 and 255.")

            message_length = len(message)
            if not 1 <= message_length <= 3:
                raise ValueError("Message length must be between 1 and 3.")

            data += bytes([message_index, message_length]) + message  # Add each message

        self.send_command("", data=data)
        response = self.read_response()
        if response and response[0] != 1:
            raise Exception("Storing serial messages failed.")

    def clear_serial_message_libraries(self):
        """
        Clears all the serial message libraries of each module.
        Restores each message to default (a message of length 1, value=index).
        """
        self.send_command(">")
        response = self.read_response()
        if response and response[0] != 1:
            raise Exception("Clearing message libraries failed.")

    def send_stored_message(self, module_index, message_index):
        """
        Sends a stored serial message (by index) to a module.


        Args:
            module_index (int): The index of the target module.
            message_index (int):  The index of the message to send.

        Raises:
            ValueError: If module_index is out of range.
        """

        if not 0 <= module_index < self.n_modules:  # Check if module index is valid
            raise ValueError("Invalid module index.")  # Clearer error message
        data = b"U" + bytes([module_index, message_index])  # 'U' is the command
        self.send_command("", data=data)

    def echo_soft_code(self, soft_code):
        """
        Echoes a USB soft code targeting the PC. Useful for debugging.

        Args:
            soft_code (int): The soft code to echo.

        Returns:
            int: The echoed soft code.

        """
        data = b"S" + bytes([soft_code])  # 'S' is the command
        self.send_command("", data=data)
        op_code = self.read_response()[0]  # Should be 2
        echoed_code = self.read_response()[0]

        if op_code != 2:
            raise Exception("Echo soft code failed")
        return echoed_code

    def send_soft_code(self, soft_code):
        """
        Sends a USB soft code targeting the state machine.

        Args:
            soft_code (int): The soft code to send.
        """
        data = b"~" + bytes([soft_code])  # '~' is the command
        self.send_command("", data=data)

    def override_input_channel(self, channel_index, new_value):
        """
        Manually overrides an input channel, creating a virtual event.

        Args:
            channel_index (int): The input channel index to override.
            new_value (int): The new value for the channel (0 = low, 1 = high).
        """

        data = b"V" + bytes([channel_index, new_value])  # 'V' is the command
        self.send_command("", data=data)

    def send_state_machine_description(self, state_machine_description):
        """
        Sends a compressed state machine description to the Bpod.
        (Implementation would be complex, and very specific to the state machine structure,
        so this provides a basic framework.  You'll need to adapt it based on the
        precise format you're using for your compressed descriptions)

        Args:
            state_machine_description (bytes): The byte string representing the state machine description.
        """
        data = b"C" + state_machine_description  # 'C' is the command

        self.send_command("", data=data)  # No immediate confirmation for 'C'

    def run_state_machine(self):
        """
        Runs the state machine on the Bpod.


        Returns:
            tuple:  A tuple containing trial start timestamp, list of (event_codes, timestamps),
                    number of cycles completed, trial end timestamp, and post-trial timestamps
                    (if applicable).

        """
        self.send_command("R")

        # Get confirmation byte (if a new state machine was sent)
        confirmation = self.read_response()  # May need to adapt to handle both cases
        # Check if the new state machine upload was successful
        if confirmation and confirmation[0] != 1:
            # Handle failure appropriately
            raise Exception("Uploading of new state machine failed")

        trial_start_timestamp = struct.unpack("<Q", self.read_response(8))[
            0
        ]  # 64-bit unsigned int

        trial_events = []
        while True:  # Main trial running loop
            op_code = self.read_response()[0]
            if op_code == 1:  # Event code
                n_events = self.read_response()[0]
                event_codes = []
                timestamps = []
                for _ in range(n_events):
                    event_code = self.read_response()[0]
                    event_codes.append(event_code)
                    if self.timestamp_transmission_scheme == 1:  # Live timestamps
                        timestamp = struct.unpack("<I", self.read_response(4))[
                            0
                        ]  # 32 bit int
                        timestamps.append(timestamp)

                trial_events.append((event_codes, timestamps))  # Store as tuple

            elif op_code == 2:  # Soft code
                soft_code = self.read_response()[0]  # Get soft code
                # Handle soft code here... (e.g., logging, updating UI)
                print(f"Soft code received: {soft_code}")

            if event_codes and event_codes[-1] == 255:  # Exit state
                n_cycles_completed = struct.unpack("<I", self.read_response(4))[
                    0
                ]  # 32-bit
                trial_end_timestamp = struct.unpack("<Q", self.read_response(8))[
                    0
                ]  # 64-bit

                post_trial_timestamps = []
                if self.timestamp_transmission_scheme == 0:  # Post trial
                    n_timestamps = struct.unpack("<H", self.read_response(2))[
                        0
                    ]  # 16-bit
                    for _ in range(n_timestamps):
                        timestamp = struct.unpack("<I", self.read_response(4))[
                            0
                        ]  # 32 bit int
                        post_trial_timestamps.append(timestamp)

                return (
                    trial_start_timestamp,
                    trial_events,
                    n_cycles_completed,
                    trial_end_timestamp,
                    post_trial_timestamps,
                )

    def force_exit_state_machine(self):
        """
        Force-exits the currently running state machine and returns partial trial data.
        The returned data format is the same as run_state_machine().
        """

        self.send_command("X")
        return (
            self.run_state_machine()
        )  # Data returned in the same format as regular run


class BpodController:
    """Simple controller wrapper for Bpod serial communication."""

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_connection = SimpleNamespace(is_open=False)
        self._logger = logging.getLogger(__name__)

    def connect(self):
        """Open the serial connection."""
        try:
            # Keep the constructor call minimal so it is easy to mock in unit tests.
            conn = serial.Serial(self.port)
            conn.baudrate = self.baudrate
            conn.timeout = self.timeout
            self.serial_connection = conn
            return conn
        except Exception:
            self._logger.exception("Failed to connect to Bpod on %s", self.port)
            raise

    def disconnect(self) -> None:
        """Close the serial connection (best effort)."""
        conn = self.serial_connection
        self.serial_connection = None
        if conn is None:
            return
        try:
            if getattr(conn, "is_open", False):
                conn.close()
        except Exception:
            self._logger.debug("Failed to close Bpod serial connection.", exc_info=True)

    def send_event(self, event_code: int) -> None:
        """Send a single event byte to the Bpod."""
        conn = self.serial_connection
        if conn is None or not getattr(conn, "is_open", False):
            self._logger.warning(
                "Bpod serial connection is not open; dropping event %s", event_code
            )
            return
        payload = bytes([int(event_code) & 0xFF])
        conn.write(payload)
