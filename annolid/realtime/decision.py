import zmq
import zmq.asyncio
import time
import logging
from enum import Enum, auto
import asyncio

from annolid.utils.logger import logger


class TrialState(Enum):
    """Defines the possible states of an experimental trial."""

    INTER_TRIAL_INTERVAL = auto()
    AWAITING_BEHAVIOR = auto()
    REWARD_PENDING = auto()
    TIMEOUT = auto()


class DecisionProcess:
    """
    The "brain" of the real-time experiment, implemented as a Finite State Machine.

    This process subscribes to detection events from the Perception module and
    makes decisions based on the current experimental state. It then publishes
    commands to the Action module to control hardware like a Bpod.

    It is designed to be CPU-bound and contains all the scientific trial logic,
    decoupled from the high-performance perception and hardware layers.
    """

    def __init__(
        self,
        perception_address: str,
        action_address: str,
        trial_timeout_s: float = 10.0,
        iti_duration_s: float = 5.0,
        target_behavior: str = "freezing",
    ):
        """
        Initializes the DecisionProcess.

        Args:
            perception_address (str): The ZMQ address to subscribe to for detections.
            action_address (str): The ZMQ address to push action commands to.
            trial_timeout_s (float): Seconds a trial can last before timing out.
            iti_duration_s (float): Seconds for the inter-trial interval.
            target_behavior (str): The specific behavior string to trigger an action.
        """
        self.perception_address = perception_address
        self.action_address = action_address
        self.trial_timeout_s = trial_timeout_s
        self.iti_duration_s = iti_duration_s
        self.target_behavior = target_behavior

        # ZMQ setup using the asyncio context
        self.context = zmq.asyncio.Context()
        self.perception_sub: zmq.asyncio.Socket = self.context.socket(zmq.SUB)
        self.action_push: zmq.asyncio.Socket = self.context.socket(zmq.PUSH)

        # State machine variables
        self.state: TrialState = TrialState.INTER_TRIAL_INTERVAL
        self.trial_start_time: float = 0.0
        self.last_state_change: float = time.time()
        self.running = True

    async def _setup(self):
        """Connects ZMQ sockets."""
        logger.info("Setting up DecisionProcess...")
        self.perception_sub.connect(self.perception_address)
        self.perception_sub.setsockopt_string(zmq.SUBSCRIBE, "detections")
        self.action_push.connect(self.action_address)
        logger.info(f"Subscribed to detections at {self.perception_address}")
        logger.info(f"Pushing actions to {self.action_address}")

    async def run(self):
        """Main execution loop for the decision process."""
        await self._setup()
        logger.info(f"Decision process started. Current state: {self.state.name}")

        while self.running:
            try:
                # This is the core of the event-driven loop.
                # We wait for either a message from perception or a timeout.
                await self.process_state_logic()

                # Check for incoming detection messages with a short timeout
                if await self.perception_sub.poll(timeout=10):  # Poll for 10ms
                    topic, msg = await self.perception_sub.recv_multipart()
                    detection = zmq.utils.jsonapi.loads(msg)
                    await self.handle_detection(detection)

            except asyncio.CancelledError:
                logger.info("Decision process cancellation requested.")
                self.running = False
            except Exception as e:
                logger.error(
                    f"An error occurred in the decision loop: {e}", exc_info=True
                )
                await asyncio.sleep(1)  # Prevent rapid-fire errors

        await self._cleanup()

    async def process_state_logic(self):
        """Handles state transitions based on time."""
        current_time = time.time()

        if self.state == TrialState.INTER_TRIAL_INTERVAL:
            if current_time - self.last_state_change >= self.iti_duration_s:
                await self._transition_to(TrialState.AWAITING_BEHAVIOR)

        elif self.state == TrialState.AWAITING_BEHAVIOR:
            if current_time - self.trial_start_time > self.trial_timeout_s:
                await self._transition_to(TrialState.TIMEOUT)

        elif self.state == TrialState.TIMEOUT:
            # After a timeout, immediately go to the inter-trial interval
            await self._transition_to(TrialState.INTER_TRIAL_INTERVAL)

    async def handle_detection(self, detection: dict):
        """Handles incoming detection events based on the current state."""
        behavior = detection.get("behavior")

        # We only act on a detection if we are in the correct state
        if (
            self.state == TrialState.AWAITING_BEHAVIOR
            and behavior == self.target_behavior
        ):
            logger.info(f"Target behavior '{self.target_behavior}' detected!")

            # Send command to the action module
            command = {"action": "trigger_reward_valve", "duration_ms": 50}
            await self.action_push.send_json(command)

            # Transition to the next state
            await self._transition_to(TrialState.REWARD_PENDING)
            # This could be followed by another state transition after a delay
            await asyncio.sleep(0.5)  # e.g., brief pause after reward
            await self._transition_to(TrialState.INTER_TRIAL_INTERVAL)

    async def _transition_to(self, new_state: TrialState):
        """Manages state transitions and associated setup/teardown logic."""
        if self.state == new_state:
            return

        logger.info(f"State transition: {self.state.name} -> {new_state.name}")
        self.state = new_state
        self.last_state_change = time.time()

        # Logic to execute upon entering a new state
        if new_state == TrialState.AWAITING_BEHAVIOR:
            self.trial_start_time = time.time()
            logger.info(
                f"New trial started. Waiting for '{self.target_behavior}' for {self.trial_timeout_s}s."
            )
        elif new_state == TrialState.INTER_TRIAL_INTERVAL:
            logger.info(f"Entering ITI for {self.iti_duration_s}s.")

    async def _cleanup(self):
        """Releases ZMQ resources cleanly."""
        logger.info("Cleaning up decision process resources...")
        if self.action_push and not self.action_push.closed:
            self.action_push.close()
        if self.perception_sub and not self.perception_sub.closed:
            self.perception_sub.close()
        if self.context and not self.context.closed:
            self.context.term()
        logger.info("Cleanup complete.")


async def main():
    """Example of how to launch the DecisionProcess."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    )

    # These addresses must match the other processes
    PERCEPTION_ADDR = "tcp://localhost:5555"
    ACTION_ADDR = "tcp://localhost:5556"

    config = {
        "perception_address": PERCEPTION_ADDR,
        "action_address": ACTION_ADDR,
        "trial_timeout_s": 10.0,
        "iti_duration_s": 5.0,
        "target_behavior": "freezing",  # Should match a class name from your model
    }

    decision_process = DecisionProcess(**config)

    try:
        await decision_process.run()
    except KeyboardInterrupt:
        logger.info("Main function caught KeyboardInterrupt.")
    finally:
        decision_process.running = False


if __name__ == "__main__":
    asyncio.run(main())
