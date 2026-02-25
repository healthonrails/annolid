"""
Inference Controller for Annolid GUI Application.

Handles UI interactions and coordinates AI inference operations.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from qtpy import QtCore

from ..interfaces.services import IInferenceService
from ..services import InferenceService

logger = logging.getLogger(__name__)


class InferenceController(QtCore.QObject):
    """
    Controller for inference-related UI operations.

    Coordinates between the UI and inference service, handling
    user interactions and business logic orchestration.
    """

    # Signals
    inference_started = QtCore.Signal(str)  # Emitted when inference starts (model_name)
    inference_completed = QtCore.Signal(
        dict
    )  # Emitted when inference completes (results)
    inference_error = QtCore.Signal(str)  # Emitted on inference errors
    progress_updated = QtCore.Signal(int, str)  # Progress updates
    model_validated = QtCore.Signal(
        bool, list
    )  # Emitted after model validation (is_valid, errors)

    def __init__(
        self,
        inference_service: Optional[IInferenceService] = None,
        parent: Optional[QtCore.QObject] = None,
    ):
        """
        Initialize the inference controller.

        Args:
            inference_service: Inference service instance
            parent: Parent QObject
        """
        super().__init__(parent)
        self._inference_service = inference_service or InferenceService()
        self._current_model_config: Optional[Dict[str, Any]] = None
        self._inference_thread: Optional[InferenceWorker] = None

    def validate_model_config(
        self, model_config: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate a model configuration.

        Args:
            model_config: Model configuration to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            is_valid, errors = self._inference_service.validate_model_config(
                model_config
            )

            self.model_validated.emit(is_valid, errors)

            if is_valid:
                self._current_model_config = model_config
                logger.info("Model configuration validated successfully")
            else:
                logger.warning(
                    f"Model configuration validation failed: {'; '.join(errors)}"
                )

            return is_valid, errors

        except Exception as e:
            error_msg = f"Failed to validate model config: {str(e)}"
            self.inference_error.emit(error_msg)
            logger.error(error_msg)
            return False, [error_msg]

    def run_inference(
        self,
        model_type: str,
        input_data: Any,
        model_config: Dict[str, Any],
        postprocessing_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Run inference asynchronously.

        Args:
            model_type: Type of model (yolo, sam, etc.)
            input_data: Input data for inference
            model_config: Model configuration
            postprocessing_config: Optional postprocessing configuration
        """
        try:
            if (
                self._inference_thread is not None
                and self._inference_thread.isRunning()
            ):
                self.inference_error.emit(
                    "Inference is already running. Cancel it before starting a new request."
                )
                return

            # Validate model config first
            is_valid, errors = self.validate_model_config(model_config)
            if not is_valid:
                self.inference_error.emit(f"Invalid model config: {'; '.join(errors)}")
                return

            # Prepare input data
            prepared_input = self._inference_service.prepare_inference_input(
                model_type, input_data, model_config
            )

            raw_inference_callable = model_config.get("raw_inference_callable")
            if raw_inference_callable is None:
                error_msg = (
                    "InferenceController.run_inference is deprecated without a real "
                    "inference callable. Use Inference Wizard/InferenceProcessor, or "
                    "set model_config['raw_inference_callable']."
                )
                self.inference_error.emit(error_msg)
                logger.error(error_msg)
                return
            if not callable(raw_inference_callable):
                error_msg = "model_config['raw_inference_callable'] must be callable."
                self.inference_error.emit(error_msg)
                logger.error(error_msg)
                return

            # Start inference in background thread
            self._inference_thread = InferenceWorker(
                inference_service=self._inference_service,
                model_type=model_type,
                input_data=prepared_input,
                model_config=model_config,
                postprocessing_config=postprocessing_config,
                raw_inference_callable=raw_inference_callable,
            )

            # Connect signals
            self._inference_thread.inference_completed.connect(
                self._on_inference_completed
            )
            self._inference_thread.inference_error.connect(self._on_inference_error)
            self._inference_thread.progress_updated.connect(self._on_progress_updated)
            self._inference_thread.finished.connect(self._on_inference_thread_finished)
            self._inference_thread.finished.connect(self._inference_thread.deleteLater)

            # Start inference
            model_name = model_config.get("identifier", "Unknown")
            self.inference_started.emit(model_name)
            self._inference_thread.start()

            logger.info(f"Inference started for model: {model_name}")

        except Exception as e:
            error_msg = f"Failed to start inference: {str(e)}"
            self.inference_error.emit(error_msg)
            logger.error(error_msg)

    def process_inference_results(
        self,
        model_type: str,
        raw_results: Any,
        model_config: Dict[str, Any],
        postprocessing_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process inference results synchronously.

        Args:
            model_type: Type of model that produced results
            raw_results: Raw results from inference
            model_config: Model configuration
            postprocessing_config: Postprocessing configuration

        Returns:
            Processed results
        """
        try:
            processed_results = self._inference_service.process_inference_results(
                model_type, raw_results, model_config, postprocessing_config
            )

            logger.info(f"Inference results processed for model type: {model_type}")
            return processed_results

        except Exception as e:
            error_msg = f"Failed to process inference results: {str(e)}"
            self.inference_error.emit(error_msg)
            logger.error(error_msg)
            return {"error": error_msg}

    def convert_results_to_annotation(
        self, results: Dict[str, Any], image_path: str, image_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Convert inference results to annotation format.

        Args:
            results: Inference results
            image_path: Path to the source image
            image_size: Image dimensions (width, height)

        Returns:
            Annotation data
        """
        try:
            annotation = self._inference_service.convert_results_to_labelme_format(
                results, image_path, image_size
            )

            logger.info(
                f"Inference results converted to annotation format for: {image_path}"
            )
            return annotation

        except Exception as e:
            error_msg = f"Failed to convert results to annotation: {str(e)}"
            self.inference_error.emit(error_msg)
            logger.error(error_msg)
            return {}

    def filter_results_by_confidence(
        self, results: Dict[str, Any], min_confidence: float
    ) -> Dict[str, Any]:
        """
        Filter inference results by confidence threshold.

        Args:
            results: Inference results to filter
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered results
        """
        try:
            filtered_results = self._inference_service.filter_results_by_confidence(
                results, min_confidence
            )

            logger.info(f"Results filtered by confidence: {min_confidence}")
            return filtered_results

        except Exception as e:
            error_msg = f"Failed to filter results: {str(e)}"
            self.inference_error.emit(error_msg)
            logger.error(error_msg)
            return results

    def merge_inference_results(
        self, results_list: List[Dict[str, Any]], merge_strategy: str = "combine"
    ) -> Dict[str, Any]:
        """
        Merge multiple inference results.

        Args:
            results_list: List of inference results to merge
            merge_strategy: Strategy for merging

        Returns:
            Merged results
        """
        try:
            merged_results = self._inference_service.merge_inference_results(
                results_list, merge_strategy
            )

            logger.info(f"Inference results merged using strategy: {merge_strategy}")
            return merged_results

        except Exception as e:
            error_msg = f"Failed to merge results: {str(e)}"
            self.inference_error.emit(error_msg)
            logger.error(error_msg)
            return {}

    def get_inference_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get statistics for inference results.

        Args:
            results: Inference results to analyze

        Returns:
            Statistics dictionary
        """
        try:
            stats = self._inference_service.get_inference_statistics(results)

            logger.info("Inference statistics calculated")
            return stats

        except Exception as e:
            error_msg = f"Failed to get inference statistics: {str(e)}"
            self.inference_error.emit(error_msg)
            logger.error(error_msg)
            return {}

    def validate_inference_results(
        self, results: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate inference results.

        Args:
            results: Inference results to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            is_valid, errors = self._inference_service.validate_inference_results(
                results
            )

            if not is_valid:
                logger.warning(
                    f"Inference results validation failed: {'; '.join(errors)}"
                )

            return is_valid, errors

        except Exception as e:
            error_msg = f"Failed to validate inference results: {str(e)}"
            self.inference_error.emit(error_msg)
            logger.error(error_msg)
            return False, [error_msg]

    def cancel_inference(self) -> None:
        """Cancel the current inference operation."""
        if self._inference_thread and self._inference_thread.isRunning():
            self._inference_thread.cancel()
            self._inference_thread.requestInterruption()
            logger.info("Inference cancelled")

    def shutdown(self, timeout_ms: int = 1000) -> None:
        """Best-effort cleanup for any running inference worker thread."""
        thread = self._inference_thread
        if thread is None:
            return

        try:
            if thread.isRunning():
                thread.cancel()
                thread.requestInterruption()
                thread.quit()
                thread.wait(max(0, int(timeout_ms)))
        except Exception as e:
            logger.warning(f"Failed to shutdown inference thread cleanly: {e}")
        finally:
            self._inference_thread = None

    def is_inference_running(self) -> bool:
        """
        Check if inference is currently running.

        Returns:
            True if inference is running, False otherwise
        """
        return self._inference_thread is not None and self._inference_thread.isRunning()

    def get_current_model_config(self) -> Optional[Dict[str, Any]]:
        """
        Get the current model configuration.

        Returns:
            Current model configuration or None
        """
        return self._current_model_config.copy() if self._current_model_config else None

    def _on_inference_completed(self, results: Dict[str, Any]) -> None:
        """Handle inference completion."""
        self.inference_completed.emit(results)
        self._inference_thread = None
        logger.info("Inference completed successfully")

    def _on_inference_error(self, error_msg: str) -> None:
        """Handle inference error."""
        self.inference_error.emit(error_msg)
        self._inference_thread = None
        logger.error(f"Inference error: {error_msg}")

    def _on_progress_updated(self, progress: int, message: str) -> None:
        """Handle progress updates."""
        self.progress_updated.emit(progress, message)

    def _on_inference_thread_finished(self) -> None:
        if (
            self._inference_thread is not None
            and not self._inference_thread.isRunning()
        ):
            self._inference_thread = None


class InferenceWorker(QtCore.QThread):
    """
    Worker thread for running inference operations.

    Runs inference in a separate thread to avoid blocking the UI.
    """

    # Signals
    inference_completed = QtCore.Signal(dict)
    inference_error = QtCore.Signal(str)
    progress_updated = QtCore.Signal(int, str)

    def __init__(
        self,
        inference_service: IInferenceService,
        model_type: str,
        input_data: Any,
        model_config: Dict[str, Any],
        postprocessing_config: Optional[Dict[str, Any]] = None,
        raw_inference_callable: Optional[
            Callable[[str, Any, Dict[str, Any], Optional[Dict[str, Any]]], Any]
        ] = None,
        parent: Optional[QtCore.QObject] = None,
    ):
        """
        Initialize the inference worker.

        Args:
            inference_service: Inference service instance
            model_type: Type of model
            input_data: Input data for inference
            model_config: Model configuration
            postprocessing_config: Postprocessing configuration
            parent: Parent QObject
        """
        super().__init__(parent)
        self._inference_service = inference_service
        self._model_type = model_type
        self._input_data = input_data
        self._model_config = model_config
        self._postprocessing_config = postprocessing_config
        self._raw_inference_callable = raw_inference_callable
        self._cancelled = False

    def run(self) -> None:
        """Run the inference operation."""
        try:
            self.progress_updated.emit(0, "Starting inference...")

            # Simulate progress (in real implementation, this would come from the model)
            self.progress_updated.emit(25, "Loading model...")

            if self._cancelled or self.isInterruptionRequested():
                return

            self.progress_updated.emit(50, "Running inference...")

            if self._cancelled or self.isInterruptionRequested():
                return

            if self._raw_inference_callable is None:
                raise RuntimeError(
                    "InferenceWorker requires a real raw_inference_callable."
                )
            raw_results = self._raw_inference_callable(
                self._model_type,
                self._input_data,
                self._model_config,
                self._postprocessing_config,
            )

            self.progress_updated.emit(75, "Processing results...")

            if self._cancelled or self.isInterruptionRequested():
                return

            # Process results using service
            processed_results = self._inference_service.process_inference_results(
                self._model_type,
                raw_results,
                self._model_config,
                self._postprocessing_config,
            )

            self.progress_updated.emit(100, "Inference completed")
            self.inference_completed.emit(processed_results)

        except Exception as e:
            error_msg = f"Inference failed: {str(e)}"
            self.inference_error.emit(error_msg)

    def cancel(self) -> None:
        """Cancel the inference operation."""
        self._cancelled = True
