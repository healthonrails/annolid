import os
import onnxruntime as ort
from annolid.utils.cancellation import (
    InferenceCancelledError,
    is_stop_requested,
    onnx_cancellable_run,
)
from annolid.utils.devices import has_gpu


class ONNXBaseModel:
    """
    A base class for working with ONNX models using onnxruntime.

    Parameters:
    - model_path (str): Path to the ONNX model file.
    - device_type (str, optional): Type of device to run inference on, 'cpu' (default) or 'gpu'.
    """

    def __init__(self, model_path: str, device_type: str = "cpu", *, stop_event=None):
        """
        Initialize the ONNXBaseModel.

        Args:
        - model_path (str): Path to the ONNX model file.
        - device_type (str, optional): Type of device to run inference on, 'cpu' (default) or 'gpu'.
        - stop_event (threading.Event, optional): Cooperative cancellation event.
        """
        if is_stop_requested(stop_event):
            raise InferenceCancelledError("ONNX session initialization cancelled.")

        try:
            available_providers = set(ort.get_available_providers())
        except Exception:
            available_providers = set()

        if "CUDAExecutionProvider" in available_providers and has_gpu():
            device_type = "gpu"
        self.sess_opts = ort.SessionOptions()
        self.sess_opts.inter_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", 1))

        # Select the execution provider based on device_type and build capabilities
        gpu_providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available_providers
            else ["CPUExecutionProvider"]
        )
        providers_dict = {
            "cpu": ["CPUExecutionProvider"],
            "gpu": gpu_providers,
        }
        self.providers = providers_dict.get(
            device_type.lower(), ["CPUExecutionProvider"]
        )

        self.ort_session = ort.InferenceSession(
            model_path,
            providers=self.providers,
            sess_options=self.sess_opts,
        )
        if is_stop_requested(stop_event):
            self.close()
            raise InferenceCancelledError("ONNX session initialization cancelled.")

    def close(self) -> None:
        """Release the underlying ONNX Runtime session reference."""
        self.ort_session = None

    def get_onnx_inference(
        self,
        blob,
        input_data=None,
        extract=True,
        squeeze=False,
        *,
        stop_event=None,
    ):
        """
        Perform inference using the ONNX model.

        Args:
        - blob: Input data for inference.
        - input_data (dict, optional): Dictionary of input data if specified.
        - extract (bool, optional): Whether to extract the output.
        - squeeze (bool, optional): Whether to squeeze the output.
        - stop_event (threading.Event, optional): Cooperative cancellation event.

        Returns:
        - Output of the inference.
        """
        if input_data is None:
            input_data = {self.get_input_name(): blob}
        outputs = onnx_cancellable_run(
            self.ort_session,
            None,
            input_data,
            stop_event=stop_event,
        )

        if extract:
            outputs = outputs[0]
        if squeeze:
            outputs = outputs.squeeze(axis=0)
        return outputs

    def get_input_name(self):
        """
        Get the name of the input node.

        Returns:
        - str: Name of the input node.
        """
        return self.ort_session.get_inputs()[0].name

    def get_input_shape(self):
        """
        Get the shape of the input tensor.

        Returns:
        - tuple: Shape of the input tensor.
        """
        return self.ort_session.get_inputs()[0].shape

    def get_output_names(self):
        """
        Get the names of the output nodes.

        Returns:
        - list: List of output node names.
        """
        return [out.name for out in self.ort_session.get_outputs()]
