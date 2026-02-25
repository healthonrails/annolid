from __future__ import annotations

import time
from typing import Any, Dict, Optional

from qtpy import QtCore

from annolid.gui.controllers.inference_controller import InferenceController


def test_inference_controller_requires_real_callable() -> None:
    controller = InferenceController()
    errors: list[str] = []
    controller.inference_error.connect(errors.append)

    controller.run_inference(
        model_type="yolo",
        input_data={"dummy": True},
        model_config={"identifier": "x", "weight_file": ""},
        postprocessing_config=None,
    )

    assert errors
    assert "deprecated without a real inference callable" in errors[-1]


def test_inference_controller_runs_with_callable() -> None:
    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    controller = InferenceController()
    completed: list[Dict[str, Any]] = []
    errors: list[str] = []
    controller.inference_completed.connect(completed.append)
    controller.inference_error.connect(errors.append)

    def raw_infer(
        model_type: str,
        input_data: Any,
        model_config: Dict[str, Any],
        post_cfg: Optional[Dict[str, Any]],
    ) -> Any:
        _ = input_data, model_config, post_cfg
        if model_type == "yolo":
            return {"boxes": []}
        return {}

    controller.run_inference(
        model_type="yolo",
        input_data={"dummy": True},
        model_config={
            "identifier": "x",
            "weight_file": "",
            "raw_inference_callable": raw_infer,
        },
        postprocessing_config=None,
    )

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        app.processEvents()
        if completed or errors:
            break

    assert not errors
    assert completed
    payload = completed[-1]
    assert payload["model_type"] == "yolo"
    assert "detections" in payload
    assert "error" in payload


def test_inference_controller_rejects_parallel_runs() -> None:
    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    controller = InferenceController()
    errors: list[str] = []
    controller.inference_error.connect(errors.append)

    def raw_infer(
        model_type: str,
        input_data: Any,
        model_config: Dict[str, Any],
        post_cfg: Optional[Dict[str, Any]],
    ) -> Any:
        _ = model_type, input_data, model_config, post_cfg
        time.sleep(0.2)
        return {}

    cfg = {"identifier": "x", "weight_file": "", "raw_inference_callable": raw_infer}
    controller.run_inference("yolo", {}, cfg)
    controller.run_inference("yolo", {}, cfg)

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        app.processEvents()
        if errors:
            break

    assert any("already running" in e for e in errors)
