from __future__ import annotations

from annolid.gui.services.inference_service import InferenceService


def test_process_inference_results_returns_stable_schema_for_unknown_model() -> None:
    service = InferenceService()
    payload = service.process_inference_results(
        model_type="unknown",
        raw_results={"x": 1},
        model_config={},
        postprocessing_config=None,
    )

    expected_keys = {
        "model_type",
        "detections",
        "masks",
        "keypoints",
        "results",
        "meta",
        "error",
    }
    assert expected_keys.issubset(set(payload.keys()))
    assert payload["model_type"] == "unknown"
    assert payload["results"] == {"x": 1}
    assert payload["error"] is None


def test_validate_inference_results_rejects_non_schema_payload() -> None:
    service = InferenceService()
    ok, errors = service.validate_inference_results({"model_type": "yolo"})
    assert ok
    assert not errors


def test_process_inference_results_parses_ultralytics_like_yolo_object() -> None:
    service = InferenceService()

    class _Boxes:
        xyxy = [[1.0, 2.0, 3.0, 4.0]]
        conf = [0.95]
        cls = [0]

    class _Result:
        boxes = _Boxes()

    payload = service.process_inference_results(
        model_type="yolo",
        raw_results=_Result(),
        model_config={"class_names": ["mouse"], "confidence_threshold": 0.1},
        postprocessing_config=None,
    )

    assert payload["model_type"] == "yolo"
    assert len(payload["detections"]) == 1
    det = payload["detections"][0]
    assert det["class_name"] == "mouse"
    assert det["bbox"] == [1.0, 2.0, 3.0, 4.0]
