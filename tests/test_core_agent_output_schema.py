import pytest

from annolid.core.output.validate import (
    AgentOutputValidationError,
    validate_agent_record,
)


def test_agent_output_schema_allows_optional_timestamp_sec():
    record = {
        "version": "Annolid",
        "video_name": "test_video.mp4",
        "frame_index": 0,
        "timestamp_sec": 0.0,
        "imagePath": "",
        "imageHeight": 10,
        "imageWidth": 10,
        "flags": {},
        "otherData": {},
        "shapes": [
            {
                "label": "mouse",
                "points": [[1, 2], [3, 4]],
                "shape_type": "rectangle",
                "group_id": 1,
                "flags": {},
            }
        ],
    }
    validate_agent_record(record)


def test_agent_output_schema_rejects_unknown_geometry_type():
    record = {
        "version": "Annolid",
        "video_name": "test_video.mp4",
        "frame_index": 0,
        "imagePath": "",
        "imageHeight": 10,
        "imageWidth": 10,
        "flags": {},
        "otherData": {},
        "shapes": [
            {
                "label": "mouse",
                "points": "not-a-list",
                "shape_type": "rectangle",
            }
        ],
    }
    with pytest.raises(AgentOutputValidationError):
        validate_agent_record(record)
