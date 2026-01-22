import pytest

from annolid.core.output.validate import (
    AgentOutputValidationError,
    validate_agent_record,
)
from annolid.core.types import BBoxGeometry, FrameRef


def test_agent_output_schema_allows_optional_timestamp_sec():
    record = {
        "schema_version": "annolid.agent_output.1",
        "type": "detection",
        "video_name": "test_video",
        "frame": FrameRef(frame_index=0, timestamp_sec=0.0).to_dict(),
        "objects": [
            {
                "label": "mouse",
                "score": 0.9,
                "geometry": BBoxGeometry("bbox", (1, 2, 3, 4)).to_dict(),
            }
        ],
    }
    validate_agent_record(record)


def test_agent_output_schema_rejects_unknown_geometry_type():
    record = {
        "schema_version": "annolid.agent_output.1",
        "type": "detection",
        "frame": FrameRef(frame_index=0).to_dict(),
        "objects": [{"geometry": {"type": "circle", "r": 3}}],
    }
    with pytest.raises(AgentOutputValidationError):
        validate_agent_record(record)
