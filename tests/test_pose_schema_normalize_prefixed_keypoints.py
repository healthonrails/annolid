from annolid.annotation.pose_schema import PoseSchema


def test_pose_schema_normalize_prefixed_keypoints_does_not_split_base_names() -> None:
    schema = PoseSchema.from_dict(
        {
            "version": "1.0",
            "keypoints": ["nose", "tail_base", "left_ear"],
            "instances": ["resident", "intruder"],
            "instance_separator": "_",
        }
    )
    schema.normalize_prefixed_keypoints()
    assert schema.instances == ["resident", "intruder"]
    assert schema.keypoints == ["nose", "tail_base", "left_ear"]


def test_pose_schema_normalize_prefixed_keypoints_only_when_fully_expanded() -> None:
    schema = PoseSchema.from_dict(
        {
            "version": "1.0",
            "keypoints": [
                "intruder_left_ear",
                "intruder_tail_base",
                "resident_left_ear",
                "resident_tail_base",
            ],
            "instances": [],
            "instance_separator": "_",
        }
    )
    schema.normalize_prefixed_keypoints()
    assert schema.instances == ["intruder", "resident"]
    assert schema.keypoints == ["left_ear", "tail_base"]
