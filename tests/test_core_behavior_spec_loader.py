from pathlib import Path

from annolid.core.behavior.spec import (
    DEFAULT_SCHEMA_FILENAME,
    default_behavior_spec,
    load_behavior_spec,
    save_behavior_spec,
)


def test_load_behavior_spec_json(tmp_path: Path):
    schema = default_behavior_spec()
    schema.behaviors[0].code = "digging"

    schema_path = tmp_path / DEFAULT_SCHEMA_FILENAME
    save_behavior_spec(schema, schema_path)

    loaded, loaded_path = load_behavior_spec(path=schema_path)
    assert loaded_path == schema_path
    assert loaded.behaviors[0].code == "digging"


def test_load_behavior_spec_yaml(tmp_path: Path):
    schema = default_behavior_spec()
    schema.categories[0].name = "Ethogram"

    schema_path = tmp_path / "project.annolid.yaml"
    save_behavior_spec(schema, schema_path)

    loaded, loaded_path = load_behavior_spec(path=schema_path)
    assert loaded_path == schema_path
    assert loaded.categories[0].name == "Ethogram"


def test_load_behavior_spec_autodiscovers_near_video(tmp_path: Path):
    schema = default_behavior_spec()
    schema.behaviors[0].name = "Rearing"

    video_path = tmp_path / "trial_01.mp4"
    video_path.write_bytes(b"")

    schema_path = tmp_path / DEFAULT_SCHEMA_FILENAME
    save_behavior_spec(schema, schema_path)

    loaded, loaded_path = load_behavior_spec(video_path=video_path)
    assert loaded_path == schema_path
    assert loaded.behaviors[0].name == "Rearing"
