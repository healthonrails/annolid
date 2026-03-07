import pytest

from annolid.interfaces.memory.adapters.settings_model import SettingsProfile


def test_settings_profile_roundtrip() -> None:
    profile = SettingsProfile(
        id="m1",
        name="Default Tracking",
        workflow="tracking",
        settings={"model": "cutie"},
        workspace_id="w1",
        tags=["default"],
    )
    payload = profile.to_dict()
    restored = SettingsProfile.from_dict(payload)
    assert restored.name == "Default Tracking"
    assert restored.settings["model"] == "cutie"
    assert restored.workflow == "tracking"


def test_settings_profile_validation() -> None:
    with pytest.raises(ValueError):
        SettingsProfile(name="", workflow="tracking", settings={}).validate()
    with pytest.raises(ValueError):
        SettingsProfile(name="x", workflow="", settings={}).validate()
