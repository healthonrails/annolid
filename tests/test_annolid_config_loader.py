from __future__ import annotations

from annolid import configs


def test_sam3_handoff_flags_are_valid_config_keys(monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        configs.logger, "warn", lambda message: warnings.append(message)
    )

    config = configs.get_config(
        """
sam3:
  use_explicit_window_reseed: false
  allow_private_state_mutation: true
"""
    )

    assert config["sam3"]["use_explicit_window_reseed"] is False
    assert config["sam3"]["allow_private_state_mutation"] is True
    assert warnings == []
