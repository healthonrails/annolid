from __future__ import annotations

from annolid.engine.registry import _REGISTRY, get_model


def test_get_model_uses_targeted_import_before_full_builtin_load(monkeypatch) -> None:
    _REGISTRY.pop("image-edit", None)

    def _fail_load_builtins():
        raise AssertionError("load_builtin_models should not be called for image-edit")

    monkeypatch.setattr(
        "annolid.engine.registry.load_builtin_models", _fail_load_builtins
    )

    plugin = get_model("image-edit", load_builtins=True)
    assert plugin.name == "image-edit"
