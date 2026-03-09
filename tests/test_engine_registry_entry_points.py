from __future__ import annotations

import argparse
from collections.abc import Callable

import pytest

from annolid.engine import registry
from annolid.engine.registry import ModelPluginBase


class _ExternalPlugin(ModelPluginBase):
    name = "external-sim"
    description = "Third-party simulation plugin."

    def add_train_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--unused-train", action="store_true")

    def add_predict_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--unused-predict", action="store_true")

    def train(self, args: argparse.Namespace) -> int:
        return 0

    def predict(self, args: argparse.Namespace) -> int:
        return 0


class _LoadedEntryPoint:
    def __init__(self, name: str, loader: Callable[[], object]) -> None:
        self.name = name
        self._loader = loader

    def load(self) -> object:
        return self._loader()


class _EntryPoints(tuple):
    def select(self, *, group: str):
        if group == registry._ENTRY_POINT_GROUP:
            return tuple(self)
        return ()


def _reset_registry_state() -> None:
    registry._REGISTRY.pop("external-sim", None)
    registry._ENTRY_POINTS_LOADED = False
    registry._LOAD_FAILURES.clear()


@pytest.fixture(autouse=True)
def _clean_registry_state():
    _reset_registry_state()
    yield
    _reset_registry_state()


def test_list_models_loads_third_party_entry_points(monkeypatch) -> None:
    monkeypatch.setattr(
        registry.importlib.metadata,
        "entry_points",
        lambda: _EntryPoints(
            (_LoadedEntryPoint("external-sim", lambda: _ExternalPlugin),)
        ),
    )

    models = {model.name: model for model in registry.list_models(load_builtins=False)}

    assert "external-sim" not in models

    models = {model.name: model for model in registry.list_models(load_builtins=True)}

    assert "external-sim" in models
    assert models["external-sim"].supports_predict is True


def test_entry_point_failures_are_recorded(monkeypatch) -> None:
    monkeypatch.setattr(
        registry.importlib.metadata,
        "entry_points",
        lambda: _EntryPoints((_LoadedEntryPoint("broken-plugin", lambda: object()),)),
    )

    failures = registry.load_entry_point_models()
    details = registry.get_load_failures()

    assert failures == ["annolid.model_plugins:broken-plugin"]
    assert "annolid.model_plugins:broken-plugin" in details
    assert "ModelPluginBase subclass" in details["annolid.model_plugins:broken-plugin"]


def test_entry_point_duplicate_names_are_reported(monkeypatch) -> None:
    registry.register_model(_ExternalPlugin)
    monkeypatch.setattr(
        registry.importlib.metadata,
        "entry_points",
        lambda: _EntryPoints(
            (
                _LoadedEntryPoint(
                    "duplicate",
                    lambda: type(
                        "OtherPlugin", (_ExternalPlugin,), {"name": "external-sim"}
                    ),
                ),
            )
        ),
    )

    failures = registry.load_entry_point_models()

    assert failures == ["annolid.model_plugins:duplicate"]
    assert (
        "Duplicate model plugin name"
        in registry.get_load_failures()["annolid.model_plugins:duplicate"]
    )
