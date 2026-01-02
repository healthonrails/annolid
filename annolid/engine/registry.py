from __future__ import annotations

import argparse
import importlib
import pkgutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Type


class ModelPluginBase(ABC):
    """Base class for train/predict wrappers.

    Plugins must be import-light: avoid importing heavy optional deps at module import time.
    Prefer importing inside `train()` / `predict()`.
    """

    name: str
    description: str = ""

    def add_train_args(self, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    def add_predict_args(self, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    def train(self, args: argparse.Namespace) -> int:
        raise NotImplementedError

    def predict(self, args: argparse.Namespace) -> int:
        raise NotImplementedError

    @classmethod
    def supports_train(cls) -> bool:
        return cls.add_train_args is not ModelPluginBase.add_train_args and cls.train is not ModelPluginBase.train

    @classmethod
    def supports_predict(cls) -> bool:
        return (
            cls.add_predict_args is not ModelPluginBase.add_predict_args
            and cls.predict is not ModelPluginBase.predict
        )


@dataclass(frozen=True)
class ModelInfo:
    name: str
    description: str
    supports_train: bool
    supports_predict: bool


_REGISTRY: Dict[str, Type[ModelPluginBase]] = {}
_BUILTINS_LOADED = False
_LOAD_FAILURES: Dict[str, str] = {}


def register_model(plugin_cls: Type[ModelPluginBase]) -> Type[ModelPluginBase]:
    name = str(getattr(plugin_cls, "name", "") or "").strip()
    if not name:
        raise ValueError(
            f"Plugin {plugin_cls!r} must define a non-empty .name")
    if name in _REGISTRY and _REGISTRY[name] is not plugin_cls:
        raise ValueError(f"Duplicate model plugin name: {name!r}")
    _REGISTRY[name] = plugin_cls
    return plugin_cls


def _iter_builtin_modules() -> Iterable[str]:
    # Auto-discover `annolid.engine.models.*` modules.
    try:
        import annolid.engine.models as models_pkg
    except Exception:
        return []
    for mod in pkgutil.iter_modules(models_pkg.__path__, models_pkg.__name__ + "."):
        yield mod.name


def load_builtin_models() -> List[str]:
    """Import built-in plugin modules and return those that failed to import."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return []
    _BUILTINS_LOADED = True

    failed: List[str] = []
    for module_name in _iter_builtin_modules():
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failed.append(module_name)
            _LOAD_FAILURES[module_name] = f"{exc.__class__.__name__}: {exc}"
    return failed


def get_load_failures() -> Dict[str, str]:
    load_builtin_models()
    return dict(_LOAD_FAILURES)


def list_models(*, load_builtins: bool = True) -> List[ModelInfo]:
    if load_builtins:
        load_builtin_models()
    out: List[ModelInfo] = []
    for name, cls in sorted(_REGISTRY.items(), key=lambda kv: kv[0]):
        out.append(
            ModelInfo(
                name=name,
                description=str(getattr(cls, "description", "") or ""),
                supports_train=bool(cls.supports_train()),
                supports_predict=bool(cls.supports_predict()),
            )
        )
    return out


def get_model(name: str, *, load_builtins: bool = True) -> ModelPluginBase:
    if load_builtins:
        load_builtin_models()
    key = str(name or "").strip()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        raise KeyError(f"Unknown model {key!r}. Available: {available}")
    return _REGISTRY[key]()
