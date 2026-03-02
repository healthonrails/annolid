"""Compatibility patches applied when launching TensorBoard from Annolid.

This module is imported automatically by Python's site machinery when it is
present on PYTHONPATH as ``sitecustomize``.

It is intentionally narrow in scope: it patches protobuf json_format helpers to
accept newer keyword arguments used by TensorBoard when running with an older
protobuf runtime.
"""

from __future__ import annotations

import inspect
import sys
import types


def _patch_protobuf_json_format() -> None:
    try:
        from google.protobuf import json_format  # type: ignore
    except Exception:
        return

    # TensorBoard (hparams plugin) may call MessageToJson(..., including_default_value_fields=...).
    # Older protobuf versions do not accept this kwarg.
    try:
        sig = inspect.signature(json_format.MessageToJson)
    except Exception:
        sig = None
    if sig is not None and "including_default_value_fields" in sig.parameters:
        return

    original_to_json = getattr(json_format, "MessageToJson", None)
    if callable(original_to_json):

        def _wrapped_message_to_json(message, *args, **kwargs):
            kwargs.pop("including_default_value_fields", None)
            return original_to_json(message, *args, **kwargs)

        # type: ignore[assignment]
        json_format.MessageToJson = _wrapped_message_to_json

    original_to_dict = getattr(json_format, "MessageToDict", None)
    if callable(original_to_dict):

        def _wrapped_message_to_dict(message, *args, **kwargs):
            kwargs.pop("including_default_value_fields", None)
            return original_to_dict(message, *args, **kwargs)

        # type: ignore[assignment]
        json_format.MessageToDict = _wrapped_message_to_dict


_patch_protobuf_json_format()


def _ensure_pkg_resources_stub() -> None:
    """Provide a tiny pkg_resources shim when setuptools is unavailable.

    TensorBoard imports ``pkg_resources`` to discover entry-point plugins.
    On minimal Python environments this module may be absent, causing startup
    to fail before TensorBoard can serve logs.
    """
    try:
        import pkg_resources  # type: ignore  # noqa: F401

        return
    except Exception:
        pass

    stub = types.ModuleType("pkg_resources")

    def _iter_entry_points(*_args, **_kwargs):
        return []

    try:
        from packaging.version import parse as _parse_version  # type: ignore
    except Exception:
        _parse_version = None

    stub.iter_entry_points = _iter_entry_points  # type: ignore[attr-defined]
    if _parse_version is not None:
        stub.parse_version = _parse_version  # type: ignore[attr-defined]
    sys.modules["pkg_resources"] = stub


_ensure_pkg_resources_stub()
