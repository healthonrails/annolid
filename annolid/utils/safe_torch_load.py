"""Safe PyTorch checkpoint loading with strict weights_only=True, SHA-256 checks, and inert metadata."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from dataclasses import FrozenInstanceError, is_dataclass
import hashlib
import hmac
import io
import logging
from pathlib import (
    Path,
    PosixPath,
    PurePath,
    PurePosixPath,
    PureWindowsPath,
    WindowsPath,
)
import pickle
import pickletools
import sys
from typing import Any, Iterable
import zipfile

import numpy as np
import torch

__all__ = ["SafeCheckpointMetadata", "safe_torch_load"]

_logger = logging.getLogger(__name__)


class SafeCheckpointMetadata:
    """Inert, zero-side-effect attribute container for unimported checkpoint config classes."""

    __orig_qualname__: str = "SafeCheckpointMetadata"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__orig_qualname__}({fields})"


_BLOCKED_PICKLE_ROOT_MODULES: frozenset[str] = frozenset(
    {
        "asyncio",
        "bdb",
        "cffi",
        "code",
        "codeop",
        "commands",
        "compileall",
        "concurrent",
        "ctypes",
        "_ctypes",
        "distutils",
        "ensurepip",
        "fcntl",
        "ftplib",
        "gc",
        "http",
        "imaplib",
        "importlib",
        "_imp",
        "inspect",
        "linecache",
        "marshal",
        "mmap",
        "multiprocessing",
        "nt",
        "os",
        "pdb",
        "pickle",
        "_pickle",
        "pip",
        "pkgutil",
        "platform",
        "poplib",
        "popen2",
        "posix",
        "pty",
        "resource",
        "runpy",
        "selectors",
        "setuptools",
        "shutil",
        "signal",
        "smtplib",
        "socket",
        "ssl",
        "subprocess",
        "sys",
        "telnetlib",
        "tempfile",
        "threading",
        "_thread",
        "trace",
        "tty",
        "urllib",
        "venv",
        "webbrowser",
        "xmlrpc",
        "zipimport",
    }
)

_SAFE_BUILTIN_NAMES: frozenset[str] = frozenset(
    {
        "bool",
        "bytearray",
        "bytes",
        "complex",
        "dict",
        "float",
        "frozenset",
        "int",
        "list",
        "object",
        "range",
        "set",
        "slice",
        "str",
        "tuple",
    }
)

_SAFE_PATHLIB_NAMES: frozenset[str] = frozenset(
    {
        "Path",
        "PosixPath",
        "PurePath",
        "PurePosixPath",
        "PureWindowsPath",
        "WindowsPath",
    }
)

_BLOCKED_SPECIFIC_CALLABLES: frozenset[tuple[str, str]] = frozenset(
    {
        ("operator", "attrgetter"),
        ("operator", "itemgetter"),
        ("operator", "methodcaller"),
        ("_operator", "attrgetter"),
        ("_operator", "itemgetter"),
        ("_operator", "methodcaller"),
        ("functools", "reduce"),
        ("_functools", "reduce"),
        ("io", "open"),
        ("io", "FileIO"),
        ("io", "TextIOWrapper"),
        ("io", "BufferedReader"),
        ("io", "BufferedWriter"),
        ("io", "BufferedRandom"),
        ("_io", "open"),
        ("_io", "FileIO"),
        ("_io", "TextIOWrapper"),
        ("_io", "BufferedReader"),
        ("_io", "BufferedWriter"),
        ("_io", "BufferedRandom"),
    }
)

_BUILTIN_TRUSTED_TYPES: frozenset[Any] = frozenset(
    {
        OrderedDict,
        defaultdict,
        Counter,
        argparse.Namespace,
        slice,
        Path,
        PosixPath,
        PurePath,
        PurePosixPath,
        PureWindowsPath,
        WindowsPath,
        np.ndarray,
        np.dtype,
    }
)


def _has_inert_setstate(cls: type) -> bool:
    if not hasattr(cls, "__setstate__"):
        return True
    if not is_dataclass(cls):
        return False
    for base in cls.__mro__:
        if base is object or "__setstate__" not in base.__dict__:
            continue
        fn = base.__dict__["__setstate__"]
        if getattr(fn, "__module__", "") != "dataclasses":
            return False
    return True


def _has_inert_setattr(cls: type) -> bool:
    if cls.__setattr__ is object.__setattr__:
        return True
    if not is_dataclass(cls):
        return False
    allowed_names = {"type", "super", "__setattr__", "FrozenInstanceError", "object"}
    allowed_freevars = {"FrozenInstanceError", "cls", "__class__"}
    for base in cls.__mro__:
        if base is object or "__setattr__" not in base.__dict__:
            continue
        if base.__dict__["__setattr__"] is object.__setattr__:
            continue
        params = getattr(base, "__dataclass_params__", None)
        if params is None or not getattr(params, "frozen", False):
            return False
        fn = base.__dict__["__setattr__"]
        code = getattr(fn, "__code__", None)
        if code is None:
            return False
        closure = getattr(fn, "__closure__", None) or ()
        fn_globals = getattr(fn, "__globals__", None) or {}
        has_frozen_err = any(
            getattr(cell, "cell_contents", None) is FrozenInstanceError
            for cell in closure
        ) or (
            isinstance(fn_globals, dict)
            and fn_globals.get("FrozenInstanceError") is FrozenInstanceError
        )
        co_names = set(getattr(code, "co_names", ()))
        co_freevars = set(getattr(code, "co_freevars", ()))
        if not (
            has_frozen_err
            and co_names.issubset(allowed_names)
            and co_freevars.issubset(allowed_freevars)
        ):
            return False
    return True


def _assert_inert_data_class(cls: Any, label: str) -> None:
    if cls in _BUILTIN_TRUSTED_TYPES:
        return
    if not isinstance(cls, type):
        raise pickle.UnpicklingError(
            f"Checkpoint global '{label}' is not a class (got {type(cls).__name__}); "
            "arbitrary callables cannot be unpickled."
        )
    if issubclass(cls, BaseException):
        raise pickle.UnpicklingError(
            f"Checkpoint class '{label}' inherits from BaseException and cannot be unpickled."
        )
    if cls.__new__ is not object.__new__:
        raise pickle.UnpicklingError(
            f"Checkpoint class '{label}' defines a custom __new__ constructor and cannot be safely unpickled."
        )
    if not _has_inert_setstate(cls):
        raise pickle.UnpicklingError(
            f"Checkpoint class '{label}' defines a custom __setstate__ hook and cannot be safely unpickled."
        )
    if not _has_inert_setattr(cls):
        raise pickle.UnpicklingError(
            f"Checkpoint class '{label}' defines a custom __setattr__ hook and cannot be safely unpickled."
        )
    if cls.__getattribute__ is not object.__getattribute__ or any(
        "__getattr__" in base.__dict__ for base in cls.__mro__
    ):
        raise pickle.UnpicklingError(
            f"Checkpoint class '{label}' defines custom attribute-lookup hooks and cannot be safely unpickled."
        )


def _add_allowed_global(
    allowed: list[Any], item: Any, alias: str | None = None
) -> None:
    if item is None:
        return
    entry: Any = (item, alias) if alias is not None else item
    if entry not in allowed:
        allowed.append(entry)


def _numpy_safe_callables() -> set[Any]:
    items: set[Any] = {np.ndarray, np.dtype}
    for mod_attr in ("_core", "core"):
        core_mod = getattr(np, mod_attr, None)
        multiarray = (
            getattr(core_mod, "multiarray", None) if core_mod is not None else None
        )
        if multiarray is not None:
            for fn_name in ("_reconstruct", "scalar"):
                fn = getattr(multiarray, fn_name, None)
                if fn is not None:
                    items.add(fn)
    for scalar_dt in (
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.str_,
        np.bytes_,
    ):
        items.add(type(np.dtype(scalar_dt)))
        items.add(scalar_dt)
    return items


def _build_safe_globals(extra_globals: Iterable[Any] = ()) -> list[Any]:
    allowed: list[Any] = []
    for container_cls in (
        OrderedDict,
        defaultdict,
        Counter,
        argparse.Namespace,
        slice,
        np.ndarray,
        np.dtype,
    ):
        _add_allowed_global(allowed, container_cls)

    for path_cls, legacy_name in (
        (Path, "pathlib.Path"),
        (PosixPath, "pathlib.PosixPath"),
        (WindowsPath, "pathlib.WindowsPath"),
        (PurePath, "pathlib.PurePath"),
        (PurePosixPath, "pathlib.PurePosixPath"),
        (PureWindowsPath, "pathlib.PureWindowsPath"),
    ):
        _add_allowed_global(allowed, path_cls)
        _add_allowed_global(allowed, path_cls, legacy_name)

    for mod_attr in ("_core", "core"):
        core_mod = getattr(np, mod_attr, None)
        multiarray = (
            getattr(core_mod, "multiarray", None) if core_mod is not None else None
        )
        if multiarray is not None:
            for fn_name in ("_reconstruct", "scalar"):
                fn = getattr(multiarray, fn_name, None)
                if fn is not None:
                    _add_allowed_global(allowed, fn)
                    _add_allowed_global(allowed, fn, f"numpy.core.multiarray.{fn_name}")
                    _add_allowed_global(
                        allowed, fn, f"numpy._core.multiarray.{fn_name}"
                    )

    for scalar_dt in (
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.str_,
        np.bytes_,
    ):
        dt_cls = type(np.dtype(scalar_dt))
        _add_allowed_global(allowed, dt_cls)
        _add_allowed_global(allowed, scalar_dt)

    np_trusted = _numpy_safe_callables()
    for item in extra_globals:
        if item is None:
            continue
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], str):
            target_obj, alias_name = item
            if target_obj not in np_trusted:
                _assert_inert_data_class(target_obj, alias_name)
            _add_allowed_global(allowed, target_obj, alias_name)
        else:
            qualname = getattr(item, "__qualname__", getattr(item, "__name__", ""))
            mod_name = getattr(item, "__module__", "")
            label = f"{mod_name}.{qualname}" if mod_name else str(qualname or item)
            if item not in np_trusted:
                _assert_inert_data_class(item, label)
            _add_allowed_global(allowed, item)
            if qualname:
                _add_allowed_global(allowed, item, f"__main__.{qualname}")
    return allowed


def _allowed_reduce_global_names() -> set[str]:
    names: set[str] = {
        "collections.OrderedDict",
        "collections.defaultdict",
        "collections.Counter",
        "_codecs.encode",
        "codecs.encode",
        "builtins.slice",
        "__builtin__.slice",
        "numpy.dtype",
        "numpy.ndarray",
        "numpy.core.multiarray._reconstruct",
        "numpy._core.multiarray._reconstruct",
        "numpy.core.multiarray.scalar",
        "numpy._core.multiarray.scalar",
    }
    for path_name in _SAFE_PATHLIB_NAMES:
        names.add(f"pathlib.{path_name}")
        names.add(f"pathlib._local.{path_name}")
    wo_mod = getattr(torch, "_weights_only_unpickler", None)
    get_allowed = getattr(wo_mod, "_get_allowed_globals", None)
    if callable(get_allowed):
        try:
            names.update(str(k) for k in get_allowed().keys())
        except Exception:
            pass
    for dt_item in _numpy_safe_callables():
        mod = getattr(dt_item, "__module__", "")
        qn = getattr(dt_item, "__qualname__", getattr(dt_item, "__name__", ""))
        if mod and qn:
            names.add(f"{mod}.{qn}")
    return names


def _assert_safe_pickle_reference(mod_name: str, name: str) -> None:
    mod_clean = str(mod_name or "").strip()
    name_clean = str(name or "").strip()
    root_mod = mod_clean.split(".", 1)[0]
    if (
        root_mod in _BLOCKED_PICKLE_ROOT_MODULES
        or mod_clean in _BLOCKED_PICKLE_ROOT_MODULES
    ):
        raise pickle.UnpicklingError(
            f"Blocked dangerous pickle module reference '{mod_clean}.{name_clean}'."
        )
    if (
        mod_clean in {"builtins", "__builtin__"}
        and name_clean not in _SAFE_BUILTIN_NAMES
    ):
        raise pickle.UnpicklingError(
            f"Blocked non-primitive builtin pickle callable '{mod_clean}.{name_clean}'."
        )
    if (
        mod_clean in {"pathlib", "pathlib._local"}
        and name_clean not in _SAFE_PATHLIB_NAMES
    ):
        raise pickle.UnpicklingError(
            f"Blocked unsafe pathlib callable '{mod_clean}.{name_clean}'."
        )
    if (mod_clean, name_clean) in _BLOCKED_SPECIFIC_CALLABLES:
        raise pickle.UnpicklingError(
            f"Blocked unsafe pickle gadget '{mod_clean}.{name_clean}'."
        )


def _inspect_pickle_bytes_for_inert_classes(
    data: bytes, allowed_reduce: set[str]
) -> list[tuple[str, str]]:
    """Disassemble pickle bytes without executing code; return (mod, name) used in NEWOBJ."""
    stack: list[tuple[str, Any]] = []
    metastack: list[list[tuple[str, Any]]] = []
    memo: dict[Any, tuple[str, Any]] = {}
    newobj_targets: list[tuple[str, str]] = []

    for opcode, arg, _pos in pickletools.genops(io.BytesIO(data)):
        op = opcode.name
        if op == "MARK":
            metastack.append(stack)
            stack = []
        elif op in {
            "SHORT_BINUNICODE",
            "BINUNICODE",
            "UNICODE",
            "SHORT_BINSTRING",
            "BINSTRING",
            "STRING",
        }:
            stack.append(("str", str(arg)))
        elif op == "NONE":
            stack.append(("none", None))
        elif op in {"GLOBAL", "INST"} and isinstance(arg, str):
            parts = arg.split(" ", 1)
            if len(parts) != 2:
                raise pickle.UnpicklingError(f"Malformed GLOBAL operand: {arg!r}")
            _assert_safe_pickle_reference(parts[0], parts[1])
            if op == "INST":
                if metastack:
                    stack = metastack.pop()
                raise pickle.UnpicklingError(
                    f"Legacy INST opcode for '{parts[0]}.{parts[1]}' is disallowed."
                )
            stack.append(("global", (parts[0], parts[1])))
        elif op == "STACK_GLOBAL":
            if len(stack) < 2:
                raise pickle.UnpicklingError("Malformed STACK_GLOBAL with empty stack.")
            name_tok = stack.pop()
            mod_tok = stack.pop()
            if mod_tok[0] != "str" or name_tok[0] != "str":
                raise pickle.UnpicklingError(
                    "Dynamic non-literal STACK_GLOBAL is disallowed."
                )
            _assert_safe_pickle_reference(mod_tok[1], name_tok[1])
            stack.append(("global", (mod_tok[1], name_tok[1])))
        elif op in {"EMPTY_TUPLE", "EMPTY_LIST", "EMPTY_DICT", "EMPTY_SET"}:
            stack.append(("container", ()))
        elif op == "TUPLE":
            items = tuple(stack)
            stack = metastack.pop() if metastack else []
            stack.append(("tuple", items))
        elif op == "TUPLE1":
            a = stack.pop() if stack else ("other", None)
            stack.append(("tuple", (a,)))
        elif op == "TUPLE2":
            b = stack.pop() if stack else ("other", None)
            a = stack.pop() if stack else ("other", None)
            stack.append(("tuple", (a, b)))
        elif op == "TUPLE3":
            c = stack.pop() if stack else ("other", None)
            b = stack.pop() if stack else ("other", None)
            a = stack.pop() if stack else ("other", None)
            stack.append(("tuple", (a, b, c)))
        elif op in {"LIST", "DICT", "FROZENSET"}:
            stack = metastack.pop() if metastack else []
            stack.append(("container", None))
        elif op in {"APPENDS", "SETITEMS", "ADDITEMS"}:
            stack = metastack.pop() if metastack else [("container", None)]
        elif op in {"APPEND", "SETITEM"}:
            if op == "SETITEM" and len(stack) >= 2:
                stack.pop()
                stack.pop()
            elif op == "APPEND" and stack:
                stack.pop()
        elif op == "POP":
            if stack:
                stack.pop()
        elif op == "POP_MARK":
            stack = metastack.pop() if metastack else []
        elif op == "DUP":
            if stack:
                stack.append(stack[-1])
        elif op in {"BINPUT", "LONG_BINPUT", "PUT"}:
            if stack:
                memo[arg] = stack[-1]
        elif op == "MEMOIZE":
            if stack:
                memo[len(memo)] = stack[-1]
        elif op in {"BINGET", "LONG_BINGET", "GET"}:
            stack.append(memo.get(arg, ("other", None)))
        elif op in {"NEWOBJ", "NEWOBJ_EX"}:
            if op == "NEWOBJ_EX" and stack:
                stack.pop()
            if stack:
                stack.pop()
            cls_tok = stack.pop() if stack else ("other", None)
            if cls_tok[0] == "global":
                mod_s, name_s = cls_tok[1]
                if (mod_s, name_s) not in newobj_targets:
                    newobj_targets.append((mod_s, name_s))
            stack.append(("instance", None))
        elif op == "REDUCE":
            args_tok = stack.pop() if stack else ("other", None)
            fn_tok = stack.pop() if stack else ("other", None)
            if fn_tok[0] != "global":
                raise pickle.UnpicklingError(
                    "Disallowed non-global REDUCE target in checkpoint."
                )
            fn_mod, fn_name = fn_tok[1]
            full_fn = f"{fn_mod}.{fn_name}"
            if full_fn in {"copyreg._reconstructor", "copy_reg._reconstructor"}:
                tup = args_tok[1] if args_tok[0] == "tuple" else ()
                if (
                    len(tup) == 3
                    and tup[0][0] == "global"
                    and tup[1]
                    in {
                        ("global", ("builtins", "object")),
                        ("global", ("__builtin__", "object")),
                    }
                    and tup[2][0] == "none"
                ):
                    c_mod, c_name = tup[0][1]
                    if (c_mod, c_name) not in newobj_targets:
                        newobj_targets.append((c_mod, c_name))
                    stack.append(("instance", None))
                    continue
                raise pickle.UnpicklingError(
                    "Disallowed non-object copyreg._reconstructor call in checkpoint."
                )
            if full_fn not in allowed_reduce:
                raise pickle.UnpicklingError(
                    f"Blocked arbitrary REDUCE callable '{full_fn}' in checkpoint."
                )
            stack.append(("reduced", full_fn))
        elif op in {"PROTO", "FRAME"}:
            continue
        elif op == "BINPERSID":
            if stack:
                stack.pop()
            stack.append(("storage", None))
        elif op == "PERSID":
            stack.append(("storage", None))
        elif op == "BUILD":
            if stack:
                stack.pop()
        elif op == "STOP":
            break
        else:
            for _ in getattr(opcode, "stack_before", ()):
                if stack:
                    stack.pop()
            for _ in getattr(opcode, "stack_after", (("other",),)):
                stack.append(("other", None))

    return newobj_targets


def _make_inert_metadata_stub(mod_name: str, qualname: str) -> type:
    short_name = qualname.split(".")[-1] or "SafeCheckpointMetadata"
    full_label = f"{mod_name}.{qualname}"
    return type(
        short_name,
        (SafeCheckpointMetadata,),
        {"__module__": mod_name, "__orig_qualname__": full_label},
    )


def _resolve_trusted_inert_globals(checkpoint_path: Path) -> list[tuple[Any, str]]:
    allowed_reduce = _allowed_reduce_global_names()
    discovered: list[tuple[str, str]] = []

    if zipfile.is_zipfile(checkpoint_path):
        with zipfile.ZipFile(checkpoint_path, "r") as zf:
            pkl_members = [m for m in zf.namelist() if m.endswith(".pkl")]
            if not pkl_members:
                raise pickle.UnpicklingError(
                    f"Archive checkpoint '{checkpoint_path}' contains no .pkl stream."
                )
            for member in pkl_members:
                for target in _inspect_pickle_bytes_for_inert_classes(
                    zf.read(member), allowed_reduce
                ):
                    if target not in discovered:
                        discovered.append(target)
    else:
        raw_bytes = checkpoint_path.read_bytes()
        for target in _inspect_pickle_bytes_for_inert_classes(
            raw_bytes, allowed_reduce
        ):
            if target not in discovered:
                discovered.append(target)

    resolved_pairs: list[tuple[Any, str]] = []
    _missing = object()
    for mod_name, qualname in discovered:
        full_name = f"{mod_name}.{qualname}"
        if full_name in allowed_reduce:
            continue
        mod_obj = sys.modules.get(mod_name)
        cur: Any = mod_obj if mod_obj is not None else _missing
        if cur is not _missing:
            for attr_part in qualname.split("."):
                cur = getattr(cur, attr_part, _missing)
                if cur is _missing:
                    break
        if cur is not _missing:
            _assert_inert_data_class(cur, full_name)
            resolved_pairs.append((cur, full_name))
        else:
            stub_cls = _make_inert_metadata_stub(mod_name, qualname)
            resolved_pairs.append((stub_cls, full_name))
    return resolved_pairs


def _verify_checkpoint_file_and_sha256(
    checkpoint_path: Path, expected_sha256: str | None = None
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint path is not a regular file: {checkpoint_path}")
    if expected_sha256 is not None:
        expected = expected_sha256.strip().lower().removeprefix("sha256:")
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
            raise ValueError(
                f"Invalid expected_sha256 digest '{expected_sha256}'; expected 64 hex chars."
            )
        hasher = hashlib.sha256()
        with checkpoint_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
        actual = hasher.hexdigest().lower()
        if not hmac.compare_digest(actual, expected):
            raise ValueError(
                f"SHA-256 mismatch for checkpoint '{checkpoint_path}': "
                f"expected {expected}, got {actual}."
            )


def safe_torch_load(
    path: str | Path,
    *,
    map_location: Any = "cpu",
    extra_globals: Iterable[Any] = (),
    trust_checkpoint: bool = False,
    expected_sha256: str | None = None,
) -> Any:
    """Safely load a PyTorch checkpoint strictly under ``weights_only=True``.

    - Never disables ``weights_only=True`` and never imports unimported modules.
    - By default allows only tensors, primitives, standard containers/NumPy/pathlib
      types, and inert data classes explicitly listed in ``extra_globals``.
    - When ``trust_checkpoint=True``, disassembles the pickle stream without executing
      code, blocks all non-tensor/non-container ``REDUCE`` callables and any class with
      custom ``__new__`` / ``__setstate__`` / ``__setattr__`` / ``__getattribute__`` hooks,
      resolves already-imported inert data classes, and materializes unimported external
      config classes into inert ``SafeCheckpointMetadata`` containers under ``weights_only=True``.
    """
    checkpoint_path = Path(path).expanduser()
    _verify_checkpoint_file_and_sha256(checkpoint_path, expected_sha256=expected_sha256)

    safe_ctx = getattr(torch.serialization, "safe_globals", None)
    if not callable(safe_ctx):
        raise RuntimeError(
            "PyTorch installation lacks torch.serialization.safe_globals required for safe loading."
        )

    dynamic_trusted: list[tuple[Any, str]] = []
    if trust_checkpoint:
        dynamic_trusted = _resolve_trusted_inert_globals(checkpoint_path)
        _logger.info(
            "Loading checkpoint '%s' with trust_checkpoint=True under strict weights_only=True "
            "(inert metadata classes=%d).",
            checkpoint_path,
            len(dynamic_trusted),
        )

    try:
        combined_globals = (*extra_globals, *dynamic_trusted)
        with safe_ctx(_build_safe_globals(combined_globals)):
            return torch.load(
                checkpoint_path, map_location=map_location, weights_only=True
            )
    except pickle.UnpicklingError as exc:
        unsafe_names: list[str] = []
        inspector = getattr(
            torch.serialization, "get_unsafe_globals_in_checkpoint", None
        )
        if callable(inspector):
            try:
                unsafe_names = list(inspector(checkpoint_path))
            except Exception:
                unsafe_names = []
        blocked_detail = (
            f" Disallowed globals detected: {unsafe_names}." if unsafe_names else ""
        )
        raise pickle.UnpicklingError(
            f"Safe checkpoint load blocked untrusted pickled objects in '{checkpoint_path}'.{blocked_detail} "
            f"Pass inert config classes via extra_globals=(...) or set trust_checkpoint=True "
            f"for inert external metadata classes. Original error: {exc}"
        ) from exc
