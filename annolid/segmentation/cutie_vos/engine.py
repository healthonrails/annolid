# cutie_engine.py
# ruff: noqa: E402

from annolid.segmentation.cutie_vos.dependencies import (
    ensure_cutie_runtime_dependencies,
)

ensure_cutie_runtime_dependencies()

import os
import shutil
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Callable, Dict, Optional, Iterator, Tuple, Union

from omegaconf import open_dict
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

# Assuming these are from the original Cutie VOS repository or your adaptation
from annolid.segmentation.cutie_vos.model.cutie import CUTIE
from annolid.segmentation.cutie_vos.model.weight_compression import (
    compress_and_factorize_cutie_in_place,
    is_compact_cutie_checkpoint,
    pack_compact_cutie_checkpoint,
)
from annolid.segmentation.cutie_vos.inference.inference_core import InferenceCore
from annolid.segmentation.cutie_vos.interactive_utils import (
    image_to_torch,
    resize_frame_for_inference,
    torch_prob_to_numpy_mask,
)

from annolid.utils.logger import logger
from annolid.utils.devices import get_device  # For device selection
from annolid.utils.model_assets import (
    ensure_cached_model_asset,
    model_downloads_dir,
    resolve_existing_model_path,
)

DEFAULT_CUTIE_WEIGHTS_FILENAME = "cutie-base-mega.pth"
ORIG_BACKUP_CUTIE_WEIGHTS_FILENAME = "cutie-base-mega.orig.pth"
SAM3_BODYPARTS_CUTIE_WEIGHTS_FILENAME = "cutie-sam3-bodyparts.pth"
SAM3_BODYPARTS_CUTIE_RELEASE_URL = (
    "https://github.com/healthonrails/annolid/releases/download/v1.6.10/"
    f"{SAM3_BODYPARTS_CUTIE_WEIGHTS_FILENAME}"
)
SAM3_BODYPARTS_CUTIE_SHA256 = (
    "9327fe2eec8b7889b91a96c66dc23c8e40600fbf66e3e1ee41243e94f910299d"
)
UPGRADED_WEIGHTS_URL_ENV = "ANNOLID_CUTIE_UPGRADED_WEIGHTS_URL"
UPGRADED_WEIGHTS_SHA256_ENV = "ANNOLID_CUTIE_UPGRADED_WEIGHTS_SHA256"
_PACKAGE_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
_ORIGINAL_CUTIE_WEIGHT_FILENAMES = {
    DEFAULT_CUTIE_WEIGHTS_FILENAME,
    ORIG_BACKUP_CUTIE_WEIGHTS_FILENAME,
}
_MANAGED_CUTIE_WEIGHT_FILENAMES = {
    SAM3_BODYPARTS_CUTIE_WEIGHTS_FILENAME,
}
_DOWNLOADABLE_CUTIE_WEIGHT_FILENAMES = (
    _ORIGINAL_CUTIE_WEIGHT_FILENAMES | _MANAGED_CUTIE_WEIGHT_FILENAMES
)
_LEGACY_PLACEHOLDER_WEIGHT_NAMES = {"cutie.pt"}


def resolve_cutie_weights_path(
    weights_path: Optional[Union[str, Path]] = None,
    *,
    weights_dir: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """Resolve an explicit Cutie checkpoint path, or return None for default weights.

    Relative filenames (such as ``cutie-sam3-bodyparts.pth``) are resolved against
    ``weights_dir`` (defaulting to ``annolid/segmentation/cutie_vos/weights``) and
    Annolid's standard model asset search roots.
    """
    raw = str(weights_path or "").strip()
    if not raw or raw.lower() in _LEGACY_PLACEHOLDER_WEIGHT_NAMES:
        return None

    base_weights_dir = (
        Path(weights_dir).expanduser()
        if weights_dir is not None
        else Path(__file__).resolve().parent / "weights"
    )
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return candidate

    if not candidate.is_absolute():
        in_weights_dir = base_weights_dir / candidate
        if in_weights_dir.is_file():
            return in_weights_dir
        by_name = base_weights_dir / candidate.name
        if by_name.is_file():
            return by_name

    resolved = resolve_existing_model_path(raw)
    if resolved is not None and resolved.is_file():
        return resolved

    # Return explicit target path so callers raise a clear FileNotFoundError
    if not candidate.is_absolute() and len(candidate.parts) == 1:
        return base_weights_dir / candidate.name
    return candidate


def _build_safe_cutie_checkpoint_globals() -> list:
    import argparse
    from collections import OrderedDict, defaultdict
    import pathlib
    from typing import Any

    allowed: list = [
        dict,
        list,
        tuple,
        set,
        frozenset,
        int,
        float,
        str,
        bool,
        bytes,
        slice,
        argparse.Namespace,
        defaultdict,
        OrderedDict,
        pathlib.Path,
        pathlib.PosixPath,
        pathlib.WindowsPath,
        Any,
    ]
    for np_attr in ("ndarray", "dtype"):
        obj = getattr(np, np_attr, None)
        if obj is not None and obj not in allowed:
            allowed.append(obj)
    np_dtypes_mod = getattr(np, "dtypes", None)
    if np_dtypes_mod is not None:
        for dt_cls in vars(np_dtypes_mod).values():
            if (
                isinstance(dt_cls, type)
                and issubclass(dt_cls, np.dtype)
                and dt_cls not in allowed
            ):
                allowed.append(dt_cls)
    for mod_name in ("numpy._core.multiarray", "numpy.core.multiarray"):
        try:
            mod = __import__(mod_name, fromlist=["scalar", "_reconstruct"])
            for sym in ("scalar", "_reconstruct"):
                fn = getattr(mod, sym, None)
                if fn is not None and fn not in allowed:
                    allowed.append(fn)
        except Exception:
            pass
    try:
        from omegaconf import DictConfig, ListConfig
        from omegaconf.base import Box, ContainerMetadata, Metadata
        from omegaconf.nodes import (
            AnyNode,
            BooleanNode,
            FloatNode,
            IntegerNode,
            StringNode,
            ValueNode,
        )

        for cls in (
            DictConfig,
            ListConfig,
            Box,
            ContainerMetadata,
            Metadata,
            AnyNode,
            ValueNode,
            BooleanNode,
            FloatNode,
            IntegerNode,
            StringNode,
        ):
            if cls not in allowed:
                allowed.append(cls)
    except Exception:
        pass
    return allowed


def safe_load_cutie_checkpoint(
    weights_path: Union[str, Path],
    *,
    map_location: Union[torch.device, str, None] = "cpu",
) -> Dict:
    """Load a Cutie checkpoint with ``weights_only=True`` enforced at all times.

    Retries once under ``torch.serialization.safe_globals(...)`` for inert training
    metadata types (``OmegaConf``, ``argparse.Namespace``, ``pathlib.Path``, ``numpy``
    scalars) used by third-party/Hydra Cutie checkpoints while keeping arbitrary code
    execution strictly blocked.
    """
    import pickle

    try:
        return torch.load(weights_path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError:
        safe_ctx = getattr(torch.serialization, "safe_globals", None)
        if not callable(safe_ctx):
            raise
        with safe_ctx(_build_safe_cutie_checkpoint_globals()):
            return torch.load(
                weights_path,
                map_location=map_location,
                weights_only=True,
            )


def unwrap_cutie_state_dict(loaded_weights: Dict) -> Dict:
    """Extract the CUTIE parameter state dict from flat or nested checkpoints,
    preserving self-describing compressed checkpoint envelopes and stripping DDP prefixes."""
    if not isinstance(loaded_weights, dict):
        return loaded_weights
    if "__annolid_cutie_format__" in loaded_weights:
        return loaded_weights
    payload = loaded_weights
    for key in ("model", "weights", "state_dict"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            payload = nested
            break
    if payload and all(
        isinstance(k, str) and k.startswith("module.") for k in payload.keys()
    ):
        payload = {k[len("module.") :]: v for k, v in payload.items()}
    return payload


def _atomic_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp.{os.getpid()}")
    try:
        shutil.copy2(src, tmp)
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_torch_save(payload: object, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp)
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)


def _is_bare_managed_cutie_weight_spec(raw_spec: Optional[Union[str, Path]]) -> bool:
    raw = str(raw_spec or "").strip()
    if not raw:
        return False
    p = Path(raw).expanduser()
    return (
        (not p.is_absolute())
        and len(p.parts) == 1
        and p.name in _DOWNLOADABLE_CUTIE_WEIGHT_FILENAMES
    )


def describe_cutie_checkpoint_status(
    weights_path: Optional[Union[str, Path]] = None,
    *,
    weights_dir: Optional[Union[str, Path]] = None,
) -> Tuple[bool, str]:
    """Return a fast, non-blocking ``(is_valid, status_text)`` summary for GUI badges."""
    raw = str(weights_path or "").strip()
    base_dir = (
        Path(weights_dir).expanduser()
        if weights_dir is not None
        else _PACKAGE_WEIGHTS_DIR
    )
    dl_dir = model_downloads_dir()

    def _find_managed(fname: str) -> Optional[Path]:
        for cand in (base_dir / fname, dl_dir / fname):
            try:
                if cand.is_file():
                    return cand
            except OSError:
                continue
        return None

    if (
        not raw
        or raw.lower() in _LEGACY_PLACEHOLDER_WEIGHT_NAMES
        or raw in _ORIGINAL_CUTIE_WEIGHT_FILENAMES
    ):
        preferred = (
            ORIG_BACKUP_CUTIE_WEIGHTS_FILENAME
            if raw == ORIG_BACKUP_CUTIE_WEIGHTS_FILENAME
            else DEFAULT_CUTIE_WEIGHTS_FILENAME
        )
        fallback = (
            DEFAULT_CUTIE_WEIGHTS_FILENAME
            if preferred == ORIG_BACKUP_CUTIE_WEIGHTS_FILENAME
            else ORIG_BACKUP_CUTIE_WEIGHTS_FILENAME
        )
        found = _find_managed(preferred) or _find_managed(fallback)
        if found is not None:
            size_mb = found.stat().st_size / (1024.0 * 1024.0)
            return (
                True,
                f"Ready locally ({size_mb:.1f} MB — Original Upstream Cutie Base Mega, 35.0M params)",
            )
        return (
            True,
            "Auto-downloads on first run (~133.9 MB — Original Upstream Cutie Base Mega)",
        )

    if raw == SAM3_BODYPARTS_CUTIE_WEIGHTS_FILENAME:
        found = _find_managed(SAM3_BODYPARTS_CUTIE_WEIGHTS_FILENAME)
        if found is not None:
            size_mb = found.stat().st_size / (1024.0 * 1024.0)
            if size_mb < 60.0:
                return (
                    True,
                    f"Ready locally ({size_mb:.1f} MB — Improved Compact SAM3 + DINOv3, 15.8M params)",
                )
            return (
                True,
                f"Ready locally ({size_mb:.1f} MB — will auto-compress to 30.3 MB on first load)",
            )
        return (
            True,
            "Auto-downloads on first run (~30.3 MB — Improved Compact SAM3 + DINOv3)",
        )

    resolved = resolve_cutie_weights_path(raw, weights_dir=base_dir)
    if resolved is not None and resolved.is_file():
        size_mb = resolved.stat().st_size / (1024.0 * 1024.0)
        return (True, f"Ready locally ({size_mb:.1f} MB — Custom: {resolved.name})")
    return (False, f"Custom checkpoint not found: {resolved or raw}")


def _is_bundle_or_readonly_dir(path: Path) -> bool:
    """Return True when ``path`` lives inside a frozen app bundle or non-writable tree."""
    if getattr(sys, "frozen", False):
        return True
    parts = set(path.parts)
    if "_internal" in parts or any(part.endswith(".app") for part in parts):
        return True
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return not os.access(probe, os.W_OK)


def _resolve_writable_cutie_cache_dir(weights_dir: Path) -> Path:
    if _is_bundle_or_readonly_dir(weights_dir):
        return model_downloads_dir()
    return weights_dir


def ensure_managed_cutie_weights_file(
    model_weights_path: Optional[Union[str, Path]],
    *,
    weights_dir: Path,
    remote_url: str,
    expected_md5: str,
    asset_downloader: Callable[..., Path] = ensure_cached_model_asset,
    try_release_asset: Optional[bool] = None,
) -> Path:
    """Resolve and materialize default or managed Cutie checkpoints on fresh installs.

    Resolution order for missing managed checkpoints (``cutie-base-mega.pth`` and
    ``cutie-sam3-bodyparts.pth``):
    1. Local ``weights_dir`` (or ``~/.annolid/workspace/downloads`` when running from a
       package/frozen install).
    2. Explicit ``ANNOLID_CUTIE_UPGRADED_WEIGHTS_URL`` (+ SHA-256) override.
    3. Canonical GitHub Release asset (``SAM3_BODYPARTS_CUTIE_RELEASE_URL`` verified
       against ``SAM3_BODYPARTS_CUTIE_SHA256``) when ``try_release_asset`` is enabled.
    4. Upstream ``cutie-base-mega.pth`` (MD5-verified) followed by automatic in-place
       HOSVD + Conv+BN folding compression in ``maybe_upgrade_and_cache_compact_cutie_weights``.
    """
    explicit = resolve_cutie_weights_path(model_weights_path, weights_dir=weights_dir)
    if explicit is not None:
        if explicit.exists():
            return explicit
        if not _is_bare_managed_cutie_weight_spec(model_weights_path):
            raise FileNotFoundError(f"Cutie model weights not found at: {explicit}")
        target_name = explicit.name
    else:
        default_in_weights = weights_dir / DEFAULT_CUTIE_WEIGHTS_FILENAME
        if default_in_weights.exists():
            return default_in_weights
        target_name = DEFAULT_CUTIE_WEIGHTS_FILENAME

    is_default_pkg_dir = False
    try:
        is_default_pkg_dir = weights_dir.resolve() == _PACKAGE_WEIGHTS_DIR.resolve()
    except OSError:
        is_default_pkg_dir = False

    if is_default_pkg_dir or _is_bundle_or_readonly_dir(weights_dir):
        user_cached = model_downloads_dir() / target_name
        if user_cached.is_file():
            return user_cached

    cache_dir = _resolve_writable_cutie_cache_dir(weights_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / target_name
    if target.is_file():
        return target

    if target.name in _ORIGINAL_CUTIE_WEIGHT_FILENAMES:
        sibling_orig_name = (
            DEFAULT_CUTIE_WEIGHTS_FILENAME
            if target.name == ORIG_BACKUP_CUTIE_WEIGHTS_FILENAME
            else ORIG_BACKUP_CUTIE_WEIGHTS_FILENAME
        )
        for search_dir in (weights_dir, cache_dir):
            sibling_orig = search_dir / sibling_orig_name
            try:
                if (
                    sibling_orig.is_file()
                    and sibling_orig.stat().st_size >= 80 * 1024 * 1024
                ):
                    _atomic_copy_file(sibling_orig, target)
                    return target
            except OSError:
                pass
        downloaded_orig = asset_downloader(
            file_name=target.name,
            url=remote_url,
            expected_md5=expected_md5,
            cache_dir=cache_dir,
        )
        return Path(downloaded_orig) if downloaded_orig is not None else target

    upgraded_url = str(os.environ.get(UPGRADED_WEIGHTS_URL_ENV, "")).strip()
    upgraded_sha = str(os.environ.get(UPGRADED_WEIGHTS_SHA256_ENV, "")).strip()
    if upgraded_url and upgraded_sha:
        return asset_downloader(
            file_name=target.name,
            url=upgraded_url,
            expected_sha256=upgraded_sha,
            cache_dir=cache_dir,
        )

    if try_release_asset is None:
        try_release_asset = asset_downloader is ensure_cached_model_asset

    if (
        try_release_asset
        and SAM3_BODYPARTS_CUTIE_RELEASE_URL
        and SAM3_BODYPARTS_CUTIE_SHA256
    ):
        try:
            downloaded_release = asset_downloader(
                file_name=target.name,
                url=SAM3_BODYPARTS_CUTIE_RELEASE_URL,
                expected_sha256=SAM3_BODYPARTS_CUTIE_SHA256,
                cache_dir=cache_dir,
            )
            if downloaded_release is not None and Path(downloaded_release).is_file():
                return Path(downloaded_release)
        except Exception as exc:
            logger.debug(
                "Pre-distilled Cutie release checkpoint unavailable at %s (%s); "
                "falling back to upstream Cutie + local structural compression.",
                SAM3_BODYPARTS_CUTIE_RELEASE_URL,
                exc,
            )

    base_path = cache_dir / DEFAULT_CUTIE_WEIGHTS_FILENAME
    if not base_path.exists():
        downloaded = asset_downloader(
            file_name=base_path.name,
            url=remote_url,
            expected_md5=expected_md5,
            cache_dir=cache_dir,
        )
        if downloaded is not None:
            base_path = Path(downloaded)

    if target.name != base_path.name and base_path.is_file() and not target.exists():
        try:
            _atomic_copy_file(base_path, target)
        except OSError:
            return base_path
    return target


def maybe_upgrade_and_cache_compact_cutie_weights(
    model: object,
    loaded_payload: object,
    *,
    model_path: Union[str, Path],
    weights_dir: Optional[Union[str, Path]] = None,
) -> None:
    """When a fresh install loads a dense managed Cutie checkpoint, compress it in-place
    (48 Conv+BN folded + HOSVD Tucker-2 + FFN pruning) and atomically cache the compact
    checkpoint while preserving ``cutie-base-mega.orig.pth``."""
    if not isinstance(model, CUTIE):
        return
    if not isinstance(loaded_payload, dict) or is_compact_cutie_checkpoint(
        loaded_payload
    ):
        return
    if "pixel_encoder.conv1.weight" not in loaded_payload:
        return

    resolved_path = Path(model_path)
    if resolved_path.name not in _MANAGED_CUTIE_WEIGHT_FILENAMES:
        return
    if not resolved_path.is_file():
        return

    raw_dir = Path(weights_dir) if weights_dir is not None else resolved_path.parent
    base_dir = _resolve_writable_cutie_cache_dir(raw_dir)
    orig_backup = base_dir / ORIG_BACKUP_CUTIE_WEIGHTS_FILENAME
    try:
        if not orig_backup.exists():
            _atomic_copy_file(resolved_path, orig_backup)
    except OSError:
        logger.debug(
            "Skipping upstream Cutie backup write in read-only directory: %s", base_dir
        )

    tucker_cfg = compress_and_factorize_cutie_in_place(model)
    compact_payload = pack_compact_cutie_checkpoint(
        model,
        tucker_cfg=tucker_cfg,
        metadata={"auto_bootstrapped_from_upstream": True},
    )
    model.load_weights(compact_payload)

    for fname in _MANAGED_CUTIE_WEIGHT_FILENAMES:
        dest = base_dir / fname
        if dest == resolved_path or not dest.exists():
            try:
                _atomic_torch_save(compact_payload, dest)
            except OSError:
                logger.debug(
                    "Skipping compact Cutie cache write in read-only directory: %s",
                    dest,
                )
        elif dest.is_file() and dest.stat().st_size > 80 * 1024 * 1024:
            try:
                _atomic_torch_save(compact_payload, dest)
            except OSError:
                pass


class CutieEngine:
    """
    A core processing engine for the Cutie VOS model.
    It loads the model, manages the InferenceCore, and processes video frames
    to yield predicted masks for a given segment.
    It does not handle file I/O for annotations or initial mask creation from shapes.
    """

    _REMOTE_MODEL_URL = (
        "https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth"
    )
    _MD5 = "a6071de6136982e396851903ab4c083a"
    _MODEL_FILENAME = DEFAULT_CUTIE_WEIGHTS_FILENAME

    def __init__(
        self,
        # e.g., mem_every, t_max_value
        cutie_config_overrides: Optional[Dict] = None,
        device: Optional[Union[torch.device, str]] = None,
        model_weights_path: Optional[Union[str, Path]] = None,
    ):
        self.device = device or get_device()
        logger.info(f"CutieEngine: Initializing on device: {self.device}")

        self.cfg = self._load_hydra_config(cutie_config_overrides)
        weights_dir = Path(__file__).resolve().parent / "weights"
        self.model_weights_path = str(
            ensure_managed_cutie_weights_file(
                model_weights_path,
                weights_dir=weights_dir,
                remote_url=self._REMOTE_MODEL_URL,
                expected_md5=self._MD5,
                asset_downloader=ensure_cached_model_asset,
            )
        )

        self.cutie_model: Optional[CUTIE] = None
        self.inference_core: Optional[InferenceCore] = None

        try:
            self.cutie_model = self._load_cutie_model()
            self.inference_core = InferenceCore(self.cutie_model, cfg=self.cfg)
            logger.info(
                "CutieEngine: Model and InferenceCore initialized successfully."
            )
        except Exception as e:
            logger.error(
                f"CutieEngine: Failed to initialize model or InferenceCore: {e}",
                exc_info=True,
            )
            # Propagate error or handle gracefully
            raise

    def _device_type(self) -> str:
        """Return device type ('cuda', 'mps', 'cpu') for autocast/gating."""
        if isinstance(self.device, torch.device):
            return self.device.type
        device_str = str(self.device).lower()
        if "cuda" in device_str:
            return "cuda"
        if "mps" in device_str:
            return "mps"
        return "cpu"

    def reset_inference_core(self) -> None:
        """Reset inference state (objects + memory) while keeping model weights loaded."""
        if self.cutie_model is None:
            raise RuntimeError(
                "CutieEngine reset requested before model initialization."
            )
        self.inference_core = InferenceCore(self.cutie_model, cfg=self.cfg)

    def _get_model_weights_path(self) -> str:
        """Determines and potentially downloads the model weights."""
        # Store weights in a known location, e.g., user's cache or package data
        package_dir = Path(__file__).resolve().parent
        # Example: annolid/segmentation/cutie_vos/weights/
        weights_dir = package_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        model_path = weights_dir / self._MODEL_FILENAME

        if not model_path.exists():
            logger.info(
                f"Cutie model weights not found at {model_path}. Downloading..."
            )
            try:
                ensure_cached_model_asset(
                    file_name=model_path.name,
                    url=self._REMOTE_MODEL_URL,
                    expected_md5=self._MD5,
                    cache_dir=weights_dir,
                )
                logger.info(f"Cutie model downloaded to {model_path}")
            except Exception as e:
                logger.error(f"Failed to download Cutie model: {e}", exc_info=True)
                raise FileNotFoundError(
                    f"Cutie model weights could not be downloaded to {model_path}"
                )
        return str(model_path)

    # OmegaConf's DictConfig is often returned
    def _load_hydra_config(self, overrides: Optional[Dict] = None) -> Dict:
        """Loads the base Hydra config for Cutie and applies overrides."""
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        # Define the relative path to your config directory FROM THE LOCATION OF THIS FILE
        # If 'config' directory is in the same directory as 'cutie_engine.py':
        relative_config_path = "config"
        # If 'config' directory is one level up from 'cutie_engine.py':
        # relative_config_path = "../config_parent_dir_name" # Adjust as needed

        try:
            # The config_path for initialize() should be relative.
            # Hydra will search from the location of the calling Python file if an absolute path
            # isn't established through other means (like being inside a Hydra app).
            initialize(
                version_base="1.3.2",
                config_path=relative_config_path,  # Use the relative path string
                job_name="cutie_engine_config",
            )
            # Loads 'eval_config.yaml' from 'config_path'
            cfg = compose(config_name="eval_config")
        except Exception as e:
            logger.error(
                f"Hydra initialization failed. Ensure '{relative_config_path}/eval_config.yaml' exists "
                f"relative to 'cutie_engine.py' or is discoverable by Hydra. Error: {e}",
                exc_info=True,
            )
            raise  # Re-raise the exception to stop execution if config is critical

        _DEFAULT_RUNTIME_FIELDS = {
            "mem_every": 5,
            "use_long_term": False,
            "max_mem_frames": 5,
            "top_k": 30,
            "stagger_updates": 5,
            "chunk_size": -1,
            "kv_storage_dtype": "auto",
            "max_kv_memory_mb": 2048,
            "amp": False,
            "flip_aug": False,
            "save_aux": False,
        }
        with open_dict(cfg):
            for default_key, default_val in _DEFAULT_RUNTIME_FIELDS.items():
                if cfg.get(default_key) is None:
                    cfg[default_key] = default_val

        if overrides:
            with open_dict(cfg):  # Allow modifications
                for key, value in overrides.items():
                    try:
                        # For nested keys, use dot notation: OmegaConf.update(cfg, "data.batch_size", value)
                        # For simple top-level keys:
                        if "." in key:  # Basic check for nested key attempt
                            parts = key.split(".")
                            current_level = cfg
                            for i, part in enumerate(parts[:-1]):
                                if part not in current_level:
                                    logger.warning(
                                        f"Nested override key part '{part}' in '{key}' not found."
                                    )
                                    current_level = None
                                    break
                                current_level = current_level[part]
                            if current_level is not None and parts[-1] in current_level:
                                current_level[parts[-1]] = value
                            elif current_level is not None:  # Key not present at leaf
                                logger.warning(
                                    f"Leaf override key part '{parts[-1]}' in '{key}' not found."
                                )
                            # else: already warned
                        elif key in cfg:
                            cfg[key] = value
                        else:
                            logger.warning(
                                f"Override key '{key}' not found in base Cutie config."
                            )
                    except Exception as e:
                        logger.error(f"Error applying override '{key}={value}': {e}")
        return cfg

    def _load_cutie_model(self) -> CUTIE:
        """Loads the CUTIE PyTorch model and its weights."""
        if not self.model_weights_path or not Path(self.model_weights_path).exists():
            raise FileNotFoundError(
                f"Cutie model weights not found at: {self.model_weights_path}"
            )

        with open_dict(self.cfg):  # Ensure weights path in cfg is correct
            self.cfg["weights"] = self.model_weights_path

        model = CUTIE(self.cfg).to(self.device).eval()

        try:
            logger.debug(f"Loading Cutie weights from: {self.cfg.weights}")
            loaded_weights = safe_load_cutie_checkpoint(
                self.cfg.weights,
                map_location=self.device,
            )

            # Handle potential nesting of weights (e.g. 'model', 'weights', 'state_dict')
            weights_to_load = unwrap_cutie_state_dict(loaded_weights)

            # Assumes CUTIE class has a load_weights method
            model.load_weights(weights_to_load)
            maybe_upgrade_and_cache_compact_cutie_weights(
                model,
                weights_to_load,
                model_path=self.cfg.weights,
            )
            # Or use model.load_state_dict(weights_to_load) directly
            # if load_weights is just a wrapper.
        except Exception as e:
            logger.error(f"Error loading Cutie model weights: {e}", exc_info=True)
            raise
        return model

    def clear_memory(self):
        """Clears the memory of the InferenceCore."""
        if self.inference_core:
            self.inference_core.clear_memory()
            logger.info("CutieEngine: InferenceCore memory cleared.")

    def clear_non_permanent_memory(self):
        """Clears only the non-permanent (working) memory of InferenceCore."""
        if self.inference_core:
            self.inference_core.clear_non_permanent_memory()
            logger.info("CutieEngine: InferenceCore non-permanent memory cleared.")

    def process_frames(
        self,
        video_capture: cv2.VideoCapture,
        start_frame_index: int,  # Actual video frame index to start reading from
        # NumPy array (H, W), integer object IDs, 0 for bg
        initial_mask_np: np.ndarray,
        # Number of distinct objects in initial_mask_np (excluding bg)
        num_objects_in_mask: int,
        # How many frames to process *including* the start_frame_index
        frames_to_propagate: int,
        pred_worker: Optional[object] = None,
        reset_core: bool = True,
    ) -> Iterator[
        Tuple[int, np.ndarray, np.ndarray]
    ]:  # (frame_idx, frame_bgr, pred_mask_np)
        """
        Processes a sequence of frames from the video_capture.

        Args:
            video_capture: Opened cv2.VideoCapture object for the video.
            start_frame_index: The video frame index from which to start processing.
                               The initial_mask_np corresponds to this frame.
            initial_mask_np: The starting mask (H, W, object IDs).
            num_objects_in_mask: Number of objects in the initial mask.
            frames_to_propagate: Total number of frames to process for this segment.
            pred_worker: Optional object with an `is_stopped()` method for early termination.

        Yields:
            Tuple[int, np.ndarray, np.ndarray]: (current_video_frame_index, frame_bgr, predicted_numpy_mask)
                                    The mask is an object ID mask (H, W).
        """
        if not self.cutie_model:
            logger.error("CutieEngine: Model not initialized.")
            return  # Or raise exception

        if not video_capture.isOpened():
            logger.error("CutieEngine: VideoCapture is not open.")
            return

        if reset_core or self.inference_core is None:
            self.reset_inference_core()

        initial_mask_np = np.asarray(initial_mask_np)
        if initial_mask_np.ndim != 2:
            raise ValueError(
                "CutieEngine initial mask must be a 2D indexed mask; "
                f"received shape {initial_mask_np.shape}."
            )
        initial_object_ids = [
            int(value) for value in np.unique(initial_mask_np) if int(value) != 0
        ]
        if not initial_object_ids:
            raise ValueError("CutieEngine initial mask has no foreground objects.")
        if len(initial_object_ids) != int(num_objects_in_mask):
            logger.warning(
                "CutieEngine initial object count mismatch: mask contains %s, caller reported %s. Using mask ids.",
                len(initial_object_ids),
                num_objects_in_mask,
            )
        uses_temporary_ids_as_object_ids = initial_object_ids == list(
            range(1, len(initial_object_ids) + 1)
        )

        try:
            max_internal_size = int(self.cfg.max_internal_size)
        except (AttributeError, TypeError, ValueError):
            max_internal_size = -1

        total_video_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_processed_in_this_call = 0
        current_video_frame_idx = start_frame_index

        with torch.inference_mode():
            with torch.amp.autocast(
                "cuda", enabled=bool(self.cfg.amp) and self._device_type() == "cuda"
            ):
                if current_video_frame_idx < 0:
                    current_video_frame_idx = 0
                if current_video_frame_idx >= total_video_frames:
                    logger.info(
                        "CutieEngine: Start frame %s is beyond video length %s.",
                        current_video_frame_idx,
                        total_video_frames,
                    )
                    return

                if (
                    int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                    != current_video_frame_idx
                ):
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_video_frame_idx)

                while True:
                    if pred_worker and pred_worker.is_stopped():
                        logger.info(
                            f"CutieEngine: Processing stopped by worker at frame {current_video_frame_idx}."
                        )
                        break

                    if frames_processed_in_this_call >= frames_to_propagate:
                        logger.info(
                            f"CutieEngine: Reached 'frames_to_propagate' ({frames_to_propagate}). Segment ended at frame {current_video_frame_idx - 1}."
                        )
                        break

                    if current_video_frame_idx >= total_video_frames:
                        logger.info(
                            f"CutieEngine: Reached end of video ({current_video_frame_idx}) while processing segment."
                        )
                        break

                    ret, frame_bgr = video_capture.read()

                    if not ret or frame_bgr is None:
                        logger.warning(
                            f"CutieEngine: Could not read frame {current_video_frame_idx}. Ending segment."
                        )
                        break

                    frame_rgb = cv2.cvtColor(
                        frame_bgr, cv2.COLOR_BGR2RGB
                    )  # Cutie expects RGB
                    output_size = tuple(frame_rgb.shape[:2])
                    frame_for_inference = resize_frame_for_inference(
                        frame_rgb,
                        max_internal_size,
                    )
                    frame_torch = image_to_torch(
                        frame_for_inference,
                        device=self.device,
                    )

                    if (
                        frames_processed_in_this_call == 0
                    ):  # First frame of this segment processing call
                        if tuple(initial_mask_np.shape) != output_size:
                            raise ValueError(
                                "CutieEngine initial mask size does not match its seed frame: "
                                f"mask={tuple(initial_mask_np.shape)}, frame={output_size}."
                            )
                        mask_torch = torch.from_numpy(
                            np.ascontiguousarray(initial_mask_np)
                        )

                        predicted_probs_torch = self.inference_core.step(
                            frame_torch,
                            mask_torch,
                            objects=initial_object_ids,
                            idx_mask=True,
                            force_permanent=True,
                            return_index_mask=True,
                            output_size=output_size,
                        )
                    else:  # Subsequent frames
                        predicted_probs_torch = self.inference_core.step(
                            frame_torch,
                            return_index_mask=True,
                            output_size=output_size,
                        )

                    if uses_temporary_ids_as_object_ids:
                        predicted_object_ids = predicted_probs_torch
                    else:
                        predicted_object_ids = self.inference_core.output_prob_to_mask(
                            predicted_probs_torch
                        )
                    predicted_mask_np = torch_prob_to_numpy_mask(predicted_object_ids)

                    yield current_video_frame_idx, frame_bgr, predicted_mask_np

                    frames_processed_in_this_call += 1
                    current_video_frame_idx += 1

        logger.info(
            f"CutieEngine: Processed {frames_processed_in_this_call} frames for segment starting at {start_frame_index}."
        )

    def cleanup(self):
        """Releases resources, particularly GPU memory if applicable."""
        logger.debug("CutieEngine: Cleaning up.")
        if hasattr(self, "inference_core") and self.inference_core is not None:
            del self.inference_core
            self.inference_core = None
        if hasattr(self, "cutie_model") and self.cutie_model is not None:
            del self.cutie_model
            self.cutie_model = None

        if self._device_type() == "cuda":
            torch.cuda.empty_cache()
