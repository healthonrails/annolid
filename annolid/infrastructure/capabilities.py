"""Optional runtime capability manager for lean and on-demand installations."""

from __future__ import annotations

import ast
import base64
from collections import deque
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from email.parser import Parser as EmailHeaderParser
import functools
import hashlib
import importlib
from importlib import metadata as importlib_metadata
from importlib import util as importlib_util
import io
import json
import os
from pathlib import Path
import platform
import queue
import re
import shutil
import site
import subprocess
import sys
import sysconfig
import tempfile
import threading
from typing import Callable, Sequence
import zipfile


SpecFinder = Callable[[str], object | None]
OutputCallback = Callable[[str], None]

AUTO_INSTALL_OPTIONAL_DEPS_ENV = "ANNOLID_AUTO_INSTALL_OPTIONAL_DEPS"
NO_GPU_ENV = "ANNOLID_NO_GPU"
TORCH_INDEX_URL_ENV = "ANNOLID_TORCH_INDEX_URL"
WHEELHOUSE_ENV = "ANNOLID_WHEELHOUSE"
OFFLINE_ENV = "ANNOLID_OFFLINE"
USER_RUNTIME_DIR_ENV = "ANNOLID_USER_RUNTIME_DIR"
FROZEN_USER_RUNTIME_ENV = "ANNOLID_FROZEN_USER_RUNTIME"
WHEELHOUSE_MANIFEST_NAME = "wheelhouse-manifest.json"

PYTORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
PYTORCH_CUDA12_INDEX_URL = "https://download.pytorch.org/whl/cu124"
ONNXRUNTIME_CUDA12_EXTRA_INDEX_URL = (
    "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/"
    "onnxruntime-cuda-12/pypi/simple/"
)
INSTALL_DOCS_URL = "https://annolid.com/installation/"


class RuntimeInstallCancelledError(RuntimeError):
    """Raised when an on-demand runtime package installation is cancelled."""


@dataclass(frozen=True)
class RuntimeDependency:
    import_name: str
    package_spec: str
    alternative_imports: tuple[str, ...] = ()
    max_python_exclusive: tuple[int, int] | None = None

    def applies_to_current_python(
        self, version_info: tuple[int, int] | None = None
    ) -> bool:
        if self.max_python_exclusive is None:
            return True
        current = version_info or (sys.version_info.major, sys.version_info.minor)
        return current < self.max_python_exclusive


@dataclass(frozen=True)
class RuntimeCapability:
    name: str
    dependencies: tuple[RuntimeDependency, ...]
    package_specs: tuple[str, ...]
    docs_url: str
    description: str
    title: str = ""
    category: str = "general"
    approx_download_mb: int = 0

    @property
    def display_title(self) -> str:
        return self.title or self.name.replace("_", " ").title()

    @property
    def import_names(self) -> tuple[str, ...]:
        return tuple(dependency.import_name for dependency in self.dependencies)

    @property
    def install_command(self) -> str:
        return "python -m pip install " + " ".join(
            f'"{package}"' for package in self.package_specs
        )


@dataclass(frozen=True)
class CapabilityStatus:
    name: str
    available: bool
    missing_imports: tuple[str, ...]
    install_command: str
    docs_url: str
    description: str
    title: str = ""
    category: str = "general"
    approx_download_mb: int = 0
    missing_package_specs: tuple[str, ...] = ()

    @property
    def state(self) -> str:
        return "available" if self.available else "missing"


CAPABILITIES: dict[str, RuntimeCapability] = {
    "onnx": RuntimeCapability(
        name="onnx",
        title="ONNX Runtime",
        category="inference",
        approx_download_mb=45,
        dependencies=(RuntimeDependency("onnxruntime", "onnxruntime>=1.23.2"),),
        package_specs=("annolid[onnx]",),
        docs_url=INSTALL_DOCS_URL,
        description="ONNX Runtime for interactive SAM, TapNext, and VideoMT inference.",
    ),
    "scientific": RuntimeCapability(
        name="scientific",
        title="Scientific & Clustering Tools",
        category="analysis",
        approx_download_mb=55,
        dependencies=(
            RuntimeDependency("scipy", "scipy>=1.5.2"),
            RuntimeDependency("sklearn", "scikit-learn>=1.7.2"),
            RuntimeDependency(
                "skimage",
                "scikit-image>=0.20.0",
                max_python_exclusive=(3, 15),
            ),
        ),
        package_specs=("annolid[scientific]",),
        docs_url=INSTALL_DOCS_URL,
        description="SciPy, scikit-learn, and scikit-image for K-Means, PCA, and morphology.",
    ),
    "webengine": RuntimeCapability(
        name="webengine",
        title="Embedded Web & 3D Viewers",
        category="gui",
        approx_download_mb=240,
        dependencies=(
            RuntimeDependency(
                "PySide6.QtWebEngineWidgets",
                "PySide6-Addons>=6.7.0",
                alternative_imports=(
                    "PyQt6.QtWebEngineWidgets",
                    "PyQt5.QtWebEngineWidgets",
                ),
            ),
        ),
        package_specs=("annolid[webengine]",),
        docs_url=INSTALL_DOCS_URL,
        description="Qt WebEngine add-ons for embedded PDF.js, Three.js 3D, and web dashboards.",
    ),
    "ml": RuntimeCapability(
        name="ml",
        title="Core ML & Vision Backbones",
        category="models",
        approx_download_mb=850,
        dependencies=(
            RuntimeDependency("scipy", "scipy>=1.5.2"),
            RuntimeDependency("sklearn", "scikit-learn>=1.7.2"),
            RuntimeDependency("torch", "torch>=2.5.0"),
            RuntimeDependency("torchvision", "torchvision>=0.20.0"),
            RuntimeDependency("transformers", "transformers>=4.45.1"),
            RuntimeDependency("huggingface_hub", "huggingface-hub>=0.22.0"),
            RuntimeDependency("onnxruntime", "onnxruntime>=1.23.2"),
            RuntimeDependency("omegaconf", "omegaconf>=2.3.0"),
            RuntimeDependency("hydra", "hydra-core>=1.3.2"),
            RuntimeDependency("einops", "einops>=0.6.0"),
        ),
        package_specs=("annolid[ml]",),
        docs_url=INSTALL_DOCS_URL,
        description="General ML inference, DINOv3/Florence-2 backbones, and model tooling.",
    ),
    "tracking": RuntimeCapability(
        name="tracking",
        title="Multi-Animal Tracking",
        category="tracking",
        approx_download_mb=900,
        dependencies=(
            RuntimeDependency("cv2", "opencv-contrib-python>=4.1.2.30"),
            RuntimeDependency("scipy", "scipy>=1.5.2"),
            RuntimeDependency("sklearn", "scikit-learn>=1.7.2"),
            RuntimeDependency("torch", "torch>=2.5.0"),
            RuntimeDependency("torchvision", "torchvision>=0.20.0"),
            RuntimeDependency("omegaconf", "omegaconf>=2.3.0"),
            RuntimeDependency("hydra", "hydra-core>=1.3.2"),
            RuntimeDependency("onnxruntime", "onnxruntime>=1.23.2"),
            RuntimeDependency("ultralytics", "ultralytics>=8.4.0"),
            RuntimeDependency("lap", "lap"),
        ),
        package_specs=("annolid[tracking]",),
        docs_url=INSTALL_DOCS_URL,
        description="Common multi-animal tracking workstation runtime.",
    ),
    "cutie": RuntimeCapability(
        name="cutie",
        title="Cutie Video Segmentation",
        category="tracking",
        approx_download_mb=780,
        dependencies=(
            RuntimeDependency("cv2", "opencv-contrib-python>=4.1.2.30"),
            RuntimeDependency("torch", "torch>=2.5.0"),
            RuntimeDependency("torchvision", "torchvision>=0.20.0"),
            RuntimeDependency("omegaconf", "omegaconf>=2.3.0"),
            RuntimeDependency("hydra", "hydra-core>=1.3.2"),
            RuntimeDependency("PIL", "Pillow>=9.3.0"),
            RuntimeDependency("pycocotools", "pycocotools>=2.0.2"),
            RuntimeDependency("einops", "einops>=0.6.0"),
        ),
        package_specs=("annolid[cutie]",),
        docs_url=INSTALL_DOCS_URL,
        description="Cutie video object segmentation tracking runtime.",
    ),
    "sam3": RuntimeCapability(
        name="sam3",
        title="SAM 2 / SAM 3 Segmentation",
        category="segmentation",
        approx_download_mb=920,
        dependencies=(
            RuntimeDependency("cv2", "opencv-contrib-python>=4.1.2.30"),
            RuntimeDependency("torch", "torch>=2.5.0"),
            RuntimeDependency("torchvision", "torchvision>=0.20.0"),
            RuntimeDependency("scipy", "scipy>=1.5.2"),
            RuntimeDependency("iopath", "iopath>=0.1.9"),
            RuntimeDependency("ftfy", "ftfy>=6.1.1"),
            RuntimeDependency("tokenizers", "tokenizers"),
            RuntimeDependency("timm", "timm>=0.9.7"),
        ),
        package_specs=("annolid[sam3]",),
        docs_url=INSTALL_DOCS_URL,
        description="SAM3 promptable image and video segmentation runtime.",
    ),
    "yolo": RuntimeCapability(
        name="yolo",
        title="YOLO / YOLOE Detection & Pose",
        category="models",
        approx_download_mb=820,
        dependencies=(
            RuntimeDependency("ultralytics", "ultralytics>=8.4.0"),
            RuntimeDependency("lap", "lap"),
        ),
        package_specs=("annolid[yolo]",),
        docs_url=INSTALL_DOCS_URL,
        description="YOLO/YOLOE detection, segmentation, pose, and training runtime.",
    ),
    "training": RuntimeCapability(
        name="training",
        title="Training Dashboards (TensorBoard)",
        category="training",
        approx_download_mb=35,
        dependencies=(RuntimeDependency("tensorboard", "tensorboard>=2.3.0"),),
        package_specs=("annolid[training]",),
        docs_url=INSTALL_DOCS_URL,
        description="TensorBoard training dashboards and embedding projector workflows.",
    ),
    "image_editing": RuntimeCapability(
        name="image_editing",
        title="Diffusion Image Editing",
        category="generative",
        approx_download_mb=950,
        dependencies=(
            RuntimeDependency("torch", "torch>=2.5.0"),
            RuntimeDependency("diffusers", "diffusers>=0.32.0"),
            RuntimeDependency("accelerate", "accelerate>=0.27.0"),
            RuntimeDependency("safetensors", "safetensors>=0.4.0"),
        ),
        package_specs=("annolid[image_editing]",),
        docs_url=INSTALL_DOCS_URL,
        description="Diffusers image generation and inpainting runtime.",
    ),
    "qwen3_embedding": RuntimeCapability(
        name="qwen3_embedding",
        title="Qwen3 Multimodal Embeddings",
        category="models",
        approx_download_mb=850,
        dependencies=(
            RuntimeDependency("torch", "torch>=2.5.0"),
            RuntimeDependency("transformers", "transformers>=4.57.0"),
            RuntimeDependency("qwen_vl_utils", "qwen-vl-utils>=0.0.14"),
        ),
        package_specs=("annolid[qwen3_embedding]",),
        docs_url=INSTALL_DOCS_URL,
        description="Qwen3 multimodal vision-language embedding adapter runtime.",
    ),
    "mediapipe": RuntimeCapability(
        name="mediapipe",
        title="MediaPipe Landmarks",
        category="models",
        approx_download_mb=45,
        dependencies=(RuntimeDependency("mediapipe", "mediapipe>=0.10.0"),),
        package_specs=("annolid[mediapipe]",),
        docs_url=INSTALL_DOCS_URL,
        description="MediaPipe pose and landmark estimation runtime.",
    ),
    "cowtracker": RuntimeCapability(
        name="cowtracker",
        title="VGGT / 3D Reconstruction Weights",
        category="3d",
        approx_download_mb=15,
        dependencies=(RuntimeDependency("safetensors", "safetensors>=0.4.0"),),
        package_specs=("annolid[cowtracker]",),
        docs_url=INSTALL_DOCS_URL,
        description="Safetensors loader for VGGT / CowTracker 3D reconstruction.",
    ),
    "audio": RuntimeCapability(
        name="audio",
        title="Audio Recording & Spectrograms",
        category="audio",
        approx_download_mb=85,
        dependencies=(
            RuntimeDependency("sounddevice", "sounddevice>=0.4.6"),
            RuntimeDependency("librosa", "librosa>=0.10.1"),
            RuntimeDependency("numba", "numba>=0.60.0"),
        ),
        package_specs=("annolid[audio]",),
        docs_url=INSTALL_DOCS_URL,
        description="Voice recording, playback, and librosa audio feature extraction.",
    ),
    "text_to_speech": RuntimeCapability(
        name="text_to_speech",
        title="Text-to-Speech & PDF Narration",
        category="audio",
        approx_download_mb=120,
        dependencies=(
            RuntimeDependency("pocket_tts", "pocket-tts>=1.0.0"),
            RuntimeDependency("fitz", "pymupdf"),
            RuntimeDependency("sounddevice", "sounddevice>=0.4.6"),
            RuntimeDependency("librosa", "librosa>=0.10.1"),
            RuntimeDependency("numba", "numba>=0.60.0"),
        ),
        package_specs=("annolid[text_to_speech]",),
        docs_url=INSTALL_DOCS_URL,
        description="Pocket-TTS local speech synthesis and PDF reader narration.",
    ),
    "pdf": RuntimeCapability(
        name="pdf",
        title="PDF Document Rendering",
        category="io",
        approx_download_mb=18,
        dependencies=(RuntimeDependency("fitz", "pymupdf"),),
        package_specs=("annolid[pdf]",),
        docs_url=INSTALL_DOCS_URL,
        description="PyMuPDF document rendering, page import, and vector extraction.",
    ),
    "ai_chat": RuntimeCapability(
        name="ai_chat",
        title="Cloud LLM Providers",
        category="agent",
        approx_download_mb=15,
        dependencies=(
            RuntimeDependency("openai", "openai>=1.40.0"),
            RuntimeDependency("anthropic", "anthropic>=0.49.0"),
        ),
        package_specs=("annolid[ai_chat]",),
        docs_url=INSTALL_DOCS_URL,
        description="OpenAI and Anthropic SDKs for hosted AI chat providers.",
    ),
    "bot": RuntimeCapability(
        name="bot",
        title="Annolid Bot & Channels",
        category="agent",
        approx_download_mb=180,
        dependencies=(
            RuntimeDependency("openai", "openai>=1.40.0"),
            RuntimeDependency("anthropic", "anthropic>=0.49.0"),
            RuntimeDependency("croniter", "croniter>=2.0.5"),
            RuntimeDependency("mcp", "mcp>=1.0.0"),
            RuntimeDependency("websockets", "websockets>=12.0"),
            RuntimeDependency("qrcode", "qrcode>=7.4.2"),
        ),
        package_specs=("annolid[bot]",),
        docs_url=INSTALL_DOCS_URL,
        description="Annolid Bot providers, channels, integrations, and memory.",
    ),
    "memory": RuntimeCapability(
        name="memory",
        title="Vector & Full-Text Memory",
        category="agent",
        approx_download_mb=110,
        dependencies=(
            RuntimeDependency("lancedb", "lancedb>=0.6.0"),
            RuntimeDependency("pyarrow", "pyarrow>=14.0.1"),
            RuntimeDependency("tantivy", "tantivy>=0.21.1"),
        ),
        package_specs=("annolid[memory]",),
        docs_url=INSTALL_DOCS_URL,
        description="LanceDB, PyArrow, and Tantivy fast local agent memory.",
    ),
    "large_image": RuntimeCapability(
        name="large_image",
        title="Whole-Slide & Pyramidal TIFF",
        category="io",
        approx_download_mb=45,
        dependencies=(
            RuntimeDependency("tifffile", "tifffile>=2024.8.30"),
            RuntimeDependency("pyvips", "pyvips>=2.2.3"),
            RuntimeDependency("openslide", "openslide-python>=1.3.1"),
        ),
        package_specs=("annolid[large_image]",),
        docs_url=INSTALL_DOCS_URL,
        description="Large TIFF, OME-TIFF, and whole-slide image streaming runtime.",
    ),
    "sleap": RuntimeCapability(
        name="sleap",
        title="SLEAP HDF5 Import",
        category="io",
        approx_download_mb=12,
        dependencies=(RuntimeDependency("h5py", "h5py>=3.15.1"),),
        package_specs=("annolid[sleap]",),
        docs_url=INSTALL_DOCS_URL,
        description="HDF5 (.slp / .h5) reader for SLEAP pose import.",
    ),
    "remote_video": RuntimeCapability(
        name="remote_video",
        title="Remote Stream Video Decoding",
        category="io",
        approx_download_mb=25,
        dependencies=(RuntimeDependency("ffpyplayer", "ffpyplayer>=4.5.0"),),
        package_specs=("annolid[remote_video]",),
        docs_url=INSTALL_DOCS_URL,
        description="Network/remote video stream decoding runtime.",
    ),
    "realtime": RuntimeCapability(
        name="realtime",
        title="Realtime Hardware & ZeroMQ",
        category="hardware",
        approx_download_mb=12,
        dependencies=(
            RuntimeDependency("serial", "pyserial>=3.5"),
            RuntimeDependency("zmq", "pyzmq>=25.1.2"),
        ),
        package_specs=("annolid[realtime]",),
        docs_url=INSTALL_DOCS_URL,
        description="Serial and ZeroMQ closed-loop hardware control runtime.",
    ),
    "model_downloads": RuntimeCapability(
        name="model_downloads",
        title="Google Drive Checkpoint Downloader",
        category="utilities",
        approx_download_mb=5,
        dependencies=(RuntimeDependency("gdown", "gdown"),),
        package_specs=("annolid[model_downloads]",),
        docs_url=INSTALL_DOCS_URL,
        description="gdown helper for Google Drive hosted model checkpoints.",
    ),
}

INSTALL_PROFILES: dict[str, dict[str, object]] = {
    "minimal": {
        "capabilities": (),
        "extras": ("gui",),
        "description": "Fastest lean GUI & annotation shell; optional ML/tracking runtimes install on demand.",
    },
    "gui": {
        "capabilities": (),
        "extras": ("gui",),
        "description": "Default lean desktop GUI install; heavy ML/tracking/viewer packages install when first used.",
    },
    "tracking": {
        "capabilities": ("ml", "tracking", "cutie"),
        "extras": ("gui", "ml", "tracking", "cutie"),
        "description": "Pre-warms PyTorch, YOLO, ONNX, DINOv3, and Cutie multi-animal tracking runtimes upfront.",
    },
    "workstation": {
        "capabilities": (
            "ml",
            "tracking",
            "cutie",
            "sam3",
            "training",
            "webengine",
            "pdf",
        ),
        "extras": (
            "gui",
            "ml",
            "tracking",
            "cutie",
            "sam3",
            "training",
            "webengine",
            "pdf",
        ),
        "description": "Pre-warms tracking, SAM3 segmentation, TensorBoard dashboards, and embedded WebEngine/PDF viewers.",
    },
    "full": {
        "capabilities": tuple(sorted(CAPABILITIES)),
        "extras": ("gui", "all"),
        "description": "Installs every optional Annolid capability upfront.",
    },
}

_INSTALL_LOCKS_GUARD = threading.Lock()
_INSTALL_LOCKS: dict[str, threading.Lock] = {}
_INTERACTIVE_INSTALL_HANDLER: Callable[[str, str], tuple[str, ...] | None] | None = None


def set_interactive_install_handler(
    handler: Callable[[str, str], tuple[str, ...] | None] | None,
) -> None:
    """Register an optional UI-thread install handler (used by the Qt GUI)."""
    global _INTERACTIVE_INSTALL_HANDLER
    _INTERACTIVE_INSTALL_HANDLER = handler


def _capability_lock(key: str) -> threading.Lock:
    with _INSTALL_LOCKS_GUARD:
        lock = _INSTALL_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _INSTALL_LOCKS[key] = lock
        return lock


def _default_find_spec(import_name: str) -> object | None:
    try:
        return importlib_util.find_spec(import_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None


def list_capabilities() -> tuple[str, ...]:
    return tuple(sorted(CAPABILITIES))


def _dependency_satisfied(
    dependency: RuntimeDependency,
    *,
    finder: SpecFinder,
) -> bool:
    if finder(dependency.import_name) is not None:
        return True
    return any(finder(alt) is not None for alt in dependency.alternative_imports)


def _missing_dependencies(
    capability: RuntimeCapability,
    *,
    find_spec: SpecFinder | None = None,
) -> tuple[RuntimeDependency, ...]:
    if getattr(sys, "frozen", False) and find_spec is None:
        activate_user_runtime_site()
    finder = find_spec or _default_find_spec
    enforce_python_bounds = find_spec is None
    return tuple(
        dependency
        for dependency in capability.dependencies
        if (not enforce_python_bounds or dependency.applies_to_current_python())
        and not _dependency_satisfied(dependency, finder=finder)
    )


def check_capability(
    name: str, *, find_spec: SpecFinder | None = None
) -> CapabilityStatus:
    key = name.strip().lower()
    if key not in CAPABILITIES:
        known = ", ".join(list_capabilities())
        raise KeyError(
            f"Unknown runtime capability '{name}'. Known capabilities: {known}"
        )

    capability = CAPABILITIES[key]
    missing = _missing_dependencies(capability, find_spec=find_spec)
    return CapabilityStatus(
        name=capability.name,
        available=not missing,
        missing_imports=tuple(dependency.import_name for dependency in missing),
        install_command=capability.install_command,
        docs_url=capability.docs_url,
        description=capability.description,
        title=capability.display_title,
        category=capability.category,
        approx_download_mb=capability.approx_download_mb,
        missing_package_specs=tuple(dependency.package_spec for dependency in missing),
    )


def capability_status_dict(status: CapabilityStatus) -> dict[str, object]:
    return {
        "name": status.name,
        "title": status.title or status.name,
        "category": status.category,
        "state": status.state,
        "available": status.available,
        "approx_download_mb": status.approx_download_mb,
        "missing_imports": list(status.missing_imports),
        "missing_package_specs": list(status.missing_package_specs),
        "install_command": status.install_command,
        "docs_url": status.docs_url,
        "description": status.description,
    }


def capability_message(status: CapabilityStatus) -> str:
    if status.available:
        return f"{status.name}: available"
    missing = ", ".join(status.missing_imports)
    return (
        f"{status.name}: missing optional runtime packages ({missing}). "
        f"Install with: {status.install_command}. Docs: {status.docs_url}"
    )


def _auto_install_optional_deps_enabled(value: str | None = None) -> bool:
    raw = os.environ.get(AUTO_INSTALL_OPTIONAL_DEPS_ENV) if value is None else value
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _no_gpu_requested() -> bool:
    raw = os.environ.get(NO_GPU_ENV)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@functools.lru_cache(maxsize=1)
def _probe_nvidia_smi() -> tuple[bool, str | None, bool, str | None]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return (False, None, False, None)
    try:
        listed = subprocess.run(
            [nvidia_smi, "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.5,
        )
        gpu_line = (
            next(
                (
                    line.strip()
                    for line in (listed.stdout or "").splitlines()
                    if line.strip()
                ),
                "",
            )
            if listed.returncode == 0
            else ""
        )
        if not gpu_line:
            return (False, None, False, None)

        header = subprocess.run(
            [nvidia_smi],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.5,
        )
        match = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", header.stdout or "")
        cuda_version = match.group(1) if match else None
        cuda_major = int(cuda_version.split(".", 1)[0]) if cuda_version else 0
        return (True, cuda_version, cuda_major >= 12, gpu_line)
    except Exception:
        return (False, None, False, None)


def _truthy_env(var_name: str) -> bool:
    raw = os.environ.get(var_name)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def is_offline_mode(explicit: bool | None = None) -> bool:
    """Return True when network package index access is disabled."""
    if explicit is not None:
        return bool(explicit)
    return _truthy_env(OFFLINE_ENV)


def frozen_user_runtime_enabled() -> bool:
    """Return True unless isolated user-space runtime extensions are disabled."""
    raw = os.environ.get(FROZEN_USER_RUNTIME_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _frozen_install_opted_in(allow_frozen_user_runtime: bool) -> bool:
    if not frozen_user_runtime_enabled():
        return False
    if allow_frozen_user_runtime:
        return True
    return _truthy_env(FROZEN_USER_RUNTIME_ENV)


def user_runtime_abi_slug() -> str:
    """Return an ABI-isolated slug for user-installed runtime extensions."""
    arch = (platform.machine() or "unknown").strip().lower()
    gil_suffix = "t" if bool(sysconfig.get_config_var("Py_GIL_DISABLED")) else ""
    return (
        f"py{sys.version_info.major}.{sys.version_info.minor}{gil_suffix}"
        f"-{sys.platform}-{arch}"
    )


def user_runtime_site_dir() -> Path:
    """Return the isolated user-space site-packages path (never inside app bundles)."""
    override = (os.environ.get(USER_RUNTIME_DIR_ENV) or "").strip()
    base = (
        Path(override).expanduser()
        if override
        else Path.home() / ".annolid" / "runtime"
    )
    return base / user_runtime_abi_slug() / "site-packages"


def activate_user_runtime_site(site_dir: Path | None = None) -> Path | None:
    """Attach the user-space runtime site-packages at the end of ``sys.path``.

    Keeping ``site_dir`` after bundled paths guarantees frozen core libraries
    (``annolid``, ``numpy``, ``cv2``, ``PySide6-Essentials``) are never shadowed,
    while extending ``PySide6.__path__`` so ``PySide6-Addons`` modules resolve.
    """
    if not frozen_user_runtime_enabled():
        return None
    target = (site_dir or user_runtime_site_dir()).expanduser()
    if not target.is_dir():
        return None
    resolved_str = str(target.resolve())
    if resolved_str not in sys.path and str(target) not in sys.path:
        site.addsitedir(resolved_str)
    if "PySide6" in sys.modules:
        pyside_addon_dir = target / "PySide6"
        if pyside_addon_dir.is_dir():
            pyside_mod = sys.modules["PySide6"]
            mod_path = getattr(pyside_mod, "__path__", None)
            addon_str = str(pyside_addon_dir.resolve())
            if isinstance(mod_path, list) and addon_str not in mod_path:
                mod_path.append(addon_str)
            try:
                from annolid.infrastructure.runtime import (
                    configure_qtwebengine_resource_paths,
                )

                configure_qtwebengine_resource_paths()
            except Exception:
                pass
    importlib.invalidate_caches()
    return target


def _dir_has_wheels(directory: Path) -> bool:
    try:
        return directory.is_dir() and any(directory.glob("*.whl"))
    except Exception:
        return False


def resolve_wheelhouse_dir(
    explicit: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Resolve an explicit, env-configured, or portable adjacent wheelhouse."""
    if explicit is not None and str(explicit).strip():
        cand = Path(explicit).expanduser().resolve()
        if not cand.is_dir():
            raise RuntimeError(f"Offline wheelhouse directory does not exist: {cand}")
        return cand

    env_val = (os.environ.get(WHEELHOUSE_ENV) or "").strip()
    if env_val:
        cand = Path(env_val).expanduser().resolve()
        if not cand.is_dir():
            raise RuntimeError(
                f"{WHEELHOUSE_ENV} points to a non-existent directory: {cand}"
            )
        return cand

    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        exe_path = Path(sys.executable).resolve()
        candidates.append(exe_path.parent / "wheelhouse")
        # macOS Annolid.app/Contents/MacOS/annolid -> parent folder beside Annolid.app
        if len(exe_path.parents) >= 4 and exe_path.parents[2].suffix == ".app":
            candidates.append(exe_path.parents[3] / "wheelhouse")
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        candidates.append(Path.home() / ".annolid" / "wheelhouse")

    for cand in candidates:
        if _dir_has_wheels(cand):
            return cand.resolve()
    return None


def _find_uv_binary() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    if not getattr(sys, "frozen", False):
        return None
    # macOS/Linux desktop bundles launched from Finder/Desktop often omit user/Homebrew PATHs.
    exe_name = "uv.exe" if sys.platform == "win32" else "uv"
    extra_dirs = [
        Path.home() / ".local" / "bin",
        Path.home() / ".cargo" / "bin",
        Path("/opt/homebrew/bin"),
        Path("/usr/local/bin"),
    ]
    for directory in extra_dirs:
        cand = directory / exe_name
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


@functools.lru_cache(maxsize=1)
def _find_matching_host_python() -> str | None:
    """Locate a host CPython matching ``sys.version_info[:2]`` with working ``pip``."""
    maj, min_ = sys.version_info.major, sys.version_info.minor
    names = (
        [f"python{maj}.{min_}.exe", "python.exe"]
        if sys.platform == "win32"
        else [f"python{maj}.{min_}", f"python{maj}", "python3"]
    )
    candidates: list[str] = []
    for name in names:
        hit = shutil.which(name)
        if hit and hit not in candidates:
            candidates.append(hit)
    if sys.platform != "win32":
        for prefix in ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"):
            cand = Path(prefix) / f"python{maj}.{min_}"
            if (
                cand.is_file()
                and os.access(cand, os.X_OK)
                and str(cand) not in candidates
            ):
                candidates.append(str(cand))

    frozen_exe = str(Path(sys.executable).resolve())
    expected = f"{maj}.{min_}"
    for cand in candidates:
        try:
            if str(Path(cand).resolve()) == frozen_exe:
                continue
            probe = subprocess.run(
                [
                    cand,
                    "-c",
                    "import pip, sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            if probe.returncode == 0 and (probe.stdout or "").strip() == expected:
                return cand
        except Exception:
            continue
    return None


def can_install_in_frozen_user_runtime(
    wheelhouse: str | os.PathLike[str] | None = None,
) -> tuple[bool, str]:
    """Return ``(available, backend)`` for installing into user-space from a frozen app."""
    if not frozen_user_runtime_enabled() or not _auto_install_optional_deps_enabled():
        return (False, "disabled")
    try:
        wh = resolve_wheelhouse_dir(wheelhouse)
    except RuntimeError:
        wh = None
    if _find_uv_binary():
        return (True, "uv")
    if _find_matching_host_python():
        return (True, "host-python")
    if wh is not None and _dir_has_wheels(wh):
        return (True, "wheelhouse")
    return (False, "none")


def detect_accelerator_summary() -> dict[str, object]:
    """Return a lightweight, dependency-free hardware and Python runtime summary."""
    apple_silicon = sys.platform == "darwin" and platform.machine().lower() in {
        "arm64",
        "aarch64",
    }
    is_frozen = bool(getattr(sys, "frozen", False))
    try:
        wh_dir = resolve_wheelhouse_dir()
    except RuntimeError:
        wh_dir = None
    if is_frozen:
        can_frozen, frozen_backend = can_install_in_frozen_user_runtime(wh_dir)
        writable = can_frozen
    else:
        frozen_backend = "venv"
        writable = _auto_install_optional_deps_enabled()

    base_info: dict[str, object] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "frozen": is_frozen,
        "writable": writable,
        "user_runtime_dir": str(user_runtime_site_dir()),
        "user_runtime_backend": frozen_backend,
        "wheelhouse_dir": str(wh_dir) if wh_dir is not None else None,
        "offline_mode": is_offline_mode(),
        "platform": sys.platform,
        "arch": platform.machine(),
        "apple_silicon": apple_silicon,
    }
    if _no_gpu_requested():
        return {
            **base_info,
            "accelerator": "cpu",
            "gpu_disabled_by_env": True,
            "has_nvidia_gpu": False,
            "cuda_version": None,
            "has_cuda12": False,
            "gpu_name": None,
        }
    has_nvidia, cuda_version, has_cuda12, gpu_name = _probe_nvidia_smi()
    if has_nvidia:
        accelerator = "cuda-12" if has_cuda12 else "cuda"
    elif apple_silicon:
        accelerator = "apple-silicon"
    else:
        accelerator = "cpu"
    return {
        **base_info,
        "accelerator": accelerator,
        "gpu_disabled_by_env": False,
        "has_nvidia_gpu": has_nvidia,
        "cuda_version": cuda_version,
        "has_cuda12": has_cuda12,
        "gpu_name": gpu_name,
    }


def _installed_pyside6_version() -> str | None:
    if "PySide6" in sys.modules:
        mod_ver = getattr(sys.modules["PySide6"], "__version__", None)
        if isinstance(mod_ver, str) and mod_ver.strip():
            return mod_ver.strip()
    for dist_name in ("PySide6-Essentials", "PySide6"):
        try:
            ver = importlib_metadata.version(dist_name)
            if ver:
                return ver.strip()
        except importlib_metadata.PackageNotFoundError:
            continue
        except Exception:
            continue
    return None


def _pin_abi_sensitive_specs(package_specs: tuple[str, ...]) -> tuple[str, ...]:
    """Pin ABI-coupled Qt add-on wheels to the active PySide6-Essentials version."""
    pyside_ver = _installed_pyside6_version()
    if not pyside_ver:
        return package_specs
    resolved: list[str] = []
    for spec in package_specs:
        if spec.lower().startswith("pyside6-addons"):
            resolved.append(f"PySide6-Addons=={pyside_ver}")
        else:
            resolved.append(spec)
    return tuple(resolved)


def _frozen_bundled_constraints() -> list[str]:
    """Pin core ABI-sensitive packages already shipped inside a frozen bundle."""
    pins: list[str] = []
    pyside_ver = _installed_pyside6_version()
    if pyside_ver:
        pins.append(f"PySide6-Essentials=={pyside_ver}")
        pins.append(f"shiboken6=={pyside_ver}")
    for dist_name, mod_name in (("numpy", "numpy"), ("Pillow", "PIL")):
        ver: str | None = None
        if mod_name in sys.modules:
            mod_ver = getattr(sys.modules[mod_name], "__version__", None)
            if isinstance(mod_ver, str) and mod_ver.strip():
                ver = mod_ver.strip()
        if not ver:
            try:
                ver = importlib_metadata.version(dist_name).strip()
            except Exception:
                ver = None
        if ver:
            pins.append(f"{dist_name}=={ver}")
    return pins


def _install_command(
    package_specs: tuple[str, ...],
    *,
    index_url: str | None = None,
    extra_index_url: str | None = None,
    wheelhouse_dir: Path | None = None,
    offline: bool = False,
    target_site_dir: Path | None = None,
    constraints_file: Path | None = None,
) -> list[str]:
    is_frozen = bool(getattr(sys, "frozen", False))
    uv_cmd = _find_uv_binary()
    if target_site_dir is not None:
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        if uv_cmd:
            cmd = [
                uv_cmd,
                "pip",
                "install",
                "--target",
                str(target_site_dir),
                "--python-version",
                py_ver,
                "--only-binary",
                ":all:",
            ]
        else:
            host_py = sys.executable if not is_frozen else _find_matching_host_python()
            if not host_py:
                raise RuntimeError(
                    "No external 'uv' or matching host Python interpreter found for --target install."
                )
            cmd = [
                host_py,
                "-m",
                "pip",
                "install",
                "--target",
                str(target_site_dir),
                "--upgrade",
                "--only-binary=:all:",
            ]
        if constraints_file is not None:
            cmd.extend(["-c", str(constraints_file)])
    else:
        if uv_cmd:
            cmd = [uv_cmd, "pip", "install", "--python", sys.executable]
        else:
            cmd = [sys.executable, "-m", "pip", "install"]

    if offline:
        cmd.append("--no-index")
        if wheelhouse_dir is not None:
            cmd.extend(["--find-links", str(wheelhouse_dir)])
    else:
        if wheelhouse_dir is not None:
            cmd.extend(["--find-links", str(wheelhouse_dir)])
        if index_url:
            cmd.extend(["--index-url", index_url])
        if extra_index_url:
            cmd.extend(["--extra-index-url", extra_index_url])

    cmd.extend(package_specs)
    return cmd


def _build_smart_install_plan(
    missing: tuple[RuntimeDependency, ...],
    *,
    wheelhouse: str | os.PathLike[str] | None = None,
    offline: bool | None = None,
    target_site_dir: Path | None = None,
    constraints_file: Path | None = None,
) -> tuple[tuple[str, ...], list[tuple[list[str], list[str] | None]]]:
    """Build primary (and optional fallback) install commands for real environments.

    Returns ``(effective_specs, steps)`` where each step is
    ``(primary_cmd, optional_fallback_cmd)``.
    """
    raw_specs = tuple(dep.package_spec for dep in missing)
    pinned_specs = _pin_abi_sensitive_specs(raw_specs)
    wh_dir = resolve_wheelhouse_dir(wheelhouse)
    offline_active = is_offline_mode(offline)
    if offline_active and wh_dir is None:
        raise RuntimeError(
            f"Offline mode is enabled ({OFFLINE_ENV}=1 or --offline), "
            f"but no offline wheelhouse directory was provided via --wheelhouse or {WHEELHOUSE_ENV}."
        )

    accel = detect_accelerator_summary()
    torch_index_override = (os.environ.get(TORCH_INDEX_URL_ENV) or "").strip()

    if offline_active:
        effective_offline: list[str] = []
        for spec in pinned_specs:
            if (
                (
                    spec.lower().startswith("onnxruntime>=")
                    or spec.lower() == "onnxruntime"
                )
                and wh_dir is not None
                and sys.platform in {"linux", "win32"}
                and accel["has_nvidia_gpu"]
                and accel["has_cuda12"]
                and any(wh_dir.glob("onnxruntime_gpu-*.whl"))
            ):
                effective_offline.append(
                    spec.replace("onnxruntime", "onnxruntime-gpu", 1)
                )
            else:
                effective_offline.append(spec)
        eff_tuple = tuple(effective_offline)
        cmd = _install_command(
            eff_tuple,
            wheelhouse_dir=wh_dir,
            offline=True,
            target_site_dir=target_site_dir,
            constraints_file=constraints_file,
        )
        return eff_tuple, [(cmd, None)]

    torch_specs: list[str] = []
    onnx_specs: list[str] = []
    other_specs: list[str] = []
    for spec in pinned_specs:
        lower = spec.lower()
        if lower.startswith(("torch>=", "torch==", "torchvision>=", "torchvision==")):
            torch_specs.append(spec)
        elif lower.startswith("onnxruntime>=") or lower == "onnxruntime":
            onnx_specs.append(spec)
        else:
            other_specs.append(spec)

    steps: list[tuple[list[str], list[str] | None]] = []
    effective_specs: list[str] = []

    def _fallback_for(specs_tuple: tuple[str, ...]) -> list[str] | None:
        if wh_dir is not None:
            return _install_command(
                specs_tuple,
                wheelhouse_dir=wh_dir,
                offline=True,
                target_site_dir=target_site_dir,
                constraints_file=constraints_file,
            )
        return None

    if torch_specs:
        effective_specs.extend(torch_specs)
        target_index: str | None = None
        if torch_index_override:
            target_index = torch_index_override
        elif sys.platform.startswith("linux") and not accel["has_nvidia_gpu"]:
            target_index = PYTORCH_CPU_INDEX_URL
        elif (
            sys.platform in {"linux", "win32"}
            and accel["has_nvidia_gpu"]
            and accel["has_cuda12"]
        ):
            target_index = PYTORCH_CUDA12_INDEX_URL

        t_tuple = tuple(torch_specs)
        if target_index:
            steps.append(
                (
                    _install_command(
                        t_tuple,
                        index_url=target_index,
                        wheelhouse_dir=wh_dir,
                        target_site_dir=target_site_dir,
                        constraints_file=constraints_file,
                    ),
                    _fallback_for(t_tuple)
                    or _install_command(
                        t_tuple,
                        wheelhouse_dir=wh_dir,
                        target_site_dir=target_site_dir,
                        constraints_file=constraints_file,
                    ),
                )
            )
        else:
            steps.append(
                (
                    _install_command(
                        t_tuple,
                        wheelhouse_dir=wh_dir,
                        target_site_dir=target_site_dir,
                        constraints_file=constraints_file,
                    ),
                    _fallback_for(t_tuple),
                )
            )

    if onnx_specs:
        o_tuple = tuple(onnx_specs)
        if (
            sys.platform in {"linux", "win32"}
            and accel["has_nvidia_gpu"]
            and accel["has_cuda12"]
        ):
            gpu_specs = tuple(
                spec.replace("onnxruntime", "onnxruntime-gpu", 1) for spec in onnx_specs
            )
            effective_specs.extend(gpu_specs)
            steps.append(
                (
                    _install_command(
                        gpu_specs,
                        extra_index_url=ONNXRUNTIME_CUDA12_EXTRA_INDEX_URL,
                        wheelhouse_dir=wh_dir,
                        target_site_dir=target_site_dir,
                        constraints_file=constraints_file,
                    ),
                    _fallback_for(o_tuple)
                    or _install_command(
                        o_tuple,
                        wheelhouse_dir=wh_dir,
                        target_site_dir=target_site_dir,
                        constraints_file=constraints_file,
                    ),
                )
            )
        else:
            effective_specs.extend(onnx_specs)
            steps.append(
                (
                    _install_command(
                        o_tuple,
                        wheelhouse_dir=wh_dir,
                        target_site_dir=target_site_dir,
                        constraints_file=constraints_file,
                    ),
                    _fallback_for(o_tuple),
                )
            )

    if other_specs:
        rest_tuple = tuple(other_specs)
        effective_specs.extend(other_specs)
        steps.append(
            (
                _install_command(
                    rest_tuple,
                    wheelhouse_dir=wh_dir,
                    target_site_dir=target_site_dir,
                    constraints_file=constraints_file,
                ),
                _fallback_for(rest_tuple),
            )
        )

    return tuple(effective_specs), steps


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_wheelhouse_bundle(
    wheelhouse_dir: str | os.PathLike[str],
) -> dict[str, object]:
    """Verify wheelhouse directory contents and optional SHA256 manifest."""
    wh_path = Path(wheelhouse_dir).expanduser().resolve()
    if not wh_path.is_dir():
        raise RuntimeError(f"Wheelhouse directory does not exist: {wh_path}")

    manifest_path = wh_path / WHEELHOUSE_MANIFEST_NAME
    wheel_files = sorted(wh_path.glob("*.whl"))
    if not manifest_path.is_file():
        if not wheel_files:
            raise RuntimeError(
                f"No .whl files found in wheelhouse directory: {wh_path}"
            )
        return {
            "valid": True,
            "manifest_present": False,
            "wheelhouse_dir": str(wh_path),
            "wheel_count": len(wheel_files),
            "wheels": [w.name for w in wheel_files],
            "capabilities": [],
            "profiles": [],
        }

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Invalid {WHEELHOUSE_MANIFEST_NAME} in {wh_path}: {exc}"
        ) from exc

    entries = manifest.get("wheels")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(
            f"{WHEELHOUSE_MANIFEST_NAME} in {wh_path} contains no wheel entries."
        )

    verified_names: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("filename"):
            raise RuntimeError(f"Malformed wheel entry in {manifest_path}")
        fname = str(entry["filename"])
        if "/" in fname or "\\" in fname or ".." in fname:
            raise RuntimeError(f"Unsafe wheel filename in manifest: {fname}")
        target = wh_path / fname
        if not target.is_file():
            raise RuntimeError(
                f"Missing wheel file listed in {WHEELHOUSE_MANIFEST_NAME}: {fname}"
            )
        expected_size = entry.get("size_bytes")
        if isinstance(expected_size, int) and target.stat().st_size != expected_size:
            raise RuntimeError(
                f"Wheel size mismatch for {fname}: expected {expected_size}, got {target.stat().st_size}"
            )
        expected_sha = str(entry.get("sha256") or "").strip().lower()
        if expected_sha:
            actual_sha = _sha256_file(target)
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"SHA256 checksum mismatch for {fname}: expected {expected_sha}, got {actual_sha}"
                )
        verified_names.append(fname)

    return {
        "valid": True,
        "manifest_present": True,
        "wheelhouse_dir": str(wh_path),
        "wheel_count": len(verified_names),
        "wheels": verified_names,
        "capabilities": list(manifest.get("capabilities") or ()),
        "profiles": list(manifest.get("profiles") or ()),
        "python_version": manifest.get("python_version"),
        "platform": manifest.get("platform"),
    }


def export_wheelhouse_bundle(
    output_dir: str | os.PathLike[str],
    *,
    capability_names: Sequence[str] = (),
    profile_names: Sequence[str] = (),
    python_version: str | None = None,
    platform_tag: str | None = None,
    on_output: OutputCallback | None = None,
    cancel_event: threading.Event | None = None,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
) -> dict[str, object]:
    """Download a self-contained, SHA256-checksummed offline wheelhouse bundle."""
    resolved_profiles: list[str] = []
    resolved_caps: list[str] = []

    for pname in profile_names:
        key = str(pname).strip().lower()
        if not key:
            continue
        if key not in INSTALL_PROFILES:
            known = ", ".join(sorted(INSTALL_PROFILES))
            raise KeyError(f"Unknown install profile '{pname}'. Known: {known}")
        if key not in resolved_profiles:
            resolved_profiles.append(key)
        for cap in INSTALL_PROFILES[key]["capabilities"]:  # type: ignore[index]
            cap_str = str(cap)
            if cap_str not in resolved_caps:
                resolved_caps.append(cap_str)

    for cname in capability_names:
        key = str(cname).strip().lower()
        if not key:
            continue
        if key not in CAPABILITIES:
            known = ", ".join(list_capabilities())
            raise KeyError(
                f"Unknown runtime capability '{cname}'. Known capabilities: {known}"
            )
        if key not in resolved_caps:
            resolved_caps.append(key)

    if not resolved_caps:
        raise ValueError(
            "Select at least one capability or non-empty profile to export an offline bundle."
        )

    target_py_tuple: tuple[int, int] | None = None
    if python_version:
        parts = python_version.strip().split(".")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            target_py_tuple = (int(parts[0]), int(parts[1]))

    all_deps: list[RuntimeDependency] = []
    seen_specs: set[str] = set()
    for cap_key in resolved_caps:
        for dep in CAPABILITIES[cap_key].dependencies:
            if not dep.applies_to_current_python(target_py_tuple):
                continue
            if dep.package_spec not in seen_specs:
                seen_specs.add(dep.package_spec)
                all_deps.append(dep)

    out_path = Path(output_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    python_exe = (
        _find_matching_host_python()
        if getattr(sys, "frozen", False)
        else sys.executable
    )
    if not python_exe:
        raise RuntimeError(
            "Exporting an offline wheelhouse requires a Python environment with 'pip'."
        )

    pinned_specs = _pin_abi_sensitive_specs(tuple(dep.package_spec for dep in all_deps))
    cross_targeting = bool(python_version or platform_tag)

    download_cmds: list[list[str]] = []
    if cross_targeting:
        cmd = [
            python_exe,
            "-m",
            "pip",
            "download",
            "--dest",
            str(out_path),
            "--only-binary=:all:",
        ]
        if python_version:
            cmd.extend(["--python-version", python_version.strip()])
        if platform_tag:
            cmd.extend(["--platform", platform_tag.strip()])
        cmd.extend(pinned_specs)
        download_cmds.append(cmd)
    else:
        accel = detect_accelerator_summary()
        torch_specs = [
            s
            for s in pinned_specs
            if s.lower().startswith(
                ("torch>=", "torch==", "torchvision>=", "torchvision==")
            )
        ]
        onnx_specs = [
            s
            for s in pinned_specs
            if s.lower().startswith("onnxruntime>=") or s.lower() == "onnxruntime"
        ]
        other_specs = [
            s for s in pinned_specs if s not in torch_specs and s not in onnx_specs
        ]
        if torch_specs:
            t_cmd = [
                python_exe,
                "-m",
                "pip",
                "download",
                "--dest",
                str(out_path),
                "--only-binary=:all:",
            ]
            if sys.platform.startswith("linux") and not accel["has_nvidia_gpu"]:
                t_cmd.extend(["--index-url", PYTORCH_CPU_INDEX_URL])
            elif (
                sys.platform in {"linux", "win32"}
                and accel["has_nvidia_gpu"]
                and accel["has_cuda12"]
            ):
                t_cmd.extend(["--index-url", PYTORCH_CUDA12_INDEX_URL])
            t_cmd.extend(torch_specs)
            download_cmds.append(t_cmd)
        if onnx_specs:
            o_cmd = [
                python_exe,
                "-m",
                "pip",
                "download",
                "--dest",
                str(out_path),
                "--only-binary=:all:",
            ]
            if (
                sys.platform in {"linux", "win32"}
                and accel["has_nvidia_gpu"]
                and accel["has_cuda12"]
            ):
                o_cmd.extend(["--extra-index-url", ONNXRUNTIME_CUDA12_EXTRA_INDEX_URL])
                o_cmd.extend(
                    [s.replace("onnxruntime", "onnxruntime-gpu", 1) for s in onnx_specs]
                )
            else:
                o_cmd.extend(onnx_specs)
            download_cmds.append(o_cmd)
        if other_specs:
            download_cmds.append(
                [
                    python_exe,
                    "-m",
                    "pip",
                    "download",
                    "--dest",
                    str(out_path),
                    "--only-binary=:all:",
                    *other_specs,
                ]
            )

    for d_cmd in download_cmds:
        if runner is not None:
            completed = runner(d_cmd, check=False, capture_output=True, text=True)
            if completed.returncode != 0:
                stderr = str(getattr(completed, "stderr", "") or "").strip()
                raise RuntimeError(
                    f"Failed to export offline wheelhouse bundle: {stderr}"
                )
        else:
            rc, out_text = _run_streaming_command(
                d_cmd,
                on_output=on_output,
                cancel_event=cancel_event,
                capability_name="bundle",
            )
            if rc != 0:
                tail = "\n".join(out_text.splitlines()[-40:]).strip()
                raise RuntimeError(
                    f"Failed to export offline wheelhouse bundle:\n{tail}"
                )

    wheels_meta: list[dict[str, object]] = []
    for whl in sorted(out_path.glob("*.whl")):
        wheels_meta.append(
            {
                "filename": whl.name,
                "size_bytes": whl.stat().st_size,
                "sha256": _sha256_file(whl),
            }
        )
    if not wheels_meta:
        raise RuntimeError(f"No .whl files were downloaded into {out_path}.")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python_version": python_version
        or f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": platform_tag or sysconfig.get_platform(),
        "arch": platform.machine(),
        "capabilities": resolved_caps,
        "profiles": resolved_profiles,
        "package_specs": list(pinned_specs),
        "wheel_count": len(wheels_meta),
        "wheels": wheels_meta,
    }
    manifest_file = out_path / WHEELHOUSE_MANIFEST_NAME
    manifest_file.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    if on_output is not None:
        on_output(
            f"Wrote {WHEELHOUSE_MANIFEST_NAME} ({len(wheels_meta)} wheels) in {out_path}"
        )
    return manifest


def _normalize_dist_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", str(name or "").strip()).lower()


def _spec_dist_name(spec: str) -> str:
    head = re.split(r"[<>=!~;\[\s@]", str(spec or "").strip(), maxsplit=1)[0]
    return _normalize_dist_name(head)


def _version_sort_key(version: str) -> tuple[tuple[int, object], ...]:
    tokens = re.findall(r"\d+|[a-zA-Z]+", version)
    return tuple((0, int(tok)) if tok.isdigit() else (1, tok.lower()) for tok in tokens)


def _wheel_compatibility_rank(
    py_tags: Sequence[str],
    abi_tags: Sequence[str],
    plat_tags: Sequence[str],
) -> int | None:
    maj, min_ = sys.version_info.major, sys.version_info.minor
    gil_disabled = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    exact_cp = f"cp{maj}{min_}"
    exact_abi = f"{exact_cp}t" if gil_disabled else exact_cp

    py_abi_score = 0
    for py_tag in py_tags:
        for abi_tag in abi_tags:
            if py_tag == exact_cp and abi_tag == exact_abi:
                py_abi_score = max(py_abi_score, 40)
            elif (
                not gil_disabled
                and abi_tag == "abi3"
                and sys.implementation.name == "cpython"
            ):
                if py_tag in {"cp3", "py3"}:
                    py_abi_score = max(py_abi_score, 30)
                elif py_tag.startswith("cp3") and py_tag[3:].isdigit():
                    if int(py_tag[3:]) <= min_:
                        py_abi_score = max(py_abi_score, 30)
            elif abi_tag == "none":
                if py_tag in {exact_cp, f"py{maj}{min_}"}:
                    py_abi_score = max(py_abi_score, 25)
                elif py_tag in {f"py{maj}", "py3", "py2.py3"}:
                    py_abi_score = max(py_abi_score, 20)
                elif py_tag.startswith("py3") and py_tag[3:].isdigit():
                    if int(py_tag[3:]) <= min_:
                        py_abi_score = max(py_abi_score, 22)
    if py_abi_score == 0:
        return None

    raw_arch = (platform.machine() or "").strip().lower()
    arch_norm = {"amd64": "x86_64"}.get(raw_arch, raw_arch)
    plat_score = 0
    for ptag in plat_tags:
        pt = ptag.lower()
        if pt == "any":
            plat_score = max(plat_score, 10)
            continue
        if sys.platform == "darwin" and pt.startswith("macosx_"):
            m = re.match(r"^macosx_(\d+)_(\d+)_(.+)$", pt)
            if not m:
                continue
            w_maj, w_min, w_arch = int(m.group(1)), int(m.group(2)), m.group(3)
            mac_ver_str = platform.mac_ver()[0] or ""
            if mac_ver_str:
                vparts = [p for p in mac_ver_str.split(".") if p.isdigit()]
                if len(vparts) >= 2 and (w_maj, w_min) > (
                    int(vparts[0]),
                    int(vparts[1]),
                ):
                    continue
            if w_arch == arch_norm:
                plat_score = max(plat_score, 30)
            elif w_arch == "universal2" and arch_norm in {"arm64", "x86_64"}:
                plat_score = max(plat_score, 25)
        elif sys.platform.startswith("linux"):
            linux_arch = {"arm64": "aarch64"}.get(arch_norm, arch_norm)
            if pt in {
                f"linux_{linux_arch}",
                f"manylinux1_{linux_arch}",
                f"manylinux2010_{linux_arch}",
                f"manylinux2014_{linux_arch}",
            } or (pt.startswith("manylinux_") and pt.endswith(f"_{linux_arch}")):
                plat_score = max(plat_score, 30)
        elif sys.platform == "win32":
            if (arch_norm in {"x86_64", "amd64"} and pt == "win_amd64") or (
                arch_norm in {"arm64", "aarch64"} and pt == "win_arm64"
            ):
                plat_score = max(plat_score, 30)

    if plat_score == 0:
        return None
    return py_abi_score + plat_score


def _version_tuple_cmp(lhs: str, rhs: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    def _nums(val: str) -> list[int]:
        return [int(x) for x in re.findall(r"\d+", val)] or [0]

    a, b = _nums(lhs), _nums(rhs)
    width = max(len(a), len(b))
    a.extend([0] * (width - len(a)))
    b.extend([0] * (width - len(b)))
    return tuple(a), tuple(b)


def _evaluate_pep508_marker(marker: str) -> bool:
    """Safely evaluate PEP 508 environment markers via AST (never ``eval``)."""
    env = {
        "os_name": os.name,
        "sys_platform": sys.platform,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "implementation_name": sys.implementation.name,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "python_full_version": platform.python_version(),
        "extra": "",
    }

    def _eval_node(node: ast.AST) -> object:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(bool(_eval_node(v)) for v in node.values)
            if isinstance(node.op, ast.Or):
                return any(bool(_eval_node(v)) for v in node.values)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not bool(_eval_node(node.operand))
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left = str(_eval_node(node.left))
            right = str(_eval_node(node.comparators[0]))
            op = node.ops[0]
            is_ver = (
                isinstance(node.left, ast.Name)
                and node.left.id in {"python_version", "python_full_version"}
            ) or (
                isinstance(node.comparators[0], ast.Name)
                and node.comparators[0].id in {"python_version", "python_full_version"}
            )
            lhs: object = left
            rhs: object = right
            if is_ver and isinstance(
                op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
            ):
                lhs, rhs = _version_tuple_cmp(left, right)
            if isinstance(op, ast.Eq):
                return lhs == rhs
            if isinstance(op, ast.NotEq):
                return lhs != rhs
            if isinstance(op, ast.Lt):
                return lhs < rhs  # type: ignore[operator]
            if isinstance(op, ast.LtE):
                return lhs <= rhs  # type: ignore[operator]
            if isinstance(op, ast.Gt):
                return lhs > rhs  # type: ignore[operator]
            if isinstance(op, ast.GtE):
                return lhs >= rhs  # type: ignore[operator]
            if isinstance(op, ast.In):
                return left in right
            if isinstance(op, getattr(ast, "Not" + "In")):
                return left not in right
        raise ValueError(f"Unsupported marker AST node: {ast.dump(node)}")

    try:
        parsed = ast.parse(marker.strip(), mode="eval")
        return bool(_eval_node(parsed))
    except Exception:
        # Fail conservatively: skip ambiguous optional markers unless extra-free.
        return "extra" not in marker


_FROZEN_BUNDLED_DIST_IMPORTS: dict[str, str] = {
    "numpy": "numpy",
    "pillow": "PIL",
    "opencv-python": "cv2",
    "opencv-contrib-python": "cv2",
    "opencv-python-headless": "cv2",
    "pyyaml": "yaml",
    "pyside6-essentials": "PySide6",
    "shiboken6": "shiboken6",
    "pycocotools": "pycocotools",
    "shapely": "shapely",
    "psutil": "psutil",
}


def _base_bundled_distributions(target_site_dir: Path) -> set[str]:
    """Return normalized distribution names already satisfied by the base runtime."""
    target_resolved = target_site_dir.resolve()
    base_paths = [
        p for p in sys.path if p and Path(p).expanduser().resolve() != target_resolved
    ]
    present: set[str] = set()
    for dist in importlib_metadata.distributions(path=base_paths):
        try:
            dname = dist.metadata["Name"]
            if dname:
                present.add(_normalize_dist_name(dname))
        except Exception:
            continue
    if getattr(sys, "frozen", False):
        for dist_norm, mod_name in _FROZEN_BUNDLED_DIST_IMPORTS.items():
            try:
                spec = importlib_util.find_spec(mod_name)
                if spec is not None:
                    present.add(dist_norm)
            except Exception:
                continue
    return present


def _read_wheel_requires_dist(whl_path: Path) -> list[str]:
    with zipfile.ZipFile(whl_path, "r") as zf:
        meta_names = [
            name
            for name in zf.namelist()
            if name.endswith(".dist-info/METADATA") and name.count("/") == 1
        ]
        if not meta_names:
            return []
        raw = zf.read(meta_names[0]).decode("utf-8", errors="replace")
    msg = EmailHeaderParser().parsestr(raw)
    reqs = msg.get_all("Requires-Dist") or []
    applicable: list[str] = []
    for req in reqs:
        parts = req.split(";", 1)
        if len(parts) == 2 and not _evaluate_pep508_marker(parts[1]):
            continue
        dist_norm = _spec_dist_name(parts[0])
        if dist_norm:
            applicable.append(dist_norm)
    return applicable


def _extract_verified_wheel(whl_path: Path, target_site_dir: Path) -> None:
    """Verify PEP 427 RECORD SHA256 digests and extract safely into ``target_site_dir``."""
    site_root = target_site_dir.expanduser().resolve()
    site_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(whl_path, "r") as zf:
        names = zf.namelist()
        record_candidates = [
            n for n in names if n.endswith(".dist-info/RECORD") and n.count("/") == 1
        ]
        if not record_candidates:
            raise RuntimeError(
                f"Invalid wheel (missing .dist-info/RECORD): {whl_path.name}"
            )
        record_member = record_candidates[0]
        record_hashes: dict[str, tuple[str | None, int | None]] = {}
        record_text = zf.read(record_member).decode("utf-8", errors="replace")
        for row in csv.reader(io.StringIO(record_text)):
            if not row:
                continue
            rec_path = row[0]
            rec_hash = row[1] if len(row) > 1 and row[1] else None
            rec_size = int(row[2]) if len(row) > 2 and row[2].isdigit() else None
            record_hashes[rec_path] = (rec_hash, rec_size)

        for info in zf.infolist():
            member = info.filename
            if info.is_dir():
                continue
            norm_parts = Path(member).parts
            if not norm_parts or ".." in norm_parts or member.startswith(("/", "\\")):
                raise RuntimeError(
                    f"Unsafe archive path in wheel {whl_path.name}: {member}"
                )

            rel_parts = list(norm_parts)
            if len(rel_parts) >= 3 and rel_parts[0].endswith(".data"):
                section = rel_parts[1]
                if section in {"purelib", "platlib"}:
                    rel_parts = rel_parts[2:]
                else:
                    # Skip console scripts / headers / data outside site-packages in frozen mode
                    continue

            dest = (site_root / Path(*rel_parts)).resolve()
            try:
                dest.relative_to(site_root)
            except ValueError as exc:
                raise RuntimeError(
                    f"Wheel path escapes site-packages in {whl_path.name}: {member}"
                ) from exc

            hasher = hashlib.sha256()
            total_bytes = 0
            dest.parent.mkdir(parents=True, exist_ok=True)
            part_dest = dest.with_name(f".{dest.name}.annolid-tmp")
            try:
                with zf.open(info, "r") as src, part_dest.open("wb") as out_f:
                    while True:
                        chunk = src.read(256 * 1024)
                        if not chunk:
                            break
                        hasher.update(chunk)
                        total_bytes += len(chunk)
                        out_f.write(chunk)

                if member != record_member and member in record_hashes:
                    exp_hash, exp_size = record_hashes[member]
                    if exp_size is not None and total_bytes != exp_size:
                        raise RuntimeError(
                            f"RECORD size mismatch in {whl_path.name} for {member}"
                        )
                    if exp_hash and exp_hash.startswith("sha256="):
                        b64 = (
                            base64.urlsafe_b64encode(hasher.digest())
                            .rstrip(b"=")
                            .decode("ascii")
                        )
                        if f"sha256={b64}" != exp_hash:
                            raise RuntimeError(
                                f"RECORD SHA256 mismatch in {whl_path.name} for {member}"
                            )
                part_dest.replace(dest)
            except Exception:
                part_dest.unlink(missing_ok=True)
                raise


def _install_wheels_from_wheelhouse_pure_python(
    package_specs: tuple[str, ...],
    *,
    wheelhouse_dir: Path,
    target_site_dir: Path,
    on_output: OutputCallback | None = None,
    cancel_event: threading.Event | None = None,
    capability_name: str = "runtime",
) -> tuple[str, ...]:
    """Zero-toolchain PEP 427/425 offline wheel installer for air-gapped frozen bundles."""
    verify_wheelhouse_bundle(wheelhouse_dir)

    # Index compatible wheels by normalized distribution name -> best (version, rank, Path)
    best_wheels: dict[str, tuple[tuple[tuple[int, object], ...], int, Path]] = {}
    for whl_path in sorted(wheelhouse_dir.glob("*.whl")):
        stem = whl_path.name[:-4]
        parts = stem.split("-")
        if len(parts) not in (5, 6):
            continue
        dist_norm = _normalize_dist_name(parts[0])
        ver_str = parts[1]
        py_tags = parts[-3].split(".")
        abi_tags = parts[-2].split(".")
        plat_tags = parts[-1].split(".")
        rank = _wheel_compatibility_rank(py_tags, abi_tags, plat_tags)
        if rank is None:
            continue
        key = (_version_sort_key(ver_str), rank)
        prev = best_wheels.get(dist_norm)
        if prev is None or key >= (prev[0], prev[1]):
            best_wheels[dist_norm] = (key[0], key[1], whl_path)

    bundled = _base_bundled_distributions(target_site_dir)
    queue_dists: deque[str] = deque()
    missing_roots: list[str] = []

    for spec in package_specs:
        dnorm = _spec_dist_name(spec)
        if dnorm in bundled:
            continue
        if dnorm not in best_wheels:
            if dnorm == "onnxruntime" and "onnxruntime-gpu" in best_wheels:
                dnorm = "onnxruntime-gpu"
            elif dnorm == "onnxruntime-gpu" and "onnxruntime" in best_wheels:
                dnorm = "onnxruntime"
            else:
                missing_roots.append(spec)
                continue
        queue_dists.append(dnorm)

    if missing_roots:
        raise RuntimeError(
            f"Offline wheelhouse {wheelhouse_dir} has no compatible wheel for "
            f"Python {sys.version_info.major}.{sys.version_info.minor} ({sys.platform}/{platform.machine()}): "
            + ", ".join(missing_roots)
        )

    ordered_wheels: list[Path] = []
    visited: set[str] = set()
    while queue_dists:
        curr = queue_dists.popleft()
        if curr in visited or curr in bundled:
            continue
        visited.add(curr)
        entry = best_wheels.get(curr)
        if entry is None:
            continue
        whl_file = entry[2]
        ordered_wheels.append(whl_file)
        for req_dist in _read_wheel_requires_dist(whl_file):
            if (
                req_dist not in visited
                and req_dist not in bundled
                and req_dist in best_wheels
            ):
                queue_dists.append(req_dist)

    for whl_file in ordered_wheels:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeInstallCancelledError(
                f"Installation of '{capability_name}' was cancelled."
            )
        if on_output is not None:
            on_output(
                f"[wheelhouse] Verifying & extracting {whl_file.name} -> {target_site_dir}"
            )
        _extract_verified_wheel(whl_file, target_site_dir)

    activate_user_runtime_site(target_site_dir)
    return package_specs


def _run_streaming_command(
    command: list[str],
    *,
    on_output: OutputCallback | None = None,
    cancel_event: threading.Event | None = None,
    capability_name: str,
) -> tuple[int, str]:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeInstallCancelledError(
            f"Installation of '{capability_name}' was cancelled."
        )

    if on_output is not None:
        on_output("$ " + " ".join(command))

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines: list[str] = []
    line_queue: queue.Queue[str | None] = queue.Queue()

    def _reader() -> None:
        stream = proc.stdout
        if stream is None:
            line_queue.put(None)
            return
        try:
            for raw_line in iter(stream.readline, ""):
                line_queue.put(raw_line)
        finally:
            try:
                stream.close()
            except Exception:
                pass
            line_queue.put(None)

    reader_thread = threading.Thread(
        target=_reader,
        name=f"annolid-cap-install-{capability_name}",
        daemon=True,
    )
    reader_thread.start()

    eof = False
    while not eof:
        if cancel_event is not None and cancel_event.is_set():
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            raise RuntimeInstallCancelledError(
                f"Installation of '{capability_name}' was cancelled."
            )
        try:
            item = line_queue.get(timeout=0.1)
        except queue.Empty:
            if proc.poll() is not None and not reader_thread.is_alive():
                break
            continue
        if item is None:
            eof = True
            break
        cleaned = item.rstrip("\r\n")
        output_lines.append(cleaned)
        if on_output is not None:
            on_output(cleaned)

    returncode = proc.wait()
    reader_thread.join(timeout=1.0)
    return returncode, "\n".join(output_lines)


def _guard_installable(
    key: str,
    *,
    auto_install: bool | None = None,
    find_spec: SpecFinder | None = None,
    allow_frozen_user_runtime: bool = False,
) -> CapabilityStatus:
    status = check_capability(key, find_spec=find_spec)
    should_install = (
        _auto_install_optional_deps_enabled()
        if auto_install is None
        else bool(auto_install)
    )
    is_frozen = bool(getattr(sys, "frozen", False))
    frozen_allowed = is_frozen and _frozen_install_opted_in(allow_frozen_user_runtime)
    if not should_install or (is_frozen and not frozen_allowed):
        message = capability_message(status)
        if is_frozen:
            message += (
                ". Frozen desktop bundles are read-only; install this feature via "
                "Settings -> Runtime & Optional Packages (isolated ~/.annolid/runtime), "
                f"set {FROZEN_USER_RUNTIME_ENV}=1 / {WHEELHOUSE_ENV}=<dir>, "
                "or run Annolid from a Python environment."
            )
        raise RuntimeError(message)
    return status


def _execute_frozen_user_runtime_install(
    missing: tuple[RuntimeDependency, ...],
    *,
    key: str,
    status: CapabilityStatus,
    wheelhouse: str | os.PathLike[str] | None = None,
    offline: bool | None = None,
    on_output: OutputCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[str, ...]:
    site_dir = user_runtime_site_dir()
    site_dir.mkdir(parents=True, exist_ok=True)
    wh_dir = resolve_wheelhouse_dir(wheelhouse)
    if wh_dir is not None and (wh_dir / WHEELHOUSE_MANIFEST_NAME).is_file():
        verify_wheelhouse_bundle(wh_dir)

    pinned_specs = _pin_abi_sensitive_specs(tuple(dep.package_spec for dep in missing))
    has_external_tool = bool(_find_uv_binary() or _find_matching_host_python())

    if not has_external_tool:
        if wh_dir is None:
            raise RuntimeError(
                capability_message(status)
                + ". Frozen desktop bundles are read-only and no external 'uv', "
                f"matching Python {sys.version_info.major}.{sys.version_info.minor}, "
                f"or offline wheelhouse ({WHEELHOUSE_ENV}) was found for user-space runtime install."
            )
        return _install_wheels_from_wheelhouse_pure_python(
            pinned_specs,
            wheelhouse_dir=wh_dir,
            target_site_dir=site_dir,
            on_output=on_output,
            cancel_event=cancel_event,
            capability_name=key,
        )

    pins = _frozen_bundled_constraints()
    tmp_constraints: Path | None = None
    try:
        if pins:
            fd, tmp_name = tempfile.mkstemp(prefix="annolid-frozen-", suffix=".txt")
            os.close(fd)
            tmp_constraints = Path(tmp_name)
            tmp_constraints.write_text("\n".join(pins) + "\n", encoding="utf-8")

        effective_specs, steps = _build_smart_install_plan(
            missing,
            wheelhouse=wh_dir,
            offline=offline,
            target_site_dir=site_dir,
            constraints_file=tmp_constraints,
        )
        for primary_cmd, fallback_cmd in steps:
            rc, combined_output = _run_streaming_command(
                primary_cmd,
                on_output=on_output,
                cancel_event=cancel_event,
                capability_name=key,
            )
            if rc != 0 and fallback_cmd is not None:
                if on_output is not None:
                    on_output("Primary index failed; retrying with fallback source...")
                rc, combined_output = _run_streaming_command(
                    fallback_cmd,
                    on_output=on_output,
                    cancel_event=cancel_event,
                    capability_name=key,
                )
            if rc != 0:
                tail = "\n".join(combined_output.splitlines()[-40:]).strip()
                detail = f"\n\npip output:\n{tail}" if tail else ""
                raise RuntimeError(capability_message(status) + detail)
    finally:
        if tmp_constraints is not None:
            tmp_constraints.unlink(missing_ok=True)

    activate_user_runtime_site(site_dir)
    return effective_specs


def install_capability_stream(
    name: str,
    *,
    auto_install: bool | None = None,
    wheelhouse: str | os.PathLike[str] | None = None,
    offline: bool | None = None,
    allow_frozen_user_runtime: bool = False,
    on_output: OutputCallback | None = None,
    cancel_event: threading.Event | None = None,
    find_spec: SpecFinder | None = None,
) -> tuple[str, ...]:
    """Install missing packages for ``name`` with live line streaming and cancellation."""
    key = name.strip().lower()
    if key not in CAPABILITIES:
        known = ", ".join(list_capabilities())
        raise KeyError(
            f"Unknown runtime capability '{name}'. Known capabilities: {known}"
        )
    capability = CAPABILITIES[key]

    with _capability_lock(key):
        if getattr(sys, "frozen", False):
            activate_user_runtime_site()
        missing = _missing_dependencies(capability, find_spec=find_spec)
        if not missing:
            return ()

        status = _guard_installable(
            key,
            auto_install=auto_install,
            find_spec=find_spec,
            allow_frozen_user_runtime=allow_frozen_user_runtime,
        )

        if getattr(sys, "frozen", False):
            effective_specs = _execute_frozen_user_runtime_install(
                missing,
                key=key,
                status=status,
                wheelhouse=wheelhouse,
                offline=offline,
                on_output=on_output,
                cancel_event=cancel_event,
            )
        else:
            wh_dir = resolve_wheelhouse_dir(wheelhouse)
            if wh_dir is not None and (wh_dir / WHEELHOUSE_MANIFEST_NAME).is_file():
                verify_wheelhouse_bundle(wh_dir)
            effective_specs, steps = _build_smart_install_plan(
                missing,
                wheelhouse=wh_dir,
                offline=offline,
            )

            for primary_cmd, fallback_cmd in steps:
                rc, combined_output = _run_streaming_command(
                    primary_cmd,
                    on_output=on_output,
                    cancel_event=cancel_event,
                    capability_name=key,
                )
                if rc != 0 and fallback_cmd is not None:
                    if on_output is not None:
                        on_output(
                            "Primary accelerator wheel index failed; retrying with fallback source..."
                        )
                    rc, combined_output = _run_streaming_command(
                        fallback_cmd,
                        on_output=on_output,
                        cancel_event=cancel_event,
                        capability_name=key,
                    )
                if rc != 0:
                    tail = "\n".join(combined_output.splitlines()[-40:]).strip()
                    detail = f"\n\npip output:\n{tail}" if tail else ""
                    raise RuntimeError(capability_message(status) + detail)

        importlib.invalidate_caches()
        still_missing = _missing_dependencies(capability, find_spec=find_spec)
        if still_missing:
            refreshed = check_capability(key, find_spec=find_spec)
            raise RuntimeError(capability_message(refreshed))

        return effective_specs


def install_profile_stream(
    profile_name: str,
    *,
    wheelhouse: str | os.PathLike[str] | None = None,
    offline: bool | None = None,
    allow_frozen_user_runtime: bool = False,
    on_output: OutputCallback | None = None,
    cancel_event: threading.Event | None = None,
    find_spec: SpecFinder | None = None,
) -> tuple[str, ...]:
    """Install all capabilities belonging to a named install profile."""
    key = profile_name.strip().lower()
    if key not in INSTALL_PROFILES:
        known = ", ".join(sorted(INSTALL_PROFILES))
        raise KeyError(f"Unknown install profile '{profile_name}'. Known: {known}")
    caps = tuple(INSTALL_PROFILES[key]["capabilities"])  # type: ignore[arg-type]
    installed: list[str] = []
    for cap_name in caps:
        specs = install_capability_stream(
            str(cap_name),
            auto_install=True,
            wheelhouse=wheelhouse,
            offline=offline,
            allow_frozen_user_runtime=allow_frozen_user_runtime,
            on_output=on_output,
            cancel_event=cancel_event,
            find_spec=find_spec,
        )
        for spec in specs:
            if spec not in installed:
                installed.append(spec)
    return tuple(installed)


def ensure_capability(
    name: str,
    *,
    reason: str = "",
    auto_install: bool | None = None,
    wheelhouse: str | os.PathLike[str] | None = None,
    offline: bool | None = None,
    allow_frozen_user_runtime: bool = False,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
    find_spec: SpecFinder | None = None,
) -> tuple[str, ...]:
    """Ensure an optional runtime exists, installing missing packages on first use.

    Returns the package specs installed during this call. Frozen desktop bundles
    never mutate the bundled application directory; unless user-space runtime
    installation is explicitly invoked (GUI prompt or ``ANNOLID_FROZEN_USER_RUNTIME=1``),
    headless calls fail fast with an actionable install command.
    """
    key = name.strip().lower()
    if key not in CAPABILITIES:
        known = ", ".join(list_capabilities())
        raise KeyError(
            f"Unknown runtime capability '{name}'. Known capabilities: {known}"
        )
    capability = CAPABILITIES[key]

    if getattr(sys, "frozen", False) and find_spec is None:
        activate_user_runtime_site()

    if (
        runner is None
        and find_spec is None
        and auto_install is not False
        and _INTERACTIVE_INSTALL_HANDLER is not None
        and (not getattr(sys, "frozen", False) or frozen_user_runtime_enabled())
        and _auto_install_optional_deps_enabled()
        and _missing_dependencies(capability)
    ):
        handled = _INTERACTIVE_INSTALL_HANDLER(key, reason)
        if handled is not None:
            return handled

    with _capability_lock(key):
        missing = _missing_dependencies(capability, find_spec=find_spec)
        if not missing:
            return ()

        status = _guard_installable(
            key,
            auto_install=auto_install,
            find_spec=find_spec,
            allow_frozen_user_runtime=allow_frozen_user_runtime,
        )

        # Preserve deterministic single-command injection contract for callers/tests
        # that pass an explicit subprocess runner.
        if runner is not None:
            wh_dir = resolve_wheelhouse_dir(wheelhouse)
            package_specs = tuple(dependency.package_spec for dependency in missing)
            command = _install_command(
                package_specs,
                wheelhouse_dir=wh_dir,
                offline=is_offline_mode(offline),
            )
            completed = runner(command, check=False, capture_output=True, text=True)
            if completed.returncode != 0:
                stderr = str(getattr(completed, "stderr", "") or "").strip()
                detail = f"\n\npip output:\n{stderr}" if stderr else ""
                raise RuntimeError(capability_message(status) + detail)

            importlib.invalidate_caches()
            still_missing = _missing_dependencies(capability, find_spec=find_spec)
            if still_missing:
                refreshed = check_capability(key, find_spec=find_spec)
                raise RuntimeError(capability_message(refreshed))
            return package_specs

        if getattr(sys, "frozen", False):
            effective_specs = _execute_frozen_user_runtime_install(
                missing,
                key=key,
                status=status,
                wheelhouse=wheelhouse,
                offline=offline,
            )
            importlib.invalidate_caches()
            still_missing = _missing_dependencies(capability, find_spec=find_spec)
            if still_missing:
                refreshed = check_capability(key, find_spec=find_spec)
                raise RuntimeError(capability_message(refreshed))
            return effective_specs

        wh_dir = resolve_wheelhouse_dir(wheelhouse)
        if wh_dir is not None and (wh_dir / WHEELHOUSE_MANIFEST_NAME).is_file():
            verify_wheelhouse_bundle(wh_dir)
        effective_specs, steps = _build_smart_install_plan(
            missing,
            wheelhouse=wh_dir,
            offline=offline,
        )
        reason_suffix = f" for {reason.strip()}" if str(reason or "").strip() else ""
        print(
            f"[annolid] Installing optional '{key}' runtime packages{reason_suffix} "
            f"({', '.join(effective_specs)})...",
            file=sys.stderr,
        )
        for primary_cmd, fallback_cmd in steps:
            completed = subprocess.run(
                primary_cmd, check=False, capture_output=True, text=True
            )
            if completed.returncode != 0 and fallback_cmd is not None:
                completed = subprocess.run(
                    fallback_cmd, check=False, capture_output=True, text=True
                )
            if completed.returncode != 0:
                stderr = str(
                    getattr(completed, "stderr", "")
                    or getattr(completed, "stdout", "")
                    or ""
                ).strip()
                detail = f"\n\npip output:\n{stderr}" if stderr else ""
                raise RuntimeError(capability_message(status) + detail)

        importlib.invalidate_caches()
        still_missing = _missing_dependencies(capability, find_spec=find_spec)
        if still_missing:
            refreshed = check_capability(key, find_spec=find_spec)
            raise RuntimeError(capability_message(refreshed))

        return effective_specs


def format_capability_report(
    names: tuple[str, ...] | None = None, *, find_spec: SpecFinder | None = None
) -> str:
    selected = names or list_capabilities()
    return "\n".join(
        capability_message(check_capability(name, find_spec=find_spec))
        for name in selected
    )


__all__ = [
    "AUTO_INSTALL_OPTIONAL_DEPS_ENV",
    "CAPABILITIES",
    "CapabilityStatus",
    "FROZEN_USER_RUNTIME_ENV",
    "INSTALL_PROFILES",
    "NO_GPU_ENV",
    "OFFLINE_ENV",
    "RuntimeCapability",
    "RuntimeDependency",
    "RuntimeInstallCancelledError",
    "TORCH_INDEX_URL_ENV",
    "USER_RUNTIME_DIR_ENV",
    "WHEELHOUSE_ENV",
    "WHEELHOUSE_MANIFEST_NAME",
    "activate_user_runtime_site",
    "can_install_in_frozen_user_runtime",
    "capability_message",
    "capability_status_dict",
    "check_capability",
    "detect_accelerator_summary",
    "ensure_capability",
    "export_wheelhouse_bundle",
    "format_capability_report",
    "frozen_user_runtime_enabled",
    "install_capability_stream",
    "install_profile_stream",
    "is_offline_mode",
    "list_capabilities",
    "resolve_wheelhouse_dir",
    "set_interactive_install_handler",
    "user_runtime_abi_slug",
    "user_runtime_site_dir",
    "verify_wheelhouse_bundle",
]
