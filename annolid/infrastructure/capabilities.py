"""Optional runtime capability manager for lean and on-demand installations."""

from __future__ import annotations

from dataclasses import dataclass
import functools
import importlib
from importlib import metadata as importlib_metadata
from importlib import util as importlib_util
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
from typing import Callable


SpecFinder = Callable[[str], object | None]
OutputCallback = Callable[[str], None]

AUTO_INSTALL_OPTIONAL_DEPS_ENV = "ANNOLID_AUTO_INSTALL_OPTIONAL_DEPS"
NO_GPU_ENV = "ANNOLID_NO_GPU"
TORCH_INDEX_URL_ENV = "ANNOLID_TORCH_INDEX_URL"

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


def detect_accelerator_summary() -> dict[str, object]:
    """Return a lightweight, dependency-free hardware and Python runtime summary."""
    apple_silicon = sys.platform == "darwin" and platform.machine().lower() in {
        "arm64",
        "aarch64",
    }
    writable = (
        not bool(getattr(sys, "frozen", False))
        and _auto_install_optional_deps_enabled()
    )
    if _no_gpu_requested():
        return {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "writable": writable,
            "platform": sys.platform,
            "arch": platform.machine(),
            "accelerator": "cpu",
            "gpu_disabled_by_env": True,
            "has_nvidia_gpu": False,
            "cuda_version": None,
            "has_cuda12": False,
            "gpu_name": None,
            "apple_silicon": apple_silicon,
        }
    has_nvidia, cuda_version, has_cuda12, gpu_name = _probe_nvidia_smi()
    if has_nvidia:
        accelerator = "cuda-12" if has_cuda12 else "cuda"
    elif apple_silicon:
        accelerator = "apple-silicon"
    else:
        accelerator = "cpu"
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "writable": writable,
        "platform": sys.platform,
        "arch": platform.machine(),
        "accelerator": accelerator,
        "gpu_disabled_by_env": False,
        "has_nvidia_gpu": has_nvidia,
        "cuda_version": cuda_version,
        "has_cuda12": has_cuda12,
        "gpu_name": gpu_name,
        "apple_silicon": apple_silicon,
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


def _install_command(
    package_specs: tuple[str, ...],
    *,
    index_url: str | None = None,
    extra_index_url: str | None = None,
) -> list[str]:
    uv_cmd = shutil.which("uv")
    if uv_cmd:
        cmd = [uv_cmd, "pip", "install", "--python", sys.executable]
    else:
        cmd = [sys.executable, "-m", "pip", "install"]
    if index_url:
        cmd.extend(["--index-url", index_url])
    if extra_index_url:
        cmd.extend(["--extra-index-url", extra_index_url])
    cmd.extend(package_specs)
    return cmd


def _build_smart_install_plan(
    missing: tuple[RuntimeDependency, ...],
) -> tuple[tuple[str, ...], list[tuple[list[str], list[str] | None]]]:
    """Build primary (and optional fallback) install commands for real environments.

    Returns ``(effective_specs, steps)`` where each step is
    ``(primary_cmd, optional_fallback_cmd)``.
    """
    raw_specs = tuple(dep.package_spec for dep in missing)
    pinned_specs = _pin_abi_sensitive_specs(raw_specs)
    accel = detect_accelerator_summary()
    torch_index_override = (os.environ.get(TORCH_INDEX_URL_ENV) or "").strip()

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

        if target_index:
            steps.append(
                (
                    _install_command(tuple(torch_specs), index_url=target_index),
                    _install_command(tuple(torch_specs)),
                )
            )
        else:
            steps.append((_install_command(tuple(torch_specs)), None))

    if onnx_specs:
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
                        gpu_specs, extra_index_url=ONNXRUNTIME_CUDA12_EXTRA_INDEX_URL
                    ),
                    _install_command(tuple(onnx_specs)),
                )
            )
        else:
            effective_specs.extend(onnx_specs)
            steps.append((_install_command(tuple(onnx_specs)), None))

    if other_specs:
        effective_specs.extend(other_specs)
        steps.append((_install_command(tuple(other_specs)), None))

    return tuple(effective_specs), steps


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
) -> CapabilityStatus:
    status = check_capability(key, find_spec=find_spec)
    should_install = (
        _auto_install_optional_deps_enabled()
        if auto_install is None
        else bool(auto_install)
    )
    if not should_install or getattr(sys, "frozen", False):
        message = capability_message(status)
        if getattr(sys, "frozen", False):
            message += (
                ". Frozen desktop bundles are read-only; install this feature in "
                "a normal Python environment and launch Annolid from that environment."
            )
        raise RuntimeError(message)
    return status


def install_capability_stream(
    name: str,
    *,
    auto_install: bool | None = None,
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
        missing = _missing_dependencies(capability, find_spec=find_spec)
        if not missing:
            return ()

        status = _guard_installable(key, auto_install=auto_install, find_spec=find_spec)
        effective_specs, steps = _build_smart_install_plan(missing)

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
                        "Primary accelerator wheel index failed; retrying with standard PyPI index..."
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
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
    find_spec: SpecFinder | None = None,
) -> tuple[str, ...]:
    """Ensure an optional runtime exists, installing missing packages on first use.

    Returns the package specs installed during this call. Frozen desktop bundles
    intentionally do not mutate themselves; they raise a clear install message
    instead so users can install the feature in a normal Python environment.
    """
    key = name.strip().lower()
    if key not in CAPABILITIES:
        known = ", ".join(list_capabilities())
        raise KeyError(
            f"Unknown runtime capability '{name}'. Known capabilities: {known}"
        )
    capability = CAPABILITIES[key]

    if (
        runner is None
        and find_spec is None
        and auto_install is not False
        and _INTERACTIVE_INSTALL_HANDLER is not None
        and not getattr(sys, "frozen", False)
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

        status = _guard_installable(key, auto_install=auto_install, find_spec=find_spec)

        # Preserve deterministic single-command injection contract for callers/tests
        # that pass an explicit subprocess runner.
        if runner is not None:
            package_specs = tuple(dependency.package_spec for dependency in missing)
            command = _install_command(package_specs)
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

        effective_specs, steps = _build_smart_install_plan(missing)
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
    "INSTALL_PROFILES",
    "NO_GPU_ENV",
    "RuntimeCapability",
    "RuntimeDependency",
    "RuntimeInstallCancelledError",
    "TORCH_INDEX_URL_ENV",
    "capability_message",
    "capability_status_dict",
    "check_capability",
    "detect_accelerator_summary",
    "ensure_capability",
    "format_capability_report",
    "install_capability_stream",
    "install_profile_stream",
    "list_capabilities",
    "set_interactive_install_handler",
]
