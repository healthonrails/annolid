"""Optional runtime capability checks for lean and frozen installations."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import util as importlib_util
import importlib
import os
import shutil
import subprocess
import sys
from typing import Callable


SpecFinder = Callable[[str], object | None]
AUTO_INSTALL_OPTIONAL_DEPS_ENV = "ANNOLID_AUTO_INSTALL_OPTIONAL_DEPS"


@dataclass(frozen=True)
class RuntimeDependency:
    import_name: str
    package_spec: str


@dataclass(frozen=True)
class RuntimeCapability:
    name: str
    dependencies: tuple[RuntimeDependency, ...]
    package_specs: tuple[str, ...]
    docs_url: str
    description: str

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

    @property
    def state(self) -> str:
        return "available" if self.available else "missing"


CAPABILITIES: dict[str, RuntimeCapability] = {
    "ml": RuntimeCapability(
        name="ml",
        dependencies=(
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
        docs_url="https://annolid.com/installation/",
        description="General ML inference and model tooling runtime.",
    ),
    "tracking": RuntimeCapability(
        name="tracking",
        dependencies=(
            RuntimeDependency("cv2", "opencv-contrib-python>=4.1.2.30"),
            RuntimeDependency("torch", "torch>=2.5.0"),
            RuntimeDependency("torchvision", "torchvision>=0.20.0"),
            RuntimeDependency("omegaconf", "omegaconf>=2.3.0"),
            RuntimeDependency("hydra", "hydra-core>=1.3.2"),
            RuntimeDependency("onnxruntime", "onnxruntime>=1.23.2"),
            RuntimeDependency("ultralytics", "ultralytics>=8.4.0"),
            RuntimeDependency("lap", "lap"),
        ),
        package_specs=("annolid[tracking]",),
        docs_url="https://annolid.com/installation/",
        description="Common tracking workstation runtime.",
    ),
    "cutie": RuntimeCapability(
        name="cutie",
        dependencies=(
            RuntimeDependency("cv2", "opencv-contrib-python>=4.1.2.30"),
            RuntimeDependency("torch", "torch>=2.5.0"),
            RuntimeDependency("torchvision", "torchvision>=0.20.0"),
            RuntimeDependency("omegaconf", "omegaconf>=2.3.0"),
            RuntimeDependency("hydra", "hydra-core>=1.3.2"),
            RuntimeDependency("PIL", "Pillow>=9.3.0,<12.0"),
            RuntimeDependency("pycocotools", "pycocotools>=2.0.2"),
            RuntimeDependency("einops", "einops>=0.6.0"),
        ),
        package_specs=("annolid[cutie]",),
        docs_url="https://annolid.com/installation/",
        description="Cutie video object segmentation tracking runtime.",
    ),
    "sam3": RuntimeCapability(
        name="sam3",
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
        docs_url="https://annolid.com/installation/",
        description="SAM3 promptable segmentation runtime.",
    ),
    "yolo": RuntimeCapability(
        name="yolo",
        dependencies=(
            RuntimeDependency("ultralytics", "ultralytics>=8.4.0"),
            RuntimeDependency("lap", "lap"),
        ),
        package_specs=("annolid[yolo]",),
        docs_url="https://annolid.com/installation/",
        description="YOLO/YOLOE detection and training runtime.",
    ),
    "bot": RuntimeCapability(
        name="bot",
        dependencies=(
            RuntimeDependency("openai", "openai>=1.40.0"),
            RuntimeDependency("anthropic", "anthropic>=0.49.0"),
            RuntimeDependency("mcp", "mcp>=1.0.0"),
            RuntimeDependency("websockets", "websockets>=12.0"),
            RuntimeDependency("qrcode", "qrcode>=7.4.2"),
        ),
        package_specs=("annolid[bot]",),
        docs_url="https://annolid.com/installation/",
        description="Annolid Bot providers, channels, integrations, and memory.",
    ),
    "large_image": RuntimeCapability(
        name="large_image",
        dependencies=(
            RuntimeDependency("tifffile", "tifffile>=2024.8.30"),
            RuntimeDependency("pyvips", "pyvips>=2.2.3"),
            RuntimeDependency("openslide", "openslide-python>=1.3.1"),
        ),
        package_specs=("annolid[large_image]",),
        docs_url="https://annolid.com/installation/",
        description="Large TIFF, OME-TIFF, and slide-image runtime.",
    ),
    "remote_video": RuntimeCapability(
        name="remote_video",
        dependencies=(RuntimeDependency("ffpyplayer", "ffpyplayer>=4.5.0"),),
        package_specs=("annolid[remote_video]",),
        docs_url="https://annolid.com/installation/",
        description="Network/remote video decoding runtime.",
    ),
}


def _default_find_spec(import_name: str) -> object | None:
    try:
        return importlib_util.find_spec(import_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None


def list_capabilities() -> tuple[str, ...]:
    return tuple(sorted(CAPABILITIES))


def _missing_dependencies(
    capability: RuntimeCapability,
    *,
    find_spec: SpecFinder | None = None,
) -> tuple[RuntimeDependency, ...]:
    finder = find_spec or _default_find_spec
    return tuple(
        dependency
        for dependency in capability.dependencies
        if finder(dependency.import_name) is None
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
    )


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


def _install_command(package_specs: tuple[str, ...]) -> list[str]:
    uv_cmd = shutil.which("uv")
    if uv_cmd:
        return [uv_cmd, "pip", "install", "--python", sys.executable, *package_specs]
    return [sys.executable, "-m", "pip", "install", *package_specs]


def ensure_capability(
    name: str,
    *,
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
    missing = _missing_dependencies(capability, find_spec=find_spec)
    if not missing:
        return ()

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

    package_specs = tuple(dependency.package_spec for dependency in missing)
    command = _install_command(package_specs)
    run = runner or subprocess.run
    completed = run(command, check=False, capture_output=True, text=True)
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
    "RuntimeDependency",
    "RuntimeCapability",
    "capability_message",
    "check_capability",
    "ensure_capability",
    "format_capability_report",
    "list_capabilities",
]
