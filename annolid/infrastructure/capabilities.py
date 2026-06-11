"""Optional runtime capability checks for lean and frozen installations."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import util as importlib_util
from typing import Callable


SpecFinder = Callable[[str], object | None]


@dataclass(frozen=True)
class RuntimeCapability:
    name: str
    import_names: tuple[str, ...]
    package_specs: tuple[str, ...]
    docs_url: str
    description: str

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
        import_names=("torch", "torchvision", "transformers", "onnxruntime"),
        package_specs=("annolid[ml]",),
        docs_url="https://annolid.com/installation/",
        description="General ML inference and model tooling runtime.",
    ),
    "tracking": RuntimeCapability(
        name="tracking",
        import_names=("torch", "torchvision", "omegaconf", "hydra", "onnxruntime"),
        package_specs=("annolid[tracking]",),
        docs_url="https://annolid.com/installation/",
        description="Common tracking workstation runtime.",
    ),
    "cutie": RuntimeCapability(
        name="cutie",
        import_names=("torch", "torchvision", "omegaconf", "hydra", "einops"),
        package_specs=("annolid[cutie]",),
        docs_url="https://annolid.com/installation/",
        description="Cutie video object segmentation tracking runtime.",
    ),
    "sam3": RuntimeCapability(
        name="sam3",
        import_names=("iopath", "ftfy", "tokenizers", "timm"),
        package_specs=("annolid[sam3]",),
        docs_url="https://annolid.com/installation/",
        description="SAM3 promptable segmentation runtime.",
    ),
    "yolo": RuntimeCapability(
        name="yolo",
        import_names=("ultralytics", "lap"),
        package_specs=("annolid[yolo]",),
        docs_url="https://annolid.com/installation/",
        description="YOLO/YOLOE detection and training runtime.",
    ),
    "bot": RuntimeCapability(
        name="bot",
        import_names=("openai", "anthropic", "mcp", "websockets", "qrcode"),
        package_specs=("annolid[bot]",),
        docs_url="https://annolid.com/installation/",
        description="Annolid Bot providers, channels, integrations, and memory.",
    ),
    "large_image": RuntimeCapability(
        name="large_image",
        import_names=("tifffile", "pyvips", "openslide"),
        package_specs=("annolid[large_image]",),
        docs_url="https://annolid.com/installation/",
        description="Large TIFF, OME-TIFF, and slide-image runtime.",
    ),
    "remote_video": RuntimeCapability(
        name="remote_video",
        import_names=("ffpyplayer",),
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
    finder = find_spec or _default_find_spec
    missing = tuple(
        import_name
        for import_name in capability.import_names
        if finder(import_name) is None
    )
    return CapabilityStatus(
        name=capability.name,
        available=not missing,
        missing_imports=missing,
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


def format_capability_report(
    names: tuple[str, ...] | None = None, *, find_spec: SpecFinder | None = None
) -> str:
    selected = names or list_capabilities()
    return "\n".join(
        capability_message(check_capability(name, find_spec=find_spec))
        for name in selected
    )


__all__ = [
    "CAPABILITIES",
    "CapabilityStatus",
    "RuntimeCapability",
    "capability_message",
    "check_capability",
    "format_capability_report",
    "list_capabilities",
]
