from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


AUTO_INSTALL_ENV = "ANNOLID_AUTO_INSTALL_CUTIE_DEPS"


@dataclass(frozen=True)
class RuntimeDependency:
    import_name: str
    package_spec: str


CUTIE_RUNTIME_DEPENDENCIES: tuple[RuntimeDependency, ...] = (
    RuntimeDependency("cv2", "opencv-contrib-python>=4.1.2.30"),
    RuntimeDependency("torch", "torch>=2.5.0"),
    RuntimeDependency("torchvision", "torchvision>=0.20.0"),
    RuntimeDependency("omegaconf", "omegaconf>=2.3.0"),
    RuntimeDependency("hydra", "hydra-core>=1.3.2"),
    RuntimeDependency("PIL", "Pillow>=9.3.0,<12.0"),
    RuntimeDependency("pycocotools", "pycocotools>=2.0.2"),
    RuntimeDependency("einops", "einops>=0.6.0"),
)


def _auto_install_enabled(value: str | None = None) -> bool:
    raw = os.environ.get(AUTO_INSTALL_ENV) if value is None else value
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def missing_cutie_runtime_dependencies(
    *,
    finder: Callable[[str], object | None] | None = None,
    dependencies: Iterable[RuntimeDependency] = CUTIE_RUNTIME_DEPENDENCIES,
) -> list[RuntimeDependency]:
    find_spec = finder or importlib.util.find_spec
    missing: list[RuntimeDependency] = []
    for dependency in dependencies:
        if find_spec(dependency.import_name) is None:
            missing.append(dependency)
    return missing


def _install_command(package_specs: Sequence[str]) -> list[str]:
    uv_cmd = shutil.which("uv")
    if uv_cmd:
        return [uv_cmd, "pip", "install", "--python", sys.executable, *package_specs]
    return [sys.executable, "-m", "pip", "install", *package_specs]


def _format_missing_message(missing: Sequence[RuntimeDependency]) -> str:
    package_specs = [dependency.package_spec for dependency in missing]
    command = " ".join(_install_command(package_specs))
    imports = ", ".join(dependency.import_name for dependency in missing)
    return (
        "Cutie tracking requires additional runtime packages that are missing "
        f"from the active Python environment: {imports}.\n"
        f"Install them with:\n  {command}\n"
        f"Automatic installation can be disabled with {AUTO_INSTALL_ENV}=0."
    )


def ensure_cutie_runtime_dependencies(
    *,
    auto_install: bool | None = None,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
    finder: Callable[[str], object | None] | None = None,
    dependencies: Iterable[RuntimeDependency] = CUTIE_RUNTIME_DEPENDENCIES,
) -> list[str]:
    """Ensure packages needed for Cutie tracking exist in the active env.

    Returns the package specs installed during this call. If everything is
    already available, returns an empty list.
    """
    missing = missing_cutie_runtime_dependencies(
        finder=finder,
        dependencies=dependencies,
    )
    if not missing:
        return []

    should_install = _auto_install_enabled() if auto_install is None else auto_install
    if not should_install or getattr(sys, "frozen", False):
        raise RuntimeError(_format_missing_message(missing))

    package_specs = [dependency.package_spec for dependency in missing]
    command = _install_command(package_specs)
    run = runner or subprocess.run
    completed = run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = str(getattr(completed, "stderr", "") or "").strip()
        detail = f"\n\npip output:\n{stderr}" if stderr else ""
        raise RuntimeError(_format_missing_message(missing) + detail)

    importlib.invalidate_caches()
    still_missing = missing_cutie_runtime_dependencies(
        finder=finder,
        dependencies=dependencies,
    )
    if still_missing:
        raise RuntimeError(_format_missing_message(still_missing))

    return package_specs
