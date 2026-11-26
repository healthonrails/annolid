from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from annolid.utils.logger import logger

RAT_NCAP_GITHUB_URL = "https://github.com/rat-ncap/rat-ncap.git"
_PROBE_CACHE_TTL_SECONDS = 120.0
_PROBE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def resolve_local_rat_ncap_repo() -> Path | None:
    """Locate an optional external clone of `rat-ncap/rat-ncap`."""
    env_value = os.environ.get("ANNOLID_RAT_NCAP_PATH", "").strip()
    candidates: list[Path] = []
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.extend(
        [
            Path.home() / "Downloads" / "rat-ncap",
            Path.home() / "Desktop" / "rat-ncap",
            Path.cwd().parent / "rat-ncap",
        ]
    )
    for candidate in candidates:
        try:
            if (
                candidate
                / "projects"
                / "rodent"
                / "envs"
                / "bodies"
                / "rat"
                / "rat.xml"
            ).exists():
                return candidate
        except OSError:
            continue
    return None


def resolve_rat_xml_path(repo_root: str | Path | None = None) -> Path | None:
    """Resolve `rat.xml` from an explicit or auto-discovered `rat-ncap` checkout."""
    root = (
        Path(repo_root).expanduser()
        if repo_root is not None
        else resolve_local_rat_ncap_repo()
    )
    if root is None:
        return None
    candidate = root / "projects" / "rodent" / "envs" / "bodies" / "rat" / "rat.xml"
    try:
        return candidate if candidate.exists() else None
    except OSError:
        return None


def default_rat_ncap_install_dir() -> Path:
    return Path.home() / "Downloads" / "rat-ncap"


def rat_ncap_runtime_python_candidates() -> list[Path]:
    candidates: list[Path] = []
    repo = resolve_local_rat_ncap_repo()
    if repo is not None:
        candidates.append(repo / ".venv" / "bin" / "python")
    candidates.extend(
        [
            Path.cwd() / ".venv" / "bin" / "python",
            Path(sys.executable),
        ]
    )
    seen: set[str] = set()
    resolved: list[Path] = []
    for candidate in candidates:
        key = str(candidate.expanduser())
        if key in seen:
            continue
        seen.add(key)
        resolved.append(Path(key))
    return resolved


def build_clone_rat_ncap_command(
    repo_url: str = RAT_NCAP_GITHUB_URL, destination: str | Path | None = None
) -> list[str]:
    dest = (
        Path(destination).expanduser()
        if destination is not None
        else default_rat_ncap_install_dir()
    )
    return ["git", "clone", str(repo_url), str(dest)]


def build_probe_rat_ncap_command(python_executable: str | Path) -> list[str]:
    return [
        str(Path(python_executable).expanduser()),
        "-m",
        "annolid.simulation.rat_ncap_live",
        "--probe",
        "--json",
    ]


def build_live_rat_ncap_command(
    python_executable: str | Path,
    *,
    out_path: str | Path,
    steps: int = 180,
    seed: int = 7,
    behavior: str = "trot",
    interface: str = "flexor_extensor",
    coat: str = "black_hooded",
    target_speed: float | None = None,
    ablation: str = "none",
    ablation_scale: float = 0.0,
    rg_noise_std: float = 0.0,
    terrain_bump_cm: float = 0.0,
) -> list[str]:
    command = [
        str(Path(python_executable).expanduser()),
        "-m",
        "annolid.simulation.rat_ncap_live",
        "--behavior",
        str(behavior),
        "--interface",
        str(interface),
        "--coat",
        str(coat),
        "--ablation",
        str(ablation),
        "--ablation-scale",
        str(float(ablation_scale)),
        "--rg-noise-std",
        str(float(rg_noise_std)),
        "--terrain-bump-cm",
        str(float(terrain_bump_cm)),
        "--out",
        str(Path(out_path).expanduser()),
        "--steps",
        str(int(steps)),
        "--seed",
        str(int(seed)),
    ]
    if target_speed is not None:
        command.extend(["--target-speed", str(float(target_speed))])
    return command


def probe_rat_ncap_runtime(python_executable: str | Path) -> dict[str, Any]:
    python_path = str(Path(python_executable).expanduser())
    cached = _PROBE_CACHE.get(python_path)
    now = time.monotonic()
    if cached is not None and (now - cached[0]) < _PROBE_CACHE_TTL_SECONDS:
        return dict(cached[1])
    command = build_probe_rat_ncap_command(python_executable)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(Path.cwd()),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    payload: dict[str, Any]
    try:
        payload = json.loads(stdout) if stdout else {}
    except json.JSONDecodeError:
        payload = {}
    payload.setdefault("python", python_path)
    payload["returncode"] = int(completed.returncode)
    if stderr:
        payload["stderr"] = stderr
    payload["ready"] = bool(payload.get("ready")) and completed.returncode == 0
    frozen = dict(payload)
    _PROBE_CACHE[python_path] = (now, frozen)
    logger.info(
        "Rat NCAP probe finished for %s in %.1fms (ready=%s, mujoco=%s)",
        python_path,
        (time.perf_counter() - started) * 1000.0,
        bool(frozen.get("ready")),
        bool(frozen.get("mujoco_available")),
    )
    return dict(frozen)


def pick_ready_rat_ncap_runtime() -> tuple[Path | None, dict[str, Any]]:
    best: dict[str, Any] = {}
    for candidate in rat_ncap_runtime_python_candidates():
        if not candidate.exists():
            continue
        payload = probe_rat_ncap_runtime(candidate)
        if payload.get("ready"):
            return candidate, payload
        if not best:
            best = payload
    return None, best


def summarize_rat_ncap_status() -> dict[str, Any]:
    repo_root = resolve_local_rat_ncap_repo()
    rat_xml = resolve_rat_xml_path(repo_root)
    candidates = []
    ready = False
    mujoco_ready = False
    for candidate in rat_ncap_runtime_python_candidates():
        record: dict[str, Any] = {
            "python": str(candidate),
            "exists": candidate.exists(),
        }
        if candidate.exists():
            probe = probe_rat_ncap_runtime(candidate)
            record.update(probe)
            ready = ready or bool(probe.get("ready"))
            mujoco_ready = mujoco_ready or bool(probe.get("mujoco_rat_ready"))
        candidates.append(record)
    return {
        "repo_root": str(repo_root) if repo_root is not None else None,
        "rat_xml_path": str(rat_xml) if rat_xml is not None else None,
        "ready": ready,
        "mujoco_rat_ready": mujoco_ready,
        "candidates": candidates,
    }
