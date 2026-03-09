from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from annolid.utils.logger import logger


FLYBODY_GITHUB_URL = "https://github.com/TuragaLab/flybody.git"
_PROBE_CACHE_TTL_SECONDS = 120.0
_PROBE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def resolve_local_flybody_repo() -> Path | None:
    env_value = os.environ.get("ANNOLID_FLYBODY_PATH", "").strip()
    candidates = []
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.append(Path.home() / "Downloads" / "flybody")
    for candidate in candidates:
        if (candidate / "flybody" / "fruitfly" / "assets" / "fruitfly.xml").exists():
            return candidate
    return None


def default_flybody_install_dir() -> Path:
    return Path.home() / "Downloads" / "flybody"


def default_flybody_runtime_venv() -> Path:
    return Path.cwd() / ".venv311"


def repo_local_flybody_python(repo_root: str | Path | None = None) -> Path | None:
    root = (
        Path(repo_root).expanduser()
        if repo_root is not None
        else resolve_local_flybody_repo()
    )
    if root is None:
        return None
    candidate = root / ".venv" / "bin" / "python"
    return candidate


def flybody_runtime_python_candidates() -> list[Path]:
    candidates = []
    repo_python = repo_local_flybody_python()
    if repo_python is not None:
        candidates.append(repo_python)
    candidates.extend(
        [
            default_flybody_runtime_venv() / "bin" / "python",
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


def build_clone_flybody_command(repo_url: str, destination: str | Path) -> list[str]:
    return ["git", "clone", str(repo_url), str(Path(destination).expanduser())]


def build_setup_flybody_command(
    *,
    repo_root: str | Path,
    flybody_path: str | Path,
    venv_dir: str | Path,
    python_version: str = "3.12",
) -> list[str]:
    script_path = Path(repo_root).expanduser() / "scripts" / "setup_flybody_uv.sh"
    return [
        "bash",
        str(script_path),
        "--flybody-path",
        str(Path(flybody_path).expanduser()),
        "--venv-dir",
        str(Path(venv_dir).expanduser()),
        "--python",
        str(python_version),
    ]


def build_probe_flybody_command(python_executable: str | Path) -> list[str]:
    return [
        str(Path(python_executable).expanduser()),
        "-m",
        "annolid.simulation.flybody_live",
        "--probe",
        "--json",
    ]


def build_live_flybody_command(
    python_executable: str | Path,
    *,
    out_path: str | Path,
    steps: int = 180,
    seed: int = 7,
    behavior: str = "walk_imitation",
) -> list[str]:
    return [
        str(Path(python_executable).expanduser()),
        "-m",
        "annolid.simulation.flybody_live",
        "--behavior",
        str(behavior),
        "--out",
        str(Path(out_path).expanduser()),
        "--steps",
        str(int(steps)),
        "--seed",
        str(int(seed)),
    ]


def probe_flybody_runtime(python_executable: str | Path) -> dict[str, Any]:
    python_path = str(Path(python_executable).expanduser())
    cached = _PROBE_CACHE.get(python_path)
    now = time.monotonic()
    if cached is not None and (now - cached[0]) < _PROBE_CACHE_TTL_SECONDS:
        logger.info("Reusing cached FlyBody probe for %s", python_path)
        return dict(cached[1])
    command = build_probe_flybody_command(python_executable)
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
        "FlyBody probe finished for %s in %.1fms (ready=%s, returncode=%s)",
        python_path,
        (time.perf_counter() - started) * 1000.0,
        bool(frozen.get("ready")),
        frozen.get("returncode"),
    )
    return dict(frozen)


def pick_ready_flybody_runtime() -> tuple[Path | None, dict[str, Any]]:
    best: dict[str, Any] = {}
    for candidate in flybody_runtime_python_candidates():
        if not candidate.exists():
            continue
        payload = probe_flybody_runtime(candidate)
        if payload.get("ready"):
            return candidate, payload
        if not best:
            best = payload
    return None, best


def summarize_flybody_status() -> dict[str, Any]:
    repo_root = resolve_local_flybody_repo()
    candidates = []
    ready = False
    for candidate in flybody_runtime_python_candidates():
        record = {
            "python": str(candidate),
            "exists": candidate.exists(),
        }
        if candidate.exists():
            probe = probe_flybody_runtime(candidate)
            record.update(probe)
            ready = ready or bool(probe.get("ready"))
        candidates.append(record)
    return {
        "repo_root": str(repo_root) if repo_root is not None else None,
        "ready": ready,
        "candidates": candidates,
    }
