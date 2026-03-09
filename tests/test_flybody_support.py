from __future__ import annotations

import json
from pathlib import Path

from annolid.gui.flybody_support import (
    build_live_flybody_command,
    build_probe_flybody_command,
    build_clone_flybody_command,
    build_setup_flybody_command,
    flybody_runtime_python_candidates,
    pick_ready_flybody_runtime,
    probe_flybody_runtime,
    repo_local_flybody_python,
    summarize_flybody_status,
)


def test_build_clone_flybody_command_uses_repo_and_destination() -> None:
    cmd = build_clone_flybody_command(
        "https://github.com/TuragaLab/flybody.git",
        "/tmp/flybody",
    )

    assert cmd == [
        "git",
        "clone",
        "https://github.com/TuragaLab/flybody.git",
        "/tmp/flybody",
    ]


def test_build_setup_flybody_command_points_to_repo_script() -> None:
    cmd = build_setup_flybody_command(
        repo_root=Path("/repo"),
        flybody_path=Path("/tmp/flybody"),
        venv_dir=Path("/repo/.venv311"),
        python_version="3.11",
    )

    assert cmd == [
        "bash",
        "/repo/scripts/setup_flybody_uv.sh",
        "--flybody-path",
        "/tmp/flybody",
        "--venv-dir",
        "/repo/.venv311",
        "--python",
        "3.11",
    ]


def test_build_probe_flybody_command_points_to_module() -> None:
    cmd = build_probe_flybody_command("/tmp/python")

    assert cmd == [
        "/tmp/python",
        "-m",
        "annolid.simulation.flybody_live",
        "--probe",
        "--json",
    ]


def test_build_live_flybody_command_points_to_module() -> None:
    cmd = build_live_flybody_command(
        "/tmp/python",
        out_path="/tmp/flybody.json",
        steps=42,
        seed=11,
        behavior="walk_on_ball",
    )

    assert cmd == [
        "/tmp/python",
        "-m",
        "annolid.simulation.flybody_live",
        "--behavior",
        "walk_on_ball",
        "--out",
        "/tmp/flybody.json",
        "--steps",
        "42",
        "--seed",
        "11",
    ]


def test_probe_flybody_runtime_parses_json(monkeypatch) -> None:
    class _Result:
        returncode = 0
        stdout = json.dumps({"ready": True, "detail": "ok"})
        stderr = ""

    monkeypatch.setattr(
        "annolid.gui.flybody_support.subprocess.run",
        lambda *args, **kwargs: _Result(),
    )

    payload = probe_flybody_runtime("/tmp/python")

    assert payload["ready"] is True
    assert payload["detail"] == "ok"
    assert payload["python"] == "/tmp/python"


def test_probe_flybody_runtime_uses_short_ttl_cache(monkeypatch) -> None:
    calls = {"count": 0}

    class _Result:
        returncode = 0
        stdout = json.dumps({"ready": True})
        stderr = ""

    def _run(*args, **kwargs):
        calls["count"] += 1
        return _Result()

    monkeypatch.setattr(
        "annolid.gui.flybody_support.subprocess.run",
        _run,
    )
    monkeypatch.setattr(
        "annolid.gui.flybody_support._PROBE_CACHE",
        {},
    )

    first = probe_flybody_runtime("/tmp/python")
    second = probe_flybody_runtime("/tmp/python")

    assert first["ready"] is True
    assert second["ready"] is True
    assert calls["count"] == 1


def test_pick_ready_flybody_runtime_returns_first_ready(
    monkeypatch, tmp_path: Path
) -> None:
    py_a = tmp_path / "a" / "python"
    py_b = tmp_path / "b" / "python"
    py_a.parent.mkdir(parents=True)
    py_b.parent.mkdir(parents=True)
    py_a.write_text("", encoding="utf-8")
    py_b.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "annolid.gui.flybody_support.flybody_runtime_python_candidates",
        lambda: [py_a, py_b],
    )
    monkeypatch.setattr(
        "annolid.gui.flybody_support.probe_flybody_runtime",
        lambda python: {"ready": str(python) == str(py_b), "python": str(python)},
    )

    python, payload = pick_ready_flybody_runtime()

    assert python == py_b
    assert payload["ready"] is True


def test_repo_local_flybody_python_points_to_repo_venv(tmp_path: Path) -> None:
    repo = tmp_path / "flybody"
    python_path = repo / ".venv" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("", encoding="utf-8")

    assert repo_local_flybody_python(repo) == python_path


def test_flybody_runtime_python_candidates_prefers_repo_venv(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "flybody"
    repo_python = repo / ".venv" / "bin" / "python"
    repo_python.parent.mkdir(parents=True)
    repo_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "annolid.gui.flybody_support.resolve_local_flybody_repo",
        lambda: repo,
    )
    monkeypatch.setattr(
        "annolid.gui.flybody_support.default_flybody_runtime_venv",
        lambda: tmp_path / "annolid-venv",
    )

    candidates = flybody_runtime_python_candidates()

    assert candidates[0] == repo_python


def test_summarize_flybody_status_reports_repo_and_candidates(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "flybody"
    repo.mkdir()
    py = tmp_path / "python"
    py.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "annolid.gui.flybody_support.resolve_local_flybody_repo",
        lambda: repo,
    )
    monkeypatch.setattr(
        "annolid.gui.flybody_support.flybody_runtime_python_candidates",
        lambda: [py],
    )
    monkeypatch.setattr(
        "annolid.gui.flybody_support.probe_flybody_runtime",
        lambda python: {"ready": True, "python": str(python)},
    )

    summary = summarize_flybody_status()

    assert summary["repo_root"] == str(repo)
    assert summary["ready"] is True
    assert summary["candidates"][0]["python"] == str(py)
    assert summary["candidates"][0]["ready"] is True
