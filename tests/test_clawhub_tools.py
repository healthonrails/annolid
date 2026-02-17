from __future__ import annotations

import asyncio
import io
import zipfile
from pathlib import Path

from annolid.core.agent.tools import clawhub


def _build_skill_zip_bytes(skill_name: str = "demo-skill") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{skill_name}/SKILL.md", "---\nname: demo-skill\n---\n")
        zf.writestr(f"{skill_name}/README.md", "demo")
    return buffer.getvalue()


def test_clawhub_search_skills_python_api(monkeypatch, tmp_path: Path) -> None:
    def _fake_get_json(url: str, timeout_sec: int = 20):
        del timeout_sec
        assert "/api/v1/search" in url
        return {
            "results": [
                {"slug": "a", "name": "A", "description": "desc-a"},
                {"slug": "b", "name": "B", "description": "desc-b"},
            ]
        }

    monkeypatch.setattr(clawhub, "_http_get_json", _fake_get_json)
    payload = asyncio.run(
        clawhub.clawhub_search_skills("behavior", limit=1, workspace=tmp_path)
    )
    assert payload["ok"] is True
    assert payload["query"] == "behavior"
    assert payload["count"] == 1
    assert payload["results"][0]["slug"] == "a"


def test_clawhub_install_skill_python_api(monkeypatch, tmp_path: Path) -> None:
    def _fake_get_bytes(url: str, timeout_sec: int = 60) -> bytes:
        del timeout_sec
        assert "/api/v1/download" in url
        return _build_skill_zip_bytes("my-skill")

    monkeypatch.setattr(clawhub, "_http_get_bytes", _fake_get_bytes)
    payload = asyncio.run(clawhub.clawhub_install_skill("my-skill", workspace=tmp_path))
    assert payload["ok"] is True
    assert payload["slug"] == "my-skill"
    assert (tmp_path / "skills" / "my-skill" / "SKILL.md").exists()


def test_run_clawhub_command_shim_dispatch(monkeypatch, tmp_path: Path) -> None:
    async def _fake_search(query: str, *, limit: int = 5, workspace=None):
        return {"ok": True, "query": query, "limit": limit, "workspace": str(workspace)}

    monkeypatch.setattr(clawhub, "clawhub_search_skills", _fake_search)
    payload = asyncio.run(
        clawhub.run_clawhub_command(
            ["search", "tracking", "--limit", "3"], workspace=tmp_path
        )
    )
    assert payload["ok"] is True
    assert payload["query"] == "tracking"
    assert payload["limit"] == 3
