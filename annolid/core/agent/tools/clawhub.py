from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from .function_base import FunctionTool


def _resolve_workspace(workspace: Path | str | None) -> Path:
    if workspace is None:
        return Path.cwd()
    return Path(workspace).expanduser().resolve()


def _api_base() -> str:
    value = str(os.environ.get("ANNOLID_CLAWHUB_API_BASE") or "https://clawhub.ai")
    value = value.strip().rstrip("/")
    return value or "https://clawhub.ai"


def _api_url(path: str, params: dict[str, str] | None = None) -> str:
    base = _api_base()
    query = urllib.parse.urlencode(params or {})
    suffix = f"{path}?{query}" if query else path
    return f"{base}{suffix}"


def _http_get_json(url: str, timeout_sec: int = 20) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "annolid-clawhub/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8"))


def _http_get_bytes(url: str, timeout_sec: int = 60) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "annolid-clawhub/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as response:
        return bytes(response.read())


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe zip member path: {member.filename}")
        zf.extractall(dest_dir)


def _locate_skill_root(root_dir: Path) -> Path | None:
    # Prefer nearest SKILL.md match for predictable installs.
    candidates = sorted(
        (
            p.parent
            for p in root_dir.rglob("SKILL.md")
            if p.is_file() and p.parent.is_dir()
        ),
        key=lambda p: len(p.parts),
    )
    if not candidates:
        return None
    return candidates[0]


async def run_clawhub_command(
    args: list[str],
    *,
    workspace: Path | str | None = None,
    timeout_sec: int = 45,
) -> dict[str, Any]:
    # Backward-compatible shim for existing callers. Internally uses Python API.
    argv = [str(v).strip() for v in args if str(v).strip()]
    if not argv:
        return {"ok": False, "error": "No ClawHub action provided."}
    action = argv[0].lower()
    if action == "search":
        query = argv[1] if len(argv) > 1 else ""
        limit = 5
        if "--limit" in argv:
            idx = argv.index("--limit")
            if idx + 1 < len(argv):
                try:
                    limit = int(argv[idx + 1])
                except Exception:
                    limit = 5
        return await clawhub_search_skills(
            query,
            limit=limit,
            workspace=workspace,
        )
    if action == "install":
        slug = argv[1] if len(argv) > 1 else ""
        resolved_workspace = workspace
        if "--workdir" in argv:
            idx = argv.index("--workdir")
            if idx + 1 < len(argv):
                resolved_workspace = argv[idx + 1]
        return await clawhub_install_skill(
            slug,
            workspace=resolved_workspace,
        )
    return {"ok": False, "error": f"Unsupported ClawHub action: {action}"}


async def clawhub_search_skills(
    query: str,
    *,
    limit: int = 5,
    workspace: Path | str | None = None,
) -> dict[str, Any]:
    query_text = str(query or "").strip()
    if not query_text:
        return {"ok": False, "error": "query is required"}
    top_k = max(1, min(int(limit or 5), 20))
    ws = _resolve_workspace(workspace)
    url = _api_url("/api/v1/search", {"q": query_text})
    try:
        data = await asyncio.wait_for(
            asyncio.to_thread(_http_get_json, url, 20),
            timeout=25.0,
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": f"ClawHub search failed: {exc}",
            "query": query_text,
            "limit": top_k,
            "workspace": str(ws),
            "source": "clawhub",
            "url": url,
        }

    if isinstance(data, dict):
        raw_items = data.get("results")
        if not isinstance(raw_items, list):
            raw_items = data.get("items")
    elif isinstance(data, list):
        raw_items = data
    else:
        raw_items = []

    results: list[dict[str, Any]] = []
    for item in raw_items[:top_k]:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "slug": str(item.get("slug") or "").strip(),
                "name": str(item.get("name") or "").strip(),
                "description": str(item.get("description") or "").strip(),
                "latest_version": str(
                    item.get("latestVersion") or item.get("version") or ""
                ).strip(),
                "author": str(item.get("author") or "").strip(),
                "url": str(item.get("url") or "").strip(),
            }
        )

    return {
        "ok": True,
        "query": query_text,
        "limit": top_k,
        "count": len(results),
        "results": results,
        "workspace": str(ws),
        "source": "clawhub",
        "url": url,
    }


async def clawhub_install_skill(
    slug: str,
    *,
    workspace: Path | str | None = None,
) -> dict[str, Any]:
    slug_text = str(slug or "").strip()
    if not slug_text:
        return {"ok": False, "error": "slug is required"}
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}", slug_text):
        return {
            "ok": False,
            "error": "Invalid skill slug format.",
            "slug": slug_text,
        }
    ws = _resolve_workspace(workspace)
    skills_dir = ws / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    download_url = _api_url("/api/v1/download", {"slug": slug_text})

    try:
        zip_bytes = await asyncio.wait_for(
            asyncio.to_thread(_http_get_bytes, download_url, 90),
            timeout=95.0,
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": f"ClawHub download failed: {exc}",
            "slug": slug_text,
            "workspace": str(ws),
            "source": "clawhub",
            "url": download_url,
        }

    try:
        with tempfile.TemporaryDirectory(
            prefix="annolid_clawhub_", dir=str(ws)
        ) as tmpdir:
            tmp_path = Path(tmpdir)
            zip_path = tmp_path / "skill.zip"
            extract_dir = tmp_path / "extract"
            extract_dir.mkdir(parents=True, exist_ok=True)
            zip_path.write_bytes(zip_bytes)
            _safe_extract_zip(zip_path, extract_dir)
            source_root = _locate_skill_root(extract_dir)
            if source_root is None:
                return {
                    "ok": False,
                    "error": "Downloaded package did not contain SKILL.md.",
                    "slug": slug_text,
                    "workspace": str(ws),
                    "source": "clawhub",
                }
            target_dir = skills_dir / slug_text
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(source_root, target_dir)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Failed to install skill package: {exc}",
            "slug": slug_text,
            "workspace": str(ws),
            "source": "clawhub",
        }

    return {
        "ok": True,
        "slug": slug_text,
        "workspace": str(ws),
        "skills_dir": str(skills_dir),
        "installed_path": str((skills_dir / slug_text).resolve()),
        "source": "clawhub",
        "url": download_url,
        "restart_hint": "Start a new Annolid Bot session to load newly installed skills.",
    }


class ClawHubSearchSkillsTool(FunctionTool):
    def __init__(self, workspace: Path | str | None = None):
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "clawhub_search_skills"

    @property
    def description(self) -> str:
        return "Search ClawHub public skill registry by natural-language query."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        payload = await clawhub_search_skills(
            str(kwargs.get("query") or ""),
            limit=int(kwargs.get("limit") or 5),
            workspace=self._workspace,
        )
        return json.dumps(payload)


class ClawHubInstallSkillTool(FunctionTool):
    def __init__(self, workspace: Path | str | None = None):
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "clawhub_install_skill"

    @property
    def description(self) -> str:
        return "Install a ClawHub skill by slug into the current agent workspace."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"slug": {"type": "string", "minLength": 1}},
            "required": ["slug"],
        }

    async def execute(self, **kwargs: Any) -> str:
        payload = await clawhub_install_skill(
            str(kwargs.get("slug") or ""),
            workspace=self._workspace,
        )
        return json.dumps(payload)
