"""GWS setup tool — bootstrap gws CLI install, auth, and skill symlinks."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from .function_base import FunctionTool
from .resolve import ensure_node_env, resolve_command

_GWS_REPO_URL = "https://github.com/googleworkspace/cli.git"

# Default local clone location
_DEFAULT_CLONE_DIR = Path("~/.annolid/gws-cli").expanduser()

# Managed skills directory used by the annolid skill loader
_MANAGED_SKILLS_DIR = Path("~/.annolid/workspace/skills").expanduser()

# Recommended starter skills
_DEFAULT_SKILLS = [
    "gws-shared",
    "gws-drive",
    "gws-gmail",
    "gws-calendar",
    "gws-sheets",
    "gws-slides",
]


class GWSSetupTool(FunctionTool):
    """Bootstrap the Google Workspace CLI, authenticate, and link skills."""

    _ACTIONS = {
        "check",
        "install",
        "auth_status",
        "link_skills",
        "update_skills",
    }

    def __init__(
        self,
        *,
        clone_dir: str | Path | None = None,
        managed_skills_dir: str | Path | None = None,
        default_skills: list[str] | None = None,
    ) -> None:
        self._clone_dir = Path(clone_dir) if clone_dir else _DEFAULT_CLONE_DIR
        self._managed_skills_dir = (
            Path(managed_skills_dir) if managed_skills_dir else _MANAGED_SKILLS_DIR
        )
        self._default_skills = list(default_skills or _DEFAULT_SKILLS)

    # -- FunctionTool interface ------------------------------------------------

    @property
    def name(self) -> str:
        return "gws_setup"

    @property
    def description(self) -> str:
        return (
            "Bootstrap the Google Workspace CLI (gws): check installation, "
            "install via npm, check auth status, link or update gws skills."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Setup action to perform.",
                    "enum": sorted(self._ACTIONS),
                },
                "skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Skill names to link (default: gws-shared, gws-drive, "
                        "gws-gmail, gws-calendar, gws-sheets).  "
                        "Use ['all'] to link every gws-* skill."
                    ),
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = str(kwargs.get("action") or "").strip().lower()
        if action not in self._ACTIONS:
            return json.dumps(
                {
                    "error": f"Unsupported action '{action}'.  Use one of: {sorted(self._ACTIONS)}"
                }
            )

        if action == "check":
            return await self._check()
        if action == "install":
            return await self._install()
        if action == "auth_status":
            return await self._auth_status()
        if action == "link_skills":
            skills = kwargs.get("skills") or self._default_skills
            return await self._link_skills(skills)
        if action == "update_skills":
            return await self._update_skills()
        return json.dumps({"error": "unknown action"})

    # -- action implementations ------------------------------------------------

    async def _check(self) -> str:
        gws_path = resolve_command("gws")
        npm_path = resolve_command("npm")
        clone_exists = self._clone_dir.is_dir()
        linked_skills = self._list_linked_skills()
        return json.dumps(
            {
                "gws_installed": gws_path is not None,
                "gws_path": gws_path or "",
                "npm_available": npm_path is not None,
                "clone_dir": str(self._clone_dir),
                "clone_exists": clone_exists,
                "linked_skills": linked_skills,
            }
        )

    async def _install(self) -> str:
        if resolve_command("gws"):
            return json.dumps({"ok": True, "message": "gws is already installed."})

        npm_bin = resolve_command("npm")
        if not npm_bin:
            return json.dumps(
                {
                    "ok": False,
                    "error": "npm is not available.  Install Node.js 18+ first.",
                }
            )

        stdout, stderr, rc = await self._run_cmd(
            [npm_bin, "install", "-g", "@googleworkspace/cli"]
        )
        if rc != 0:
            return json.dumps({"ok": False, "error": stderr or stdout})
        return json.dumps({"ok": True, "message": "gws installed successfully."})

    async def _auth_status(self) -> str:
        gws_bin = resolve_command("gws")
        if not gws_bin:
            return json.dumps({"ok": False, "error": "gws is not installed."})
        stdout, stderr, rc = await self._run_cmd([gws_bin, "auth", "status"])
        return json.dumps(
            {
                "ok": rc == 0,
                "stdout": stdout,
                "stderr": stderr,
            }
        )

    async def _link_skills(self, skills: list[str]) -> str:
        # 1. Clone repo if not present
        if not self._clone_dir.is_dir():
            stdout, stderr, rc = await self._run_cmd(
                ["git", "clone", "--depth", "1", _GWS_REPO_URL, str(self._clone_dir)]
            )
            if rc != 0:
                return json.dumps(
                    {"ok": False, "error": f"git clone failed: {stderr or stdout}"}
                )

        skills_src = self._clone_dir / "skills"
        if not skills_src.is_dir():
            return json.dumps(
                {
                    "ok": False,
                    "error": f"skills directory not found in {self._clone_dir}",
                }
            )

        # 2. Ensure managed skills dir exists
        self._managed_skills_dir.mkdir(parents=True, exist_ok=True)

        # 3. Resolve which skills to link
        if skills == ["all"]:
            skill_dirs = sorted(
                d.name
                for d in skills_src.iterdir()
                if d.is_dir() and d.name.startswith("gws-")
            )
        else:
            skill_dirs = [str(s).strip() for s in skills if str(s).strip()]

        linked: list[str] = []
        skipped: list[str] = []
        errors: list[str] = []

        for skill_name in skill_dirs:
            src = skills_src / skill_name
            dst = self._managed_skills_dir / skill_name

            if not src.is_dir():
                errors.append(f"{skill_name}: source not found at {src}")
                continue

            if dst.exists() or dst.is_symlink():
                # Already linked or exists — skip
                skipped.append(skill_name)
                continue

            try:
                dst.symlink_to(src)
                linked.append(skill_name)
            except OSError as exc:
                errors.append(f"{skill_name}: symlink failed — {exc}")

        return json.dumps(
            {
                "ok": not errors,
                "linked": linked,
                "skipped": skipped,
                "errors": errors,
            }
        )

    async def _update_skills(self) -> str:
        if not self._clone_dir.is_dir():
            return json.dumps(
                {
                    "ok": False,
                    "error": "Clone directory does not exist.  Run link_skills first.",
                }
            )

        stdout, stderr, rc = await self._run_cmd(
            ["git", "-C", str(self._clone_dir), "pull", "--ff-only"]
        )
        if rc != 0:
            return json.dumps({"ok": False, "error": stderr or stdout})
        return json.dumps(
            {
                "ok": True,
                "message": "gws-cli repo updated.  Symlinked skills are now current.",
                "git_output": stdout,
            }
        )

    # -- helpers ---------------------------------------------------------------

    def _list_linked_skills(self) -> list[str]:
        if not self._managed_skills_dir.is_dir():
            return []
        return sorted(
            d.name
            for d in self._managed_skills_dir.iterdir()
            if d.name.startswith("gws-") and (d.is_symlink() or d.is_dir())
        )

    @staticmethod
    async def _run_cmd(cmd: list[str], *, timeout: int = 120) -> tuple[str, str, int]:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=ensure_node_env(),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except FileNotFoundError:
            return "", f"command not found: {cmd[0]}", 127
        except asyncio.TimeoutError:
            return "", f"command timed out after {timeout}s", 1

        return (
            (stdout or b"").decode("utf-8", errors="replace").strip(),
            (stderr or b"").decode("utf-8", errors="replace").strip(),
            proc.returncode or 0,
        )


__all__ = ["GWSSetupTool"]
