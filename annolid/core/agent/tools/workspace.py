"""Google Workspace CLI (gws) tool for the annolid agent.

Wraps the ``gws`` CLI binary so the agent can interact with Google Drive,
Gmail, Calendar, Sheets, Docs, Chat and other Workspace APIs.  All output
comes back as structured JSON.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from .function_base import FunctionTool
from .resolve import ensure_node_env, resolve_command


class GoogleWorkspaceTool(FunctionTool):
    """Execute Google Workspace commands via the ``gws`` CLI."""

    _ALLOWED_SERVICES = {
        "drive",
        "gmail",
        "calendar",
        "sheets",
        "docs",
        "slides",
        "chat",
        "tasks",
        "people",
        "forms",
        "keep",
        "meet",
        "admin",
    }

    def __init__(
        self,
        *,
        allowed_services: list[str] | None = None,
        timeout_seconds: int = 60,
    ) -> None:
        self._allowed_services = (
            set(allowed_services) if allowed_services else self._ALLOWED_SERVICES
        )
        self._timeout = timeout_seconds

    # -- FunctionTool interface ------------------------------------------------

    @property
    def name(self) -> str:
        return "google_workspace"

    @property
    def description(self) -> str:
        return (
            "Execute Google Workspace commands via the gws CLI.  "
            "Supports Drive, Gmail, Calendar, Sheets, Docs, Chat and more.  "
            "Returns structured JSON."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": (
                        "Workspace service to call, e.g. drive, gmail, "
                        "calendar, sheets, docs, chat."
                    ),
                    "enum": sorted(self._allowed_services),
                },
                "resource": {
                    "type": "string",
                    "description": (
                        "API resource within the service, e.g. 'files' for "
                        "drive, 'users' for gmail, 'events' for calendar."
                    ),
                },
                "method": {
                    "type": "string",
                    "description": (
                        "Method to invoke on the resource, e.g. 'list', "
                        "'create', 'get', 'delete', 'send'."
                    ),
                },
                "params": {
                    "type": "string",
                    "description": (
                        "JSON-encoded query parameters passed via --params, "
                        "e.g. '{\"pageSize\": 10}'."
                    ),
                },
                "json_body": {
                    "type": "string",
                    "description": (
                        "JSON-encoded request body passed via --json, "
                        'e.g. \'{"properties": {"title": "My Doc"}}\'.'
                    ),
                },
                "extra_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Extra CLI flags, e.g. ['--page-all', '--dry-run']."
                    ),
                },
            },
            "required": ["service", "resource", "method"],
        }

    @classmethod
    def is_available(cls) -> bool:
        """Return True when the ``gws`` binary can be found."""
        return resolve_command("gws") is not None

    async def execute(self, **kwargs: Any) -> str:
        service = str(kwargs.get("service") or "").strip().lower()
        resource = str(kwargs.get("resource") or "").strip()
        method = str(kwargs.get("method") or "").strip()

        if not service or not resource or not method:
            return json.dumps(
                {"error": "service, resource and method are all required."}
            )

        if service not in self._allowed_services:
            return json.dumps(
                {
                    "error": f"Service '{service}' is not allowed.  "
                    f"Allowed: {sorted(self._allowed_services)}"
                }
            )

        gws_bin = resolve_command("gws")
        if gws_bin is None:
            return json.dumps(
                {
                    "error": "gws CLI not found.  Install with: "
                    "npm install -g @googleworkspace/cli"
                }
            )

        cmd = [gws_bin, service, resource, method]

        params = str(kwargs.get("params") or "").strip()
        if params:
            cmd.extend(["--params", params])

        json_body = str(kwargs.get("json_body") or "").strip()
        if json_body:
            cmd.extend(["--json", json_body])

        extra_flags = kwargs.get("extra_flags")
        if isinstance(extra_flags, list):
            for flag in extra_flags:
                cmd.append(str(flag))

        return await self._run_gws(cmd)

    # -- internals -------------------------------------------------------------

    async def _run_gws(self, cmd: list[str]) -> str:
        """Shell out to ``gws`` and return the collected output."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=ensure_node_env(),
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._timeout
            )
        except FileNotFoundError:
            return json.dumps(
                {
                    "error": "gws CLI not found.  Install with: "
                    "npm install -g @googleworkspace/cli"
                }
            )
        except asyncio.TimeoutError:
            return json.dumps(
                {"error": f"gws command timed out after {self._timeout}s"}
            )

        out_text = (stdout or b"").decode("utf-8", errors="replace").strip()
        err_text = (stderr or b"").decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            return json.dumps(
                {
                    "error": f"gws exited with code {proc.returncode}",
                    "stderr": err_text or "(no stderr)",
                    "stdout": out_text[:2000] if out_text else "",
                }
            )

        # Cap extremely long outputs
        if len(out_text) > 50_000:
            out_text = (
                out_text[:50_000] + "\n\n[WARNING: output truncated at 50 000 chars]"
            )

        return out_text or "(no output)"


__all__ = ["GoogleWorkspaceTool"]
