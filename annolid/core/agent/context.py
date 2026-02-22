from __future__ import annotations

import base64
import mimetypes
import platform
from datetime import datetime
import re
from pathlib import Path
import time
from typing import Any, List, Mapping, Optional, Sequence

from .memory import AgentMemoryStore
from .skills import AgentSkillsLoader
from annolid.utils.logger import logger


class AgentContextBuilder:
    """Build system prompts and LLM message payloads for conversational loops."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.memory = AgentMemoryStore(self.workspace)
        self.skills = AgentSkillsLoader(self.workspace)
        self._bootstrap_cache: Optional[str] = None

    def build_system_prompt(self, skill_names: Optional[List[str]] = None) -> str:
        started = time.perf_counter()
        parts: List[str] = []
        parts.append(self._get_identity())

        t0 = time.perf_counter()
        bootstrap = self._load_bootstrap_files()
        t1 = time.perf_counter()
        if bootstrap:
            parts.append(bootstrap)

        memory_context = self.memory.get_memory_context()
        t2 = time.perf_counter()
        if memory_context:
            parts.append(f"# Memory\n\n{memory_context}")

        always_skills = self.skills.get_always_skills()
        t3 = time.perf_counter()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")
        t4 = time.perf_counter()

        if skill_names:
            selected = self.skills.load_skills_for_context(skill_names)
            if selected:
                parts.append(f"# Requested Skills\n\n{selected}")
        t5 = time.perf_counter()

        skills_summary = self.skills.build_skills_summary()
        t6 = time.perf_counter()
        if skills_summary:
            parts.append(
                "# Skills\n\n"
                "The following skills extend capabilities. To use a skill, read its "
                "`SKILL.md` file via `read_file`.\n\n"
                f"{skills_summary}"
            )
        result = "\n\n---\n\n".join(parts)
        logger.info(
            "annolid-bot profile system_prompt workspace=%s bootstrap_ms=%.1f memory_ms=%.1f always_skill_scan_ms=%.1f always_skill_load_ms=%.1f requested_skill_load_ms=%.1f skills_summary_ms=%.1f total_ms=%.1f chars=%d",
            str(self.workspace),
            (t1 - t0) * 1000.0,
            (t2 - t1) * 1000.0,
            (t3 - t2) * 1000.0,
            (t4 - t3) * 1000.0,
            (t5 - t4) * 1000.0,
            (t6 - t5) * 1000.0,
            (t6 - started) * 1000.0,
            len(result),
        )
        return result

    def build_messages(
        self,
        *,
        history: Sequence[Mapping[str, Any]],
        current_message: str,
        skill_names: Optional[List[str]] = None,
        media: Optional[List[str]] = None,
        channel: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> List[dict[str, Any]]:
        messages: List[dict[str, Any]] = []
        system_prompt = self.build_system_prompt(skill_names)
        if channel and chat_id:
            system_prompt += (
                "\n\n## Current Session\n"
                f"Channel: {self._redact_session_value(channel)}\n"
                f"Chat ID: {self._redact_session_value(chat_id)}"
            )
        messages.append({"role": "system", "content": system_prompt})
        messages.extend([dict(m) for m in history])
        user_content = self._build_user_content(current_message, media)
        messages.append({"role": "user", "content": user_content})
        return messages

    def _get_identity(self) -> str:
        local_now = datetime.now().astimezone()
        tz_name = local_now.tzname() or "local"
        now = local_now.strftime("%Y-%m-%d %H:%M (%A)")
        tz_offset = local_now.strftime("%z")
        pretty_offset = (
            f"{tz_offset[:3]}:{tz_offset[3:]}" if len(tz_offset) == 5 else tz_offset
        )
        now_iso = local_now.isoformat(timespec="seconds")
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = (
            f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, "
            f"Python {platform.python_version()}"
        )
        return (
            "# Annolid Agent\n\n"
            "You are an Annolid assistant with tool access for annotation workflows.\n\n"
            f"## Current Time\n{now} ({tz_name})\n\n"
            "Use the following local datetime as source of truth for relative date "
            f"phrases (today/tomorrow): {now_iso} (UTC{pretty_offset}).\n\n"
            f"## Runtime\n{runtime}\n\n"
            f"## Workspace\n{workspace_path}\n"
            "You are operating within an isolated project sandbox. Do not attempt to read or modify files outside of this workspace path.\n\n"
            "For camera checks in non-GUI channels, use `camera_snapshot` to probe and save a frame, "
            "then use `email` with `attachment_paths` to send the saved snapshot.\n"
            "Do not claim these tools are unavailable before trying them.\n"
        )

    def _load_bootstrap_files(self) -> str:
        if self._bootstrap_cache is not None:
            return self._bootstrap_cache
        parts: List[str] = []
        for filename in self.BOOTSTRAP_FILES:
            p = self.workspace / filename
            if not p.exists():
                continue
            content = p.read_text(encoding="utf-8")
            parts.append(f"## {filename}\n\n{content}")
        self._bootstrap_cache = "\n\n".join(parts)
        return self._bootstrap_cache

    def _build_user_content(
        self, text: str, media: Optional[List[str]]
    ) -> str | List[dict[str, Any]]:
        if not media:
            return text
        images: List[dict[str, Any]] = []
        for path in media:
            p = Path(path)
            mime, _ = mimetypes.guess_type(path)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
            images.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            )
        if not images:
            return text
        images.append({"type": "text", "text": text})
        return images

    @staticmethod
    def _redact_session_value(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
            local, _, domain = text.partition("@")
            masked_local = (
                f"{local[:2]}***{local[-1:]}" if len(local) > 3 else f"{local[:1]}***"
            )
            return f"{masked_local}@{domain}"
        if len(text) <= 6:
            return "***"
        return f"{text[:3]}***{text[-3:]}"

    def redact_session_value(self, value: str) -> str:
        return self._redact_session_value(value)

    def build_user_content(
        self, text: str, media: Optional[List[str]] = None
    ) -> str | List[dict[str, Any]]:
        """Public wrapper used by loops/providers."""
        return self._build_user_content(text, media)
