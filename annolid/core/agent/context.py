from __future__ import annotations

import base64
import mimetypes
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

from .memory import AgentMemoryStore
from .skills import AgentSkillsLoader


class AgentContextBuilder:
    """Build system prompts and LLM message payloads for conversational loops."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.memory = AgentMemoryStore(self.workspace)
        self.skills = AgentSkillsLoader(self.workspace)

    def build_system_prompt(self, skill_names: Optional[List[str]] = None) -> str:
        parts: List[str] = []
        parts.append(self._get_identity())

        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        memory_context = self.memory.get_memory_context()
        if memory_context:
            parts.append(f"# Memory\n\n{memory_context}")

        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        if skill_names:
            selected = self.skills.load_skills_for_context(skill_names)
            if selected:
                parts.append(f"# Requested Skills\n\n{selected}")

        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(
                "# Skills\n\n"
                "The following skills extend capabilities. To use a skill, read its "
                "`SKILL.md` file via `read_file`.\n\n"
                f"{skills_summary}"
            )
        return "\n\n---\n\n".join(parts)

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
                f"\n\n## Current Session\nChannel: {channel}\nChat ID: {chat_id}"
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
            f"## Runtime\n{runtime}\n\n"
            f"## Workspace\n{workspace_path}\n"
        )

    def _load_bootstrap_files(self) -> str:
        parts: List[str] = []
        for filename in self.BOOTSTRAP_FILES:
            p = self.workspace / filename
            if not p.exists():
                continue
            content = p.read_text(encoding="utf-8")
            parts.append(f"## {filename}\n\n{content}")
        return "\n\n".join(parts)

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

    def build_user_content(
        self, text: str, media: Optional[List[str]] = None
    ) -> str | List[dict[str, Any]]:
        """Public wrapper used by loops/providers."""
        return self._build_user_content(text, media)
