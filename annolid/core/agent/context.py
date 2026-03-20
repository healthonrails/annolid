from __future__ import annotations

import base64
import mimetypes
import os
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
    DEFAULT_SYSTEM_PROMPT_MAX_CHARS = 24_000
    DEFAULT_AUTO_SKILL_TOP_K = 3

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.memory = AgentMemoryStore(self.workspace)
        self.skills = AgentSkillsLoader(self.workspace)
        self._bootstrap_cache: Optional[str] = None
        self._system_prompt_max_chars = self._read_int_env(
            "ANNOLID_AGENT_SYSTEM_PROMPT_MAX_CHARS",
            self.DEFAULT_SYSTEM_PROMPT_MAX_CHARS,
            minimum=1_200,
        )
        self._auto_skill_top_k = self._read_int_env(
            "ANNOLID_AGENT_AUTO_SKILL_TOP_K",
            self.DEFAULT_AUTO_SKILL_TOP_K,
            minimum=0,
        )

    def build_system_prompt(
        self,
        skill_names: Optional[List[str]] = None,
        task_hint: Optional[str] = None,
    ) -> str:
        started = time.perf_counter()
        parts: List[str] = []
        identity = self._get_identity()
        parts.append(identity)

        t0 = time.perf_counter()
        bootstrap = self._load_bootstrap_files()
        t1 = time.perf_counter()
        auto_skill_names: List[str] = []
        explicit_skill_names = [s for s in (skill_names or []) if str(s).strip()]
        if not explicit_skill_names and task_hint and self._auto_skill_top_k > 0:
            auto_skill_names = self.skills.suggest_skills_for_task(
                str(task_hint),
                top_k=self._auto_skill_top_k,
            )

        memory_context = self.memory.get_memory_context()
        t2 = time.perf_counter()

        always_skills = self.skills.get_always_skills()
        t3 = time.perf_counter()
        always_content = ""
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
        t4 = time.perf_counter()

        requested_content = ""
        resolved_skill_names = explicit_skill_names or auto_skill_names
        if resolved_skill_names:
            requested_content = self.skills.load_skills_for_context(
                resolved_skill_names
            )
        t5 = time.perf_counter()

        skills_summary = self.skills.build_skills_summary()
        t6 = time.perf_counter()

        section_map: dict[str, str] = {}
        if bootstrap:
            section_map["bootstrap"] = bootstrap
        if memory_context:
            section_map["memory"] = f"# Memory\n\n{memory_context}"
        if always_content:
            section_map["active_skills"] = f"# Active Skills\n\n{always_content}"
        if requested_content:
            title = (
                "# Requested Skills"
                if explicit_skill_names
                else "# Auto-selected Skills"
            )
            section_map["requested_skills"] = f"{title}\n\n{requested_content}"
        if skills_summary:
            section_map["skills_summary"] = (
                "# Skills\n\n"
                "The following skills extend capabilities. To use a skill, read its "
                "`SKILL.md` file via `read_file`.\n\n"
                f"{skills_summary}"
            )
        fitted = self._fit_system_prompt_sections(
            identity=identity,
            section_map=section_map,
            max_chars=self._system_prompt_max_chars,
        )
        for key in (
            "bootstrap",
            "memory",
            "active_skills",
            "requested_skills",
            "skills_summary",
        ):
            value = fitted.get(key)
            if value:
                parts.append(value)
        result = "\n\n---\n\n".join(parts)
        logger.info(
            "annolid-bot profile system_prompt workspace=%s bootstrap_ms=%.1f memory_ms=%.1f always_skill_scan_ms=%.1f always_skill_load_ms=%.1f requested_skill_load_ms=%.1f skills_summary_ms=%.1f total_ms=%.1f chars=%d max_chars=%d auto_skills=%d explicit_skills=%d",
            str(self.workspace),
            (t1 - t0) * 1000.0,
            (t2 - t1) * 1000.0,
            (t3 - t2) * 1000.0,
            (t4 - t3) * 1000.0,
            (t5 - t4) * 1000.0,
            (t6 - t5) * 1000.0,
            (t6 - started) * 1000.0,
            len(result),
            self._system_prompt_max_chars,
            len(auto_skill_names),
            len(explicit_skill_names),
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
        system_prompt = self.build_system_prompt(
            skill_names=skill_names,
            task_hint=current_message,
        )
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
            f"## Memory\n- Long-term memory: {workspace_path}/memory/MEMORY.md\n"
            f"- History log: {workspace_path}/memory/HISTORY.md (grep-searchable; each entry should start with [YYYY-MM-DD HH:MM]).\n\n"
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

    @staticmethod
    def _read_int_env(name: str, default: int, *, minimum: int = 0) -> int:
        raw = os.getenv(name, "").strip()
        if not raw:
            return max(minimum, int(default))
        try:
            value = int(raw)
        except ValueError:
            return max(minimum, int(default))
        return max(minimum, value)

    @staticmethod
    def _truncate_section(text: str, max_chars: int, label: str) -> str:
        cleaned = str(text or "")
        if max_chars <= 0:
            return ""
        if len(cleaned) <= max_chars:
            return cleaned
        marker = f"\n...[{label} truncated to fit system prompt budget]"
        keep = max(0, max_chars - len(marker))
        return cleaned[:keep].rstrip() + marker

    def _fit_system_prompt_sections(
        self,
        *,
        identity: str,
        section_map: Mapping[str, str],
        max_chars: int,
    ) -> dict[str, str]:
        if max_chars <= 0:
            return dict(section_map)
        divider = "\n\n---\n\n"
        section_order = [
            ("bootstrap", 0.35),
            ("memory", 0.14),
            ("active_skills", 0.2),
            ("requested_skills", 0.16),
            ("skills_summary", 0.15),
        ]
        present = [
            (name, weight) for name, weight in section_order if section_map.get(name)
        ]
        if not present:
            return {}
        divider_count = len(present)
        overhead = len(identity) + (divider_count * len(divider))
        available = max_chars - overhead
        if available <= 0:
            return {key: "" for key in section_map.keys()}
        weight_total = sum(weight for _, weight in present) or 1.0
        budgets: dict[str, int] = {}
        consumed = 0
        for idx, (name, weight) in enumerate(present):
            if idx == len(present) - 1:
                budget = max(0, available - consumed)
            else:
                budget = max(120, int((available * weight) / weight_total))
                consumed += budget
            budgets[name] = budget
        fitted: dict[str, str] = {}
        for name in section_map.keys():
            raw = section_map.get(name) or ""
            if not raw:
                fitted[name] = ""
                continue
            budget = budgets.get(name, len(raw))
            fitted[name] = self._truncate_section(raw, budget, name)

        result = divider.join([identity, *[fitted[n] for n, _ in present if fitted[n]]])
        if len(result) <= max_chars:
            return fitted

        overflow = len(result) - max_chars
        for name, _ in reversed(present):
            current = fitted.get(name, "")
            if not current:
                continue
            target = max(120, len(current) - overflow)
            fitted[name] = self._truncate_section(current, target, name)
            result = divider.join(
                [identity, *[fitted[n] for n, _ in present if fitted[n]]]
            )
            if len(result) <= max_chars:
                break
            overflow = len(result) - max_chars
        return fitted
