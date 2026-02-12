from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

BUILTIN_SKILLS_DIR = Path(__file__).parent / "skills"


class AgentSkillsLoader:
    """Load and summarize skills from workspace and optional builtins."""

    def __init__(self, workspace: Path, builtin_skills_dir: Optional[Path] = None):
        self.workspace = Path(workspace)
        self.workspace_skills = self.workspace / "skills"
        self.builtin_skills = (
            Path(builtin_skills_dir)
            if builtin_skills_dir is not None
            else BUILTIN_SKILLS_DIR
        )

    def list_skills(self, filter_unavailable: bool = True) -> List[Dict[str, str]]:
        skills: List[Dict[str, str]] = []
        if self.workspace_skills.exists():
            for skill_dir in self.workspace_skills.iterdir():
                if not skill_dir.is_dir():
                    continue
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    skills.append(
                        {
                            "name": skill_dir.name,
                            "path": str(skill_file),
                            "source": "workspace",
                        }
                    )

        if self.builtin_skills and self.builtin_skills.exists():
            for skill_dir in self.builtin_skills.iterdir():
                if not skill_dir.is_dir():
                    continue
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists() and not any(
                    s["name"] == skill_dir.name for s in skills
                ):
                    skills.append(
                        {
                            "name": skill_dir.name,
                            "path": str(skill_file),
                            "source": "builtin",
                        }
                    )

        if filter_unavailable:
            return [
                s
                for s in skills
                if self._check_requirements(self._get_skill_meta(s["name"]))
            ]
        return skills

    def load_skill(self, name: str) -> Optional[str]:
        workspace_skill = self.workspace_skills / name / "SKILL.md"
        if workspace_skill.exists():
            return workspace_skill.read_text(encoding="utf-8")
        builtin_skill = self.builtin_skills / name / "SKILL.md"
        if builtin_skill.exists():
            return builtin_skill.read_text(encoding="utf-8")
        return None

    def load_skills_for_context(self, skill_names: List[str]) -> str:
        parts: List[str] = []
        for name in skill_names:
            content = self.load_skill(name)
            if not content:
                continue
            content = self._strip_frontmatter(content)
            parts.append(f"### Skill: {name}\n\n{content}")
        return "\n\n---\n\n".join(parts)

    def build_skills_summary(self) -> str:
        all_skills = self.list_skills(filter_unavailable=False)
        if not all_skills:
            return ""

        def esc(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        lines = ["<skills>"]
        for s in all_skills:
            name = esc(s["name"])
            desc = esc(self._get_skill_description(s["name"]))
            path = s["path"]
            skill_meta = self._get_skill_meta(s["name"])
            available = self._check_requirements(skill_meta)
            lines.append(f'  <skill available="{str(available).lower()}">')
            lines.append(f"    <name>{name}</name>")
            lines.append(f"    <description>{desc}</description>")
            lines.append(f"    <location>{path}</location>")
            if not available:
                missing = self._get_missing_requirements(skill_meta)
                if missing:
                    lines.append(f"    <requires>{esc(missing)}</requires>")
            lines.append("  </skill>")
        lines.append("</skills>")
        return "\n".join(lines)

    def get_always_skills(self) -> List[str]:
        out: List[str] = []
        for s in self.list_skills(filter_unavailable=True):
            meta = self.get_skill_metadata(s["name"]) or {}
            skill_meta = self._parse_agent_metadata(meta.get("metadata", ""))
            if skill_meta.get("always") or meta.get("always"):
                out.append(s["name"])
        return out

    def get_skill_metadata(self, name: str) -> Optional[dict]:
        content = self.load_skill(name)
        if not content:
            return None
        if content.startswith("---"):
            match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
            if match:
                meta: Dict[str, str] = {}
                for line in match.group(1).split("\n"):
                    if ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    meta[key.strip()] = value.strip().strip("\"'")
                return meta
        return None

    def _get_skill_description(self, name: str) -> str:
        meta = self.get_skill_metadata(name) or {}
        return str(meta.get("description") or name)

    @staticmethod
    def _strip_frontmatter(content: str) -> str:
        if content.startswith("---"):
            match = re.match(r"^---\n.*?\n---\n", content, re.DOTALL)
            if match:
                return content[match.end() :].strip()
        return content

    @staticmethod
    def _parse_agent_metadata(raw: str) -> dict:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data.get("annolid", data.get("nanobot", {})) or {}
        except Exception:
            return {}
        return {}

    def _check_requirements(self, skill_meta: dict) -> bool:
        requires = skill_meta.get("requires", {})
        for b in requires.get("bins", []):
            if not shutil.which(str(b)):
                return False
        for env in requires.get("env", []):
            if not os.environ.get(str(env)):
                return False
        return True

    def _get_skill_meta(self, name: str) -> dict:
        meta = self.get_skill_metadata(name) or {}
        return self._parse_agent_metadata(str(meta.get("metadata", "")))

    def _get_missing_requirements(self, skill_meta: dict) -> str:
        missing: List[str] = []
        requires = skill_meta.get("requires", {})
        for b in requires.get("bins", []):
            if not shutil.which(str(b)):
                missing.append(f"CLI: {b}")
        for env in requires.get("env", []):
            if not os.environ.get(str(env)):
                missing.append(f"ENV: {env}")
        return ", ".join(missing)
