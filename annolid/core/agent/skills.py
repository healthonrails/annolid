from __future__ import annotations

import json
import os
import platform
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from annolid.core.agent.config.loader import get_config_path, load_config
from annolid.core.agent.utils.helpers import get_agent_data_path

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

BUILTIN_SKILLS_DIR = Path(__file__).parent / "skills"
MANAGED_SKILLS_DIR = get_agent_data_path() / "skills"


class AgentSkillsLoader:
    """Load and summarize skills from workspace and optional builtins."""

    def __init__(
        self,
        workspace: Path,
        builtin_skills_dir: Optional[Path] = None,
        managed_skills_dir: Optional[Path] = None,
    ):
        self.workspace = Path(workspace)
        self.workspace_skills = self.workspace / "skills"
        self.builtin_skills = (
            Path(builtin_skills_dir)
            if builtin_skills_dir is not None
            else BUILTIN_SKILLS_DIR
        )
        self.managed_skills = (
            Path(managed_skills_dir)
            if managed_skills_dir is not None
            else MANAGED_SKILLS_DIR
        )
        self._snapshot: Optional[List[Dict[str, str]]] = None
        self._config_dict_cache: Optional[Dict[str, Any]] = None

    def refresh_snapshot(self) -> None:
        self._snapshot = self._build_skill_snapshot()
        self._config_dict_cache = None

    def list_skills(self, filter_unavailable: bool = True) -> List[Dict[str, str]]:
        if self._snapshot is None:
            self.refresh_snapshot()
        skills = list(self._snapshot or [])
        if filter_unavailable:
            return [
                s
                for s in skills
                if self._check_requirements(self._get_skill_meta_by_path(s["path"]))
            ]
        return skills

    def load_skill(self, name: str) -> Optional[str]:
        for skill in self.list_skills(filter_unavailable=False):
            if skill.get("name") != name:
                continue
            path = Path(skill["path"])
            if path.exists():
                return path.read_text(encoding="utf-8")
        return None

    def load_skills_for_context(self, skill_names: List[str]) -> str:
        parts: List[str] = []
        indexed = {s["name"]: s for s in self.list_skills(filter_unavailable=True)}
        for name in skill_names:
            skill = indexed.get(name)
            if not skill:
                continue
            skill_meta = self._get_skill_meta_by_path(skill["path"])
            if bool(skill_meta.get("disable_model_invocation")):
                continue
            content = Path(skill["path"]).read_text(encoding="utf-8")
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
            desc = esc(self._get_skill_description(s["path"]))
            path = s["path"]
            skill_meta = self._get_skill_meta_by_path(s["path"])
            if bool(skill_meta.get("disable_model_invocation")):
                continue
            available = self._check_requirements(skill_meta)
            lines.append(f'  <skill available="{str(available).lower()}">')
            lines.append(f"    <name>{name}</name>")
            lines.append(f"    <description>{desc}</description>")
            lines.append(f"    <location>{path}</location>")
            if not bool(skill_meta.get("user_invocable", True)):
                lines.append("    <user_invocable>false</user_invocable>")
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
            skill_meta = self._get_skill_meta_by_path(s["path"])
            if bool(skill_meta.get("disable_model_invocation")):
                continue
            if skill_meta.get("always"):
                out.append(s["name"])
        return out

    def get_skill_metadata(self, name: str) -> Optional[dict]:
        for skill in self.list_skills(filter_unavailable=False):
            if skill.get("name") != name:
                continue
            return self._get_frontmatter(skill["path"])
        return None

    def _get_skill_description(self, path: str) -> str:
        meta = self._get_frontmatter(path) or {}
        fallback = Path(path).parent.name
        return str(meta.get("description") or fallback)

    @staticmethod
    def _strip_frontmatter(content: str) -> str:
        if content.startswith("---"):
            match = re.match(r"^---\n.*?\n---\n", content, re.DOTALL)
            if match:
                return content[match.end() :].strip()
        return content

    @staticmethod
    def _parse_agent_metadata(raw: Any) -> dict:
        parsed: Dict[str, Any] = {}
        if isinstance(raw, str):
            try:
                loaded = json.loads(raw)
                if isinstance(loaded, dict):
                    parsed = loaded
            except Exception:
                parsed = {}
        elif isinstance(raw, dict):
            parsed = dict(raw)
        annolid = parsed.get("annolid")
        openclaw = parsed.get("openclaw")
        nanobot = parsed.get("nanobot")
        merged: Dict[str, Any] = {}
        for candidate in (nanobot, annolid, openclaw):
            if isinstance(candidate, dict):
                merged.update(candidate)
        return merged or parsed

    def _check_requirements(self, skill_meta: dict) -> bool:
        if bool(skill_meta.get("always")):
            return True
        allowed_os = skill_meta.get("os", [])
        if isinstance(allowed_os, str):
            allowed_os = [allowed_os]
        if isinstance(allowed_os, list) and allowed_os:
            current = platform.system().lower()
            norm_current = "darwin" if current == "darwin" else current
            allowed = {str(v).strip().lower() for v in allowed_os if str(v).strip()}
            if allowed and norm_current not in allowed:
                return False
        requires = skill_meta.get("requires", {})
        bins = requires.get("bins", [])
        if isinstance(bins, str):
            bins = [bins]
        for b in bins:
            if not shutil.which(str(b)):
                return False
        any_bins = requires.get("any_bins", requires.get("anyBins", []))
        if isinstance(any_bins, str):
            any_bins = [any_bins]
        if isinstance(any_bins, list) and any_bins:
            if not any(shutil.which(str(b)) for b in any_bins):
                return False
        env_list = requires.get("env", [])
        if isinstance(env_list, str):
            env_list = [env_list]
        for env in env_list:
            if not os.environ.get(str(env)):
                return False
        config_list = requires.get("config", [])
        if isinstance(config_list, str):
            config_list = [config_list]
        if isinstance(config_list, list):
            config = self._get_runtime_config_dict()
            for key_path in config_list:
                if not self._config_path_truthy(config, str(key_path)):
                    return False
        return True

    def _get_skill_meta_by_path(self, path: str) -> dict:
        meta = self._get_frontmatter(path) or {}
        metadata = self._parse_agent_metadata(meta.get("metadata", {}))
        out: Dict[str, Any] = dict(metadata)
        for key in (
            "always",
            "os",
            "requires",
            "user-invocable",
            "user_invocable",
            "disable-model-invocation",
            "disable_model_invocation",
            "command-dispatch",
            "command_dispatch",
            "command-tool",
            "command_tool",
            "command-arg-mode",
            "command_arg_mode",
        ):
            if key in meta:
                out[key] = meta[key]
        out["user_invocable"] = bool(
            out.get("user_invocable", out.get("user-invocable", True))
        )
        out["disable_model_invocation"] = bool(
            out.get(
                "disable_model_invocation",
                out.get("disable-model-invocation", False),
            )
        )
        out["command_dispatch"] = str(
            out.get("command_dispatch", out.get("command-dispatch", ""))
        ).strip()
        out["command_tool"] = str(
            out.get("command_tool", out.get("command-tool", ""))
        ).strip()
        out["command_arg_mode"] = str(
            out.get("command_arg_mode", out.get("command-arg-mode", ""))
        ).strip()
        return out

    def _get_missing_requirements(self, skill_meta: dict) -> str:
        missing: List[str] = []
        requires = skill_meta.get("requires", {})
        bins = requires.get("bins", [])
        if isinstance(bins, str):
            bins = [bins]
        for b in bins:
            if not shutil.which(str(b)):
                missing.append(f"CLI: {b}")
        any_bins = requires.get("any_bins", requires.get("anyBins", []))
        if isinstance(any_bins, str):
            any_bins = [any_bins]
        if isinstance(any_bins, list) and any_bins:
            if not any(shutil.which(str(b)) for b in any_bins):
                missing.append(f"CLI(any): {'|'.join(str(b) for b in any_bins)}")
        env_list = requires.get("env", [])
        if isinstance(env_list, str):
            env_list = [env_list]
        for env in env_list:
            if not os.environ.get(str(env)):
                missing.append(f"ENV: {env}")
        config_list = requires.get("config", [])
        if isinstance(config_list, str):
            config_list = [config_list]
        if isinstance(config_list, list):
            config = self._get_runtime_config_dict()
            for key_path in config_list:
                if not self._config_path_truthy(config, str(key_path)):
                    missing.append(f"CONFIG: {key_path}")
        return ", ".join(missing)

    def _build_skill_snapshot(self) -> List[Dict[str, str]]:
        by_name: Dict[str, Dict[str, str]] = {}
        for source, root in self._iter_skill_roots_by_precedence():
            if not root.exists():
                continue
            for skill_dir in root.iterdir():
                if not skill_dir.is_dir():
                    continue
                skill_file = skill_dir / "SKILL.md"
                if not skill_file.exists():
                    continue
                name = skill_dir.name
                if name in by_name:
                    continue
                by_name[name] = {
                    "name": name,
                    "path": str(skill_file),
                    "source": source,
                }
        return [by_name[k] for k in sorted(by_name.keys())]

    def _iter_skill_roots_by_precedence(self) -> List[tuple[str, Path]]:
        roots: List[tuple[str, Path]] = [
            ("workspace", self.workspace_skills),
            ("managed", self.managed_skills),
            ("builtin", self.builtin_skills),
        ]
        for idx, path in enumerate(self._load_extra_skill_dirs()):
            roots.append((f"extra:{idx}", path))
        return roots

    @staticmethod
    def _read_frontmatter(content: str) -> Dict[str, Any]:
        if not content.startswith("---"):
            return {}
        match = re.match(r"^---\n(.*?)\n---(?:\n|$)", content, re.DOTALL)
        if not match:
            return {}
        raw = match.group(1)
        if yaml is not None:
            try:
                parsed = yaml.safe_load(raw) or {}
                if isinstance(parsed, dict):
                    return dict(parsed)
            except Exception:
                pass
        meta: Dict[str, str] = {}
        for line in raw.split("\n"):
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip("\"'")
        return meta

    def _get_frontmatter(self, path: str) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            return {}
        try:
            content = p.read_text(encoding="utf-8")
        except OSError:
            return {}
        return self._read_frontmatter(content)

    @staticmethod
    def _load_extra_skill_dirs() -> List[Path]:
        extras: List[Path] = []
        env_raw = str(os.getenv("ANNOLID_SKILLS_EXTRA_DIRS") or "").strip()
        if env_raw:
            for part in env_raw.split(os.pathsep):
                entry = str(part or "").strip()
                if entry:
                    extras.append(Path(entry).expanduser())

        try:
            cfg_path = get_config_path()
            if cfg_path.exists():
                payload = json.loads(cfg_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    skills = payload.get("skills") or {}
                    load = skills.get("load") if isinstance(skills, dict) else {}
                    extra_dirs = (
                        load.get("extraDirs") if isinstance(load, dict) else None
                    ) or (load.get("extra_dirs") if isinstance(load, dict) else None)
                    if isinstance(extra_dirs, list):
                        for item in extra_dirs:
                            entry = str(item or "").strip()
                            if entry:
                                extras.append(Path(entry).expanduser())
        except Exception:
            pass

        out: List[Path] = []
        seen: set[str] = set()
        for path in extras:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            out.append(path)
        return out

    def _get_runtime_config_dict(self) -> Dict[str, Any]:
        if self._config_dict_cache is not None:
            return dict(self._config_dict_cache)
        try:
            cfg = load_config()
            self._config_dict_cache = cfg.to_dict()
        except Exception:
            self._config_dict_cache = {}
        return dict(self._config_dict_cache)

    @staticmethod
    def _config_path_truthy(config: Dict[str, Any], dotted: str) -> bool:
        text = str(dotted or "").strip()
        if not text:
            return False
        cursor: Any = config
        for part in text.split("."):
            key = str(part).strip()
            if not key:
                return False
            if not isinstance(cursor, dict) or key not in cursor:
                return False
            cursor = cursor[key]
        return bool(cursor)
