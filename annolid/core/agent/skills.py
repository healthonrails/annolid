from __future__ import annotations

import json
import os
import platform
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from annolid.core.agent.config.loader import get_config_path, load_config
from annolid.core.agent.skill_registry.registry import (
    SkillRegistry as RuntimeSkillRegistry,
)
from annolid.core.agent.skill_registry.schema import (
    SkillLoadConfig,
    validate_skill_manifest,
)
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
        watch: Optional[bool] = None,
        watch_poll_seconds: Optional[float] = None,
        skill_retrieval_mode: Optional[str] = None,
        embedding_model_path: Optional[str] = None,
        embedding_client: Any = None,
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
        self.registry = RuntimeSkillRegistry(
            workspace=self.workspace,
            builtin_skills_dir=self.builtin_skills,
            managed_skills_dir=self.managed_skills,
            parse_meta=self._parse_skill_meta,
            read_frontmatter_from_path=self._read_frontmatter_from_path,
            get_config_path=get_config_path,
            watch=watch,
            watch_poll_seconds=watch_poll_seconds,
        )
        self._config_dict_cache: Optional[Dict[str, Any]] = None
        env_mode = str(os.getenv("ANNOLID_AGENT_SKILL_RETRIEVAL_MODE", "")).strip()
        self._skill_retrieval_mode = (
            str(skill_retrieval_mode or env_mode or "lexical").strip().lower()
        )
        self._embedding_model_path = str(
            embedding_model_path
            or os.getenv("ANNOLID_AGENT_SKILL_EMBEDDING_MODEL", "")
            or "all-MiniLM-L6-v2"
        ).strip()
        self._embedding_client = embedding_client
        self._embedding_cache_signature: Optional[tuple[str, ...]] = None
        self._embedding_cache_pairs: list[tuple[str, list[float]]] = []

    def refresh_snapshot(self) -> None:
        self.registry.refresh()
        self._config_dict_cache = None
        self._embedding_cache_signature = None
        self._embedding_cache_pairs = []

    def list_skills(self, filter_unavailable: bool = True) -> List[Dict[str, Any]]:
        skills = self.registry.list_skills()
        if filter_unavailable:
            return [
                s
                for s in skills
                if self._check_requirements(
                    s.get("parsed_meta", self._get_skill_meta_by_path(s["path"]))
                )
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
            skill_meta = skill.get(
                "parsed_meta", self._get_skill_meta_by_path(skill["path"])
            )
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
            desc = esc(s.get("description", self._get_skill_description(s["path"])))
            path = s["path"]
            skill_meta = s.get("parsed_meta", self._get_skill_meta_by_path(s["path"]))
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
            skill_meta = s.get("parsed_meta", self._get_skill_meta_by_path(s["path"]))
            if bool(skill_meta.get("disable_model_invocation")):
                continue
            if skill_meta.get("always"):
                out.append(s["name"])
        return out

    def get_skill_metadata(self, name: str) -> Optional[dict]:
        for skill in self.list_skills(filter_unavailable=False):
            if skill.get("name") != name:
                continue
            return skill.get("raw_meta", self._get_frontmatter(skill["path"]))
        return None

    def suggest_skills_for_task(
        self, task_description: str, top_k: int = 3
    ) -> List[str]:
        """
        Suggest relevant skill names for a task description.
        Supports lexical matching (default) and optional embedding retrieval.
        """
        text = str(task_description or "").strip().lower()
        if not text:
            return []
        k = max(0, int(top_k))
        if k == 0:
            return []
        if self._skill_retrieval_mode == "embedding":
            suggested = self._suggest_skills_embedding(text, k)
            if suggested:
                return suggested
        if self._skill_retrieval_mode == "hybrid":
            suggested = self._suggest_skills_hybrid(text, k)
            if suggested:
                return suggested
        return self._suggest_skills_lexical(text, k)

    def _suggest_skills_lexical(self, text: str, k: int) -> List[str]:
        ranked = self._rank_skills_lexical(text)
        return [name for name, _ in ranked[:k]]

    def _rank_skills_lexical(self, text: str) -> List[tuple[str, float]]:
        task_tokens = {
            token for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", text) if token
        }
        scored: List[tuple[str, float]] = []
        for skill in self.list_skills(filter_unavailable=True):
            name = str(skill.get("name") or "").strip()
            if not name:
                continue
            desc = str(skill.get("description") or "").strip()
            haystack = f"{name} {desc}".lower()
            score = 0
            if name.lower() in text:
                score += 4
            if task_tokens:
                haystack_tokens = {
                    token
                    for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", haystack)
                    if token
                }
                score += len(task_tokens & haystack_tokens)
            if score > 0:
                scored.append((name, float(score)))
        if not scored:
            return []
        max_score = max(score for _, score in scored) or 1.0
        ranked = [(name, float(score) / float(max_score)) for name, score in scored]
        ranked.sort(key=lambda row: (-row[1], row[0].lower()))
        return ranked

    def _suggest_skills_embedding(self, text: str, k: int) -> List[str]:
        ranked = self._rank_skills_embedding(text)
        return [name for name, _ in ranked[:k]]

    def _rank_skills_embedding(self, text: str) -> List[tuple[str, float]]:
        skills = self.list_skills(filter_unavailable=True)
        if not skills:
            return []
        signature = tuple(
            f"{str(s.get('name') or '').strip()}|{str(s.get('description') or '').strip()}"
            for s in skills
            if str(s.get("name") or "").strip()
        )
        if not signature:
            return []
        model = self._get_embedding_model()
        if model is None:
            return []
        if self._embedding_cache_signature != signature:
            pairs: list[tuple[str, list[float]]] = []
            texts: list[str] = []
            names: list[str] = []
            for s in skills:
                name = str(s.get("name") or "").strip()
                if not name:
                    continue
                desc = str(s.get("description") or "").strip()
                texts.append(f"{name}. {desc}".strip())
                names.append(name)
            vectors = self._encode_texts(model, texts)
            if len(vectors) != len(names):
                return []
            for name, vec in zip(names, vectors):
                pairs.append((name, vec))
            self._embedding_cache_signature = signature
            self._embedding_cache_pairs = pairs
        query_vectors = self._encode_texts(model, [text])
        if not query_vectors:
            return []
        query = query_vectors[0]
        scored: list[tuple[float, str]] = []
        for name, vec in self._embedding_cache_pairs:
            sim = self._cosine_similarity(query, vec)
            scored.append((sim, name))
        scored.sort(key=lambda row: (-row[0], row[1].lower()))
        if not scored:
            return []
        best = max(sim for sim, _ in scored)
        worst = min(sim for sim, _ in scored)
        if best == worst:
            return [(name, 1.0) for _, name in scored]
        span = float(best - worst) or 1.0
        return [(name, (float(sim) - float(worst)) / span) for sim, name in scored]

    def _suggest_skills_hybrid(self, text: str, k: int) -> List[str]:
        lexical = self._rank_skills_lexical(text)
        embedding = self._rank_skills_embedding(text)
        if not embedding:
            return [name for name, _ in lexical[:k]]
        combined: dict[str, float] = {}
        for name, score in lexical:
            combined[name] = combined.get(name, 0.0) + (0.4 * float(score))
        for name, score in embedding:
            combined[name] = combined.get(name, 0.0) + (0.6 * float(score))
        ranked = sorted(combined.items(), key=lambda row: (-row[1], row[0].lower()))
        return [name for name, _ in ranked[:k]]

    def _get_embedding_model(self) -> Any:
        if self._embedding_client is not None:
            return self._embedding_client
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            return None
        try:
            self._embedding_client = SentenceTransformer(self._embedding_model_path)
        except Exception:
            return None
        return self._embedding_client

    @staticmethod
    def _encode_texts(model: Any, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        encoded: Any
        try:
            encoded = model.encode(
                list(texts),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except TypeError:
            encoded = model.encode(list(texts))
        except Exception:
            return []
        rows = encoded.tolist() if hasattr(encoded, "tolist") else encoded
        if not isinstance(rows, list):
            return []
        out: list[list[float]] = []
        for row in rows:
            if not isinstance(row, (list, tuple)):
                return []
            try:
                out.append([float(v) for v in row])
            except Exception:
                return []
        return out

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(float(a) * float(b) for a, b in zip(left, right))
        left_norm = sum(float(a) * float(a) for a in left) ** 0.5
        right_norm = sum(float(b) * float(b) for b in right) ** 0.5
        if left_norm <= 0.0 or right_norm <= 0.0:
            return 0.0
        return dot / (left_norm * right_norm)

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

    @classmethod
    def _parse_skill_meta(cls, frontmatter: Dict[str, Any]) -> Dict[str, Any]:
        metadata = cls._parse_agent_metadata(frontmatter.get("metadata", {}))
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
            if key in frontmatter:
                out[key] = frontmatter[key]
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

    def _check_requirements(self, skill_meta: dict) -> bool:
        if not bool(skill_meta.get("__manifest_valid", True)):
            return False
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

        if bins or requires.get("any_bins", requires.get("anyBins", [])):
            from annolid.core.agent.tools.resolve import resolve_command

        for b in bins:
            if not resolve_command(str(b)):
                return False
        any_bins = requires.get("any_bins", requires.get("anyBins", []))
        if isinstance(any_bins, str):
            any_bins = [any_bins]
        if isinstance(any_bins, list) and any_bins:
            if not any(resolve_command(str(b)) for b in any_bins):
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
        parsed = self._parse_skill_meta(meta)
        manifest = validate_skill_manifest(meta)
        parsed["__manifest_valid"] = bool(manifest.valid)
        parsed["__manifest_errors"] = list(manifest.errors)
        return parsed

    def _get_missing_requirements(self, skill_meta: dict) -> str:
        missing: List[str] = []
        manifest_errors = skill_meta.get("__manifest_errors")
        if isinstance(manifest_errors, list) and manifest_errors:
            missing.append("MANIFEST: " + "; ".join(str(e) for e in manifest_errors))
        requires = skill_meta.get("requires", {})
        bins = requires.get("bins", [])
        if isinstance(bins, str):
            bins = [bins]

        if bins or requires.get("any_bins", requires.get("anyBins", [])):
            from annolid.core.agent.tools.resolve import resolve_command

        for b in bins:
            if not resolve_command(str(b)):
                missing.append(f"CLI: {b}")
        any_bins = requires.get("any_bins", requires.get("anyBins", []))
        if isinstance(any_bins, str):
            any_bins = [any_bins]
        if isinstance(any_bins, list) and any_bins:
            if not any(resolve_command(str(b)) for b in any_bins):
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

    @classmethod
    def _read_frontmatter_from_path(cls, path: str) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            return {}
        try:
            content = p.read_text(encoding="utf-8")
        except OSError:
            return {}
        return cls._read_frontmatter(content)

    def _get_frontmatter(self, path: str) -> Dict[str, Any]:
        return self._read_frontmatter_from_path(path)

    @staticmethod
    def _load_extra_skill_dirs() -> List[Path]:
        cfg = SkillLoadConfig.from_sources(get_config_path=get_config_path)
        return list(cfg.extra_dirs)

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
