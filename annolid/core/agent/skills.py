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

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

BUILTIN_SKILLS_DIR = Path(__file__).parent / "skills"
_DEFAULT_MANAGED_SKILLS_DIR = Path.home() / ".annolid" / "skills"
MANAGED_SKILLS_DIR = _DEFAULT_MANAGED_SKILLS_DIR

_LEXICAL_STOPWORDS = frozenset(
    {
        "about",
        "above",
        "add",
        "after",
        "again",
        "all",
        "also",
        "and",
        "any",
        "are",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "can",
        "check",
        "could",
        "current",
        "currently",
        "during",
        "each",
        "etc",
        "few",
        "find",
        "for",
        "from",
        "further",
        "get",
        "had",
        "has",
        "have",
        "help",
        "her",
        "here",
        "him",
        "his",
        "how",
        "into",
        "its",
        "just",
        "look",
        "make",
        "may",
        "might",
        "more",
        "most",
        "must",
        "need",
        "nor",
        "not",
        "now",
        "off",
        "once",
        "only",
        "onto",
        "open",
        "other",
        "our",
        "out",
        "over",
        "own",
        "per",
        "please",
        "read",
        "right",
        "run",
        "same",
        "set",
        "shall",
        "she",
        "should",
        "show",
        "some",
        "such",
        "take",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "too",
        "under",
        "use",
        "used",
        "using",
        "very",
        "via",
        "want",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "write",
        "you",
        "your",
    }
)

_SHORT_DOMAIN_TOKENS = frozenset(
    {
        "3d",
        "ai",
        "ci",
        "gh",
        "id",
        "io",
        "ir",
        "l4",
        "ml",
        "os",
        "pr",
        "qc",
        "t4",
        "ui",
    }
)


def get_managed_skills_dir() -> Path:
    """Resolve managed skills directory lazily without import-time filesystem writes."""
    if MANAGED_SKILLS_DIR != _DEFAULT_MANAGED_SKILLS_DIR:
        return Path(MANAGED_SKILLS_DIR)
    return Path.home() / ".annolid" / "skills"


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
            else get_managed_skills_dir()
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
            path = esc(str(s["path"]))
            skill_meta = s.get("parsed_meta", self._get_skill_meta_by_path(s["path"]))
            if bool(skill_meta.get("disable_model_invocation")):
                continue
            available = self._check_requirements(skill_meta)
            extras = ""
            if not bool(skill_meta.get("user_invocable", True)):
                extras += "<user_invocable>false</user_invocable>"
            if not available:
                missing = self._get_missing_requirements(skill_meta)
                if missing:
                    extras += f"<requires>{esc(missing)}</requires>"
            lines.append(
                f'<skill available="{str(available).lower()}">'
                f"<name>{name}</name>"
                f"<description>{desc}</description>"
                f"<location>{path}</location>"
                f"{extras}</skill>"
            )
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
        scored = self.suggest_skills_for_task_scored(task_description, top_k=top_k)
        return [str(row.get("name") or "") for row in scored if str(row.get("name"))]

    def suggest_skills_for_task_scored(
        self, task_description: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Suggest skills with score/strategy metadata for profiling and UX."""
        text = str(task_description or "").strip().lower()
        if not text:
            return []
        k = max(0, int(top_k))
        if k == 0:
            return []
        skills = [
            s
            for s in self.list_skills(filter_unavailable=True)
            if not bool(
                (
                    s.get("parsed_meta")
                    if isinstance(s.get("parsed_meta"), dict)
                    else self._get_skill_meta_by_path(s["path"])
                ).get("disable_model_invocation")
            )
        ]
        ranked, strategy = self._rank_skills_for_mode(text, skills)
        if not ranked:
            return []
        indexed = {
            str(row.get("name") or "").strip(): row
            for row in skills
            if str(row.get("name") or "").strip()
        }
        out: List[Dict[str, Any]] = []
        for name, score in ranked[:k]:
            skill = indexed.get(name, {})
            out.append(
                {
                    "name": name,
                    "score": float(score),
                    "strategy": strategy,
                    "source": str(skill.get("source") or ""),
                    "description": str(skill.get("description") or ""),
                    "path": str(skill.get("path") or ""),
                }
            )
        return out

    def describe_skill_pool(self, *, preview_limit: int = 100) -> Dict[str, Any]:
        all_skills = self.list_skills(filter_unavailable=False)
        available_names = {
            str(row.get("name") or "")
            for row in self.list_skills(filter_unavailable=True)
            if str(row.get("name") or "")
        }
        source_counts: Dict[str, int] = {}
        unavailable: List[Dict[str, str]] = []
        always: List[str] = []
        for row in all_skills:
            source = str(row.get("source") or "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            if name in available_names:
                if bool(
                    (row.get("parsed_meta") or {}).get("always")
                    if isinstance(row.get("parsed_meta"), dict)
                    else False
                ):
                    always.append(name)
                continue
            parsed_meta = row.get("parsed_meta")
            if not isinstance(parsed_meta, dict):
                parsed_meta = self._get_skill_meta_by_path(str(row.get("path") or ""))
            missing = self._get_missing_requirements(parsed_meta)
            unavailable.append(
                {
                    "name": name,
                    "source": source,
                    "reason": str(missing or "requirements_not_met"),
                }
            )
        rows_preview = [
            {
                "name": str(row.get("name") or ""),
                "source": str(row.get("source") or ""),
                "description": str(row.get("description") or ""),
                "available": str(row.get("name") or "") in available_names,
            }
            for row in all_skills[: max(1, int(preview_limit))]
        ]
        return {
            "retrieval_mode": str(self._skill_retrieval_mode or "lexical"),
            "counts": {
                "total": len(all_skills),
                "available": len(available_names),
                "unavailable": max(0, len(all_skills) - len(available_names)),
                "always": len(always),
            },
            "source_counts": source_counts,
            "always_skills": sorted(set(always)),
            "preview": rows_preview,
            "preview_truncated": max(0, len(all_skills) - len(rows_preview)),
            "unavailable_skills": unavailable[: max(1, int(preview_limit))],
        }

    def _rank_skills_for_mode(
        self,
        text: str,
        skills: Sequence[Dict[str, Any]],
    ) -> tuple[List[tuple[str, float]], str]:
        mode = str(self._skill_retrieval_mode or "lexical").strip().lower()
        if mode == "embedding":
            ranked = self._rank_skills_embedding(text, skills)
            if ranked:
                return ranked, "embedding"
            return self._rank_skills_lexical(text, skills), "embedding->lexical"
        if mode == "hybrid":
            ranked = self._rank_skills_hybrid(text, skills)
            if ranked:
                return ranked, "hybrid"
            return self._rank_skills_lexical(text, skills), "hybrid->lexical"
        return self._rank_skills_lexical(text, skills), "lexical"

    def _suggest_skills_lexical(self, text: str, k: int) -> List[str]:
        ranked = self._rank_skills_lexical(text)
        return [name for name, _ in ranked[:k]]

    @staticmethod
    def _tokenize_lexical(text: str) -> set[str]:
        lowered = str(text or "").lower()
        tokens: set[str] = set()
        for raw in re.findall(r"[a-z0-9]+", lowered):
            if raw in _LEXICAL_STOPWORDS:
                continue
            if len(raw) >= 3 or raw in _SHORT_DOMAIN_TOKENS:
                tokens.add(raw)
                if (
                    len(raw) >= 4
                    and raw.endswith("s")
                    and not raw.endswith(("ss", "us", "is"))
                ):
                    stem = raw[:-1]
                    if stem not in _LEXICAL_STOPWORDS and (
                        len(stem) >= 3 or stem in _SHORT_DOMAIN_TOKENS
                    ):
                        tokens.add(stem)
        return tokens

    @staticmethod
    def _extract_skill_tags(skill: Dict[str, Any]) -> list[str]:
        parsed = skill.get("parsed_meta")
        if not isinstance(parsed, dict):
            return []
        raw_tags = parsed.get("tags")
        if not isinstance(raw_tags, (list, tuple)):
            return []
        return [str(t).strip().lower() for t in raw_tags if str(t).strip()]

    def _rank_skills_lexical(
        self,
        text: str,
        skills: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[tuple[str, float]]:
        lowered_text = str(text or "").lower()
        normalized_text = re.sub(r"[^a-z0-9]+", " ", lowered_text).strip()
        padded_query = f" {normalized_text} " if normalized_text else ""
        task_tokens = self._tokenize_lexical(lowered_text)
        if "http://" in lowered_text or "https://" in lowered_text:
            task_tokens.add("url")
        scored: List[tuple[str, float]] = []
        candidates = list(skills or self.list_skills(filter_unavailable=True))
        for skill in candidates:
            name = str(skill.get("name") or "").strip()
            if not name:
                continue
            desc = str(skill.get("description") or "").strip()
            tags = self._extract_skill_tags(skill)
            score = 0.0
            name_lower = name.lower()
            name_phrase = re.sub(r"[^a-z0-9]+", " ", name_lower).strip()
            if (
                name_lower
                and re.search(
                    rf"(?<![a-z0-9]){re.escape(name_lower)}(?![a-z0-9])", lowered_text
                )
            ) or (name_phrase and padded_query and f" {name_phrase} " in padded_query):
                score += 6.0
            if task_tokens or padded_query:
                name_tokens = self._tokenize_lexical(name_lower)
                name_hits = task_tokens & name_tokens
                score += 2.5 * len(name_hits)

                single_tag_tokens: set[str] = set()
                matched_exact_phrases = 0
                matched_bag_phrases = 0
                for tag in tags:
                    tag_phrase = re.sub(r"[^a-z0-9]+", " ", tag).strip()
                    if not tag_phrase:
                        continue
                    if " " in tag_phrase:
                        if padded_query and f" {tag_phrase} " in padded_query:
                            matched_exact_phrases += 1
                        else:
                            # Match multi-word tags by unordered tokens only when
                            # every non-stopword component appears in the query,
                            # preventing generic words like "open" or "field" from
                            # triggering unrelated skills on their own.
                            raw_parts = [
                                part
                                for part in tag_phrase.split()
                                if len(part) >= 3 or part in _SHORT_DOMAIN_TOKENS
                            ]
                            significant_parts = [
                                part
                                for part in raw_parts
                                if part not in _LEXICAL_STOPWORDS
                            ]
                            if len(significant_parts) >= 2 and all(
                                (part in task_tokens)
                                or (
                                    len(part) >= 4
                                    and part.endswith("s")
                                    and part[:-1] in task_tokens
                                )
                                for part in significant_parts
                            ):
                                matched_bag_phrases += 1
                    else:
                        single_tag_tokens.update(self._tokenize_lexical(tag))
                score += 3.5 * matched_exact_phrases
                score += 2.5 * matched_bag_phrases
                tag_hits = (task_tokens & single_tag_tokens) - name_hits
                score += 2.0 * len(tag_hits)

                desc_tokens = self._tokenize_lexical(desc)
                desc_hits = (task_tokens & desc_tokens) - name_hits - tag_hits
                score += 1.0 * len(desc_hits)
            if score > 0.0:
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

    def _rank_skills_embedding(
        self,
        text: str,
        skills: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[tuple[str, float]]:
        candidates = list(skills or self.list_skills(filter_unavailable=True))
        if not candidates:
            return []
        signature = tuple(
            f"{str(s.get('name') or '').strip()}|{str(s.get('description') or '').strip()}|{','.join(self._extract_skill_tags(s))}"
            for s in candidates
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
            for s in candidates:
                name = str(s.get("name") or "").strip()
                if not name:
                    continue
                desc = str(s.get("description") or "").strip()
                tags = self._extract_skill_tags(s)
                tag_suffix = f" ({', '.join(tags)})" if tags else ""
                texts.append(f"{name}. {desc}{tag_suffix}".strip())
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
        skills = self.list_skills(filter_unavailable=True)
        lexical = self._rank_skills_lexical(text, skills)
        embedding = self._rank_skills_embedding(text, skills)
        if not embedding:
            return [name for name, _ in lexical[:k]]
        combined: dict[str, float] = {}
        for name, score in lexical:
            combined[name] = combined.get(name, 0.0) + (0.4 * float(score))
        for name, score in embedding:
            combined[name] = combined.get(name, 0.0) + (0.6 * float(score))
        ranked = sorted(combined.items(), key=lambda row: (-row[1], row[0].lower()))
        return [name for name, _ in ranked[:k]]

    def _rank_skills_hybrid(
        self,
        text: str,
        skills: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[tuple[str, float]]:
        candidates = list(skills or self.list_skills(filter_unavailable=True))
        lexical = self._rank_skills_lexical(text, candidates)
        embedding = self._rank_skills_embedding(text, candidates)
        if not embedding:
            return lexical
        combined: dict[str, float] = {}
        for name, score in lexical:
            combined[name] = combined.get(name, 0.0) + (0.4 * float(score))
        for name, score in embedding:
            combined[name] = combined.get(name, 0.0) + (0.6 * float(score))
        return sorted(combined.items(), key=lambda row: (-row[1], row[0].lower()))

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
            "tags",
            "keywords",
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
        raw_tags: list[str] = []
        for source in (metadata, frontmatter):
            for tag_key in ("tags", "keywords"):
                val = source.get(tag_key)
                if isinstance(val, str):
                    raw_tags.extend(p.strip() for p in val.split(",") if p.strip())
                elif isinstance(val, (list, tuple)):
                    raw_tags.extend(str(p).strip() for p in val if str(p).strip())
        if raw_tags:
            seen_tags: set[str] = set()
            deduped_tags: list[str] = []
            for tag in raw_tags:
                key_t = tag.lower()
                if key_t in seen_tags:
                    continue
                seen_tags.add(key_t)
                deduped_tags.append(key_t)
            out["tags"] = deduped_tags
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
        requires, requires_error = self._get_requires_mapping(skill_meta)
        if requires_error:
            return False
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
        requires, requires_error = self._get_requires_mapping(skill_meta)
        if requires_error:
            missing.append(requires_error)
            return ", ".join(missing)
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
    def _get_requires_mapping(skill_meta: dict) -> tuple[dict, str]:
        requires = skill_meta.get("requires", {})
        if requires is None:
            return {}, ""
        if isinstance(requires, dict):
            return requires, ""
        return {}, "requires must be an object"

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
