from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from annolid.core.agent.observability import emit_governance_event
from annolid.core.agent.security_policy import require_signed_skills

from .schema import SkillLoadConfig, SkillRecord, validate_skill_manifest
from .watcher import SkillRegistryWatcher


class SkillRegistry:
    """Registry of discoverable skills with precedence + optional hot reload."""

    def __init__(
        self,
        workspace: Path,
        *,
        builtin_skills_dir: Path,
        managed_skills_dir: Path,
        parse_meta: Callable[[Dict[str, Any]], Dict[str, Any]],
        read_frontmatter_from_path: Callable[[str], Dict[str, Any]],
        get_config_path: Callable[[], Path],
        watch: Optional[bool] = None,
        watch_poll_seconds: Optional[float] = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.workspace_skills = self.workspace / "skills"
        self.builtin_skills = Path(builtin_skills_dir)
        self.managed_skills = Path(managed_skills_dir)
        self._parse_meta = parse_meta
        self._read_frontmatter_from_path = read_frontmatter_from_path

        cfg = SkillLoadConfig.from_sources(get_config_path=get_config_path)
        self.extra_skill_dirs = list(cfg.extra_dirs)
        self.watch_enabled = bool(cfg.watch if watch is None else watch)
        poll_seconds = (
            cfg.poll_seconds if watch_poll_seconds is None else watch_poll_seconds
        )
        self.watch_poll_seconds = max(0.0, float(poll_seconds))
        self._watcher = SkillRegistryWatcher(poll_seconds=self.watch_poll_seconds)

        self._snapshot: List[SkillRecord] = []

    def refresh(self, *, trigger: str = "manual") -> None:
        before_names = [row.name for row in self._snapshot]
        self._snapshot = self._build_snapshot()
        after_names = [row.name for row in self._snapshot]
        self._emit_snapshot_change_event(
            before_names=before_names,
            after_names=after_names,
            trigger=trigger,
        )
        self._watcher.reset(
            [root for _, root in self._iter_skill_roots_by_precedence()]
        )

    def refresh_if_needed(self) -> bool:
        if not self.watch_enabled:
            return False
        roots = [root for _, root in self._iter_skill_roots_by_precedence()]
        if self._watcher.changed(roots):
            self.refresh(trigger="watch")
            return True
        return False

    def list_skills(self) -> List[Dict[str, Any]]:
        if not self._snapshot:
            self.refresh()
        else:
            self.refresh_if_needed()
        return [row.to_dict() for row in self._snapshot]

    def iter_roots(self) -> Sequence[Tuple[str, Path]]:
        return self._iter_skill_roots_by_precedence()

    def _build_snapshot(self) -> List[SkillRecord]:
        by_name: Dict[str, SkillRecord] = {}
        for source, root in self._iter_skill_roots_by_precedence():
            if not root.exists() or not root.is_dir():
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
                path_str = str(skill_file)
                meta = self._read_frontmatter_from_path(path_str)
                enforce_signature = bool(
                    require_signed_skills() and source != "builtin"
                )
                manifest = validate_skill_manifest(
                    meta,
                    skill_path=skill_file,
                    require_signature=enforce_signature,
                )
                parsed_meta = self._parse_meta(meta)
                parsed_meta["__manifest_valid"] = bool(manifest.valid)
                parsed_meta["__manifest_errors"] = list(manifest.errors)
                by_name[name] = SkillRecord(
                    name=name,
                    path=path_str,
                    source=source,
                    description=str(meta.get("description") or name),
                    parsed_meta=parsed_meta,
                    raw_meta=dict(meta),
                    manifest_valid=bool(manifest.valid),
                    manifest_errors=list(manifest.errors),
                )
        return [by_name[k] for k in sorted(by_name.keys())]

    def _iter_skill_roots_by_precedence(self) -> List[Tuple[str, Path]]:
        roots: List[Tuple[str, Path]] = [
            ("workspace", self.workspace_skills),
            ("managed", self.managed_skills),
            ("builtin", self.builtin_skills),
        ]
        for idx, path in enumerate(self.extra_skill_dirs):
            roots.append((f"extra:{idx}", path))
        return roots

    def _emit_snapshot_change_event(
        self,
        *,
        before_names: List[str],
        after_names: List[str],
        trigger: str,
    ) -> None:
        before_set = set(before_names)
        after_set = set(after_names)
        added = sorted(name for name in after_set if name not in before_set)
        removed = sorted(name for name in before_set if name not in after_set)
        if not added and not removed:
            return
        emit_governance_event(
            event_type="skills",
            action="snapshot",
            outcome="ok",
            actor="system",
            details={
                "workspace": str(self.workspace),
                "trigger": str(trigger or "manual"),
                "count_before": len(before_names),
                "count_after": len(after_names),
                "added": added,
                "removed": removed,
            },
        )
