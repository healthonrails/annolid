from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List


@dataclass(frozen=True)
class SkillRecord:
    name: str
    path: str
    source: str
    description: str
    parsed_meta: Dict[str, Any]
    raw_meta: Dict[str, Any]
    manifest_valid: bool = True
    manifest_errors: List[str] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "source": self.source,
            "description": self.description,
            "parsed_meta": dict(self.parsed_meta),
            "raw_meta": dict(self.raw_meta),
            "manifest_valid": bool(self.manifest_valid),
            "manifest_errors": list(self.manifest_errors or []),
        }


@dataclass(frozen=True)
class SkillLoadConfig:
    extra_dirs: List[Path]
    watch: bool
    poll_seconds: float

    @staticmethod
    def read_bool_env(name: str) -> bool | None:
        raw = str(os.getenv(name, "")).strip().lower()
        if not raw:
            return None
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
        return None

    @classmethod
    def from_sources(
        cls,
        *,
        get_config_path: Callable[[], Path],
    ) -> "SkillLoadConfig":
        extra_dirs: List[Path] = []
        watch = False
        poll_seconds = 1.0

        env_extra = str(os.getenv("ANNOLID_SKILLS_EXTRA_DIRS") or "").strip()
        if env_extra:
            for part in env_extra.split(os.pathsep):
                entry = str(part or "").strip()
                if entry:
                    extra_dirs.append(Path(entry).expanduser())

        env_poll = str(os.getenv("ANNOLID_SKILLS_WATCH_POLL_SECONDS", "")).strip()
        if env_poll:
            try:
                poll_seconds = max(0.0, float(env_poll))
            except ValueError:
                pass

        env_watch = cls.read_bool_env("ANNOLID_SKILLS_LOAD_WATCH")
        if env_watch is None:
            env_watch = cls.read_bool_env("ANNOLID_SKILLS_WATCH")

        try:
            cfg_path = get_config_path()
            if cfg_path.exists():
                payload = json.loads(cfg_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    skills = payload.get("skills") or {}
                    load = skills.get("load") if isinstance(skills, dict) else {}
                    if isinstance(load, dict):
                        cfg_extra = load.get("extraDirs") or load.get("extra_dirs")
                        if isinstance(cfg_extra, list):
                            for item in cfg_extra:
                                entry = str(item or "").strip()
                                if entry:
                                    extra_dirs.append(Path(entry).expanduser())
                        if "watch" in load and env_watch is None:
                            watch = bool(load.get("watch"))
                        cfg_poll = load.get("pollSeconds", load.get("poll_seconds"))
                        if cfg_poll is not None and not env_poll:
                            try:
                                poll_seconds = max(0.0, float(cfg_poll))
                            except (TypeError, ValueError):
                                pass
        except Exception:
            pass

        if env_watch is not None:
            watch = bool(env_watch)

        deduped: List[Path] = []
        seen: set[str] = set()
        for path in extra_dirs:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(path)
        return cls(extra_dirs=deduped, watch=watch, poll_seconds=poll_seconds)


@dataclass(frozen=True)
class SkillManifestValidation:
    valid: bool
    errors: List[str]


def _extract_signature_fields(frontmatter: Dict[str, Any]) -> tuple[str, str]:
    payload = dict(frontmatter or {})
    signature = str(payload.get("signature") or "").strip()
    signature_alg = str(
        payload.get("signature_alg") or payload.get("signatureAlg") or "none"
    ).strip()
    return signature, signature_alg


def _strip_frontmatter(content: str) -> str:
    text = str(content or "")
    if not text.startswith("---"):
        return text
    match = re.match(r"^---\n.*?\n---\n?", text, re.DOTALL)
    if match:
        return text[match.end() :]
    return text


def _verify_skill_signature(
    *,
    frontmatter: Dict[str, Any],
    skill_path: Path | None,
    require_signature: bool,
) -> List[str]:
    errors: List[str] = []
    signature, signature_alg = _extract_signature_fields(frontmatter)
    has_signature = bool(signature)
    if require_signature and not has_signature:
        errors.append("signature is required in production mode")
        return errors
    if not has_signature:
        return errors
    if str(signature_alg or "").strip().lower() not in {"hmac-sha256", "hmac_sha256"}:
        errors.append("unsupported signature_alg (expected hmac-sha256)")
        return errors
    if skill_path is None:
        errors.append("skill_path is required to verify signature")
        return errors
    secret = str(os.getenv("ANNOLID_SKILL_SIGNING_KEY") or "").strip()
    if not secret:
        errors.append("missing ANNOLID_SKILL_SIGNING_KEY for signature verification")
        return errors
    try:
        body = _strip_frontmatter(skill_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"failed to read skill file for signature verification: {exc}")
        return errors
    message = body.replace("\r\n", "\n").encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(digest, signature):
        errors.append("skill signature verification failed")
    return errors


def validate_skill_manifest(
    frontmatter: Dict[str, Any],
    *,
    skill_path: Path | None = None,
    require_signature: bool = False,
) -> SkillManifestValidation:
    payload = dict(frontmatter or {})
    errors: List[str] = []

    description = payload.get("description")
    if description is not None and not isinstance(description, str):
        errors.append("description must be a string when provided")

    metadata = payload.get("metadata")
    if metadata is not None and not isinstance(metadata, (dict, str)):
        errors.append("metadata must be an object or JSON string")

    if "always" in payload and not isinstance(payload.get("always"), bool):
        errors.append("always must be a boolean")

    if "disable-model-invocation" in payload and not isinstance(
        payload.get("disable-model-invocation"), bool
    ):
        errors.append("disable-model-invocation must be a boolean")
    if "disable_model_invocation" in payload and not isinstance(
        payload.get("disable_model_invocation"), bool
    ):
        errors.append("disable_model_invocation must be a boolean")

    if "user-invocable" in payload and not isinstance(
        payload.get("user-invocable"), bool
    ):
        errors.append("user-invocable must be a boolean")
    if "user_invocable" in payload and not isinstance(
        payload.get("user_invocable"), bool
    ):
        errors.append("user_invocable must be a boolean")

    if "os" in payload:
        os_value = payload.get("os")
        if isinstance(os_value, str):
            os_list = [os_value]
        elif isinstance(os_value, list):
            os_list = os_value
        else:
            os_list = []
            errors.append("os must be a string or list of strings")
        if os_list and not all(
            isinstance(item, str) and str(item).strip() for item in os_list
        ):
            errors.append("os entries must be non-empty strings")

    requires = payload.get("requires")
    if requires is not None and not isinstance(requires, dict):
        errors.append("requires must be an object when provided")

    errors.extend(
        _verify_skill_signature(
            frontmatter=payload,
            skill_path=skill_path,
            require_signature=bool(require_signature),
        )
    )

    return SkillManifestValidation(valid=len(errors) == 0, errors=errors)
