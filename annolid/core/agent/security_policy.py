from __future__ import annotations

import os
import re


# Docker Hub's multi-architecture index for the official Ubuntu 24.04 image,
# verified on 2026-07-24. Updating the sandbox base is an explicit, reviewable
# dependency change instead of an implicit mutation of a floating tag.
DEFAULT_SANDBOX_CONTAINER_IMAGE = (
    "ubuntu:24.04@"
    "sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90"
)
_CONTAINER_DIGEST_RE = re.compile(r"@sha256:[0-9a-fA-F]{64}$")


def _read_bool_env(name: str) -> bool | None:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return None
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return None


def is_production_mode() -> bool:
    explicit = _read_bool_env("ANNOLID_PRODUCTION_MODE")
    if explicit is not None:
        return bool(explicit)
    env_name = str(os.getenv("ANNOLID_ENV", "")).strip().lower()
    return env_name in {"prod", "production"}


def require_signed_updates() -> bool:
    explicit = _read_bool_env("ANNOLID_REQUIRE_SIGNED_UPDATES")
    if explicit is not None:
        return bool(explicit)
    return is_production_mode()


def require_signed_skills() -> bool:
    explicit = _read_bool_env("ANNOLID_REQUIRE_SIGNED_SKILLS")
    if explicit is not None:
        return bool(explicit)
    return is_production_mode()


def bot_update_requires_operator_consent() -> bool:
    explicit = _read_bool_env("ANNOLID_BOT_UPDATE_REQUIRE_CONSENT")
    if explicit is not None:
        return bool(explicit)
    return True


def operator_consent_phrase() -> str:
    return (
        str(os.getenv("ANNOLID_OPERATOR_UPDATE_CONSENT_PHRASE", "")).strip()
        or "APPROVE_ANNOLID_CORE_UPDATE"
    )


def has_operator_consent(value: str) -> bool:
    return str(value or "").strip() == operator_consent_phrase()


def is_digest_pinned_container_image(value: object) -> bool:
    """Return whether a container reference ends in an immutable SHA-256 digest."""
    return bool(_CONTAINER_DIGEST_RE.search(str(value or "").strip()))
