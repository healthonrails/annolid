from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from typing import Any, Dict

from annolid.version import get_version

from .manifest import UpdateManifest


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    reason: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "details": dict(self.details),
        }


def _parse_version_obj(text: str):
    raw = str(text or "").strip()
    if not raw:
        return None
    try:  # pragma: no cover
        from packaging.version import Version

        return Version(raw)
    except Exception:
        nums = []
        for p in raw.split("."):
            n = "".join(ch for ch in p if ch.isdigit())
            nums.append(int(n) if n else 0)

        class _Fallback:
            def __init__(self, s: str, parts: tuple[int, ...]):
                self.s = s
                self.parts = parts

            def __lt__(self, other: "_Fallback") -> bool:
                return self.parts < other.parts

            def __str__(self) -> str:
                return self.s

        return _Fallback(raw, tuple(nums))


def _verify_hmac_signature(manifest: UpdateManifest, secret: str) -> bool:
    payload = {
        "project": manifest.project,
        "channel": manifest.channel,
        "version": manifest.version,
        "artifact_url": manifest.artifact_url,
        "artifact_sha256": manifest.artifact_sha256,
    }
    message = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return hmac.compare_digest(digest, str(manifest.signature or "").strip())


def verify_manifest(
    manifest: UpdateManifest,
    *,
    require_signature: bool = False,
) -> VerificationResult:
    current = str(get_version() or "0.0.0")
    curr_obj = _parse_version_obj(current)
    next_obj = _parse_version_obj(manifest.version)
    if next_obj is None:
        return VerificationResult(
            False, "invalid_manifest_version", {"version": manifest.version}
        )
    if curr_obj is not None and not (curr_obj < next_obj):
        return VerificationResult(
            False,
            "not_newer_than_current",
            {"current_version": current, "manifest_version": manifest.version},
        )

    if not str(manifest.artifact_sha256 or "").strip():
        return VerificationResult(False, "missing_artifact_sha256", {})

    has_signature = bool(str(manifest.signature or "").strip())
    if require_signature and not has_signature:
        return VerificationResult(False, "signature_required_missing", {})

    sig_alg = str(manifest.signature_alg or "none").strip().lower()
    if has_signature:
        if sig_alg not in {"hmac-sha256", "hmac_sha256"}:
            return VerificationResult(
                False, "unsupported_signature_alg", {"signature_alg": sig_alg}
            )
        secret = str(os.getenv("ANNOLID_UPDATE_SIGNING_KEY") or "").strip()
        if not secret:
            return VerificationResult(
                False, "missing_signing_key", {"env": "ANNOLID_UPDATE_SIGNING_KEY"}
            )
        if not _verify_hmac_signature(manifest, secret):
            return VerificationResult(False, "signature_verification_failed", {})

    return VerificationResult(
        True,
        "ok",
        {
            "current_version": current,
            "manifest_version": manifest.version,
            "signature_present": has_signature,
            "signature_alg": sig_alg,
        },
    )
