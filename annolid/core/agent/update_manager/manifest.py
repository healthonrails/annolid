from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List
from urllib.request import urlopen


@dataclass(frozen=True)
class UpdateManifest:
    project: str
    channel: str
    version: str
    release_date: str
    artifact_url: str
    artifact_sha256: str
    signature: str
    signature_alg: str
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project": self.project,
            "channel": self.channel,
            "version": self.version,
            "release_date": self.release_date,
            "artifact_url": self.artifact_url,
            "artifact_sha256": self.artifact_sha256,
            "signature": self.signature,
            "signature_alg": self.signature_alg,
            "source": self.source,
        }


def _normalize_channel(channel: str) -> str:
    text = str(channel or "stable").strip().lower()
    if text not in {"stable", "beta", "dev"}:
        return "stable"
    return text


def _parse_version_obj(text: str):
    raw = str(text or "").strip()
    if not raw:
        return None
    try:  # pragma: no cover - optional dep
        from packaging.version import Version

        return Version(raw)
    except Exception:
        parts = []
        for p in raw.split("."):
            n = "".join(ch for ch in p if ch.isdigit())
            parts.append(int(n) if n else 0)

        class _Fallback:
            def __init__(self, s: str, nums: tuple[int, ...]):
                self.s = s
                self.nums = nums
                self.is_prerelease = any(ch.isalpha() for ch in s)

            def __lt__(self, other: "_Fallback") -> bool:
                return self.nums < other.nums

            def __str__(self) -> str:
                return self.s

        return _Fallback(raw, tuple(parts))


def _version_allowed(version_obj: Any, *, channel: str) -> bool:
    prerelease = bool(getattr(version_obj, "is_prerelease", False))
    if channel == "stable":
        return not prerelease
    return True


def _extract_best_file(files: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not files:
        return {}
    wheels = [f for f in files if str(f.get("filename", "")).endswith(".whl")]
    candidates = wheels or files
    # prefer non-yanked and upload with hash
    candidates = sorted(
        candidates,
        key=lambda f: (
            bool(f.get("yanked", False)),
            0 if str((f.get("digests") or {}).get("sha256", "")).strip() else 1,
        ),
    )
    return candidates[0] if candidates else {}


def fetch_channel_manifest(
    *,
    project: str = "annolid",
    channel: str = "stable",
    timeout_s: float = 4.0,
) -> UpdateManifest:
    resolved_channel = _normalize_channel(channel)
    url = f"https://pypi.org/pypi/{project}/json"
    with urlopen(url, timeout=max(0.5, float(timeout_s))) as resp:  # noqa: S310
        payload = json.loads(resp.read().decode("utf-8"))

    releases = payload.get("releases") if isinstance(payload, dict) else {}
    if not isinstance(releases, dict):
        raise RuntimeError("Invalid release metadata payload.")

    parsed: List[tuple[Any, str]] = []
    for version_text in releases.keys():
        obj = _parse_version_obj(str(version_text))
        if obj is None:
            continue
        if not _version_allowed(obj, channel=resolved_channel):
            continue
        parsed.append((obj, str(version_text)))
    if not parsed:
        raise RuntimeError(f"No releases found for channel={resolved_channel}.")

    parsed.sort(key=lambda row: row[0])
    _version_obj, latest = parsed[-1]
    files = releases.get(latest)
    if not isinstance(files, list):
        files = []
    chosen = _extract_best_file([f for f in files if isinstance(f, dict)])
    digests = chosen.get("digests") if isinstance(chosen, dict) else {}
    if not isinstance(digests, dict):
        digests = {}

    # PyPI does not provide a detached signature in JSON metadata.
    # `signature` can be supplied by private channels/overrides later.
    return UpdateManifest(
        project=project,
        channel=resolved_channel,
        version=latest,
        release_date=str(chosen.get("upload_time_iso_8601") or ""),
        artifact_url=str(chosen.get("url") or ""),
        artifact_sha256=str(digests.get("sha256") or ""),
        signature="",
        signature_alg="none",
        source=url,
    )
