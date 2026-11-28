from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from annolid.version import get_version

_PYPI_PROJECT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
ANNOLID_RELEASES_URL = "https://github.com/healthonrails/annolid/releases"


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


def release_page_url(project: str = "annolid", version: str = "") -> str:
    project_name = str(project or "annolid").strip() or "annolid"
    ver = str(version or "").strip()
    if project_name.lower() == "annolid":
        if ver:
            return f"{ANNOLID_RELEASES_URL}/tag/v{ver.lstrip('v')}"
        return ANNOLID_RELEASES_URL
    encoded = quote(project_name, safe="")
    if ver:
        return f"https://pypi.org/project/{encoded}/{quote(ver, safe='')}/"
    return f"https://pypi.org/project/{encoded}/"


def _normalize_channel(channel: str) -> str:
    text = str(channel or "stable").strip().lower()
    if text not in {"stable", "beta", "dev"}:
        return "stable"
    return text


def _pad_version_tuple(
    a: tuple[int, ...], b: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    size = max(len(a), len(b))
    return a + (0,) * (size - len(a)), b + (0,) * (size - len(b))


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
                left, right = _pad_version_tuple(self.nums, other.nums)
                if left != right:
                    return left < right
                # Prerelease sorts before final release of the same numeric tuple.
                return bool(self.is_prerelease) and not bool(other.is_prerelease)

            def __eq__(self, other: object) -> bool:
                if not isinstance(other, _Fallback):
                    return NotImplemented
                left, right = _pad_version_tuple(self.nums, other.nums)
                return left == right and bool(self.is_prerelease) == bool(
                    other.is_prerelease
                )

            def __str__(self) -> str:
                return self.s

        return _Fallback(raw, tuple(parts))


def _version_allowed(version_obj: Any, *, channel: str) -> bool:
    prerelease = bool(getattr(version_obj, "is_prerelease", False))
    if channel == "stable":
        return not prerelease
    return True


def _has_usable_files(files: Any) -> bool:
    if not isinstance(files, list):
        return False
    dicts = [f for f in files if isinstance(f, dict)]
    if not dicts:
        return False
    return any(not bool(f.get("yanked", False)) for f in dicts)


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
    project_name = str(project or "").strip()
    if not _PYPI_PROJECT_RE.fullmatch(project_name):
        raise ValueError(f"Invalid package project name: {project!r}")
    resolved_channel = _normalize_channel(channel)
    url = f"https://pypi.org/pypi/{quote(project_name, safe='')}/json"
    req = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": f"annolid/{str(get_version() or '0.0.0')}",
        },
    )
    try:
        with urlopen(req, timeout=max(0.5, float(timeout_s))) as resp:  # noqa: S310
            try:
                raw = resp.read(_MAX_MANIFEST_BYTES + 1)
            except TypeError:
                raw = resp.read()
            if len(raw) > _MAX_MANIFEST_BYTES:
                raise RuntimeError(
                    "Release manifest response exceeded maximum allowed size."
                )
            payload = json.loads(raw.decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(
            f"Release server returned HTTP {exc.code} for {project_name!r} ({url})."
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not reach release server ({url}): {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"Timed out checking for updates after {float(timeout_s):.1f}s ({url})."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON in release metadata from {url}: {exc}"
        ) from exc

    releases = payload.get("releases") if isinstance(payload, dict) else {}
    if not isinstance(releases, dict):
        raise RuntimeError("Invalid release metadata payload.")

    usable_parsed: List[tuple[Any, str]] = []
    all_parsed: List[tuple[Any, str]] = []
    for version_text, release_files in releases.items():
        obj = _parse_version_obj(str(version_text))
        if obj is None:
            continue
        if not _version_allowed(obj, channel=resolved_channel):
            continue
        row = (obj, str(version_text))
        all_parsed.append(row)
        if _has_usable_files(release_files):
            usable_parsed.append(row)

    parsed = usable_parsed or all_parsed
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
        project=project_name,
        channel=resolved_channel,
        version=latest,
        release_date=str(chosen.get("upload_time_iso_8601") or ""),
        artifact_url=str(chosen.get("url") or ""),
        artifact_sha256=str(digests.get("sha256") or ""),
        signature="",
        signature_alg="none",
        source=url,
    )
