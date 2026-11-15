from __future__ import annotations

from hashlib import sha256
import hashlib
from pathlib import Path
import urllib.request


def _is_google_drive_url(url: str) -> bool:
    lowered = str(url or "").strip().lower()
    return "drive.google.com" in lowered or "docs.google.com" in lowered


def _http_chunked_download(url: str, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    part_path = target.with_name(f"{target.name}.part")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Annolid-ModelAssets/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            with part_path.open("wb") as out_f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)
        part_path.replace(target)
    except Exception:
        part_path.unlink(missing_ok=True)
        raise
    return str(target)


class _StdlibGdownShim:
    @staticmethod
    def cached_download(
        url: str,
        path: str,
        quiet: bool = True,
        fuzzy: bool = True,
        **_kwargs: object,
    ) -> str:
        global gdown
        if _is_google_drive_url(url):
            from annolid.infrastructure.capabilities import ensure_capability

            ensure_capability(
                "model_downloads",
                reason="Google Drive model checkpoint download",
            )
            import gdown as real_gdown

            gdown = real_gdown
            return str(
                real_gdown.cached_download(url, str(path), quiet=quiet, fuzzy=fuzzy)
            )
        return _http_chunked_download(url, Path(path))


try:
    import gdown
except ImportError:
    gdown = _StdlibGdownShim()  # type: ignore[assignment]


def annolid_workspace_dir() -> Path:
    return Path.home() / ".annolid" / "workspace"


def model_downloads_dir() -> Path:
    return annolid_workspace_dir() / "downloads"


def candidate_model_paths(path_value: str | Path) -> list[Path]:
    raw = str(path_value or "").strip()
    if not raw:
        return []

    path = Path(raw).expanduser()
    candidates: list[Path] = [path]
    if not path.is_absolute():
        candidates.append(annolid_workspace_dir() / path)
        try:
            candidates.append((Path.cwd() / path).resolve())
        except Exception:
            candidates.append(Path.cwd() / path)

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def resolve_existing_model_path(path_value: str | Path) -> Path | None:
    for candidate in candidate_model_paths(path_value):
        try:
            if candidate.exists():
                return candidate
        except Exception:
            continue
    return None


def sha256sum(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5sum(path: str | Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - used for legacy published checksums
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cached_download(url: str, target: Path, *, quiet: bool, fuzzy: bool) -> None:
    try:
        gdown.cached_download(url, str(target), quiet=quiet, fuzzy=fuzzy)
    except TypeError as exc:
        message = str(exc)
        if "fuzzy" not in message or "unexpected keyword argument" not in message:
            raise
        gdown.cached_download(url, str(target), quiet=quiet)


def ensure_cached_model_asset(
    *,
    file_name: str,
    url: str,
    expected_sha256: str | None = None,
    expected_md5: str | None = None,
    cache_dir: str | Path | None = None,
    quiet: bool = True,
    fuzzy: bool = True,
) -> Path:
    if not file_name:
        raise ValueError("file_name must be non-empty")
    if not url:
        raise ValueError("url must be non-empty")
    sha_expected = str(expected_sha256 or "").strip().lower()
    md5_expected = str(expected_md5 or "").strip().lower()
    if bool(sha_expected) == bool(md5_expected):
        raise ValueError(
            "Provide exactly one checksum: expected_sha256 or expected_md5"
        )

    target_dir = Path(cache_dir).expanduser() if cache_dir else model_downloads_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / file_name

    def _matches_checksum(path: Path) -> bool:
        if sha_expected:
            return sha256sum(path) == sha_expected
        return md5sum(path) == md5_expected

    if target.is_file() and _matches_checksum(target):
        return target

    target.unlink(missing_ok=True)
    _cached_download(url, target, quiet=quiet, fuzzy=fuzzy)

    if not target.is_file():
        raise RuntimeError(f"Failed to download model asset: {url}")

    if not _matches_checksum(target):
        if sha_expected:
            got = sha256sum(target)
            expected_text = f"sha256={sha_expected}"
            got_text = f"sha256={got}"
        else:
            got = md5sum(target)
            expected_text = f"md5={md5_expected}"
            got_text = f"md5={got}"
        target.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded asset checksum mismatch for {target.name}: "
            f"expected {expected_text}, got {got_text}"
        )

    return target
