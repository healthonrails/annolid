#!/usr/bin/env python3
"""Unified local build, preview, watch, and validation workflow for Annolid's website and docs.

This tool mirrors `.github/workflows/healthonrails-site-sync.yml` locally so contributors
can edit `website/` and `docs/` side-by-side and verify the exact published layout:

- `/`                  -> `website/index.html`
- `/website-assets/`   -> `website/assets/`
- `/portal/`           -> MkDocs Material documentation portal
- `/<docs-route>/`     -> root compatibility docs routes + legacy redirects

Usage:
    python scripts/site_workflow.py stage
    python scripts/site_workflow.py preview --watch --port 8080
    python scripts/site_workflow.py check
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE_DIR = REPO_ROOT / ".site_preview"
REQUIRED_LANDING_TITLE = "Annolid | Research-Grade Animal Behavior AI"
DEFAULT_MAX_ASSET_KB = 5500
WARN_ASSET_KB = 2000

_MACHINE_PATH_RE = re.compile(
    r"(?:/Users/([A-Za-z0-9._-]+)/|/home/([A-Za-z0-9._-]+)/|[A-Za-z]:\\\\Users\\\\([A-Za-z0-9._-]+)\\\\)"
)
_DOC_PLACEHOLDER_USERS = {
    "you",
    "your",
    "yourname",
    "your_user",
    "your-user",
    "user",
    "username",
    "name",
    "alice",
    "bob",
    "lab",
    "researcher",
}
_HTML_ATTR_RE = re.compile(
    r"""(?:src|href|data-src|data-image|data-full)\s*=\s*["']([^"']+)["']""",
    re.IGNORECASE,
)
_CSS_URL_RE = re.compile(r"""url\(\s*["']?([^"')]+)["']?\s*\)""", re.IGNORECASE)
_MD_IMAGE_RE = re.compile(r"""!\[[^\]]*\]\(([^)\s]+)(?:\s+["'][^"']*["'])?\)""")
_HTML_IMG_SRC_RE = re.compile(r"""<img[^>]+src\s*=\s*["']([^"']+)["']""", re.IGNORECASE)


@dataclass
class CheckResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _find_mkdocs_executable(repo_root: Path = REPO_ROOT) -> list[str]:
    """Resolve MkDocs inside `.venv` first, falling back to current interpreter/PATH."""
    venv_mkdocs = repo_root / ".venv" / "bin" / "mkdocs"
    if venv_mkdocs.is_file():
        return [str(venv_mkdocs)]
    venv_win = repo_root / ".venv" / "Scripts" / "mkdocs.exe"
    if venv_win.is_file():
        return [str(venv_win)]
    which_mkdocs = shutil.which("mkdocs")
    if which_mkdocs:
        return [which_mkdocs]
    return [sys.executable, "-m", "mkdocs"]


def build_mkdocs(
    site_dir: Path,
    *,
    repo_root: Path = REPO_ROOT,
    strict: bool = True,
) -> None:
    """Run MkDocs build into `site_dir`."""
    cmd = [
        *_find_mkdocs_executable(repo_root),
        "build",
        "--clean",
        "--config-file",
        str(repo_root / "mkdocs.yml"),
        "--site-dir",
        str(site_dir),
    ]
    if strict:
        cmd.append("--strict")

    env = dict(os.environ)
    env["NO_MKDOCS_2_WARNING"] = "1"
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"MkDocs build failed (exit {proc.returncode}):\n{detail}")


def _copy_tree_replace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".DS_Store", "__pycache__"))


def sync_landing_page(
    website_dir: Path,
    dest_dir: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> None:
    """Sync `website/index.html` and `website/assets/` -> `<dest>/website-assets/`."""
    index_src = website_dir / "index.html"
    assets_src = website_dir / "assets"
    if not index_src.is_file():
        raise FileNotFoundError(f"Missing landing page source: {index_src}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    if assets_src.is_dir():
        _copy_tree_replace(assets_src, dest_dir / "website-assets")
    shutil.copy2(index_src, dest_dir / "index.html")
    redirect_script = repo_root / "scripts" / "write_legacy_redirects.sh"
    if redirect_script.is_file() and os.name != "nt":
        subprocess.run(
            ["bash", str(redirect_script), str(dest_dir)],
            cwd=str(repo_root),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def stage_combined_site(
    dest_dir: Path = DEFAULT_STAGE_DIR,
    *,
    repo_root: Path = REPO_ROOT,
    strict: bool = True,
    skip_mkdocs: bool = False,
) -> Path:
    """Build and stage combined landing page + `/portal/` MkDocs output."""
    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    website_dir = repo_root / "website"

    if not skip_mkdocs:
        with tempfile.TemporaryDirectory(prefix="annolid-docs-build-") as tmp_dir:
            docs_build = Path(tmp_dir) / "site"
            build_mkdocs(docs_build, repo_root=repo_root, strict=strict)

            # Mirror root compatibility routes (preserving landing page + website-assets)
            for item in docs_build.iterdir():
                if item.name in {
                    "index.html",
                    "website-assets",
                    "book",
                    "CNAME",
                    ".git",
                }:
                    continue
                target = dest_dir / item.name
                if item.is_dir():
                    _copy_tree_replace(item, target)
                else:
                    shutil.copy2(item, target)

            # Mirror canonical `/portal/` docs mount
            _copy_tree_replace(docs_build, dest_dir / "portal")

    sync_landing_page(website_dir, dest_dir, repo_root=repo_root)
    (dest_dir / ".nojekyll").touch()
    return dest_dir


def _iter_website_references(website_dir: Path) -> Iterable[tuple[Path, str]]:
    """Yield `(source_file, ref)` pairs referenced by HTML/CSS/JS inside `website/`."""
    index_path = website_dir / "index.html"
    if index_path.is_file():
        text = index_path.read_text(encoding="utf-8")
        for match in _HTML_ATTR_RE.finditer(text):
            yield index_path, match.group(1).strip()

    assets_dir = website_dir / "assets"
    if assets_dir.is_dir():
        for css_file in sorted(assets_dir.rglob("*.css")):
            text = css_file.read_text(encoding="utf-8")
            for match in _CSS_URL_RE.finditer(text):
                yield css_file, match.group(1).strip()


def _resolve_portal_target(portal_root: Path, url_path: str) -> Path | None:
    """Resolve a `/portal/...` URL path against a built MkDocs directory."""
    clean = unquote(urlparse(url_path).path)
    if not clean.startswith("/portal"):
        return None
    rel = clean[len("/portal") :].lstrip("/")
    if not rel:
        candidate = portal_root / "index.html"
        return candidate if candidate.is_file() else None
    direct = portal_root / rel
    if direct.is_file():
        return direct
    if direct.is_dir() and (direct / "index.html").is_file():
        return direct / "index.html"
    if (portal_root / f"{rel.rstrip('/')}" / "index.html").is_file():
        return portal_root / f"{rel.rstrip('/')}" / "index.html"
    return None


def _iter_published_markdown_files(repo_root: Path) -> Iterable[Path]:
    docs_dir = repo_root / "docs"
    excluded_top = {"source", "_build", "examples"}
    excluded_files = {
        "TODO_Qwen3VL_Ollama.md",
        "event_labeling_plan.md",
        "gui_decomposition_plan.md",
        "updating_jupyter_book_and_github_pages.md",
    }
    for md_path in sorted(docs_dir.rglob("*.md")):
        rel = md_path.relative_to(docs_dir)
        if rel.parts and rel.parts[0] in excluded_top:
            continue
        if rel.name in excluded_files:
            continue
        yield md_path


def _find_non_placeholder_machine_paths(text: str) -> list[str]:
    matches: list[str] = []
    for match in _MACHINE_PATH_RE.finditer(text):
        user = (match.group(1) or match.group(2) or match.group(3) or "").lower()
        if user not in _DOC_PLACEHOLDER_USERS:
            matches.append(match.group(0))
    return matches


def _strip_fenced_code(markdown_text: str) -> str:
    """Remove fenced code blocks so regex checks only inspect prose and markup."""
    lines: list[str] = []
    in_fence = False
    for line in markdown_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(line)
    return "\n".join(lines)


def check_site(
    *,
    repo_root: Path = REPO_ROOT,
    max_asset_kb: int = DEFAULT_MAX_ASSET_KB,
    skip_mkdocs: bool = False,
) -> CheckResult:
    """Validate website + docs contracts, asset links, sizes, and path hygiene."""
    result = CheckResult()
    website_dir = repo_root / "website"
    docs_dir = repo_root / "docs"

    # 1. Required source files
    required_files = [
        website_dir / "index.html",
        docs_dir / "index.md",
        docs_dir / "requirements_mkdocs.txt",
        repo_root / "mkdocs.yml",
    ]
    for path in required_files:
        if not path.is_file():
            result.errors.append(
                f"Missing required source file: {path.relative_to(repo_root)}"
            )

    if not result.ok:
        return result

    # 2. Landing page title contract (checked by healthonrails-site-sync.yml)
    index_html = (website_dir / "index.html").read_text(encoding="utf-8")
    if REQUIRED_LANDING_TITLE not in index_html:
        result.errors.append(
            f"website/index.html must contain '{REQUIRED_LANDING_TITLE}' "
            "(required by .github/workflows/healthonrails-site-sync.yml)"
        )

    # 3. Validate website asset refs and collect `/portal/` refs
    portal_refs: list[tuple[Path, str]] = []
    checked_website_refs = 0
    for src_file, ref in _iter_website_references(website_dir):
        if not ref or ref.startswith(
            ("#", "http://", "https://", "mailto:", "data:", "javascript:")
        ):
            continue
        checked_website_refs += 1
        rel_src = src_file.relative_to(repo_root)
        parsed_path = unquote(urlparse(ref).path)
        if parsed_path.startswith("website-assets/") or parsed_path.startswith(
            "/website-assets/"
        ):
            rel_asset = parsed_path.lstrip("/")[len("website-assets/") :]
            target = website_dir / "assets" / rel_asset
            if not target.is_file():
                result.errors.append(
                    f"{rel_src}: broken website-assets reference '{ref}' "
                    f"(expected {target.relative_to(repo_root)})"
                )
        elif parsed_path.startswith("/portal"):
            portal_refs.append((src_file, ref))
        elif parsed_path in {"/", "/book/", "/book"}:
            continue
        elif src_file.suffix.lower() == ".css":
            target = (src_file.parent / parsed_path).resolve()
            if not target.is_file():
                result.errors.append(f"{rel_src}: broken CSS asset reference '{ref}'")

    result.stats["website_refs_checked"] = checked_website_refs

    # 4. Validate local image/media links in published Markdown docs
    checked_md_images = 0
    for md_path in _iter_published_markdown_files(repo_root):
        raw_text = md_path.read_text(encoding="utf-8")
        cleaned = _strip_fenced_code(raw_text)
        rel_md = md_path.relative_to(repo_root)

        for pattern in (_MD_IMAGE_RE, _HTML_IMG_SRC_RE):
            for match in pattern.finditer(cleaned):
                ref = match.group(1).strip().strip("<>")
                if not ref or ref.startswith(("http://", "https://", "data:", "#")):
                    continue
                checked_md_images += 1
                ref_path = unquote(urlparse(ref).path)
                if ref_path.startswith("/"):
                    continue
                target = (md_path.parent / ref_path).resolve()
                if not target.is_file():
                    result.errors.append(
                        f"{rel_md}: missing image/media '{ref}' (resolved to {target})"
                    )

        # Check for leaked machine-local absolute paths
        for leaked in _find_non_placeholder_machine_paths(raw_text):
            result.errors.append(
                f"{rel_md}: machine-local absolute path detected: '{leaked}'"
            )

    result.stats["docs_images_checked"] = checked_md_images

    # 5. Check website + generated HTML examples for machine-local paths
    hygiene_files: list[Path] = [website_dir / "index.html"]
    hygiene_files.extend(sorted((website_dir / "assets").rglob("*.css")))
    hygiene_files.extend(sorted((website_dir / "assets").rglob("*.js")))
    examples_dir = docs_dir / "assets" / "reports"
    if examples_dir.is_dir():
        hygiene_files.extend(sorted(examples_dir.rglob("*.html")))
    for path in hygiene_files:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for leaked in _find_non_placeholder_machine_paths(text):
            result.errors.append(
                f"{path.relative_to(repo_root)}: machine-local absolute path detected: '{leaked}'"
            )

    # 6. Asset size budgets (strict for website + showcase/assets, advisory for legacy docs/imgs)
    strict_roots = [
        docs_dir / "imgs" / "showcase",
        docs_dir / "assets",
        website_dir / "assets",
    ]
    advisory_roots = [docs_dir / "imgs"]
    seen_assets: set[Path] = set()
    asset_count = 0
    for root in strict_roots:
        if not root.is_dir():
            continue
        for asset in sorted(root.rglob("*")):
            if not asset.is_file() or asset.name.startswith("."):
                continue
            resolved = asset.resolve()
            seen_assets.add(resolved)
            asset_count += 1
            size_kb = asset.stat().st_size / 1024.0
            rel_asset = asset.relative_to(repo_root)
            if size_kb > max_asset_kb:
                result.errors.append(
                    f"{rel_asset}: {size_kb:.0f} KB exceeds max allowed {max_asset_kb} KB"
                )
            elif size_kb > WARN_ASSET_KB:
                result.warnings.append(
                    f"{rel_asset}: {size_kb:.0f} KB (consider optimizing under {WARN_ASSET_KB} KB)"
                )
    for root in advisory_roots:
        if not root.is_dir():
            continue
        for asset in sorted(root.rglob("*")):
            if not asset.is_file() or asset.name.startswith("."):
                continue
            resolved = asset.resolve()
            if resolved in seen_assets:
                continue
            asset_count += 1
            size_kb = asset.stat().st_size / 1024.0
            rel_asset = asset.relative_to(repo_root)
            if size_kb > max_asset_kb:
                result.warnings.append(
                    f"{rel_asset}: legacy asset is {size_kb:.0f} KB (above {max_asset_kb} KB target)"
                )
    result.stats["assets_checked"] = asset_count

    # 7. Strict MkDocs build + verify website `/portal/` links resolve in built output
    if not skip_mkdocs:
        with tempfile.TemporaryDirectory(prefix="annolid-site-check-") as tmp_dir:
            built_portal = Path(tmp_dir) / "site"
            try:
                build_mkdocs(built_portal, repo_root=repo_root, strict=True)
            except RuntimeError as exc:
                result.errors.append(str(exc))
                return result

            for src_file, ref in portal_refs:
                if _resolve_portal_target(built_portal, ref) is None:
                    result.errors.append(
                        f"{src_file.relative_to(repo_root)}: '/portal/' link '{ref}' "
                        "does not match any built MkDocs page or asset"
                    )
            result.stats["portal_links_verified"] = len(portal_refs)

    return result


def _snapshot_mtimes(paths: Sequence[Path]) -> dict[str, float]:
    state: dict[str, float] = {}
    for root in paths:
        if root.is_file():
            state[str(root)] = root.stat().st_mtime
        elif root.is_dir():
            for child in root.rglob("*"):
                if child.is_file() and not child.name.startswith("."):
                    try:
                        state[str(child)] = child.stat().st_mtime
                    except OSError:
                        pass
    return state


def _watch_loop(
    dest_dir: Path,
    *,
    repo_root: Path,
    stop_event: threading.Event,
    interval: float = 1.0,
) -> None:
    website_dir = repo_root / "website"
    docs_paths = [repo_root / "docs", repo_root / "mkdocs.yml"]
    prev_website = _snapshot_mtimes([website_dir])
    prev_docs = _snapshot_mtimes(docs_paths)

    while not stop_event.wait(interval):
        cur_website = _snapshot_mtimes([website_dir])
        cur_docs = _snapshot_mtimes(docs_paths)

        if cur_docs != prev_docs:
            prev_docs = cur_docs
            prev_website = cur_website
            print(
                "[watch] Docs changed — rebuilding MkDocs and re-staging...", flush=True
            )
            try:
                stage_combined_site(dest_dir, repo_root=repo_root, strict=False)
                print("[watch] Re-staged combined site.", flush=True)
            except Exception as exc:
                print(f"[watch] Rebuild error: {exc}", file=sys.stderr, flush=True)
        elif cur_website != prev_website:
            prev_website = cur_website
            print("[watch] Website changed — syncing landing page...", flush=True)
            try:
                sync_landing_page(website_dir, dest_dir, repo_root=repo_root)
                print("[watch] Synced website/ in place.", flush=True)
            except Exception as exc:
                print(f"[watch] Sync error: {exc}", file=sys.stderr, flush=True)


class _NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


def serve_preview(
    dest_dir: Path = DEFAULT_STAGE_DIR,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    watch: bool = False,
    open_browser: bool = False,
    no_stage: bool = False,
    repo_root: Path = REPO_ROOT,
) -> None:
    """Stage (unless `--no-stage`) and serve the combined Annolid site locally."""
    if not no_stage:
        print(f"Staging combined Annolid site into {dest_dir}...", flush=True)
        stage_combined_site(dest_dir, repo_root=repo_root, strict=True)

    handler = functools.partial(_NoCacheHTTPRequestHandler, directory=str(dest_dir))
    server = http.server.ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{server.server_port}/"

    print(f"Serving combined Annolid site at {url}", flush=True)
    print(f"  Landing page : {url}", flush=True)
    print(f"  Docs portal  : {url}portal/", flush=True)
    print(
        f"  mouse083 demo: {url}portal/tutorials/mouse083_resident_intruder_showcase/",
        flush=True,
    )
    if watch:
        print(
            "Watching website/, docs/, and mkdocs.yml for live updates (Ctrl+C to stop).",
            flush=True,
        )
    else:
        print("Press Ctrl+C to stop.", flush=True)

    stop_event = threading.Event()
    watcher: threading.Thread | None = None
    if watch:
        watcher = threading.Thread(
            target=_watch_loop,
            args=(dest_dir,),
            kwargs={"repo_root": repo_root, "stop_event": stop_event},
            daemon=True,
        )
        watcher.start()

    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping preview server.")
    finally:
        stop_event.set()
        server.shutdown()
        server.server_close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage, preview, watch, and validate Annolid's combined website and MkDocs portal."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_stage = sub.add_parser(
        "stage", help="Build MkDocs and stage combined website + portal output."
    )
    p_stage.add_argument(
        "--dest", type=Path, default=DEFAULT_STAGE_DIR, help="Output staging directory."
    )
    p_stage.add_argument(
        "--no-strict", action="store_true", help="Run MkDocs without --strict."
    )
    p_stage.add_argument(
        "--skip-mkdocs", action="store_true", help="Only sync website/ into dest."
    )

    p_preview = sub.add_parser(
        "preview", help="Stage and serve combined website + portal locally."
    )
    p_preview.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_STAGE_DIR,
        help="Staging directory to serve.",
    )
    p_preview.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)."
    )
    p_preview.add_argument(
        "--port", type=int, default=8080, help="Bind port (default: 8080)."
    )
    p_preview.add_argument(
        "--watch",
        action="store_true",
        help="Watch website/ and docs/ for auto-updates.",
    )
    p_preview.add_argument(
        "--open", action="store_true", help="Open preview URL in default browser."
    )
    p_preview.add_argument(
        "--no-stage", action="store_true", help="Serve existing staged directory as-is."
    )

    p_check = sub.add_parser(
        "check",
        help="Validate website + docs contracts, links, sizes, and MkDocs strict build.",
    )
    p_check.add_argument(
        "--max-asset-kb",
        type=int,
        default=DEFAULT_MAX_ASSET_KB,
        help=f"Maximum allowed single asset size in KB (default: {DEFAULT_MAX_ASSET_KB}).",
    )
    p_check.add_argument(
        "--skip-mkdocs",
        action="store_true",
        help="Skip MkDocs build (fast static checks only).",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "stage":
        dest = stage_combined_site(
            args.dest,
            strict=not args.no_strict,
            skip_mkdocs=args.skip_mkdocs,
        )
        print(f"Staged combined Annolid site at: {dest}")
        return 0

    if args.command == "preview":
        serve_preview(
            args.dest,
            host=args.host,
            port=args.port,
            watch=args.watch,
            open_browser=args.open,
            no_stage=args.no_stage,
        )
        return 0

    if args.command == "check":
        res = check_site(
            max_asset_kb=args.max_asset_kb,
            skip_mkdocs=args.skip_mkdocs,
        )
        for warning in res.warnings:
            print(f"WARNING: {warning}")
        for error in res.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        stats_text = ", ".join(f"{k}={v}" for k, v in sorted(res.stats.items()))
        if res.ok:
            print(f"Site & docs check PASSED ({stats_text}).")
            return 0
        print(
            f"Site & docs check FAILED with {len(res.errors)} error(s).",
            file=sys.stderr,
        )
        return 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
