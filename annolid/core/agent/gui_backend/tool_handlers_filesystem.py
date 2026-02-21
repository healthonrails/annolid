from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict


def rename_file_tool(
    *,
    source_path: str,
    new_name: str,
    new_path: str,
    use_active_file: bool,
    overwrite: bool,
    get_pdf_state: Callable[[], Dict[str, Any]],
    get_active_video_path: Callable[[], str],
    workspace: Path,
    run_rename: Callable[[str, str, str, bool], str],
    reopen_pdf: Callable[[Path], bool],
) -> Dict[str, Any]:
    def _resolve_user_path(path_text: str) -> Path:
        candidate = Path(path_text).expanduser()
        if not candidate.is_absolute():
            candidate = (workspace / candidate).expanduser()
        return candidate

    source_text = str(source_path or "").strip()
    target_name = str(new_name or "").strip()
    target_path = str(new_path or "").strip()
    if not target_name and not target_path:
        return {"ok": False, "error": "Provide a new_name or new_path for rename."}

    current_path: Path | None = None
    if source_text:
        current_path = _resolve_user_path(source_text)
    elif bool(use_active_file):
        pdf_state = get_pdf_state()
        if isinstance(pdf_state, dict) and bool(pdf_state.get("ok")):
            active_pdf_path = str(pdf_state.get("path") or "").strip()
            if active_pdf_path:
                current_path = Path(active_pdf_path).expanduser()
        if current_path is None:
            active_video = str(get_active_video_path() or "").strip()
            if active_video:
                current_path = Path(active_video).expanduser()

    if current_path is None:
        return {
            "ok": False,
            "error": "No source file provided and no active file found.",
        }
    if not current_path.exists() or not current_path.is_file():
        return {"ok": False, "error": f"Source file is missing: {current_path}"}

    if target_name and not Path(target_name).suffix and current_path.suffix:
        target_name = f"{target_name}{current_path.suffix}"

    result_text = str(
        run_rename(
            str(current_path),
            target_name,
            target_path,
            bool(overwrite),
        )
        or ""
    )
    if not result_text.startswith("Successfully renamed "):
        return {
            "ok": False,
            "error": result_text or "Rename failed.",
            "path": str(current_path),
            "requested_new_name": target_name,
            "requested_new_path": target_path,
        }

    if target_path:
        resolved_new_path = _resolve_user_path(target_path)
    else:
        resolved_new_path = current_path.with_name(target_name)
    reopened = False
    if resolved_new_path.exists() and resolved_new_path.suffix.lower() == ".pdf":
        reopened = bool(reopen_pdf(resolved_new_path))

    return {
        "ok": True,
        "renamed": True,
        "old_path": str(current_path),
        "new_path": str(resolved_new_path),
        "reopened": reopened,
    }
