from __future__ import annotations

from pathlib import Path


class WorkspacePathError(ValueError):
    """Raised when a requested agent workspace violates the configured root."""


def resolve_workspace_dir(
    workspace: str | Path | None,
    *,
    root: str | Path | None,
) -> Path:
    """Resolve an agent workspace directory under the configured workspace root."""
    root_path = Path(root or Path.cwd()).expanduser().resolve()
    raw = str(workspace or "").strip()
    candidate = root_path if not raw else Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root_path / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise WorkspacePathError(
            f"Workspace {resolved} is outside configured root {root_path}."
        ) from exc
    if not resolved.exists():
        raise WorkspacePathError(f"Workspace does not exist: {resolved}")
    if not resolved.is_dir():
        raise WorkspacePathError(f"Workspace is not a directory: {resolved}")
    return resolved


__all__ = ["WorkspacePathError", "resolve_workspace_dir"]
