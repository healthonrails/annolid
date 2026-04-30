"""Shared command-resolution helpers for agent tools.

On macOS, node version managers (nvm, volta, fnm) install binaries in
per-user directories that are **not** on the PATH inherited by
non-interactive Python processes (e.g. the annolid GUI or scheduled runs).

``resolve_command`` probes those locations so tools that depend on ``node``,
``npx`` and related node tools still work regardless of how they were installed.
"""

from __future__ import annotations

import glob
import os
import shutil
from functools import lru_cache
from typing import Optional


# Well-known binary search directories, checked in order.
# Newest-version-first sorting is applied for glob-pattern paths.
_EXTRA_BIN_DIRS: list[str] = []


def _probe_patterns(command: str) -> list[str]:
    """Return glob patterns for well-known node-manager locations."""
    home = os.path.expanduser("~")
    return [
        # nvm (most common on macOS)
        os.path.join(home, ".nvm", "versions", "node", "*", "bin", command),
        # volta
        os.path.join(home, ".volta", "bin", command),
        # fnm
        os.path.join(
            home, ".fnm", "node-versions", "*", "installation", "bin", command
        ),
        # Homebrew Apple-Silicon and Intel
        os.path.join("/opt", "homebrew", "bin", command),
        # System-wide fallback
        os.path.join("/usr", "local", "bin", command),
    ]


@lru_cache(maxsize=64)
def resolve_command(command: str) -> Optional[str]:
    """Resolve *command* to an absolute path.

    1. Try the standard ``shutil.which`` first (fast path for normal PATHs).
    2. Probe common Node.js version-manager directories.
    3. Return ``None`` if the command cannot be found anywhere.

    Results are cached for the lifetime of the process so the filesystem
    is probed at most once per unique command name.
    """
    # Fast path: already on PATH
    found = shutil.which(command)
    if found:
        return found

    # Probe well-known locations
    for pattern in _probe_patterns(command):
        matches = sorted(glob.glob(pattern), reverse=True)  # newest first
        for match in matches:
            if os.path.isfile(match) and os.access(match, os.X_OK):
                return match

    return None


def ensure_node_env() -> dict[str, str]:
    """Return a minimal env dict that includes the resolved node/npm bin dir.

    If ``node`` is found via probing, its parent directory is prepended to
    ``PATH`` so that child processes (e.g. ``npx``) can also find
    sibling binaries like ``npm``.

    Returns the current ``os.environ`` with the augmented PATH, or the
    current environment unmodified if node is already on PATH.
    """
    env = dict(os.environ)
    node_path = resolve_command("node")
    if node_path is None:
        return env

    node_bin_dir = os.path.dirname(node_path)
    current_path = env.get("PATH", "")

    if node_bin_dir not in current_path.split(os.pathsep):
        env["PATH"] = node_bin_dir + os.pathsep + current_path

    return env


__all__ = ["resolve_command", "ensure_node_env"]
