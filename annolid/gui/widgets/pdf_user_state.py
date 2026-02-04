from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from qtpy import QtCore

from annolid.utils.logger import logger


STATE_VERSION = 1


def pdf_state_key(pdf_path: Path) -> str:
    try:
        resolved = str(pdf_path.expanduser().resolve())
    except Exception:
        resolved = str(pdf_path)
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()
    return digest[:32]


def pdf_state_dir() -> Path:
    root = ""
    try:
        root = str(
            QtCore.QStandardPaths.writableLocation(
                QtCore.QStandardPaths.AppDataLocation
            )
        )
    except Exception:
        root = ""
    if not root:
        root = str(Path.home() / ".annolid")
    base = Path(root) / "pdf_states"
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return base


def pdf_state_path(pdf_path: Path) -> Path:
    return pdf_state_dir() / f"{pdf_state_key(pdf_path)}.json"


def load_pdf_state(pdf_path: Path) -> Dict[str, Any]:
    path = pdf_state_path(pdf_path)
    if not path.exists():
        return {}
    try:
        data = path.read_text(encoding="utf-8")
        obj = json.loads(data)
        if isinstance(obj, dict):
            return obj
    except Exception as exc:
        logger.debug("Failed to load PDF state %s: %s", path, exc)
    return {}


def save_pdf_state(pdf_path: Path, state: Dict[str, Any]) -> None:
    path = pdf_state_path(pdf_path)
    payload = dict(state or {})
    payload.setdefault("version", STATE_VERSION)
    payload.setdefault("updatedAt", time.time())
    try:
        payload.setdefault("pdf", {})
        pdf_meta = payload.get("pdf")
        if isinstance(pdf_meta, dict):
            try:
                stat = pdf_path.stat()
                pdf_meta.setdefault("path", str(pdf_path))
                pdf_meta.setdefault("size", int(stat.st_size))
                pdf_meta.setdefault("mtime", float(stat.st_mtime))
            except Exception:
                pdf_meta.setdefault("path", str(pdf_path))
    except Exception:
        pass

    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(str(tmp), str(path))
    except Exception as exc:
        logger.debug("Failed to save PDF state %s: %s", path, exc)


def delete_pdf_state(pdf_path: Path) -> None:
    path = pdf_state_path(pdf_path)
    try:
        if path.exists():
            path.unlink()
    except Exception as exc:
        logger.debug("Failed to delete PDF state %s: %s", path, exc)
