from __future__ import annotations

import os

from qtpy import QtWidgets

from annolid.domain.memory.models import MemoryHit
from annolid.gui.widgets import memory_manager_dialog as mm


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


def test_memory_manager_refresh_populates_table(monkeypatch) -> None:
    _ensure_qapp()

    class DummyRetrieval:
        def search_memory(self, **kwargs):
            _ = kwargs
            return [
                MemoryHit(
                    id="m1",
                    text="Use tail_base",
                    score=0.9,
                    scope="dataset:1",
                    category="annotation_rule",
                    source="annotation",
                    importance=0.8,
                    timestamp_ms=0,
                    tags=["pose"],
                    metadata={},
                )
            ]

    monkeypatch.setattr(mm, "get_retrieval_service", lambda: DummyRetrieval())
    monkeypatch.setattr(mm, "get_memory_service", lambda: None)

    dialog = mm.MemoryManagerDialog()
    try:
        assert dialog.table.rowCount() == 1
        assert dialog.table.item(0, 0).text() == "m1"
        assert "Loaded 1 memories" in dialog.status_label.text()
    finally:
        dialog.close()


def test_memory_manager_add_edit_delete(monkeypatch) -> None:
    _ensure_qapp()
    state = {
        "records": [
            {
                "id": "m1",
                "text": "Initial note",
                "scope": "project:1",
                "category": "project_note",
                "source": "project",
                "importance": 0.5,
                "tags": [],
                "metadata": {},
            }
        ]
    }

    class DummyRetrieval:
        def search_memory(self, **kwargs):
            _ = kwargs
            rows = []
            for row in state["records"]:
                rows.append(
                    MemoryHit(
                        id=row["id"],
                        text=row["text"],
                        score=row["importance"],
                        scope=row["scope"],
                        category=row["category"],
                        source=row["source"],
                        importance=row["importance"],
                        timestamp_ms=0,
                        tags=row["tags"],
                        metadata=row["metadata"],
                    )
                )
            return rows

    class DummyService:
        def store_memory(self, **payload):
            state["records"].append(
                {
                    "id": "m2",
                    "text": payload["text"],
                    "scope": payload["scope"],
                    "category": payload["category"],
                    "source": payload["source"],
                    "importance": payload["importance"],
                    "tags": payload["tags"],
                    "metadata": payload["metadata"],
                }
            )
            return "m2"

        def update_memory(self, memory_id, patch):
            for row in state["records"]:
                if row["id"] == memory_id:
                    row.update(patch)
                    return True
            return False

        def delete_memory(self, memory_id):
            before = len(state["records"])
            state["records"] = [
                row for row in state["records"] if row["id"] != memory_id
            ]
            return len(state["records"]) < before

    class FakeEditor:
        _queue = []

        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)
            self._payload = FakeEditor._queue.pop(0)

        def exec_(self):
            return QtWidgets.QDialog.Accepted

        def payload(self):
            return self._payload

    monkeypatch.setattr(mm, "get_retrieval_service", lambda: DummyRetrieval())
    monkeypatch.setattr(mm, "get_memory_service", lambda: DummyService())
    monkeypatch.setattr(mm, "_MemoryRecordEditorDialog", FakeEditor)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.Yes,
    )

    dialog = mm.MemoryManagerDialog()
    try:
        # Add
        FakeEditor._queue.append(
            {
                "text": "Added note",
                "scope": "project:1",
                "category": "project_note",
                "source": "project",
                "importance": 0.6,
                "tags": [],
                "metadata": {},
            }
        )
        dialog.add_record()
        assert any(r["id"] == "m2" for r in state["records"])

        # Edit selected row
        dialog.table.selectRow(0)
        FakeEditor._queue.append(
            {
                "text": "Edited note",
                "scope": "project:1",
                "category": "project_note",
                "source": "project",
                "importance": 0.9,
                "tags": ["edited"],
                "metadata": {"a": 1},
            }
        )
        dialog.edit_selected_record()
        assert state["records"][0]["text"] == "Edited note"
        assert state["records"][0]["metadata"]["a"] == 1

        # Delete selected row
        dialog.table.selectRow(0)
        dialog.delete_selected_record()
        assert all(r["id"] != "m1" for r in state["records"])
    finally:
        dialog.close()
