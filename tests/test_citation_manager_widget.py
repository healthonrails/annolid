from __future__ import annotations

from pathlib import Path

from qtpy import QtWidgets

from annolid.gui.widgets.citation_manager_widget import CitationManagerDialog
from annolid.utils.citations import BibEntry, save_bibtex


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_citation_manager_save_from_context_forwards_verify_flag(
    tmp_path: Path,
) -> None:
    _app()
    captured: dict[str, object] = {}

    def _save_from_context(**kwargs):
        captured.update(kwargs)
        return {
            "ok": True,
            "created": True,
            "key": "annolid2024",
            "validation": {
                "checked": True,
                "verified": True,
                "provider": "crossref",
                "score": 0.91,
            },
            "verification": {"status": "verified", "integrity_score": 0.91},
            "verification_report": str(
                tmp_path / ".annolid_cache/citation_verification/refs_annolid2024.json"
            ),
        }

    dialog = CitationManagerDialog(
        default_bib_path_getter=lambda: tmp_path / "refs.bib",
        save_from_context=_save_from_context,
    )
    dialog.verify_after_save_checkbox.setChecked(True)
    dialog._save_from_active_context("pdf")
    assert captured["verify_after_save"] is True
    assert "verify: verified" in dialog.status_label.text().lower()
    dialog.close()


def test_citation_manager_verify_bib_generates_batch_report(
    monkeypatch, tmp_path: Path
) -> None:
    _app()
    bib_path = tmp_path / "refs.bib"
    save_bibtex(
        bib_path,
        [
            BibEntry(
                entry_type="article",
                key="annolid2024",
                fields={"title": "Annolid Toolkit", "year": "2024"},
            ),
            BibEntry(
                entry_type="article",
                key="other2020",
                fields={"title": "Other Work", "year": "2020"},
            ),
        ],
    )

    def _fake_validate(fields, timeout_s=1.8):
        title = str(fields.get("title") or "").lower()
        if "annolid" in title:
            return {
                "checked": True,
                "verified": True,
                "provider": "crossref",
                "score": 0.95,
                "message": "strong match",
                "candidate": {"title": str(fields.get("title") or "")},
            }
        return {
            "checked": True,
            "verified": False,
            "provider": "crossref",
            "score": 0.1,
            "message": "no match",
            "candidate": {},
        }

    import annolid.gui.widgets.citation_manager_widget as widget_mod

    monkeypatch.setattr(widget_mod, "validate_citation_metadata", _fake_validate)

    dialog = CitationManagerDialog(
        default_bib_path_getter=lambda: bib_path,
        save_from_context=lambda **kwargs: {"ok": False, "error": "unused"},
    )
    dialog._verify_bib_file()
    report_path = (
        tmp_path / ".annolid_cache" / "citation_verification" / "refs_batch.json"
    )
    assert report_path.exists()
    status = dialog.status_label.text().lower()
    assert "verified 2 citation(s)" in status
    assert "report: refs_batch.json" in status
    dialog.close()
