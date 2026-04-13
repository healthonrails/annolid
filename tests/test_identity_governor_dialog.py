from __future__ import annotations

import os
from pathlib import Path

import pytest
from qtpy import QtWidgets

from annolid.postprocessing import IdentityCorrection, IdentityGovernorResult
from annolid.gui.widgets.identity_governor_dialog import IdentityGovernorDialog


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


def test_identity_governor_dialog_rejects_invalid_policy_json() -> None:
    _ensure_qapp()
    dialog = IdentityGovernorDialog(initial_annotation_dir=Path.cwd())
    dialog.policy_edit.setPlainText("{invalid-json")
    with pytest.raises(ValueError):
        dialog._parse_policy()


def test_identity_governor_dialog_preview_populates_results(
    tmp_path, monkeypatch
) -> None:
    _ensure_qapp()
    annotation_dir = tmp_path / "session"
    annotation_dir.mkdir()
    (annotation_dir / "session_000000000.json").write_text(
        '{"shapes":[{"label":"alpha","instance_label":"alpha","track_id":"1","points":[[0,0],[1,0],[1,1]]}]}',
        encoding="utf-8",
    )
    report_path = annotation_dir / "identity_governor_report.json"

    dialog = IdentityGovernorDialog(initial_annotation_dir=annotation_dir)
    dialog.annotation_dir_edit.setText(str(annotation_dir))
    dialog.report_path_edit.setText(str(report_path))
    dialog._insert_selected_template()

    seen: dict[str, object] = {}

    def _fake_run_identity_governor(**kwargs):
        seen.update(kwargs)
        return IdentityGovernorResult(
            annotation_dir=annotation_dir,
            dry_run=True,
            scanned_files=6,
            scanned_observations=24,
            proposed_corrections=(
                IdentityCorrection(
                    track_id="2",
                    frame_start=10,
                    frame_end=30,
                    observed_label="beta",
                    corrected_label="alpha",
                    rule_name="alpha_when_right",
                    rule_frame_start=20,
                    rule_frame_end=30,
                    observation_count=21,
                ),
            ),
            updated_files=0,
            updated_shapes=0,
            report_path=report_path,
        )

    monkeypatch.setattr(
        "annolid.gui.widgets.identity_governor_dialog.run_identity_governor",
        _fake_run_identity_governor,
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.Ok,
    )

    dialog._run(apply_changes=False)

    assert seen["annotation_dir"] == annotation_dir
    assert seen["apply_changes"] is False
    assert dialog.corrections_table.rowCount() == 1
    assert "Corrections: 1" in dialog.summary_label.text()
    assert dialog.open_report_btn.isEnabled() is True


def test_identity_governor_dialog_snippets_insert_expected_payload() -> None:
    _ensure_qapp()
    dialog = IdentityGovernorDialog(initial_annotation_dir=Path.cwd())
    assert dialog.template_combo.count() >= 4

    dialog.template_combo.setCurrentIndex(2)  # 3-vole social assay
    dialog._insert_selected_template()
    payload = dialog._parse_policy()
    assert "rules" in payload
    assert len(payload["rules"]) >= 3
    assert "canonical_track_ids" in payload
    assert set(payload["canonical_track_ids"].keys()) >= {"rover", "stim_a", "stim_b"}

    dialog.template_combo.setCurrentIndex(3)  # Distance-only fallback
    dialog._insert_selected_template()
    payload = dialog._parse_policy()
    assert len(payload.get("rules", [])) == 2
