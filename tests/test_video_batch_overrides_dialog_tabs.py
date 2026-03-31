from __future__ import annotations

from pathlib import Path

from qtpy.QtWidgets import QApplication

from annolid.gui.widgets.video_batch_overrides_dialog import (
    VideoBatchReviewDialog,
)


def _default_settings() -> dict[str, object]:
    return {
        "scale_factor": 0.5,
        "fps": None,
        "apply_denoise": False,
        "auto_contrast": False,
        "auto_contrast_strength": 1.0,
        "crop_params": None,
    }


def test_video_batch_review_dialog_uses_linear_flow() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    dialog = VideoBatchReviewDialog(
        video_paths=["/tmp/a.mp4", "/tmp/b.mp4"],
        default_settings=_default_settings(),
    )

    assert not hasattr(dialog, "tabs")
    assert dialog.progress_label.text() == "Video 1 of 2"
    assert dialog.save_next_button.text() == "Save & Next"
    assert dialog.skip_button.text() == "Skip"
    assert dialog.previous_button.isEnabled() is False


def test_video_batch_review_dialog_saves_then_advances(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    first = tmp_path / "a.mp4"
    second = tmp_path / "b.mp4"
    first.write_bytes(b"fake")
    second.write_bytes(b"fake")

    dialog = VideoBatchReviewDialog(
        video_paths=[str(first), str(second)],
        default_settings=_default_settings(),
    )
    dialog.scale_factor_text.setText("0.25")

    dialog._save_current_video_and_advance()

    overrides = dialog.overrides()
    assert str(first) in overrides
    assert overrides[str(first)]["scale_factor"] == 0.25
    assert dialog.progress_label.text() == "Video 2 of 2"
    assert dialog.save_next_button.text() == "Save & Finish"


def test_video_batch_review_dialog_skip_uses_folder_defaults(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    first = tmp_path / "a.mp4"
    second = tmp_path / "b.mp4"
    first.write_bytes(b"fake")
    second.write_bytes(b"fake")

    dialog = VideoBatchReviewDialog(
        video_paths=[str(first), str(second)],
        default_settings=_default_settings(),
        existing_overrides={str(first): {**_default_settings(), "scale_factor": 0.25}},
    )

    dialog._skip_current_video()

    overrides = dialog.overrides()
    assert str(first) not in overrides
    assert dialog.progress_label.text() == "Video 2 of 2"
    assert dialog.current_status_label.text().startswith("using folder defaults")
