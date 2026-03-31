from __future__ import annotations

from pathlib import Path

from qtpy.QtWidgets import QApplication

from annolid.gui.widgets.downsample_videos_dialog import VideoRescaleWidget


def test_downsample_dialog_uses_tabs() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    widget = VideoRescaleWidget()

    assert widget.tabs.count() == 4
    assert widget.tabs.tabText(0) == "Input / Output"
    assert widget.tabs.tabText(1) == "Processing"
    assert widget.tabs.tabText(2) == "Summary"
    assert widget.tabs.tabText(3) == "Run"


def test_downsample_dialog_summary_tab_exists() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    widget = VideoRescaleWidget()

    assert widget.summary_header_label.text().startswith("Review the current batch")
    assert "Input: none" in widget.summary_input_label.text()
    assert "Per-video review" in widget.summary_overrides_label.text()


def test_downsample_dialog_enables_review_and_crop_preview_after_input_selection(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    widget = VideoRescaleWidget()

    assert widget.crop_preview_button.isEnabled() is False
    assert widget.per_video_review_button.isEnabled() is False

    widget.input_video_path = str(video)
    widget.input_folder_path = ""
    widget.workflow.update_input_selection_label()

    assert widget.crop_preview_button.isEnabled() is True
    assert widget.per_video_review_button.isEnabled() is False


def test_downsample_dialog_enables_review_for_folder_input(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    folder = tmp_path / "videos"
    folder.mkdir()
    (folder / "clip.mp4").write_bytes(b"fake")

    widget = VideoRescaleWidget()
    widget.input_folder_path = str(folder)
    widget.input_video_path = ""
    widget.workflow.update_input_selection_label()

    assert widget.per_video_review_button.isEnabled() is True
    assert "Per-video review" in widget.per_video_review_label.text()


def test_downsample_dialog_summary_shows_derived_output_folder(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    folder = tmp_path / "videos"
    folder.mkdir()
    (folder / "clip.mp4").write_bytes(b"fake")

    widget = VideoRescaleWidget()
    widget.input_folder_path = str(folder)
    widget.input_video_path = ""
    widget.workflow.update_input_selection_label()
    widget.workflow.update_summary_tab()

    assert str(tmp_path / "videos_downsampled") in widget.summary_output_label.text()
