from __future__ import annotations

from qtpy.QtWidgets import QApplication

from annolid.gui.widgets.video_processing_settings_widget import (
    VideoProcessingSettingsWidget,
)


def test_video_processing_settings_widget_round_trip() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    widget = VideoProcessingSettingsWidget()
    widget.apply_settings(
        {
            "scale_factor": 0.25,
            "fps": 15.0,
            "apply_denoise": True,
            "auto_contrast": True,
            "auto_contrast_strength": 1.25,
            "crop_params": (10, 20, 300, 200),
        }
    )

    settings = widget.collect_settings()

    assert settings == {
        "scale_factor": 0.25,
        "fps": 15.0,
        "apply_denoise": True,
        "auto_contrast": True,
        "auto_contrast_strength": 1.25,
        "crop_params": (10, 20, 300, 200),
    }


def test_video_processing_settings_widget_rejects_invalid_scale() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    widget = VideoProcessingSettingsWidget()
    widget.scale_factor_text.setText("invalid")

    assert widget.collect_settings() is None


def test_video_processing_settings_widget_does_not_popup_on_invalid_scale(
    monkeypatch,
) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    widget = VideoProcessingSettingsWidget()
    widget.scale_factor_text.setText("invalid")

    monkeypatch.setattr(
        "annolid.gui.widgets.video_processing_settings_widget.QMessageBox.warning",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("scale validation should not open a popup")
        ),
    )

    assert widget.collect_settings() is None


def test_video_processing_settings_widget_disables_preview_without_input() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    widget = VideoProcessingSettingsWidget()

    assert widget.crop_checkbox.isChecked() is False
    assert widget.crop_preview_button.isEnabled() is False
    assert widget.crop_x_text.isEnabled() is False


def test_video_processing_settings_widget_enables_preview_when_input_available() -> (
    None
):
    app = QApplication.instance() or QApplication([])
    _ = app
    widget = VideoProcessingSettingsWidget()

    widget.set_crop_preview_available(True)

    assert widget.crop_preview_button.isEnabled() is True
    assert widget.crop_preview_button.toolTip() == ""
