from __future__ import annotations

from pathlib import Path

from qtpy import QtCore

import annolid.gui.widgets.video_rescale_workflow as workflow_mod


class _DummyCropDialog:
    def __init__(self, *_args, **_kwargs) -> None:
        self._accepted = True

    def exec_(self):
        return workflow_mod.QDialog.Accepted

    def getCropCoordinates(self):
        return (10, 20, 100, 80)


class _DummyDialog(QtCore.QObject):
    def __init__(self) -> None:
        super().__init__()
        self.input_video_path = ""
        self.input_folder_path = ""
        self.input_selection_label = _Label()
        self.fps_text = _Text()
        self.crop_x_text = _Text()
        self.crop_y_text = _Text()
        self.crop_width_text = _Text()
        self.crop_height_text = _Text()
        self.crop_section_label = _Label()
        self.crop_preview_button = _Button()
        self.crop_checkbox = type(
            "_Check",
            (),
            {
                "setChecked": lambda self, _value: setattr(self, "checked", _value),
                "setStyleSheet": lambda self, _style: setattr(self, "style", _style),
            },
        )()
        self.crop_checkbox.checked = False


def test_preview_and_crop_enables_crop_checkbox(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    dialog = _DummyDialog()
    dialog.input_video_path = str(video)

    monkeypatch.setattr(
        workflow_mod, "temporary_first_frame_image", lambda _p: _NoopContext()
    )
    monkeypatch.setattr(workflow_mod, "CropDialog", _DummyCropDialog)
    monkeypatch.setattr(
        workflow_mod.QMessageBox, "information", lambda *args, **kwargs: None
    )

    workflow = workflow_mod.VideoRescaleWorkflow(dialog)
    workflow.preview_and_crop()

    assert dialog.crop_checkbox.checked is True
    assert "2f855a" in dialog.crop_section_label.style
    assert "2f855a" in dialog.crop_preview_button.style


class _Label:
    def __init__(self) -> None:
        self.text = ""
        self.style = ""

    def setText(self, text):
        self.text = text

    def setStyleSheet(self, style):
        self.style = style


class _Text(_Label):
    pass


class _Button(_Label):
    def setStyleSheet(self, style):
        self.style = style


class _NoopContext:
    def __enter__(self):
        return "frame.png"

    def __exit__(self, exc_type, exc, tb):
        return False
