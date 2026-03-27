from __future__ import annotations

from pathlib import Path

import annolid.gui.widgets.video_frame_preview as preview_mod


def test_temporary_first_frame_image_cleans_up(tmp_path: Path, monkeypatch) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake")
    temp_paths: list[str] = []

    class _FakeCapture:
        def read(self):
            return True, object()

        def release(self):
            return None

    def _fake_imwrite(path, _frame):
        temp_paths.append(path)
        Path(path).write_bytes(b"frame")
        return True

    monkeypatch.setattr(preview_mod.cv2, "VideoCapture", lambda _p: _FakeCapture())
    monkeypatch.setattr(preview_mod.cv2, "imwrite", _fake_imwrite)

    with preview_mod.temporary_first_frame_image(str(video_path)) as image_path:
        assert Path(image_path).exists()
        assert image_path in temp_paths

    assert not Path(image_path).exists()
