from __future__ import annotations

from pathlib import Path

import annolid.utils.video_processing_reports as reports_mod


def test_save_processing_summary_for_single_video_path(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "clip.mp4").write_bytes(b"fake")
    (output_dir / "ignore.txt").write_text("skip", encoding="utf-8")

    class _FakeCapture:
        def isOpened(self):
            return True

        def get(self, prop):
            mapping = {
                reports_mod.cv2.CAP_PROP_FRAME_WIDTH: 1920,
                reports_mod.cv2.CAP_PROP_FRAME_HEIGHT: 1080,
                reports_mod.cv2.CAP_PROP_FPS: 30.0,
                reports_mod.cv2.CAP_PROP_FRAME_COUNT: 300,
                reports_mod.cv2.CAP_PROP_FOURCC: 1234,
            }
            return mapping.get(prop, 0)

        def release(self):
            return None

    monkeypatch.setattr(reports_mod.cv2, "VideoCapture", lambda _p: _FakeCapture())

    reports_mod.save_processing_summary(
        str(output_dir),
        video_paths=[str(output_dir / "clip.mp4")],
        scale_factor=0.5,
        fps=15.0,
        apply_denoise=False,
        auto_contrast=True,
        auto_contrast_strength=1.25,
        crop_params=(10, 20, 300, 200),
        command_log={"clip.mp4": "ffmpeg -y -i clip.mp4"},
    )

    csv_path = output_dir / "metadata.csv"
    md_path = output_dir / "clip.md"
    assert csv_path.exists()
    assert md_path.exists()
    assert not (output_dir / "ignore.md").exists()

    csv_text = csv_path.read_text(encoding="utf-8")
    md_text = md_path.read_text(encoding="utf-8")
    assert "clip.mp4" in csv_text
    assert "Scale Factor: 0.5" in md_text
    assert "FFmpeg Command" in md_text
