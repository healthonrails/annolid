"""Tests for VideoFFmpegProcessTool (annolid.core.agent.tools.function_ffmpeg)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from typing import Any


from annolid.core.agent.tools.function_ffmpeg import (
    VideoFFmpegProcessTool,
    _apply_preset,
)


# ---------------------------------------------------------------------------
# _apply_preset unit tests
# ---------------------------------------------------------------------------


class TestApplyPreset:
    def test_improve_quality_defaults(self):
        result = _apply_preset("improve_quality", {})
        assert result["denoise"] is True
        assert result["auto_contrast"] is True
        assert result["contrast_strength"] == 1.0

    def test_auto_contrast_defaults(self):
        result = _apply_preset("auto_contrast", {})
        assert result["auto_contrast"] is True
        assert result.get("denoise") is None or result.get("denoise") is False

    def test_downsample_defaults(self):
        result = _apply_preset("downsample", {})
        assert result["scale_factor"] == 0.5
        assert result["_halve_fps"] is True

    def test_denoise_defaults(self):
        result = _apply_preset("denoise", {})
        assert result["denoise"] is True

    def test_explicit_overrides_preset(self):
        result = _apply_preset("downsample", {"scale_factor": 0.25})
        assert result["scale_factor"] == 0.25

    def test_custom_returns_explicit_only(self):
        result = _apply_preset("custom", {"denoise": True})
        assert result["denoise"] is True
        assert "auto_contrast" not in result

    def test_none_values_do_not_override(self):
        result = _apply_preset("improve_quality", {"denoise": None})
        assert result["denoise"] is True


# ---------------------------------------------------------------------------
# Tool integration tests (monkeypatched)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestVideoFFmpegProcessTool:
    def _make_tool(self, tmp_path: Path) -> VideoFFmpegProcessTool:
        return VideoFFmpegProcessTool(
            allowed_dir=tmp_path,
            allowed_read_roots=[tmp_path],
        )

    async def test_missing_input_returns_error(self, tmp_path: Path) -> None:
        tool = self._make_tool(tmp_path)
        result_str = await tool.execute(path=str(tmp_path / "nonexistent.mp4"))
        result = json.loads(result_str)
        assert "error" in result
        assert "not found" in result["error"].lower()

    async def test_not_a_file_returns_error(self, tmp_path: Path) -> None:
        tool = self._make_tool(tmp_path)
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        result_str = await tool.execute(path=str(subdir))
        result = json.loads(result_str)
        assert "error" in result

    async def test_path_outside_allowed_dir_returns_error(self, tmp_path: Path) -> None:
        tool = self._make_tool(tmp_path)
        result_str = await tool.execute(path="/etc/passwd")
        result = json.loads(result_str)
        assert "error" in result

    async def test_overwrite_guard(self, tmp_path: Path) -> None:
        tool = self._make_tool(tmp_path)
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake")
        output = tmp_path / "clip_processed.mp4"
        output.write_bytes(b"existing")
        result_str = await tool.execute(
            path=str(video),
            output_path=str(output),
            overwrite=False,
        )
        result = json.loads(result_str)
        assert "error" in result
        assert "overwrite" in result["error"].lower()

    async def test_crop_action_requires_crop_object(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "annolid.core.media.video.get_video_fps",
            lambda _: 30.0,
        )
        tool = self._make_tool(tmp_path)
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake")
        result_str = await tool.execute(
            path=str(video),
            action="crop",
        )
        result = json.loads(result_str)
        assert "error" in result
        assert "crop" in result["error"].lower()

    async def test_successful_processing(self, tmp_path: Path, monkeypatch) -> None:
        tool = self._make_tool(tmp_path)
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake")

        monkeypatch.setattr(
            "annolid.core.media.video.get_video_fps",
            lambda _: 30.0,
        )

        def fake_compress(
            input_folder,
            output_folder,
            scale_factor,
            fps=None,
            apply_denoise=False,
            auto_contrast=False,
            auto_contrast_strength=1.0,
            crop_x=None,
            crop_y=None,
            crop_width=None,
            crop_height=None,
        ):
            out = Path(output_folder) / "clip.mp4"
            out.write_bytes(b"processed")
            return {"clip.mp4": "ffmpeg -y -i clip.mp4 -c:v libx264 clip.mp4"}

        monkeypatch.setattr(
            "annolid.utils.videos.compress_and_rescale_video",
            fake_compress,
        )

        result_str = await tool.execute(
            path=str(video),
            action="improve_quality",
        )
        result = json.loads(result_str)
        assert result.get("success") is True
        assert result["action"] == "improve_quality"
        assert result["denoise"] is True
        assert result["auto_contrast"] is True
        assert Path(result["output_path"]).exists()

    async def test_downsample_halves_fps(self, tmp_path: Path, monkeypatch) -> None:
        tool = self._make_tool(tmp_path)
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake")

        monkeypatch.setattr(
            "annolid.core.media.video.get_video_fps",
            lambda _: 30.0,
        )

        captured: dict[str, Any] = {}

        def fake_compress(
            input_folder,
            output_folder,
            scale_factor,
            fps=None,
            apply_denoise=False,
            auto_contrast=False,
            auto_contrast_strength=1.0,
            crop_x=None,
            crop_y=None,
            crop_width=None,
            crop_height=None,
        ):
            captured["fps"] = fps
            captured["scale_factor"] = scale_factor
            out = Path(output_folder) / "clip.mp4"
            out.write_bytes(b"processed")
            return {"clip.mp4": "ffmpeg -y -i clip.mp4 clip.mp4"}

        monkeypatch.setattr(
            "annolid.utils.videos.compress_and_rescale_video",
            fake_compress,
        )

        result_str = await tool.execute(path=str(video), action="downsample")
        result = json.loads(result_str)
        assert result.get("success") is True
        assert captured["fps"] == 15.0
        assert captured["scale_factor"] == 0.5

    async def test_all_encoders_fail_returns_error(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        tool = self._make_tool(tmp_path)
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake")

        monkeypatch.setattr(
            "annolid.core.media.video.get_video_fps",
            lambda _: 30.0,
        )

        def fake_compress(**kwargs):
            return {}

        monkeypatch.setattr(
            "annolid.utils.videos.compress_and_rescale_video",
            fake_compress,
        )

        result_str = await tool.execute(path=str(video), action="denoise")
        result = json.loads(result_str)
        assert "error" in result

    def test_schema_is_valid(self) -> None:
        tool = VideoFFmpegProcessTool()
        schema = tool.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "video_ffmpeg_process"
        assert "path" in schema["function"]["parameters"]["properties"]
        assert "path" in schema["function"]["parameters"]["required"]
