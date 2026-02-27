from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from annolid.utils import model_assets


def test_resolve_existing_model_path_prefers_workspace_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    fake_home = tmp_path / "home"
    workspace_file = (
        fake_home
        / ".annolid"
        / "workspace"
        / "downloads"
        / "videomt_yt_2019_vit_small_52.8.onnx"
    )
    workspace_file.parent.mkdir(parents=True, exist_ok=True)
    workspace_file.write_bytes(b"x")

    monkeypatch.setattr("annolid.utils.model_assets.Path.home", lambda: fake_home)
    resolved = model_assets.resolve_existing_model_path(
        "downloads/videomt_yt_2019_vit_small_52.8.onnx"
    )
    assert resolved == workspace_file


def test_ensure_cached_model_asset_downloads_and_verifies_sha256(
    tmp_path: Path, monkeypatch
) -> None:
    payload = b"videomt-onnx-test-payload"
    expected = sha256(payload).hexdigest()

    def _fake_cached_download(
        url: str, output: str, quiet: bool = True, fuzzy: bool = True
    ):
        _ = (url, quiet, fuzzy)
        Path(output).write_bytes(payload)
        return output

    monkeypatch.setattr(
        "annolid.utils.model_assets.gdown.cached_download",
        _fake_cached_download,
    )

    out = model_assets.ensure_cached_model_asset(
        file_name="model.onnx",
        url="https://example.invalid/model.onnx",
        expected_sha256=expected,
        cache_dir=tmp_path / "downloads",
    )
    assert out.is_file()
    assert out.read_bytes() == payload
