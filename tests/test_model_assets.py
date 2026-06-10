from __future__ import annotations

from hashlib import md5, sha256
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


def test_ensure_cached_model_asset_retries_without_fuzzy_for_gdown_versions(
    tmp_path: Path, monkeypatch
) -> None:
    payload = b"tapnext-onnx-test-payload"
    expected = sha256(payload).hexdigest()
    calls: list[dict[str, object]] = []

    def _fake_cached_download(url: str, path: str, quiet: bool = True, **kwargs):
        calls.append({"url": url, "path": path, "quiet": quiet, **kwargs})
        if "fuzzy" in kwargs:
            raise TypeError("download() got an unexpected keyword argument 'fuzzy'")
        Path(path).write_bytes(payload)
        return path

    monkeypatch.setattr(
        "annolid.utils.model_assets.gdown.cached_download",
        _fake_cached_download,
    )

    out = model_assets.ensure_cached_model_asset(
        file_name="tapnext.onnx",
        url="https://github.com/healthonrails/annolid/releases/download/v1.6.6/tapnext.onnx",
        expected_sha256=expected,
        cache_dir=tmp_path / "downloads",
    )

    assert out.read_bytes() == payload
    assert calls == [
        {
            "url": "https://github.com/healthonrails/annolid/releases/download/v1.6.6/tapnext.onnx",
            "path": str(out),
            "quiet": True,
            "fuzzy": True,
        },
        {
            "url": "https://github.com/healthonrails/annolid/releases/download/v1.6.6/tapnext.onnx",
            "path": str(out),
            "quiet": True,
        },
    ]


def test_ensure_cached_model_asset_verifies_md5_without_forwarding_to_gdown(
    tmp_path: Path, monkeypatch
) -> None:
    payload = b"cutie-pth-test-payload"
    expected = md5(payload).hexdigest()  # noqa: S324 - testing legacy checksum flow
    calls: list[dict[str, object]] = []

    def _fake_cached_download(url: str, path: str, quiet: bool = True, **kwargs):
        calls.append({"url": url, "path": path, "quiet": quiet, **kwargs})
        if "md5" in kwargs:
            raise TypeError("download() got an unexpected keyword argument 'md5'")
        Path(path).write_bytes(payload)
        return path

    monkeypatch.setattr(
        "annolid.utils.model_assets.gdown.cached_download",
        _fake_cached_download,
    )

    out = model_assets.ensure_cached_model_asset(
        file_name="cutie-base-mega.pth",
        url="https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth",
        expected_md5=expected,
        cache_dir=tmp_path / "downloads",
    )

    assert out.read_bytes() == payload
    assert calls == [
        {
            "url": "https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth",
            "path": str(out),
            "quiet": True,
            "fuzzy": True,
        }
    ]
