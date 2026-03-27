from __future__ import annotations

import asyncio
import json
from pathlib import Path

from annolid.core.agent.tools.box import BoxTool
from annolid.utils.box import BoxApiError, BoxOAuthTokens


def test_box_tool_requires_access_or_refresh_credentials() -> None:
    async def _run() -> None:
        tool = BoxTool(access_token="")
        payload = json.loads(await tool.execute(action="list_folder_items"))
        assert payload["ok"] is False
        assert "refresh credentials are unavailable" in payload["error"].lower()

    asyncio.run(_run())


def test_box_tool_lists_folder_items() -> None:
    async def _run() -> None:
        tool = BoxTool(access_token="token")

        def _fake_list_folder_items(**kwargs):
            assert kwargs["folder_id"] == "0"
            return {
                "total_count": 1,
                "entries": [{"id": "1", "name": "demo.txt", "type": "file"}],
                "limit": 100,
                "offset": 0,
            }

        tool._client.list_folder_items = _fake_list_folder_items  # type: ignore[method-assign]
        payload = json.loads(
            await tool.execute(action="list_folder_items", folder_id="0")
        )
        assert payload["ok"] is True
        assert payload["total_count"] == 1
        assert payload["entries"][0]["name"] == "demo.txt"

    asyncio.run(_run())


def test_box_tool_download_file_writes_local_path(tmp_path: Path) -> None:
    async def _run() -> None:
        tool = BoxTool(access_token="token", allowed_dir=tmp_path)

        def _fake_download_file_content(**kwargs):
            assert kwargs["file_id"] == "123"
            return b"annolid-box"

        tool._client.download_file_content = _fake_download_file_content  # type: ignore[method-assign]
        out_path = tmp_path / "download.bin"
        payload = json.loads(
            await tool.execute(
                action="download_file",
                file_id="123",
                destination_path=str(out_path),
            )
        )
        assert payload["ok"] is True
        assert out_path.read_bytes() == b"annolid-box"
        assert payload["bytes_written"] == len(b"annolid-box")

    asyncio.run(_run())


def test_box_tool_upload_conflict_requires_overwrite(tmp_path: Path) -> None:
    async def _run() -> None:
        tool = BoxTool(access_token="token", allowed_dir=tmp_path)
        src = tmp_path / "demo.txt"
        src.write_text("hello", encoding="utf-8")

        def _fake_upload_content(**kwargs):
            raise BoxApiError(
                status=409,
                message="item_name_in_use",
                payload={
                    "context_info": {
                        "conflicts": {"id": "555", "type": "file", "name": "demo.txt"}
                    }
                },
            )

        tool._client.upload_file_content = _fake_upload_content  # type: ignore[method-assign]
        payload = json.loads(
            await tool.execute(
                action="upload_file",
                folder_id="0",
                file_path=str(src),
                overwrite=False,
            )
        )
        assert payload["ok"] is False
        assert "already exists" in payload["error"].lower()

    asyncio.run(_run())


def test_box_tool_upload_conflict_overwrite_uploads_new_version(tmp_path: Path) -> None:
    async def _run() -> None:
        tool = BoxTool(access_token="token", allowed_dir=tmp_path)
        src = tmp_path / "demo.txt"
        src.write_text("hello", encoding="utf-8")

        def _fake_upload_content(**kwargs):
            raise BoxApiError(
                status=409,
                message="item_name_in_use",
                payload={"context_info": {"conflicts": {"id": "555"}}},
            )

        def _fake_upload_version(*, file_id: str, source_path: Path):
            assert file_id == "555"
            assert source_path == src
            return {
                "entries": [{"id": "555", "type": "file", "name": source_path.name}]
            }

        tool._client.upload_file_content = _fake_upload_content  # type: ignore[method-assign]
        tool._client.upload_file_version = _fake_upload_version  # type: ignore[method-assign]
        payload = json.loads(
            await tool.execute(
                action="upload_file",
                folder_id="0",
                file_path=str(src),
                overwrite=True,
            )
        )
        assert payload["ok"] is True
        assert payload["uploaded"]["id"] == "555"

    asyncio.run(_run())


def test_box_tool_can_refresh_when_access_token_missing() -> None:
    async def _run() -> None:
        tool = BoxTool(
            access_token="",
            client_id="cid",
            client_secret="csecret",
            refresh_token="rtok",
        )

        def _fake_refresh_access_token():
            tool._client.access_token = "new-token"
            tokens = BoxOAuthTokens(
                access_token="new-token",
                refresh_token="new-refresh",
            )
            tool._on_token_refresh(tokens)
            return tokens

        def _fake_list_folder_items(**kwargs):
            return {"total_count": 0, "entries": [], "limit": 100, "offset": 0}

        tool._client.refresh_access_token = _fake_refresh_access_token  # type: ignore[method-assign]
        tool._client.list_folder_items = _fake_list_folder_items  # type: ignore[method-assign]

        payload = json.loads(
            await tool.execute(action="list_folder_items", folder_id="0")
        )
        assert payload["ok"] is True
        assert payload["token_refreshed"] is True

    asyncio.run(_run())
