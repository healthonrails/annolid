"""Tests for Colab backends (CLI, REST API, Drive/Notebook), service, CLI, and agent tool."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import urllib.error
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from annolid.colab.auth import ColabAuthManager, ColabAuthRequiredError
from annolid.colab.backends import (
    ColabApiBackend,
    ColabCliBackend,
    ColabDriveNotebookBackend,
)
from annolid.colab.models import (
    ColabPredictRequest,
    ColabRuntimeInfo,
    ColabRuntimeSpec,
    ColabTrainRequest,
)
from annolid.colab.service import ColabWorkflowService
from annolid.core.agent.config.schema import GoogleColabConfig, ToolsConfig
from annolid.core.agent.tools.annolid_run import AnnolidRunTool
from annolid.core.agent.tools.colab import GoogleColabTool
from annolid.engine.cli import main as annolid_run_main


@pytest.fixture(autouse=True)
def _isolate_home_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent Colab service/CLI tests from touching the real user's HOME config."""
    fake_home = tmp_path / "isolated_home"
    fake_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(fake_home / ".config"))


def _make_tiny_dataset(root: Path) -> Path:
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "images" / "train" / "img1.jpg").write_bytes(b"jpg")
    (root / "labels" / "train" / "img1.txt").write_text(
        "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
    )
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/train",
                "names": {0: "mouse"},
            }
        ),
        encoding="utf-8",
    )
    return data_yaml


def _make_fake_output_zip(path: Path, files: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)


def test_colab_cli_backend_end_to_end_via_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_yaml = _make_tiny_dataset(tmp_path / "ds")
    auth = ColabAuthManager(
        credentials_file=str(tmp_path / "c.json"),
        token_file=str(tmp_path / "t.json"),
    )
    monkeypatch.setattr(auth, "find_colab_cli", lambda: "/usr/local/bin/colab")

    cli_backend = ColabCliBackend(auth)
    recorded_cmds: list[list[str]] = []

    def _fake_run(cmd, capture_output=True, text=True, timeout=300):
        recorded_cmds.append(list(cmd))
        if "download" in cmd:
            local_dest = Path(cmd[-1])
            _make_fake_output_zip(
                local_dest,
                {
                    "run/weights/best.pt": b"trained-weights",
                    "run/results.csv": b"epoch,loss\n1,0.1\n",
                },
            )
            return SimpleNamespace(returncode=0, stdout="downloaded", stderr="")
        if "url" in cmd:
            return SimpleNamespace(
                returncode=0,
                stdout="https://colab.research.google.com/runtime/1\n",
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    class _FakePopen:
        def __init__(self, cmd, **_kwargs):
            recorded_cmds.append(list(cmd))
            self.stdout = iter(
                [
                    "[annolid-colab] Training epoch 1/1\n",
                    'ANNOLID_COLAB_RESULT_JSON={"status": "completed", "artifact_count": 2}\n',
                ]
            )
            self._rc = 0

        def wait(self):
            return self._rc

        def poll(self):
            return self._rc

        def terminate(self):
            pass

    monkeypatch.setattr("annolid.colab.backends.subprocess.run", _fake_run)
    monkeypatch.setattr("annolid.colab.backends.subprocess.Popen", _FakePopen)

    svc = ColabWorkflowService(
        auth_manager=auth,
        home_dir=tmp_path / "colab_home",
        cli_backend=cli_backend,
    )
    logs: list[str] = []
    rec = svc.run_train(
        ColabTrainRequest(
            model="yolo",
            dataset_path=str(data_yaml),
            weights="yolo11n-seg.pt",
            epochs=5,
            accelerator="T4",
            backend="cli",
            output_dir=str(tmp_path / "out"),
        ),
        log_callback=logs.append,
    )

    assert rec.status == "completed"
    assert rec.backend == "cli"
    assert rec.result_summary["artifact_count"] == 2
    assert (tmp_path / "out" / "run" / "weights" / "best.pt").is_file()
    # Runtime should be stopped cleanly when keep_runtime=False
    assert any("stop" in cmd for cmd in recorded_cmds)
    assert svc.get_job(rec.job_id) is not None


def test_colab_api_backend_specs_runtime_contents_and_ws_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    auth = ColabAuthManager(
        credentials_file=str(tmp_path / "c.json"),
        token_file=str(tmp_path / "t.json"),
    )
    monkeypatch.setattr(auth, "get_colab_api_access_token", lambda **_k: "tok-123")

    api = ColabApiBackend(auth)
    api._CHUNK_BYTES = 16  # force chunked upload path
    api._RETRY_BASE_SECONDS = 0.0
    monkeypatch.setattr(
        api,
        "_stream_download_binary",
        lambda rt, *_a, **_k: (False, rt),  # exercise /api/contents fallback path
    )

    calls: list[tuple[str, str, dict, object]] = []
    out_zip_bytes = tmp_path / "remote_out.zip"
    _make_fake_output_zip(
        out_zip_bytes, {"predictions/frame_0001.json": b'{"shapes": []}'}
    )

    def _fake_request(method, url, *, headers, body=None, timeout=None):
        calls.append((method, url, dict(headers), body))
        if url.endswith("/runtimespecs"):
            return {
                "runtimeSpecs": [
                    {
                        "key": {
                            "variant": "VARIANT_GPU",
                            "accelerator": "L4",
                            "shape": "SHAPE_STANDARD",
                        },
                        "eligible": True,
                    }
                ]
            }
        if "/runtimes?" in url and method == "POST":
            return {
                "name": "runtimes/rt-abc",
                "runtimeSpec": {
                    "variant": "VARIANT_GPU",
                    "accelerator": "L4",
                    "shape": "SHAPE_STANDARD",
                },
                "connectionInfo": {
                    "url": "https://proxy.colab.test",
                    "token": "proxy-secret",
                    "expireTime": "2026-11-24T15:00:00Z",
                },
            }
        if url.endswith("/api/kernels") and method == "POST":
            return {"id": "kern-1"}
        if "/api/contents/" in url and method == "GET":
            return {
                "format": "base64",
                "content": base64.b64encode(out_zip_bytes.read_bytes()).decode("ascii"),
            }
        return {}

    monkeypatch.setattr(api, "_request_json", _fake_request)

    class _FakeWS:
        def __init__(self):
            self._msg_id = ""
            self._queue: list[str] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

        async def send(self, raw: str):
            sent = json.loads(raw)
            self._msg_id = sent["header"]["msg_id"]
            self._queue = [
                json.dumps(
                    {
                        "msg_type": "stream",
                        "parent_header": {"msg_id": self._msg_id},
                        "content": {"text": "hello from colab api\n"},
                    }
                ),
                json.dumps(
                    {
                        "msg_type": "execute_reply",
                        "parent_header": {"msg_id": self._msg_id},
                        "content": {"status": "ok"},
                    }
                ),
                json.dumps(
                    {
                        "msg_type": "status",
                        "parent_header": {"msg_id": self._msg_id},
                        "content": {"execution_state": "idle"},
                    }
                ),
            ]

        async def recv(self):
            return self._queue.pop(0)

    import sys

    fake_ws_mod = SimpleNamespace(
        connect=lambda url, additional_headers=None, max_size=None: _FakeWS()
    )
    monkeypatch.setitem(sys.modules, "websockets", fake_ws_mod)

    specs = api.list_runtime_specs()
    assert specs == [
        ColabRuntimeSpec(
            variant="VARIANT_GPU",
            accelerator="L4",
            shape="SHAPE_STANDARD",
            eligible=True,
        )
    ]

    rt = api.create_runtime(accelerator="L4")
    assert rt.name == "runtimes/rt-abc"
    assert rt.proxy_token == "proxy-secret"
    # Public dict never leaks proxy token
    assert "proxy_token" not in rt.to_public_dict()

    # Chunked upload uses X-Colab-Runtime-Proxy-Token
    big_file = tmp_path / "payload.bin"
    big_file.write_bytes(b"A" * 40)
    api.upload_file(
        runtime=rt, local_path=big_file, remote_path="/content/job/payload.bin"
    )
    put_chunks = [c for c in calls if c[0] == "PUT" and "payload.bin" in c[1]]
    assert len(put_chunks) == 3
    assert put_chunks[0][2]["X-Colab-Runtime-Proxy-Token"] == "proxy-secret"
    assert put_chunks[-1][3]["chunk"] == -1

    # Execute python via WS
    streamed: list[str] = []
    out_text = api.execute_python(
        runtime=rt, code="print('hi')", log_callback=streamed.append
    )
    assert "hello from colab api" in out_text
    assert any("hello from colab api" in s for s in streamed)

    # Download file
    dl_target = tmp_path / "downloaded.zip"
    api.download_file(
        runtime=rt, remote_path="/content/job/output_bundle.zip", local_path=dl_target
    )
    assert dl_target.is_file()

    # Delete runtime
    api.delete_runtime(rt.name)
    assert any(c[0] == "DELETE" and c[1].endswith("/runtimes/rt-abc") for c in calls)


def test_colab_drive_notebook_backend_upload_and_pull(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video-bytes")

    auth = ColabAuthManager(
        credentials_file=str(tmp_path / "c.json"),
        token_file=str(tmp_path / "t.json"),
    )

    uploaded_files: list[str] = []

    class _FakeDriveTool:
        def ensure_folder_path_sync(self, folder_path: str) -> str:
            return "folder-999"

        def upload_local_file_sync(self, local_path, **_kwargs):
            name = Path(local_path).name
            uploaded_files.append(name)
            fid = "nb-file-42" if name.endswith(".ipynb") else "zip-file-77"
            return {"status": "uploaded", "file": {"id": fid, "name": name}}

        def list_folder_files_sync(self, folder_id: str):
            return [
                {"id": "res-json", "name": "job_result.json"},
                {"id": "res-zip", "name": "output_bundle.zip"},
            ]

        def download_remote_file_sync(self, file_id: str, local_path: Path):
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if file_id == "res-json":
                local_path.write_text(
                    json.dumps({"status": "completed", "artifact_count": 1}),
                    encoding="utf-8",
                )
            else:
                _make_fake_output_zip(local_path, {"predictions/frame_0.json": b"{}"})
            return {"file_id": file_id, "local_path": str(local_path)}

    monkeypatch.setattr(auth, "create_drive_tool", lambda **_k: _FakeDriveTool())
    nb_backend = ColabDriveNotebookBackend(auth)
    monkeypatch.setattr(nb_backend, "is_drive_ready", lambda: True)

    svc = ColabWorkflowService(
        auth_manager=auth,
        home_dir=tmp_path / "colab_home",
        notebook_backend=nb_backend,
    )
    rec = svc.run_predict(
        ColabPredictRequest(
            model="yolo_labelme",
            source_path=str(video),
            weights="yolo11n-seg.pt",
            backend="notebook",
            upload_to_drive=True,
            open_browser=False,
        )
    )
    assert rec.status == "notebook_ready"
    assert rec.drive_notebook_id == "nb-file-42"
    assert rec.colab_url == "https://colab.research.google.com/drive/nb-file-42"
    assert any(name.endswith(".ipynb") for name in uploaded_files)
    assert any(name.endswith(".zip") for name in uploaded_files)

    pulled = svc.pull_results(rec.job_id, output_dir=tmp_path / "pulled")
    assert pulled.status == "completed"
    assert (tmp_path / "pulled" / "predictions" / "frame_0.json").is_file()


def test_annolid_run_colab_cli_commands_and_agent_tool(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    data_yaml = _make_tiny_dataset(tmp_path / "ds")
    colab_home = tmp_path / "colab_home"

    rc = annolid_run_main(
        ["colab", "--colab-home", str(colab_home), "status", "--json"]
    )
    assert rc == 0
    st = json.loads(capsys.readouterr().out)
    assert "auth" in st and "jobs_count" in st

    rc_nb = annolid_run_main(
        [
            "colab",
            "--colab-home",
            str(colab_home),
            "notebook",
            "--mode",
            "train",
            "--model",
            "yolo",
            "--data",
            str(data_yaml),
            "--weights",
            "yolo11n-seg.pt",
            "--epochs",
            "10",
            "--json",
        ]
    )
    assert rc_nb == 0
    nb_rec = json.loads(capsys.readouterr().out)
    assert nb_rec["status"] == "notebook_ready"
    assert Path(nb_rec["notebook_path"]).is_file()

    rc_jobs = annolid_run_main(
        ["colab", "--colab-home", str(colab_home), "jobs", "--json"]
    )
    assert rc_jobs == 0
    jobs = json.loads(capsys.readouterr().out)
    assert len(jobs) == 1
    assert jobs[0]["job_id"] == nb_rec["job_id"]

    # AnnolidRunTool mutation gating for colab commands
    assert AnnolidRunTool._is_mutating(["colab", "status"]) is False
    assert AnnolidRunTool._is_mutating(["colab", "auth", "status"]) is False
    assert AnnolidRunTool._is_mutating(["colab", "templates"]) is False
    assert AnnolidRunTool._is_mutating(["colab", "auth", "login"]) is True
    assert (
        AnnolidRunTool._is_mutating(["colab", "train", "yolo", "--data", "d.yaml"])
        is True
    )
    assert AnnolidRunTool._is_mutating(["colab", "stop", "sess1"]) is True

    # Config schema roundtrip
    tools_cfg = ToolsConfig.from_dict(
        {
            "google_colab": {
                "enabled": True,
                "backend": "api",
                "defaultAccelerator": "L4",
                "driveFolder": "Lab/colab",
            }
        }
    )
    assert tools_cfg.google_colab == GoogleColabConfig(
        enabled=True,
        backend="api",
        default_accelerator="L4",
        drive_folder="Lab/colab",
    )
    assert tools_cfg.to_dict()["google_colab"]["default_accelerator"] == "L4"

    # GoogleColabTool execution
    tool = GoogleColabTool(allowed_dir=tmp_path)
    assert tool.name == "google_colab"
    tpl_json = json.loads(asyncio.run(tool.execute(action="list_templates")))
    assert len(tpl_json["templates"]) >= 4
    doc_json = json.loads(asyncio.run(tool.execute(action="doctor")))
    assert "checks" in doc_json and "recommended_backend" in doc_json
    assert AnnolidRunTool._is_mutating(["colab", "doctor"]) is False
    assert AnnolidRunTool._is_mutating(["colab", "doctor", "--live"]) is False


def test_colab_api_proxy_token_refresh_streaming_download_and_kernel_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    auth = ColabAuthManager(
        credentials_file=str(tmp_path / "c.json"),
        token_file=str(tmp_path / "t.json"),
    )
    monkeypatch.setattr(auth, "get_colab_api_access_token", lambda **_k: "tok-123")
    api = ColabApiBackend(auth)
    api._RETRY_BASE_SECONDS = 0.0

    # Expiring runtime triggers automatic refresh before proxy calls
    soon = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
    later = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    expiring_rt = ColabRuntimeInfo(
        name="runtimes/rt-exp",
        backend="api",
        proxy_url="https://proxy.colab.test",
        proxy_token="old-token",
        expire_time=soon,
    )
    fresh_rt = ColabRuntimeInfo(
        name="runtimes/rt-exp",
        backend="api",
        proxy_url="https://proxy.colab.test",
        proxy_token="fresh-token",
        expire_time=later,
    )
    assert api.is_proxy_token_expiring(expiring_rt) is True
    assert api.is_proxy_token_expiring(fresh_rt) is False

    refresh_calls: list[str] = []
    monkeypatch.setattr(
        api,
        "get_runtime",
        lambda name: (refresh_calls.append(name), fresh_rt)[1],
    )

    # Streaming binary download via /files/ endpoint
    streamed_urls: list[tuple[str, str]] = []
    expected_bytes = b"PK\x03\x04binary-zip-stream-payload" * 20

    class _FakeResp:
        def __init__(self, payload: bytes):
            self._buf = io.BytesIO(payload)

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self, n: int = -1) -> bytes:
            return self._buf.read(n)

    def _fake_urlopen(req, timeout=None):
        streamed_urls.append(
            (req.full_url, req.headers.get("X-colab-runtime-proxy-token", ""))
        )
        return _FakeResp(expected_bytes)

    monkeypatch.setattr("annolid.colab.backends.urllib.request.urlopen", _fake_urlopen)

    out_path = tmp_path / "streamed.zip"
    updated_rt = api.download_file(
        runtime=expiring_rt,
        remote_path="/content/annolid_jobs/j1/output_bundle.zip",
        local_path=out_path,
    )
    assert refresh_calls == ["runtimes/rt-exp"]
    assert updated_rt.proxy_token == "fresh-token"
    assert out_path.read_bytes() == expected_bytes
    assert streamed_urls[0][0].endswith("/files/annolid_jobs/j1/output_bundle.zip")
    assert streamed_urls[0][1] == "fresh-token"

    # Transient HTTP 503 retry in _request_json
    attempts = {"count": 0}

    def _flaky_urlopen(req, timeout=None):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise urllib.error.HTTPError(
                req.full_url, 503, "Service Unavailable", {}, io.BytesIO(b"busy")
            )
        return _FakeResp(b'{"runtimeSpecs": []}')

    monkeypatch.setattr("annolid.colab.backends.urllib.request.urlopen", _flaky_urlopen)
    assert api.list_runtime_specs() == []
    assert attempts["count"] == 2

    # Cancellation sends kernel interrupt
    interrupted: list[str] = []
    monkeypatch.setattr(
        api,
        "interrupt_kernel",
        lambda _rt, kid: interrupted.append(kid),
    )

    class _SlowWS:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

        async def send(self, _raw: str):
            pass

        async def recv(self):
            await asyncio.sleep(0.05)
            return "{}"

    import sys

    monkeypatch.setitem(
        sys.modules,
        "websockets",
        SimpleNamespace(
            connect=lambda url,
            additional_headers=None,
            max_size=None,
            ping_interval=None,
            ping_timeout=None: _SlowWS()
        ),
    )
    cancel_ev = threading.Event()
    cancel_ev.set()
    with pytest.raises(InterruptedError, match="cancelled"):
        asyncio.run(
            api._execute_on_kernel_ws(
                runtime=fresh_rt,
                kernel_id="kern-cancel",
                code="import time; time.sleep(10)",
                cancel_event=cancel_ev,
            )
        )
    assert interrupted == ["kern-cancel"]


def test_colab_doctor_offline_validation_and_live_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    from annolid.core.agent.tools.google_auth import GoogleOAuthHelper

    creds = tmp_path / "client.json"
    token = tmp_path / "token.json"
    creds.write_text(json.dumps({"type": "service_account"}), encoding="utf-8")
    monkeypatch.setattr(
        GoogleOAuthHelper, "is_oauth_available", staticmethod(lambda: True)
    )

    auth = ColabAuthManager(credentials_file=str(creds), token_file=str(token))
    monkeypatch.setattr(auth, "find_colab_cli", lambda: None)
    svc = ColabWorkflowService(auth_manager=auth, home_dir=tmp_path / "colab_home")

    bad_report = svc.run_diagnostics(live=False)
    assert bad_report["ok"] is False
    client_check = next(
        c for c in bad_report["checks"] if c["name"] == "oauth_client_json"
    )
    assert client_check["ok"] is False
    assert "service_account" in client_check["detail"]

    # Now supply valid desktop OAuth client + token and verify live probes
    creds.write_text(
        json.dumps(
            {
                "installed": {
                    "client_id": "cid-123",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            }
        ),
        encoding="utf-8",
    )
    token.write_text(
        json.dumps(
            {
                "token": "t",
                "refresh_token": "r",
                "scopes": [
                    "https://www.googleapis.com/auth/colaboratory",
                    "https://www.googleapis.com/auth/drive",
                    "https://www.googleapis.com/auth/userinfo.email",
                    "openid",
                ],
            }
        ),
        encoding="utf-8",
    )
    token.chmod(0o600)

    monkeypatch.setattr(
        svc,
        "list_runtime_specs",
        lambda: [
            ColabRuntimeSpec(variant="VARIANT_GPU", accelerator="T4", eligible=True),
        ],
    )
    monkeypatch.setattr(
        svc.notebook_backend,
        "probe_drive_account",
        lambda: {"email": "researcher@example.org", "display_name": "Researcher"},
    )

    live_report = svc.run_diagnostics(live=True)
    assert live_report["ok"] is True
    assert live_report["live_probes"]["colab_api"]["ok"] is True
    assert (
        live_report["live_probes"]["google_drive"]["email"] == "researcher@example.org"
    )

    # Verify CLI doctor subcommand exits 0 on valid offline config
    rc = annolid_run_main(
        [
            "colab",
            "--credentials",
            str(creds),
            "--token",
            str(token),
            "--colab-home",
            str(tmp_path / "colab_home"),
            "doctor",
            "--json",
        ]
    )
    assert rc == 0
    cli_report = json.loads(capsys.readouterr().out)
    assert cli_report["ok"] is True


def test_cli_interactive_login_prompt_and_auto_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    from annolid.colab import cli as colab_cli
    from annolid.colab.models import ColabJobRecord

    data_yaml = _make_tiny_dataset(tmp_path / "ds")
    colab_home = tmp_path / "colab_home"

    # Simulate interactive terminal via isatty() so real _can_prompt_interactive() logic is tested
    monkeypatch.setattr(colab_cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(colab_cli.sys.stdout, "isatty", lambda: True)

    login_calls: list[dict] = []
    train_backends: list[str] = []

    class _FakeService:
        def __init__(self):
            self.logged_in = False

        def auth_status(self):
            return {
                "recommended_backend": "cli" if self.logged_in else "notebook",
                "summary": "No direct Colab runtime auth configured yet.",
                "colab_cli": {"installed": self.logged_in},
                "oauth_colab": {"ready": False, "has_credentials_file": False},
                "oauth_drive": {"ready": False},
                "adc": {"ready": False, "gcloud_installed": False},
                "cli_ready": self.logged_in,
                "api_ready": False,
                "drive_ready": False,
                "credentials_file": str(tmp_path / "creds.json"),
                "token_file": str(tmp_path / "tok.json"),
            }

        def login(self, **kwargs):
            login_calls.append(kwargs)
            self.logged_in = True
            return {"ok": True, "method": kwargs.get("method", "colab-cli")}

        def run_train(self, req, **_kwargs):
            train_backends.append(req.backend)
            return ColabJobRecord(
                job_id="train_demo_1",
                kind="train",
                model=req.model,
                backend=req.backend,
                accelerator=req.accelerator,
                status="completed",
            )

    fake_svc = _FakeService()
    monkeypatch.setattr(colab_cli, "_make_service", lambda _args: fake_svc)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "1")

    rc = colab_cli.main(
        [
            "--colab-home",
            str(colab_home),
            "train",
            "yolo",
            "--data",
            str(data_yaml),
        ]
    )
    assert rc == 0
    assert len(login_calls) == 1
    assert login_calls[0]["method"] == "colab-cli"
    assert login_calls[0]["install_cli_if_missing"] is True
    assert train_backends == ["cli"]
    out = capsys.readouterr().out
    assert "Google Colab Sign-In Required" in out
    assert "train_demo_1" in out

    # Case 2: Runtime ColabAuthRequiredError (e.g. stale ADC) prompts and allows fallback to notebook
    train_backends.clear()

    def _stale_then_ok(req, **_kwargs):
        train_backends.append(req.backend)
        if req.backend != "notebook":
            raise ColabAuthRequiredError(
                "ADC token expired (invalid_grant)", backend="api"
            )
        return ColabJobRecord(
            job_id="train_nb_2",
            kind="train",
            model=req.model,
            backend="notebook",
            accelerator=req.accelerator,
            status="notebook_ready",
        )

    fake_svc.run_train = _stale_then_ok  # type: ignore[method-assign]
    monkeypatch.setattr("builtins.input", lambda _prompt="": "4")
    rc2 = colab_cli.main(
        [
            "--colab-home",
            str(colab_home),
            "train",
            "yolo",
            "--data",
            str(data_yaml),
            "--backend",
            "api",
        ]
    )
    assert rc2 == 0
    assert train_backends == ["notebook"]
    capsys.readouterr()

    # Case 3: --no-prompt + --json reports structured auth_required without calling input()
    monkeypatch.setattr(
        "builtins.input",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not prompt")),
    )

    def _always_auth_err(_req, **_kwargs):
        raise ColabAuthRequiredError("Token revoked", backend="api")

    fake_svc.run_train = _always_auth_err  # type: ignore[method-assign]
    rc3 = colab_cli.main(
        [
            "--no-prompt",
            "--colab-home",
            str(colab_home),
            "train",
            "yolo",
            "--data",
            str(data_yaml),
            "--backend",
            "api",
            "--json",
        ]
    )
    assert rc3 == 1
    err_payload = json.loads(capsys.readouterr().out)
    assert err_payload["ok"] is False
    assert err_payload["auth_required"] is True
    assert "Token revoked" in err_payload["error"]


def test_colab_cli_backend_session_parsing_timeout_flag_and_remote_error_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    prefs_file = tmp_path / "auth_prefs.json"
    auth = ColabAuthManager(
        credentials_file=str(tmp_path / "c.json"),
        token_file=str(tmp_path / "t.json"),
        prefs_file=prefs_file,
    )
    monkeypatch.setattr(auth, "find_colab_cli", lambda: "/usr/local/bin/colab")
    auth._save_auth_prefs({"colab_cli_auth_mode": "adc"})
    assert auth.resolve_colab_cli_auth_mode() == "adc"

    backend = ColabCliBackend(auth)

    # Empty server response banner should not be parsed as a session
    monkeypatch.setattr(
        "annolid.colab.backends.subprocess.run",
        lambda cmd, **_kw: SimpleNamespace(
            returncode=0,
            stdout="[colab] No active sessions found on server.\n",
            stderr="",
        ),
    )
    assert backend.list_sessions() == []

    # Real google-colab-cli==0.8.0 session line format
    monkeypatch.setattr(
        "annolid.colab.backends.subprocess.run",
        lambda cmd, **_kw: SimpleNamespace(
            returncode=0,
            stdout=(
                "[my-gpu] gpu-t4-s-abc123 | Hardware: T4 | Shape: Standard | Variant: GPU\n"
                "[?] gpu-l4-s-orphan | Hardware: L4 | Shape: High-RAM | Variant: GPU\n"
            ),
            stderr="",
        ),
    )
    sessions = backend.list_sessions()
    assert len(sessions) == 2
    assert sessions[0].session_name == "my-gpu"
    assert sessions[0].accelerator == "T4"
    assert sessions[1].name == "gpu-l4-s-orphan"
    assert sessions[1].accelerator == "L4"

    # exec_script passes `--timeout 0` and raises on ANNOLID_COLAB_ERROR_JSON even with exit 0
    seen_cmd: list[str] = []

    class _FakeErrPopen:
        def __init__(self, cmd, **_kwargs):
            seen_cmd.extend(cmd)
            self.stdout = iter(
                [
                    "Traceback...\n",
                    'ANNOLID_COLAB_ERROR_JSON={"job_id": "j1", "status": "failed", "error": "boom"}\n',
                ]
            )

        def wait(self):
            return 0

        def poll(self):
            return 0

        def terminate(self):
            pass

    monkeypatch.setattr("annolid.colab.backends.subprocess.Popen", _FakeErrPopen)
    with pytest.raises(RuntimeError, match="boom"):
        backend.exec_script(session_name="my-gpu", script_path=tmp_path / "run.py")
    assert "--auth=adc" in seen_cmd
    assert "--timeout" in seen_cmd and "0" in seen_cmd

    # stop_runtime unassigns server-side orphan endpoint when `colab stop` reports not found
    unassigned: list[str] = []
    monkeypatch.setattr(
        "annolid.colab.backends.subprocess.run",
        lambda cmd, **_kw: SimpleNamespace(
            returncode=0,
            stdout="[colab] Session 'gpu-t4-s-orphan' not found.\n",
            stderr="",
        ),
    )
    monkeypatch.setattr(
        backend,
        "_unassign_orphan_endpoint",
        lambda ep: (unassigned.append(ep), True)[1],
    )
    backend.stop_runtime("gpu-t4-s-orphan")
    assert unassigned == ["gpu-t4-s-orphan"]

    # create_runtime timeout cleans up newly created orphan endpoint automatically
    import subprocess as _sp

    unassigned.clear()
    list_calls = {"n": 0}

    def _list_before_after():
        list_calls["n"] += 1
        if list_calls["n"] == 1:
            return []
        return [
            ColabRuntimeInfo(
                name="gpu-t4-s-leaked",
                session_name="?",
                backend="cli",
                accelerator="T4",
                state="ACTIVE",
            )
        ]

    monkeypatch.setattr(backend, "list_sessions", _list_before_after)
    monkeypatch.setattr(
        backend,
        "_run_capture",
        lambda args, timeout=300: (_ for _ in ()).throw(
            _sp.TimeoutExpired(cmd=args, timeout=timeout)
        ),
    )
    with pytest.raises(RuntimeError, match="timed out"):
        backend.create_runtime(session_name="slow-vm", accelerator="T4")
    assert unassigned == ["gpu-t4-s-leaked"]

    # ColabApiBackend condenses Google JSON error payloads cleanly
    summary = ColabApiBackend._summarize_error_body(
        json.dumps(
            {
                "error": {
                    "code": 403,
                    "status": "PERMISSION_DENIED",
                    "message": "The colaboratory.googleapis.com API requires a quota project.",
                }
            }
        )
    )
    assert summary == (
        "PERMISSION_DENIED: The colaboratory.googleapis.com API requires a quota project."
    )
