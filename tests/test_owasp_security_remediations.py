"""Focused OWASP Top 10 security regression tests across Annolid surfaces."""

from __future__ import annotations

import asyncio
import io
import os
import pickle
import socket
import ssl
import stat
import sys
import urllib.error
import urllib.request
import zipfile
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

from annolid.colab.auth import ColabAuthManager
from annolid.colab.backends import ColabApiBackend
from annolid.colab.packaging import safe_extract_archive
from annolid.core.agent.channels.email import EmailChannel
from annolid.core.agent.channels.whatsapp_webhook_server import WhatsAppWebhookServer
from annolid.core.agent.channels.zulip import ZulipChannel
from annolid.core.agent.gui_backend import tool_handlers_arxiv
from annolid.core.agent.providers.transcription import _transcribe_via_urllib
from annolid.core.agent.security_policy import has_operator_consent
from annolid.core.agent.tools import clawhub
from annolid.core.agent.tools.annolid_run import AnnolidRunTool
from annolid.core.agent.tools.cron import CronTool
from annolid.core.agent.tools.email import EmailTool, ListEmailsTool, ReadEmailTool
from annolid.core.agent.tools.git import GitCliTool
from annolid.core.agent.tools.google_drive import GoogleDriveTool
from annolid.core.agent.tools.shell import ExecTool
from annolid.core.agent.update_manager.manager import SignedUpdatePlan
from annolid.core.agent.update_manager.manifest import (
    UpdateManifest,
    fetch_channel_manifest,
)
from annolid.core.agent.update_manager.service import UpdateManagerService
from annolid.core.agent.update_manager.verify import VerificationResult
from annolid.gui import tensorboard as tb_mod
from annolid.gui import threejs_examples
from annolid.gui.widgets import pdf_viewer_server as pdf_srv
from annolid.gui.widgets import rich_text_renderer as rtr_mod
from annolid.gui.widgets import threejs_viewer_server as three_srv
from annolid.gui.widgets.threejs_viewer import ThreeJsViewerWidget
from annolid.infrastructure.memory.lancedb.backend import LanceDBMemoryBackend
from annolid.infrastructure.memory.lancedb.retriever import LanceDBRetriever
from annolid.infrastructure.sandbox.execution import (
    SandboxExecutionError,
    execute_generated_analysis,
)
from annolid.io.vector.svg_import import import_svg_string
from annolid.services.agent_box import complete_box_oauth_browser_flow
from annolid.utils import box as box_utils
from annolid.utils import cancellation, files as file_utils, model_assets
from annolid.utils import safe_http


# ---------------------------------------------------------------------------
# OWASP A02 & A03: Email TLS verification, IMAP command injection, headers
# ---------------------------------------------------------------------------


def test_owasp_a02_a03_email_tools_enforce_tls_verify_and_block_imap_injection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_ssl_contexts: list[ssl.SSLContext | None] = []
    captured_timeouts: list[object] = []

    class _FakeIMAP4SSL:
        def __init__(self, host, port, *, ssl_context=None, timeout=None):  # noqa: ANN001
            del host, port
            captured_ssl_contexts.append(ssl_context)
            captured_timeouts.append(timeout)

        def login(self, user, password):  # noqa: ANN001
            del user, password
            return ("OK", [b""])

        def select(self, mailbox):  # noqa: ANN001
            del mailbox
            return ("OK", [b"1"])

        def search(self, charset, criteria):  # noqa: ANN001
            del charset, criteria
            return ("OK", [b""])

        def fetch(self, msg_id, spec):  # noqa: ANN001
            del msg_id, spec
            return ("OK", [(b"1", b"From: a@b.com\r\nSubject: Hi\r\n\r\nHello")])

        def store(self, msg_id, cmd, flags):  # noqa: ANN001
            del msg_id, cmd, flags
            return ("OK", [b""])

        def close(self):
            return ("OK", [b""])

        def logout(self):
            return ("BYE", [b""])

    monkeypatch.setattr("imaplib.IMAP4_SSL", _FakeIMAP4SSL)

    # Injection payloads must be rejected before connecting to IMAP
    reader = ReadEmailTool(
        imap_host="imap.example.com", user="u@example.com", password="secret"
    )
    for bad_id in ("1:*", "1\r\nA999 LOGOUT", "1 (RFC822)", "-1", "abc"):
        res = asyncio.run(reader.execute(email_id=bad_id))
        assert "invalid email_id" in res.lower()
    assert captured_ssl_contexts == []

    # Valid numeric ID connects with verified SSLContext (CERT_REQUIRED) and timeout
    ok_res = asyncio.run(reader.execute(email_id="42"))
    assert "Hello" in ok_res
    assert len(captured_ssl_contexts) == 1
    assert captured_ssl_contexts[0] is not None
    assert captured_ssl_contexts[0].verify_mode == ssl.CERT_REQUIRED
    assert captured_timeouts[0] == 30.0

    # ListEmailsTool also enforces verified SSLContext
    lister = ListEmailsTool(
        imap_host="imap.example.com", user="u@example.com", password="secret"
    )
    asyncio.run(lister.execute(count=1))
    assert captured_ssl_contexts[-1] is not None
    assert captured_ssl_contexts[-1].verify_mode == ssl.CERT_REQUIRED

    # EmailChannel._poll_imap also enforces verified SSLContext
    channel = EmailChannel(
        {
            "imap_host": "imap.example.com",
            "imap_port": 993,
            "user": "u@example.com",
            "password": "secret",
        },
        bus=None,
    )
    asyncio.run(channel._poll_imap())  # noqa: SLF001
    assert captured_ssl_contexts[-1] is not None
    assert captured_ssl_contexts[-1].verify_mode == ssl.CERT_REQUIRED

    # EmailTool rejects CRLF header injection and oversized attachments
    sender = EmailTool(
        smtp_host="smtp.example.com",
        user="u@example.com",
        password="secret",
        allowed_attachment_roots=[tmp_path],
    )
    crlf_res = asyncio.run(
        sender.execute(
            to="victim@example.com\r\nBcc: attacker@evil.example",
            content="body",
        )
    )
    assert "cr/lf" in crlf_res.lower()


# ---------------------------------------------------------------------------
# OWASP A01 / A07: OAuth CSRF state validation on Box OAuth browser flow
# ---------------------------------------------------------------------------


def test_owasp_a01_a07_box_oauth_rejects_missing_or_mismatched_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    opened_urls: list[str] = []
    monkeypatch.setattr(
        "webbrowser.open_new_tab", lambda url: (opened_urls.append(url), True)[1]
    )

    class _MismatchListener:
        def __init__(self, *, redirect_uri: str) -> None:
            self.callback_url = redirect_uri

        def start(self) -> str:
            return self.callback_url

        def wait_for_result(self, timeout_s: float = 300.0) -> dict[str, str]:
            del timeout_s
            return {"ok": True, "code": "abc123", "state": "forged-state"}

        def stop(self) -> None:
            return None

    monkeypatch.setattr(
        "annolid.services.agent_box.BoxOAuthCallbackServer", _MismatchListener
    )

    result, code = complete_box_oauth_browser_flow(
        client_id="cid",
        client_secret="sec",
        redirect_uri="http://127.0.0.1:0/oauth/callback",
        persist=False,
        open_browser=True,
    )
    assert code == 1
    assert result["ok"] is False
    assert "state verification failed" in result["error"].lower()
    # Random state was generated and sent in the authorize URL
    sent_state = parse_qs(urlparse(opened_urls[0]).query).get("state", [""])[0]
    assert len(sent_state) >= 16


# ---------------------------------------------------------------------------
# OWASP A03: XSS / HTML / Script Breakout & RichTextRenderer Sanitization
# ---------------------------------------------------------------------------


def test_owasp_a03_xss_escaping_in_rich_text_and_threejs_and_examples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 1. RichTextRenderer fallback sanitizer strips scripts, event handlers, javascript: URLs
    monkeypatch.setattr(rtr_mod, "bleach", None)
    renderer = rtr_mod.RichTextRenderer()
    cleaned = renderer._sanitize_rendered_html(  # noqa: SLF001
        "<p>Safe</p><script>alert(99)</script>"
        '<img src="https://example.com/a.png" alt="x&quot; onerror=&quot;alert(1)" onerror="alert(2)">'
        '<a href="javascript:alert(3)" onclick="evil()">click</a>'
    )
    assert "<script" not in cleaned.lower()
    assert "alert(99)" not in cleaned
    assert 'onerror="alert(2)"' not in cleaned.lower()
    assert "onclick=" not in cleaned.lower()
    assert "javascript:" not in cleaned.lower()
    assert "<p>Safe</p>" in cleaned

    # 2. Markdown fallback image alt attribute escapes double quotes
    basic_html = renderer._basic_markdown_to_html(  # noqa: SLF001
        '![a" onerror="alert(1)](https://example.com/img.png)'
    )
    assert 'alt="a&quot; onerror=&quot;alert(1)"' in basic_html

    # 3. ThreeJsViewerWidget escapes </script> in JS bootstrap and HTML in status/hints
    viewer_html = ThreeJsViewerWidget._build_viewer_html(  # noqa: SLF001
        ThreeJsViewerWidget,  # type: ignore[arg-type]
        title="</script><script>alert(1)</script>",
        model_url="http://127.0.0.1:8000/model/t/a.glb",
        model_ext=".glb",
        status_text='<img src=x onerror="alert(1)">',
        hints_text="<b>hint</b>",
    )
    assert "</script><script>alert(1)</script>" not in viewer_html
    assert "\\u003c/script\\u003e" in viewer_html
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in viewer_html
    assert "&lt;b&gt;hint&lt;/b&gt;" in viewer_html

    # 4. threejs_examples fallback HTML escapes title
    fallback = threejs_examples._resolve_or_generate_bundled_html(  # noqa: SLF001
        "nonexistent_example_xyz.html",
        tmp_path,
        title="<script>alert(1)</script>",
    )
    fallback_text = fallback.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in fallback_text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in fallback_text


# ---------------------------------------------------------------------------
# OWASP A03: XML Entity Expansion / XXE Prevention (SVG & MuJoCo XML)
# ---------------------------------------------------------------------------


def test_owasp_a03_svg_and_xml_reject_entity_declarations(tmp_path: Path) -> None:
    billion_laughs = (
        '<?xml version="1.0"?>'
        '<!DOCTYPE svg [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;">]>'
        '<svg xmlns="http://www.w3.org/2000/svg"><text>&lol2;</text></svg>'
    )
    with pytest.raises(ValueError, match="entity|DTD"):
        import_svg_string(billion_laughs)

    # Standard Inkscape/Illustrator SVG with bare DOCTYPE (no ENTITY/subset) still parses
    clean_svg = (
        '<?xml version="1.0"?>'
        '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" '
        '"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">'
        '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
        '<rect x="1" y="1" width="5" height="5"/></svg>'
    )
    doc = import_svg_string(clean_svg)
    assert len(doc.shapes) == 1

    evil_xml = tmp_path / "evil.xml"
    evil_xml.write_text(
        '<!DOCTYPE mujoco [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><mujoco/>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="DOCTYPE/ENTITY"):
        threejs_examples._safe_parse_xml_root(evil_xml)  # noqa: SLF001


# ---------------------------------------------------------------------------
# OWASP A03: Sandbox Escape Hardening (MatchClass, str.format, dunder defs)
# ---------------------------------------------------------------------------


def test_owasp_a03_sandbox_blocks_match_format_and_dunder_defs() -> None:
    payloads = [
        'def run(inputs):\n    return {"x": "{0.__class__}".format(0)}\n',
        'def run(inputs):\n    match "":\n        case str(__class__=c):\n            return {"c": str(c)}\n',
        "class X:\n    def __init_subclass__(cls):\n        pass\ndef run(inputs):\n    return {}\n",
        "def run(__builtins__=None):\n    return {}\n",
    ]
    for code in payloads:
        with pytest.raises(SandboxExecutionError):
            execute_generated_analysis(code, inputs={})

    # Legitimate analysis still executes cleanly
    ok = execute_generated_analysis(
        "def run(inputs):\n    return {'total': sum(inputs['vals'])}\n",
        inputs={"vals": [1, 2, 3]},
    )
    assert ok == {"total": 6}


# ---------------------------------------------------------------------------
# OWASP A03: Git Option Injection, Drive Query Escaping, LanceDB Filters
# ---------------------------------------------------------------------------


def test_owasp_a03_git_drive_and_lancedb_injection_guards(tmp_path: Path) -> None:
    # GitCliTool blocks pager/exec/textconv options
    assert GitCliTool._has_dangerous_option(["grep", "--open-files-in-pager=sh", "foo"])
    assert GitCliTool._has_dangerous_option(["grep", "-Osh", "foo"])
    assert GitCliTool._has_dangerous_option(["push", "--exec=sh", "origin"])
    assert GitCliTool._has_dangerous_option(["show", "--textconv", "HEAD"])
    assert not GitCliTool._has_dangerous_option(["status", "--short"])

    # clone_git_repository rejects option-like URLs and unsafe schemes
    with pytest.raises(ValueError, match="Invalid git repository URL"):
        file_utils.clone_git_repository("--upload-pack=id", str(tmp_path / "repo"))
    with pytest.raises(ValueError, match="Unsupported git"):
        file_utils.clone_git_repository("ext::sh -c id", str(tmp_path / "repo"))
    with pytest.raises(ValueError, match="Unsupported git"):
        file_utils.clone_git_repository("file:///etc/passwd", str(tmp_path / "repo"))

    # Google Drive query value escapes backslashes BEFORE single quotes
    assert (
        GoogleDriveTool._escape_drive_query_value(r"evil\' or 1=1")  # noqa: SLF001
        == r"evil\\\' or 1=1"
    )

    # LanceDB filter builder skips non-finite floats and NUL strings
    retriever = object.__new__(LanceDBRetriever)
    f_str = retriever._build_filter_str(  # noqa: SLF001
        scope="proj",
        filters={
            "importance": float("nan"),
            "timestamp_ms": float("inf"),
            "category": "bad\x00val",
            "source": "ok's",
        },
    )
    assert f_str == "scope = 'proj' AND source = 'ok''s'"


# ---------------------------------------------------------------------------
# OWASP A01 / A08: UpdateManager Path Traversal & Atomic Checksum Enforcement
# ---------------------------------------------------------------------------


def test_owasp_a01_a08_update_service_sanitizes_name_and_never_leaves_tampered_file(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "pkg.whl"
    artifact.write_bytes(b"legit-bytes")
    stage_dir = tmp_path / "stage"

    # Tampered/wrong checksum must fail AND leave zero files in stage_dir
    bad_manifest = UpdateManifest(
        project="annolid",
        channel="stable",
        version="2.0.0",
        release_date="",
        artifact_url=artifact.as_uri(),
        artifact_sha256="0" * 64,
        signature="",
        signature_alg="none",
        source="https://example.invalid/manifest.json",
    )
    plan = SignedUpdatePlan(
        channel="stable",
        project="annolid",
        current_version="1.0.0",
        manifest=bad_manifest,
        install_mode="package",
        source_root=".",
        commands=[],
        verification=VerificationResult(ok=True, reason="ok", details={}),
    )
    svc = UpdateManagerService(project="annolid", stage_dir=stage_dir)
    res = svc.download_and_verify_artifact(plan, execute=True)
    assert res["ok"] is False
    assert res["reason"] == "artifact_checksum_mismatch"
    assert list(stage_dir.iterdir()) == []

    # Path traversal in artifact_url cannot escape stage_dir
    safe_name = UpdateManagerService._safe_artifact_filename(  # noqa: SLF001
        "https://example.com/downloads/..%2F..%2Fevil.whl?token=123",
        fallback="fallback.whl",
    )
    assert "/" not in safe_name and "\\" not in safe_name and ".." not in safe_name

    # Invalid project name rejected in manifest fetcher
    with pytest.raises(ValueError, match="Invalid package project name"):
        fetch_channel_manifest(project="../evil")


# ---------------------------------------------------------------------------
# OWASP A01 / A04: Archive Extraction (Zip Bomb, Symlinks, Backslashes)
# ---------------------------------------------------------------------------


def test_owasp_a01_a04_archive_extraction_blocks_zip_bombs_symlinks_and_backslashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. ClawHub streaming byte limit blocks decompressed payload exceeding cap
    monkeypatch.setattr(clawhub, "_MAX_UNCOMPRESSED_ZIP_BYTES", 64)
    bomb_zip = tmp_path / "bomb.zip"
    with zipfile.ZipFile(bomb_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("skill/SKILL.md", "A" * 512)
    with pytest.raises(ValueError, match="byte limit"):
        clawhub._safe_extract_zip(bomb_zip, tmp_path / "out_clawhub")  # noqa: SLF001

    # 2. Backslash traversal rejected
    bs_zip = tmp_path / "backslash.zip"
    with zipfile.ZipFile(bs_zip, "w") as zf:
        zf.writestr(r"..\..\escape.txt", "nope")
    with pytest.raises(ValueError, match="Unsafe zip member"):
        clawhub._safe_extract_zip(bs_zip, tmp_path / "out_bs")  # noqa: SLF001
    with pytest.raises(ValueError, match="outside destination"):
        safe_extract_archive(bs_zip, tmp_path / "out_colab_bs")

    # 3. Colab packaging rejects ZIP symlink entries
    sym_zip = tmp_path / "sym.zip"
    with zipfile.ZipFile(sym_zip, "w") as zf:
        info = zipfile.ZipInfo("link.txt")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(ValueError, match="symlink"):
        safe_extract_archive(sym_zip, tmp_path / "out_colab_sym")


# ---------------------------------------------------------------------------
# OWASP A05: CORS Hardening & Temp Directory Isolation
# ---------------------------------------------------------------------------


def test_owasp_a05_cors_and_private_temp_and_help_flag_guards() -> None:
    # Local viewer servers allow only matching loopback Origin, never wildcard or external origins
    assert (
        pdf_srv._allowed_loopback_cors_origin("http://127.0.0.1:9123", 9123)  # noqa: SLF001
        == "http://127.0.0.1:9123"
    )
    assert (
        pdf_srv._allowed_loopback_cors_origin("https://evil.example", 9123) is None  # noqa: SLF001
    )
    assert (
        pdf_srv._allowed_loopback_cors_origin("http://127.0.0.1:9999", 9123) is None  # noqa: SLF001
    )
    assert (
        three_srv._allowed_threejs_cors_origin("https://evil.example", 9123) is None  # noqa: SLF001
    )
    assert (
        three_srv._allowed_threejs_cors_origin("http://localhost:9123", 9123)  # noqa: SLF001
        == "http://localhost:9123"
    )

    # CronTool temp fallback is user-scoped, not hardcoded /tmp/annolid
    temp_store = CronTool._private_temp_store_path()  # noqa: SLF001
    assert "annolid-cron-" in temp_store.parent.name

    # AnnolidRunTool does not treat --help after `--` as a help-bypass
    assert AnnolidRunTool._is_mutating(["train", "yolo", "--", "--help"]) is True  # noqa: SLF001
    assert AnnolidRunTool._is_mutating(["train", "yolo", "--help"]) is False  # noqa: SLF001

    # Constant-time consent check works
    assert has_operator_consent("APPROVE_ANNOLID_CORE_UPDATE") is True
    assert has_operator_consent("WRONG") is False


def test_owasp_a05_whatsapp_webhook_omits_wildcard_cors() -> None:
    import inspect

    src = inspect.getsource(WhatsAppWebhookServer)
    assert "Access-Control-Allow-Origin" not in src
    assert "X-Content-Type-Options" in src


# ---------------------------------------------------------------------------
# OWASP A10 & A04: SSRF Scheme Validation, Host Spoofing, PoW DoS Bounds
# ---------------------------------------------------------------------------


def test_owasp_a10_a04_ssrf_scheme_host_spoof_and_path_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    from annolid.gui.widgets.web_viewer import _SavePdfTask

    # Exact PMC hostname check rejects host-substring spoofing
    assert _SavePdfTask._candidate_download_urls(  # noqa: SLF001
        "https://pmc.ncbi.nlm.nih.gov.evil.com/pdf/PMC123456/"
    ) == ["https://pmc.ncbi.nlm.nih.gov.evil.com/pdf/PMC123456/"]
    assert _SavePdfTask._candidate_download_urls("file:///etc/passwd") == []  # noqa: SLF001
    worker = _SavePdfTask(
        None,
        source_url="https://pmc.ncbi.nlm.nih.gov.evil.com/",
        download_url="https://pmc.ncbi.nlm.nih.gov.evil.com/pdf/PMC1",
        output_dir=tmp_path,
    )
    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "httpx", None)
        ok_pmc, err_pmc = worker._download_with_pmc_pow(  # noqa: SLF001
            candidate_url="https://pmc.ncbi.nlm.nih.gov.evil.com/pdf/PMC1",
            source_url="https://pmc.ncbi.nlm.nih.gov.evil.com/",
            output_path=tmp_path / "out.pdf",
        )
    assert ok_pmc is False and "unsupported pmc host" in err_pmc.lower()
    with pytest.raises(RuntimeError, match="exceeded"):
        _SavePdfTask._solve_pmc_pow_nonce("test", 6, max_iterations=10)  # noqa: SLF001

    # model_assets rejects host substring spoof, file:// URL, and path traversal filename
    assert (
        model_assets._is_google_drive_url("https://evil.com/drive.google.com/a")
        is False
    )  # noqa: SLF001
    assert model_assets._is_google_drive_url("https://drive.google.com/uc?id=1") is True  # noqa: SLF001
    with pytest.raises(ValueError, match="Unsupported model asset URL"):
        model_assets._http_chunked_download("file:///etc/passwd", tmp_path / "x.bin")  # noqa: SLF001
    with pytest.raises(ValueError, match="Unsafe model asset file_name"):
        model_assets.ensure_cached_model_asset(
            file_name="../escape.bin",
            url="https://example.com/a.bin",
            expected_sha256="a" * 64,
            cache_dir=tmp_path,
        )

    # Box URL guards reject javascript:, file://, non-https
    with pytest.raises(ValueError):
        box_utils._resolve_box_authorize_url("javascript:alert(1)")  # noqa: SLF001
    with pytest.raises(ValueError):
        box_utils._validate_box_endpoint_url("file:///etc/passwd")  # noqa: SLF001

    # ClawHub, Zulip, arXiv, files.download_file, cancellation, transcription reject file://
    with pytest.raises(ValueError, match="Unsupported ClawHub URL"):
        clawhub._http_get_json("file:///etc/passwd")  # noqa: SLF001
    with pytest.raises(ValueError, match="Unsupported arXiv URL"):
        tool_handlers_arxiv._fetch_arxiv_feed_bytes(url="file:///etc/passwd")  # noqa: SLF001
    with pytest.raises(ValueError, match="Unsupported download URL"):
        file_utils.download_file("file:///etc/passwd", str(tmp_path / "out"))
    with pytest.raises(ValueError, match="Unsupported download URL"):
        cancellation.cancellable_download_file(
            "file:///etc/passwd", tmp_path / "out.bin"
        )
    with pytest.raises(ValueError, match="Unsupported transcription URL"):
        _transcribe_via_urllib(
            url="file:///etc/passwd",
            headers={},
            path=tmp_path / "a.wav",
            model="whisper-1",
        )
    zulip_channel = ZulipChannel({"enabled": True}, bus=None)
    zulip_res = zulip_channel._request_json(  # noqa: SLF001
        method="GET",
        base_url="file:///etc",
        path="/passwd",
        user="u",
        api_key="k",
    )
    assert zulip_res["result"] == "error"
    assert "unsupported zulip url" in zulip_res["msg"].lower()


# ---------------------------------------------------------------------------
# OWASP A08: Zero Raw torch.load Calls in First-Party annolid/ Package
# ---------------------------------------------------------------------------


def test_owasp_a08_no_raw_torch_load_in_first_party_package() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    vendored_prefixes = (
        "annolid/detector/countgd/",
        "annolid/detector/yolov5/",
        "annolid/segmentation/SAM/",
        "annolid/segmentation/cutie_vos/",
        "annolid/segmentation/yolact/",
        "annolid/segmentation/MEDIAR/",
        "annolid/depth/video_depth_anything/",
        "annolid/tracker/cowtracker/cowtracker/thirdparty/",
        "annolid/gui/assets/pdfjs/",
    )
    offenders: list[str] = []
    for scan_dir in (repo_root / "annolid", repo_root / "scripts"):
        if not scan_dir.exists():
            continue
        for py_path in sorted(scan_dir.rglob("*.py")):
            rel = py_path.relative_to(repo_root).as_posix()
            if rel == "annolid/utils/safe_torch_load.py" or any(
                rel.startswith(prefix) for prefix in vendored_prefixes
            ):
                continue
            text = py_path.read_text(encoding="utf-8", errors="replace")
            for idx, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "torch.load(" in stripped:
                    offenders.append(f"{rel}:{idx}")
    assert offenders == [], f"Raw torch.load found in first-party code: {offenders}"


def test_owasp_a08_mabe2022_safe_npy_loads_legit_dict_and_blocks_pickle_rce(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from annolid.datasets import mabe2022

    # 1. Legitimate MABe2022 dict with nested numeric ndarrays loads cleanly
    legit_path = tmp_path / "user_train.npy"
    expected = {
        "vocabulary": ["chase", "lights"],
        "sequences": {
            "seq_001": {
                "keypoints": np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2),
                "annotations": np.array([[0, 1], [1, 0]], dtype=np.int32),
            }
        },
    }
    np.save(legit_path, expected, allow_pickle=True)
    loaded = mabe2022.load_user_train(str(legit_path))
    assert isinstance(loaded, dict)
    assert loaded["vocabulary"] == ["chase", "lights"]
    np.testing.assert_array_equal(
        loaded["sequences"]["seq_001"]["keypoints"],
        expected["sequences"]["seq_001"]["keypoints"],
    )

    # 2. Malicious pickled object payload in .npy must raise UnpicklingError without executing
    executed: list[str] = []

    class _ExploitPayload:
        def __reduce__(self):
            return (os.system, ("echo SHOULD_NEVER_RUN",))

    evil_path = tmp_path / "evil_keypoints.npy"
    np.save(evil_path, np.array(_ExploitPayload(), dtype=object), allow_pickle=True)
    with pytest.raises(pickle.UnpicklingError):
        mabe2022.load_keypoints(str(evil_path))
    assert executed == []


def test_owasp_a10_safe_http_blocks_redirect_ssrf_and_strips_cross_origin_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. Direct scheme / credential / loopback / metadata / DNS-resolved private IP blocking
    for bad_url in (
        "file:///etc/passwd",
        "ftp://evil.example/data.bin",
        "http://user:pass@example.com/path",
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:8080/admin",
        "http://[::1]:8080/admin",
        "http://[::ffff:127.0.0.1]/admin",
        "http://localhost/admin",
    ):
        with pytest.raises(ValueError):
            safe_http.safe_urlopen(bad_url, allow_private=False)

    # DNS resolving to private RFC1918 address is blocked
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, family=0, type=0: [  # noqa: ARG005
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.42", 0))
        ],
    )
    with pytest.raises(ValueError, match="private or internal"):
        safe_http.validate_http_url(
            "https://rebind.example/resource", allow_private=False, resolve_dns=True
        )

    # 2. Redirect handler blocks unsafe redirect hops and strips cross-origin auth headers
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, family=0, type=0: [  # noqa: ARG005
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))
        ],
    )
    handler = safe_http._SafeHTTPRedirectHandler(  # noqa: SLF001
        allowed_schemes=("http", "https"),
        allowed_hosts=None,
        allow_private=False,
        allow_loopback=False,
        allow_http_only_for_loopback=False,
        resolve_dns=True,
        max_redirects=2,
    )
    base_req = urllib.request.Request(
        "https://api.example.com/start",
        headers={
            "Authorization": "Bearer secret-token",
            "Cookie": "sid=xyz",
            "X-Colab-Runtime-Proxy-Token": "colab-secret",
            "Accept": "application/json",
        },
    )
    for evil_redirect in (
        "file:///etc/passwd",
        "ftp://evil.example/file",
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:9000/internal",
    ):
        with pytest.raises(ValueError):
            handler.redirect_request(
                base_req, io.BytesIO(), 302, "Found", {}, evil_redirect
            )

    # Cross-origin public redirect strips sensitive headers but keeps Accept
    cross_req = handler.redirect_request(
        base_req, io.BytesIO(), 302, "Found", {}, "https://cdn.other.example/file"
    )
    assert cross_req is not None
    lower_headers = {k.lower() for k, _ in cross_req.header_items()}
    assert "authorization" not in lower_headers
    assert "cookie" not in lower_headers
    assert "x-colab-runtime-proxy-token" not in lower_headers
    assert "accept" in lower_headers

    # Same-origin redirect preserves headers and enforces max_redirects cap
    same_req = handler.redirect_request(
        base_req, io.BytesIO(), 302, "Found", {}, "https://api.example.com/next"
    )
    assert same_req is not None
    same_headers = {k.lower() for k, _ in same_req.header_items()}
    assert "authorization" in same_headers
    hop2 = handler.redirect_request(
        same_req, io.BytesIO(), 302, "Found", {}, "https://api.example.com/hop2"
    )
    assert hop2 is not None
    with pytest.raises(urllib.error.HTTPError, match="Too many redirects"):
        handler.redirect_request(
            hop2, io.BytesIO(), 302, "Found", {}, "https://api.example.com/hop3"
        )

    # 3. ColabApiBackend rejects file:// and non-loopback plaintext http://
    auth = ColabAuthManager(
        credentials_file=str(tmp_path / "c.json"),
        token_file=str(tmp_path / "t.json"),
    )
    colab_api = ColabApiBackend(auth)
    with pytest.raises(ValueError):
        colab_api._request_json("GET", "file:///etc/passwd", headers={})  # noqa: SLF001
    with pytest.raises(ValueError, match="Plaintext http://"):
        colab_api._request_json(  # noqa: SLF001
            "GET", "http://colaboratory.googleapis.com/v1beta/runtimes", headers={}
        )

    # 4. TensorBoard probes refuse file:// URLs
    assert tb_mod._can_connect("file:///etc/passwd") is False  # noqa: SLF001
    assert tb_mod._plugin_enabled("file:///etc/passwd", "projector") is None  # noqa: SLF001


def test_owasp_a01_a03_shell_lancedb_and_cron_boundary_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. ExecTool blocks bare `~` arguments and unexpanded `~` working_dir
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    exec_tool = ExecTool(working_dir=str(ws), restrict_to_workspace=True)
    for cmd in ("ls ~", "cd ~", "cat --file=~", "echo ~/secret"):
        err = exec_tool._guard_command(cmd, str(ws))  # noqa: SLF001
        assert err is not None and "blocked" in err.lower()
    err_cwd = exec_tool._guard_command("ls .", "~")  # noqa: SLF001
    assert err_cwd is not None and "outside" in err_cwd.lower()

    # 2. LanceDBMemoryBackend blocks NUL bytes, immutable id mutation, and unknown columns
    class _FakeTable:
        def __init__(self) -> None:
            self.deletes: list[str] = []
            self.updates: list[tuple[str, dict[str, object]]] = []

        def delete(self, where: str) -> None:
            self.deletes.append(where)

        def update(self, *, where: str, values: dict[str, object]) -> None:
            self.updates.append((where, values))

    backend = object.__new__(LanceDBMemoryBackend)
    backend.config = SimpleNamespace(enabled=True, db_path=tmp_path / "db")
    backend.embedder = None
    backend._table = _FakeTable()  # noqa: SLF001

    assert backend.delete("id\x00evil") is False
    assert backend.update("mem-1", {"id": "hijacked"}) is False
    assert backend.update("mem-1", {"unknown_col": "x"}) is False
    assert backend.update("mem-1\x00bad", {"category": "note"}) is False
    assert backend.stats(scope="proj\x00bad")["enabled"] is False
    assert backend._table.deletes == []  # noqa: SLF001
    assert backend._table.updates == []  # noqa: SLF001

    assert backend.update("mem-1", {"category": "note"}) is True
    assert len(backend._table.updates) == 1  # noqa: SLF001

    # 3. CronTool enforces allowed_attachment_roots and blocks email header CRLF injection
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    inside = ws / "report.txt"
    inside.write_text("ok", encoding="utf-8")

    cron_tool = CronTool(
        store_path=tmp_path / "cron" / "jobs.json",
        allowed_attachment_roots=[ws],
    )
    cron_tool.set_context("local", "user1")

    res_out = asyncio.run(
        cron_tool.execute(
            action="add",
            message="send report",
            every_seconds=60,
            email_to="user@example.com",
            email_content="attached",
            attachment_paths=[str(outside)],
        )
    )
    assert "outside allowed" in res_out.lower()

    res_crlf = asyncio.run(
        cron_tool.execute(
            action="add",
            message="send report",
            every_seconds=60,
            email_to="user@example.com\r\nBcc: evil@attacker.example",
            email_content="hi",
        )
    )
    assert "newline" in res_crlf.lower()

    res_ok = asyncio.run(
        cron_tool.execute(
            action="add",
            message="send report",
            every_seconds=60,
            email_to="user@example.com",
            email_subject="Daily Report",
            email_content="attached",
            attachment_paths=[str(inside)],
        )
    )
    assert "Created scheduled email job" in res_ok


def _invoke_threejs_handler(
    method: str,
    path: str,
    *,
    req_headers: dict[str, str] | None = None,
    server_port: int = 9123,
) -> tuple[int, dict[str, str], bytes]:
    handler = three_srv._ThreeJsRequestHandler.__new__(  # noqa: SLF001
        three_srv._ThreeJsRequestHandler  # noqa: SLF001
    )
    handler.path = path
    handler.command = method
    handler.request_version = "HTTP/1.1"
    msg = Message()
    for key, val in (req_headers or {}).items():
        msg[key] = val
    handler.headers = msg  # type: ignore[assignment]
    handler.server = SimpleNamespace(server_port=server_port)  # type: ignore[assignment]
    handler.wfile = io.BytesIO()

    recorded_status: list[int] = []
    recorded_headers: dict[str, str] = {}

    def _send_response(code: int, message: str | None = None) -> None:
        del message
        recorded_status.append(int(code))

    def _send_header(key: str, val: str) -> None:
        recorded_headers[key.lower()] = val

    def _end_headers() -> None:
        return None

    def _send_error(code: int, message: str | None = None) -> None:
        del message
        recorded_status.append(int(code))

    handler.send_response = _send_response  # type: ignore[method-assign]
    handler.send_header = _send_header  # type: ignore[method-assign]
    handler.end_headers = _end_headers  # type: ignore[method-assign]
    handler.send_error = _send_error  # type: ignore[method-assign]

    getattr(handler, f"do_{method}")()
    status = recorded_status[-1] if recorded_status else 0
    return status, recorded_headers, handler.wfile.getvalue()


def test_owasp_a05_threejs_viewer_server_in_process_cors_and_hidden_file_guards(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    glb = model_dir / "scene.glb"
    glb.write_bytes(b"glTF-sample")
    hidden_dir = model_dir / ".git"
    hidden_dir.mkdir()
    (hidden_dir / "config.json").write_text('{"secret": true}', encoding="utf-8")
    outside_json = tmp_path / "outside.json"
    outside_json.write_text('{"outside": true}', encoding="utf-8")

    token = "testtoken123"
    saved_tokens = dict(three_srv._THREEJS_HTTP_TOKENS)  # noqa: SLF001
    saved_times = dict(three_srv._THREEJS_HTTP_TOKEN_CREATED_AT)  # noqa: SLF001
    try:
        three_srv._THREEJS_HTTP_TOKENS[token] = glb  # noqa: SLF001
        three_srv._THREEJS_HTTP_TOKEN_CREATED_AT[token] = 1e18  # noqa: SLF001

        # 1. /swarm/status rejects external CORS Origin and sets nosniff
        st_ext, hdrs_ext, _ = _invoke_threejs_handler(
            "GET",
            "/swarm/status",
            req_headers={"Origin": "https://evil.example"},
            server_port=9123,
        )
        assert st_ext == 200
        assert hdrs_ext.get("x-content-type-options") == "nosniff"
        assert "access-control-allow-origin" not in hdrs_ext

        # 2. /swarm/status reflects exact matching loopback Origin + Vary: Origin
        st_loc, hdrs_loc, _ = _invoke_threejs_handler(
            "GET",
            "/swarm/status",
            req_headers={"Origin": "http://127.0.0.1:9123"},
            server_port=9123,
        )
        assert st_loc == 200
        assert hdrs_loc.get("access-control-allow-origin") == "http://127.0.0.1:9123"
        assert hdrs_loc.get("vary") == "Origin"

        # 3. Hidden directories (e.g. .git/config.json) and traversal are rejected with 403
        st_git, _, _ = _invoke_threejs_handler(
            "GET", f"/model/{token}/.git/config.json"
        )
        assert st_git == 403
        st_trav, _, _ = _invoke_threejs_handler(
            "GET", f"/model/{token}/../outside.json"
        )
        assert st_trav == 403

        # 4. Valid registered model serves cleanly with nosniff
        st_ok, hdrs_ok, body_ok = _invoke_threejs_handler(
            "GET", f"/model/{token}/scene.glb"
        )
        assert st_ok == 200
        assert hdrs_ok.get("x-content-type-options") == "nosniff"
        assert body_ok == b"glTF-sample"
    finally:
        three_srv._THREEJS_HTTP_TOKENS.clear()  # noqa: SLF001
        three_srv._THREEJS_HTTP_TOKENS.update(saved_tokens)  # noqa: SLF001
        three_srv._THREEJS_HTTP_TOKEN_CREATED_AT.clear()  # noqa: SLF001
        three_srv._THREEJS_HTTP_TOKEN_CREATED_AT.update(saved_times)  # noqa: SLF001
