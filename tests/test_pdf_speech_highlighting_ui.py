from __future__ import annotations

from email.message import Message
import io
import json
import math
from pathlib import Path
import subprocess
import time

import pytest
from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.widgets.pdf_manager import PdfManager
from annolid.gui.widgets.pdf_reader_controls import PdfReaderControlsWidget
from annolid.gui.widgets.pdf_speech_controller import _SpeakTextTask
from annolid.gui.widgets.pdf_user_state import load_pdf_state, save_pdf_state
from annolid.gui.widgets.pdf_viewer_bridge import _PdfReaderBridge, _SpeakToken
from annolid.gui.widgets.pdf_viewer_impl import PdfViewerWidget
from annolid.gui.widgets.pdf_viewer_server import (
    _ALLOWED_PDFJS_ASSETS,
    _MAX_HTTP_PDF_TOKENS,
    _MAX_HTTP_VIEWER_TOKENS,
    _PDFJS_HTTP_TOKENS,
    _PDFJS_HTTP_VIEWERS,
    _PdfJsHttpRequestHandler,
    _pdfjs_asset_path,
    _resolve_pdf_byte_range,
    _store_pdfjs_http_token,
    _store_pdfjs_http_viewer_html,
)

_JSC_PATH = Path(
    "/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc"
)
_SPEECH_JS = Path("annolid/gui/assets/pdfjs/annolid_pdf_speech.js")
_VIEWER_JS = Path("annolid/gui/assets/pdfjs/annolid_viewer.js")


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _RecordingPage:
    def __init__(self) -> None:
        self.scripts: list[str] = []

    def runJavaScript(self, script: str, callback=None) -> None:  # noqa: N802
        self.scripts.append(script)
        if callable(callback):
            callback({})


class _RecordingWebView:
    def __init__(self) -> None:
        self._page = _RecordingPage()

    def page(self) -> _RecordingPage:
        return self._page


@pytest.mark.skipif(not _JSC_PATH.exists(), reason="JavaScriptCore jsc unavailable")
def test_js_pdf_speech_layout_sentence_and_keyword_logic() -> None:
    script = f"""
load({json.dumps(str(_SPEECH_JS.resolve()))});
const S = AnnolidPdfSpeech;

// 1. Academic & CJK sentence splitting
const ranges = S.splitTextIntoSentenceRanges(
  "As in Smith et al., 2024 (Fig. 2, p < 0.05), value is 3.14. Second sentence!"
);
if (ranges.length !== 2) throw new Error("Expected 2 sentences, got " + ranges.length);

// 2. Paragraph splitting regression: gapLimit must use LAST line height, not cumulative paragraph height!
const lines = [];
for (let i = 0; i < 6; i++) {{
  const y = 10 + i * 14;
  lines.push({{ text: "Para one line " + i + ".", spans: [i], xMin: 50, xMax: 250, yMin: y, yMax: y + 12, h: 12, yCenter: y + 6 }});
}}
// Normal paragraph break of 18px after 6 lines (cumulative height was 82px; old bug merged this!)
lines.push({{ text: "Para two starts here.", spans: [6], xMin: 50, xMax: 250, yMin: 110, yMax: 122, h: 12, yCenter: 116 }});
const paras = S.linesToParagraphs(lines, 1, true);
if (paras.length !== 2) throw new Error("Expected 2 paragraphs, got " + paras.length);

// 3. Two-column textContent extraction must NOT merge left & right columns on the same Y!
const items = [
  // Full-width title at top (y=740)
  {{ str: "Full Width Paper Title Spanning Page", transform: [1,0,0,14,50,740], width: 480, height: 14 }},
];
for (let row = 0; row < 6; row++) {{
  const y = 680 - row * 16;
  items.push({{ str: "Left col line " + row, transform: [1,0,0,10,50,y], width: 180, height: 10 }});
  items.push({{ str: "Right col line " + row, transform: [1,0,0,10,320,y], width: 180, height: 10 }});
}}
// Full-width bottom caption (y=500)
items.push({{ str: "Figure 1 full width bottom caption across both columns", transform: [1,0,0,10,50,500], width: 480, height: 10 }});
const extracted = S.extractLinesFromTextContent({{ items }});
const ordered = S.orderLinesForReading(extracted, 600);
if (ordered[0].text.indexOf("Full Width Paper Title") < 0) {{
  throw new Error("Title should come first, got: " + ordered[0].text);
}}
const firstRightIdx = ordered.findIndex((l) => l.text.indexOf("Right col") >= 0);
const lastLeftIdx = ordered.map((l) => l.text.indexOf("Left col") >= 0).lastIndexOf(true);
if (!(lastLeftIdx > 0 && firstRightIdx > lastLeftIdx)) {{
  throw new Error("Expected all left column lines before right column: " + JSON.stringify(ordered.map((l) => l.text)));
}}
if (ordered[ordered.length - 1].text.indexOf("Figure 1") < 0) {{
  throw new Error("Bottom wide caption should come last, got: " + ordered[ordered.length - 1].text);
}}

// 4. Mid-span sentence splitting preserves exact spanSlices char offsets
const sents = S.splitParagraphEntriesIntoSentences(
  [{{ idx: 7, rawText: "First sentence ends. Second begins here." }}],
  1
);
if (sents.length !== 2) throw new Error("Expected 2 split sentences from single span, got " + sents.length);
if (sents[0].spanSlices[0].startOffset !== 0 || sents[0].spanSlices[0].endOffset >= 25) {{
  throw new Error("Bad first slice: " + JSON.stringify(sents[0].spanSlices));
}}
if (sents[1].spanSlices[0].startOffset < 20) {{
  throw new Error("Bad second slice: " + JSON.stringify(sents[1].spanSlices));
}}

// 5. Boundary-safe keyword matching + CJK similarity
const kw = S.findKeywordRanges("strategy tracks rat behavior and 多巴胺", ["rat", "多巴胺"]);
if (kw.length !== 2 || kw[0].keyword !== "rat" || kw[1].keyword !== "多巴胺") {{
  throw new Error("Unexpected keyword matches: " + JSON.stringify(kw));
}}
const cjkSim = S.scoreTextSimilarity("小鼠在开放场地中表现出探索行为", "小鼠在开放场地中表现探索行为");
if (cjkSim < 0.7) throw new Error("Expected high CJK similarity, got " + cjkSim);

// 6. Weighted timing sums exactly to durationMs
const toks = S.tokenizeWords("First, dopamine rises quickly.", ["dopamine"]);
const sched = S.buildWordTimingSchedule(toks, 1500, 45);
const totalDur = sched.reduce((acc, s) => acc + s.durationMs, 0);
if (totalDur !== 1500) throw new Error("Expected total duration 1500, got " + totalDur);
if (!sched.some((s) => s.isKeyword && s.token.text === "dopamine")) {{
  throw new Error("Expected dopamine flagged as keyword");
}}

print("JS_SPEECH_TESTS_OK");
"""
    proc = subprocess.run(
        [str(_JSC_PATH), "-e", script],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "JS_SPEECH_TESTS_OK" in proc.stdout


def test_viewer_js_escapes_notes_html_and_removes_dead_column_helper() -> None:
    source = _VIEWER_JS.read_text(encoding="utf-8")
    assert "function _annolidGroupIntoColumns(" not in source
    assert "const safeTitle = _annolidEscapeHtml(" in source
    assert "const title = _annolidEscapeHtml(" in source
    assert "const sub = _annolidEscapeHtml(" in source
    assert "window.__annolidStartSpeechHighlight = async function" in source
    assert "window.__annolidSetKeywordHighlightConfig = function" in source


def test_pdf_viewer_speech_session_guard_ignores_stale_callbacks() -> None:
    _app()
    widget = PdfViewerWidget.__new__(PdfViewerWidget)
    widget._speaking = True
    widget._speak_session_id = 2
    widget._highlight_mode = "web-sentence"
    widget._web_view = _RecordingWebView()
    widget._reader_chunk_base = 0
    widget._reader_queue_offset = 0
    widget._reader_queue = ["Active sentence."]
    widget._reader_pages = [1]
    widget._reader_total = 1
    widget._reader_state = "reading"
    widget._web_sentence_span_groups = [[5]]
    widget._web_sentence_span_slices = []
    widget._highlight_keywords = ["Active"]
    widget._keyword_highlighting_enabled = True
    widget._auto_keyword_highlighting_enabled = False
    widget._word_highlight_timer = None
    widget._word_highlight_units = []
    widget._word_highlight_durations_ms = []
    widget._word_highlight_schedule = []
    widget._word_highlight_index = 0
    widget._word_highlight_active_idx = -1
    widget._word_highlight_started_at = 0.0
    widget._active_speak_token = None
    widget._reader_pending_restart = None
    widget._reader_stop_requested = False
    widget._reader_pause_requested = False

    # Stale finish & chunk callbacks from session 1 must be completely ignored.
    widget._on_speak_chunk_for_session(1, 0, 500)
    widget._on_speak_finished_for_session(1)
    assert widget._speaking is True
    assert widget._web_view.page().scripts == []

    # Current session 2 callback triggers unified highlight with keywords + durationMs.
    widget._on_speak_chunk_for_session(2, 0, 900)
    scripts = widget._web_view.page().scripts
    assert any("__annolidStartSpeechHighlight" in s for s in scripts)
    assert any('"durationMs": 900' in s for s in scripts)
    assert any('"keywords": ["Active"]' in s for s in scripts)


def test_speak_web_selection_resets_stale_reader_chunk_base() -> None:
    _app()
    widget = PdfViewerWidget.__new__(PdfViewerWidget)
    widget._speaking = False
    widget._current_page = 2
    widget._reader_chunk_base = 7  # Stale offset from prior click-to-read
    widget._reader_queue_offset = 4
    widget._reader_queue = ["old"] * 10
    widget._reader_pages = [1] * 10
    widget._reader_spans = [[0]] * 10
    widget._web_sentence_span_groups = []
    widget._web_sentence_span_slices = []
    widget._web_selected_span_text = {}
    widget._highlight_mode = None
    spoken: list[tuple[str, list[str] | None]] = []
    widget._speak_text = lambda text, *, chunks=None: spoken.append((text, chunks))

    class _SelectionPage:
        def runJavaScript(self, _script: str, callback) -> None:  # noqa: N802
            callback(
                {
                    "texts": ["Dopamine rises.", "Freezing drops."],
                    "indices": [10, 11],
                }
            )

    class _SelectionWebView:
        def page(self):
            return _SelectionPage()

    widget._web_view = _SelectionWebView()
    widget._speak_web_selection("Dopamine rises. Freezing drops.")

    assert widget._reader_chunk_base == 0
    assert widget._reader_queue_offset == 0
    assert widget._highlight_mode == "web-sentence"
    assert widget._web_sentence_span_groups == [[10], [11]]
    assert spoken == [
        ("Dopamine rises. Freezing drops.", ["Dopamine rises.", "Freezing drops."])
    ]


def test_fallback_text_view_renders_sentence_keyword_and_active_word_highlights() -> (
    None
):
    _app()
    widget = PdfViewerWidget.__new__(PdfViewerWidget)
    QtWidgets.QWidget.__init__(widget)
    widget.text_view = QtWidgets.QTextEdit(widget)
    widget.text_view.setPlainText(
        "Dopamine rises rapidly while freezing drops across trials."
    )
    widget._doc = None
    widget._pdf_path = None
    widget._web_view = None
    widget._web_profile = None
    widget._pdfjs_active = False
    widget._speaking = True
    widget._highlight_mode = "text-sentence"
    widget._text_sentence_spans = [(0, len(widget.text_view.toPlainText()))]
    widget._active_text_sentence_span = None
    widget._active_text_keyword_spans = []
    widget._highlight_keywords = ["dopamine", "freezing"]
    widget._keyword_highlighting_enabled = True
    widget._auto_keyword_highlighting_enabled = False
    widget._word_highlight_timer = None
    widget._word_highlight_units = []
    widget._word_highlight_durations_ms = []
    widget._word_highlight_schedule = []
    widget._word_highlight_index = 0
    widget._word_highlight_active_idx = -1
    widget._word_highlight_started_at = 0.0

    widget._on_speak_chunk(0)
    widget._on_speak_chunk_timing(0, 1200)

    selections = widget.text_view.extraSelections()
    # Includes: 2 page-level keywords + 1 sentence + 2 sentence keywords + 1 active word ("Dopamine")
    assert len(selections) >= 4
    # First spoken word "Dopamine" IS a keyword -> rendered bold with coral highlight.
    last_sel = selections[-1]
    assert last_sel.cursor.selectedText() == "Dopamine"
    assert last_sel.format.fontWeight() == QtGui.QFont.Bold

    widget._stop_word_highlight()
    assert widget._word_highlight_timer is None
    widget.close()


def test_pdf_reader_bridge_page_sync_updates_viewer_counts() -> None:
    _app()
    widget = PdfViewerWidget.__new__(PdfViewerWidget)
    QtWidgets.QWidget.__init__(widget)
    widget._doc = None
    widget._current_page = 0
    widget._web_total_pages = 0
    emitted: list[tuple[int, int]] = []
    widget.page_changed.connect(lambda cur, tot: emitted.append((cur, tot)))

    bridge = _PdfReaderBridge(widget)
    meta = bridge.metaObject()
    assert meta.indexOfSlot("onPageChanged(int,int)") >= 0

    bridge.onPageChanged(5, 18)
    assert widget.current_page_index() == 4
    assert widget.page_count() == 18
    assert emitted == [(4, 18)]
    widget.close()


def test_pdf_reader_controls_keyword_signals_and_state() -> None:
    _app()
    controls = PdfReaderControlsWidget()
    kw_events: list[list[str]] = []
    enabled_events: list[bool] = []
    auto_events: list[bool] = []

    controls.keywords_changed.connect(lambda kws: kw_events.append(list(kws)))
    controls.keyword_highlighting_changed.connect(
        lambda flag: enabled_events.append(bool(flag))
    )
    controls.auto_keywords_changed.connect(lambda flag: auto_events.append(bool(flag)))

    controls.keywords_edit.setText("dopamine, open field, dopamine")
    assert kw_events[-1] == ["dopamine", "open field"]

    controls.auto_keywords_checkbox.setChecked(True)
    assert auto_events[-1] is True

    controls.highlight_keywords_checkbox.setChecked(False)
    assert enabled_events[-1] is False
    assert controls.keywords_edit.isEnabled() is False

    # Programmatic sync must not re-emit signals.
    prev_len = len(kw_events)
    controls.set_keywords(["freezing"])
    assert len(kw_events) == prev_len
    assert controls.keywords_edit.text() == "freezing"
    controls.close()


def test_pdf_viewer_server_whitelists_speech_js_and_bounds_caches(
    tmp_path: Path,
) -> None:
    assert "annolid_pdf_speech.js" in _ALLOWED_PDFJS_ASSETS
    speech_asset = _pdfjs_asset_path("annolid_pdf_speech.js")
    assert speech_asset is not None and speech_asset.is_file()
    assert _pdfjs_asset_path("../pyproject.toml") is None

    # Empty PDFs must produce Content-Length 0 with status 200 (never 1 byte).
    assert _resolve_pdf_byte_range(0, "") == (200, 0, -1, 0)
    assert _resolve_pdf_byte_range(0, "bytes=0-10") == (200, 0, -1, 0)
    assert _resolve_pdf_byte_range(100, "") == (200, 0, 99, 100)
    assert _resolve_pdf_byte_range(100, "bytes=10-19") == (206, 10, 19, 10)
    assert _resolve_pdf_byte_range(100, "bytes=-15") == (206, 85, 99, 15)
    assert _resolve_pdf_byte_range(100, "bytes=bad-range") == (200, 0, 99, 100)

    saved_tokens = dict(_PDFJS_HTTP_TOKENS)
    saved_viewers = dict(_PDFJS_HTTP_VIEWERS)
    try:
        _PDFJS_HTTP_TOKENS.clear()
        _PDFJS_HTTP_VIEWERS.clear()
        first_tok = ""
        for idx in range(_MAX_HTTP_PDF_TOKENS + 5):
            tok = _store_pdfjs_http_token(tmp_path / f"doc_{idx}.pdf")
            if idx == 0:
                first_tok = tok
        assert len(_PDFJS_HTTP_TOKENS) == _MAX_HTTP_PDF_TOKENS
        assert first_tok not in _PDFJS_HTTP_TOKENS

        first_view = ""
        for idx in range(_MAX_HTTP_VIEWER_TOKENS + 4):
            tok = _store_pdfjs_http_viewer_html(f"<html>{idx}</html>")
            if idx == 0:
                first_view = tok
        assert len(_PDFJS_HTTP_VIEWERS) == _MAX_HTTP_VIEWER_TOKENS
        assert first_view not in _PDFJS_HTTP_VIEWERS
    finally:
        _PDFJS_HTTP_TOKENS.clear()
        _PDFJS_HTTP_TOKENS.update(saved_tokens)
        _PDFJS_HTTP_VIEWERS.clear()
        _PDFJS_HTTP_VIEWERS.update(saved_viewers)


def test_load_pdf_cancels_active_speech_session_before_switching(
    tmp_path: Path, monkeypatch
) -> None:
    _app()
    pdf_path = tmp_path / "next.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n%%EOF\n")

    widget = PdfViewerWidget.__new__(PdfViewerWidget)
    QtWidgets.QWidget.__init__(widget)
    widget._doc = None
    widget._pdf_path = None
    widget._web_view = None
    widget._web_profile = None
    widget._use_web_engine = False
    widget._pdfjs_active = False
    widget._speak_session_id = 3
    widget._speaking = True
    token = _SpeakToken(session_id=3)
    widget._active_speak_token = token
    widget._reader_pending_restart = None
    widget._reader_stop_requested = False
    widget._reader_pause_requested = False
    widget._reader_queue = ["old"]
    widget._reader_pages = [1]
    widget._reader_spans = [[0]]
    widget._reader_span_slices = []
    widget._reader_queue_offset = 0
    widget._reader_chunk_base = 0
    widget._reader_index = 0
    widget._reader_total = 1
    widget._reader_state = "reading"
    widget._highlight_mode = "web-sentence"
    widget._web_sentence_span_groups = [[0]]
    widget._web_sentence_span_slices = []
    widget._web_selected_span_text = {}
    widget._text_sentence_spans = []
    widget._active_text_sentence_span = None
    widget._active_text_keyword_spans = []
    widget._word_highlight_timer = None
    widget._word_highlight_units = []
    widget._word_highlight_durations_ms = []
    widget._word_highlight_schedule = []
    widget._word_highlight_index = 0
    widget._word_highlight_active_idx = -1
    widget._word_highlight_started_at = 0.0
    widget._highlight_keywords = []
    widget._keyword_highlighting_enabled = True
    widget._auto_keyword_highlighting_enabled = False
    widget._pdf_repair_attempted = set()
    widget._bookmarks = []
    widget._reading_log = []
    widget._pdf_user_state = {}
    widget._pdf_user_state_pending = None

    opened: list[Path] = []
    monkeypatch.setattr(widget, "_maybe_repair_pdf", lambda p, reason: p)
    monkeypatch.setattr(
        widget, "_open_with_pymupdf", lambda p, source_path=None: opened.append(p)
    )
    monkeypatch.setattr(widget, "_append_reading_log_event", lambda _e: None)
    monkeypatch.setattr(
        "annolid.gui.widgets.pdf_viewer_impl.load_pdf_state", lambda _p: {}
    )
    monkeypatch.setattr(
        "annolid.utils.audio_playback.stop_audio_playback", lambda: None
    )

    widget.load_pdf(str(pdf_path))
    assert token.cancelled is True
    assert widget._speak_session_id == 4
    assert widget._speaking is False
    assert opened == [pdf_path]
    widget.close()


def test_pdf_manager_preserves_user_voice_within_same_language_family(
    monkeypatch,
) -> None:
    calls: list[tuple[str, str, bool]] = []

    class _FakeTtsControls:
        def __init__(self, lang: str, voice: str) -> None:
            self._lang = lang
            self._voice = voice

        def current_settings(self) -> dict[str, str]:
            return {"lang": self._lang, "voice": self._voice}

        def set_language_and_voice(
            self, *, lang: str, voice: str, persist: bool = True
        ) -> None:
            calls.append((lang, voice, persist))
            self._lang = lang
            self._voice = voice

    manager = PdfManager.__new__(PdfManager)
    manager.pdf_tts_controls = _FakeTtsControls("en-gb", "bf_emma")

    # English document with existing British English voice -> preserve user's voice.
    monkeypatch.setattr(
        PdfManager,
        "_extract_pdf_text_sample",
        staticmethod(
            lambda _p: "Mouse tracking in an open field arena shows robust locomotion."
        ),
    )
    manager._auto_select_tts_from_pdf_language("en_paper.pdf")
    assert calls == []

    # Switching to a Chinese PDF -> auto-switch language and voice.
    monkeypatch.setattr(
        PdfManager,
        "_extract_pdf_text_sample",
        staticmethod(lambda _p: "实验记录了小鼠在开放场地中的多巴胺释放与探索行为。"),
    )
    monkeypatch.setattr(
        PdfManager, "_suggest_default_voice", staticmethod(lambda _lang: "zf_xiaobei")
    )
    manager._auto_select_tts_from_pdf_language("zh_paper.pdf")
    assert calls == [("zh", "zf_xiaobei", True)]


def test_pdf_user_state_persists_speech_keywords_atomically(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "annolid.gui.widgets.pdf_user_state.pdf_state_dir",
        lambda: tmp_path / "states",
    )
    (tmp_path / "states").mkdir(parents=True, exist_ok=True)
    pdf_path = tmp_path / "demo.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n%%EOF\n")

    save_pdf_state(
        pdf_path,
        {
            "speechKeywords": {
                "keywords": ["dopamine", "freezing"],
                "enabled": True,
                "autoKeywords": True,
            }
        },
    )
    loaded = load_pdf_state(pdf_path)
    assert loaded["speechKeywords"]["keywords"] == ["dopamine", "freezing"]
    assert loaded["speechKeywords"]["autoKeywords"] is True
    assert list((tmp_path / "states").glob("*.tmp")) == []


def test_build_pdfjs_viewer_html_includes_speech_helper_before_viewer() -> None:
    from annolid.gui.widgets.pdf_viewer_impl import _build_pdfjs_viewer_html

    html = _build_pdfjs_viewer_html(
        asset_base_url="http://127.0.0.1:9999",
        pdf_key="k",
        initial_state={},
        pdf_url="http://127.0.0.1:9999/pdf/abc",
        pdf_b64="",
        pdf_title="paper.pdf",
    )
    speech_idx = html.find("http://127.0.0.1:9999/pdfjs/annolid_pdf_speech.js")
    viewer_idx = html.find("http://127.0.0.1:9999/pdfjs/annolid_viewer.js")
    assert speech_idx > 0
    assert viewer_idx > speech_idx


@pytest.mark.skipif(not _JSC_PATH.exists(), reason="JavaScriptCore jsc unavailable")
def test_js_viewer_real_dom_canvas_highlighting_audio_pauses_and_zoom_restore() -> None:
    """Execute the real annolid_viewer.js DOM/Range/Canvas speech highlight engine in jsc."""
    script = rf"""
// Controllable monotonic clock & RAF scheduler
let nowMs = 1000;
let nextRafId = 1;
const rafQueue = new Map();
const timeoutQueue = [];
const performance = {{ now: () => nowMs }};
function requestAnimationFrame(cb) {{
  const id = nextRafId++;
  rafQueue.set(id, cb);
  return id;
}}
function cancelAnimationFrame(id) {{ rafQueue.delete(id); }}
function setTimeout(cb, ms) {{
  const delay = Number(ms) || 0;
  if (delay <= 20) {{
    Promise.resolve().then(cb);
    return 0;
  }}
  timeoutQueue.push(cb);
  return timeoutQueue.length;
}}
function clearTimeout(_id) {{}}

function flushRafAt(targetMs) {{
  nowMs = targetMs;
  const entries = Array.from(rafQueue.entries());
  rafQueue.clear();
  for (let i = 0; i < entries.length; i++) {{
    entries[i][1](nowMs);
  }}
}}

// Minimal Canvas 2D context recording every fill/stroke/underline op + color
class MockCtx {{
  constructor() {{
    this.ops = [];
    this.fillStyle = "";
    this.strokeStyle = "";
    this.lineWidth = 1;
    this._path = [];
  }}
  save() {{}}
  restore() {{}}
  setTransform() {{}}
  clearRect(x, y, w, h) {{ this.ops.push({{ op: "clearRect", x, y, w, h }}); }}
  fillRect(x, y, w, h) {{ this.ops.push({{ op: "fillRect", fillStyle: this.fillStyle, x, y, w, h }}); }}
  beginPath() {{ this._path = []; }}
  moveTo(x, y) {{ this._path.push({{ x, y }}); }}
  lineTo(x, y) {{ this._path.push({{ x, y }}); }}
  quadraticCurveTo(_cx, _cy, x, y) {{ this._path.push({{ x, y }}); }}
  closePath() {{}}
  fill() {{
    const xs = this._path.map((p) => p.x);
    const ys = this._path.map((p) => p.y);
    this.ops.push({{
      op: "fill",
      fillStyle: this.fillStyle,
      x: Math.min(...xs),
      y: Math.min(...ys),
      w: Math.max(...xs) - Math.min(...xs),
      h: Math.max(...ys) - Math.min(...ys),
    }});
  }}
  stroke() {{
    const xs = this._path.map((p) => p.x);
    const ys = this._path.map((p) => p.y);
    this.ops.push({{
      op: "stroke",
      strokeStyle: this.strokeStyle,
      x: Math.min(...xs),
      y: Math.min(...ys),
      w: Math.max(...xs) - Math.min(...xs),
      h: Math.max(...ys) - Math.min(...ys),
    }});
  }}
}}

class MockClassList {{
  constructor(el) {{ this._el = el; this._set = new Set(); }}
  add(...names) {{ names.forEach((n) => this._set.add(n)); this._sync(); }}
  remove(...names) {{ names.forEach((n) => this._set.delete(n)); this._sync(); }}
  toggle(name, force) {{
    const on = (force == null) ? !this._set.has(name) : !!force;
    if (on) this._set.add(name); else this._set.delete(name);
    this._sync();
    return on;
  }}
  contains(name) {{ return this._set.has(name); }}
  _sync() {{ this._el._className = Array.from(this._set).join(" "); }}
}}

class MockTextNode {{
  constructor(value, parent) {{
    this.nodeType = 3;
    this.nodeValue = String(value || "");
    this.parentNode = parent || null;
  }}
}}

class MockElement {{
  constructor(tagName) {{
    this.tagName = String(tagName || "div").toUpperCase();
    this.nodeType = 1;
    this.children = [];
    this.childNodes = [];
    this.parentNode = null;
    this.dataset = {{}};
    this.style = {{}};
    this._attrs = {{}};
    this._listeners = {{}};
    this._className = "";
    this.classList = new MockClassList(this);
    this._textContent = "";
    this._rect = {{ left: 0, top: 0, width: 600, height: 800, right: 600, bottom: 800 }};
    this.clientWidth = 600;
    this.clientHeight = 800;
    this.scrollWidth = 600;
    this.scrollHeight = 2400;
    this.scrollTop = 0;
    this.offsetTop = 0;
    this.width = 0;
    this.height = 0;
    this._ctx = null;
  }}
  get className() {{ return this._className; }}
  set className(v) {{
    this._className = String(v || "");
    this.classList._set = new Set(this._className.split(/\s+/).filter(Boolean));
  }}
  get textContent() {{
    if (this.childNodes.length) {{
      return this.childNodes.map((n) => (n.nodeType === 3 ? n.nodeValue : n.textContent)).join("");
    }}
    return this._textContent;
  }}
  set textContent(v) {{
    this._textContent = String(v || "");
    this.children = [];
    this.childNodes = [new MockTextNode(this._textContent, this)];
  }}
  set innerHTML(v) {{
    if (!v) {{ this.children = []; this.childNodes = []; this._textContent = ""; }}
  }}
  setAttribute(k, v) {{ this._attrs[k] = String(v); }}
  getAttribute(k) {{ return Object.prototype.hasOwnProperty.call(this._attrs, k) ? this._attrs[k] : null; }}
  appendChild(child) {{
    child.parentNode = this;
    this.childNodes.push(child);
    if (child.nodeType === 1) this.children.push(child);
    return child;
  }}
  addEventListener(type, fn) {{
    if (!this._listeners[type]) this._listeners[type] = [];
    this._listeners[type].push(fn);
  }}
  dispatchEvent(type, ev) {{
    const list = this._listeners[type] || [];
    for (let i = 0; i < list.length; i++) list[i](ev || {{ target: this, stopPropagation() {{}}, preventDefault() {{}} }});
  }}
  getBoundingClientRect() {{ return Object.assign({{}}, this._rect); }}
  getClientRects() {{ return [this.getBoundingClientRect()]; }}
  getContext(_kind) {{
    if (!this._ctx) this._ctx = new MockCtx();
    return this._ctx;
  }}
  closest(sel) {{
    let cur = this;
    while (cur) {{
      if (sel === ".page" && cur.classList.contains("page")) return cur;
      cur = cur.parentNode;
    }}
    return null;
  }}
  querySelectorAll(sel) {{
    const out = [];
    const walk = (node) => {{
      for (let i = 0; i < node.children.length; i++) {{
        const c = node.children[i];
        if (sel === "span" && c.tagName === "SPAN") out.push(c);
        else if (sel === ".textLayer span" && c.tagName === "SPAN" && c.parentNode && c.parentNode.classList.contains("textLayer")) out.push(c);
        else if (sel === ".page" && c.classList.contains("page")) out.push(c);
        walk(c);
      }}
    }};
    walk(this);
    return out;
  }}
  querySelector(sel) {{
    const m = /^\.page\[data-page-number='(\d+)'\]$/.exec(sel);
    if (m) {{
      const pages = this.querySelectorAll(".page");
      for (let i = 0; i < pages.length; i++) {{
        if (pages[i].getAttribute("data-page-number") === m[1]) return pages[i];
      }}
      return null;
    }}
    const all = this.querySelectorAll(sel);
    return all.length ? all[0] : null;
  }}
}}

// Build mock DOM + browser globals
const viewerContainer = new MockElement("div");
viewerContainer.setAttribute("id", "viewerContainer");
const zoomInBtn = new MockElement("button");
const elementsById = {{
  viewerContainer,
  annolidZoomIn: zoomInBtn,
}};
const docListeners = {{}};
const winListeners = {{}};
const NodeFilter = {{ SHOW_TEXT: 4 }};
const document = {{
  body: new MockElement("body"),
  documentElement: {{ clientWidth: 800, clientHeight: 900 }},
  readyState: "complete",
  baseURI: "http://127.0.0.1:9999/viewer/tok",
  createElement(tag) {{ return new MockElement(tag); }},
  getElementById(id) {{ return elementsById[id] || null; }},
  querySelector(sel) {{ return viewerContainer.querySelector(sel); }},
  querySelectorAll(sel) {{ return viewerContainer.querySelectorAll(sel); }},
  addEventListener(type, fn) {{ (docListeners[type] = docListeners[type] || []).push(fn); }},
  createTreeWalker(root, _mask) {{
    const textNodes = [];
    const walk = (n) => {{
      const kids = n.childNodes || [];
      for (let i = 0; i < kids.length; i++) {{
        if (kids[i].nodeType === 3) textNodes.push(kids[i]);
        else walk(kids[i]);
      }}
    }};
    walk(root);
    let idx = 0;
    return {{ nextNode() {{ return idx < textNodes.length ? textNodes[idx++] : null; }} }};
  }},
  createRange() {{
    const rng = {{ _sNode: null, _sOff: 0, _eNode: null, _eOff: 0 }};
    rng.setStart = (n, o) => {{ rng._sNode = n; rng._sOff = o; }};
    rng.setEnd = (n, o) => {{ rng._eNode = n; rng._eOff = o; }};
    rng.getClientRects = () => {{
      const span = rng._sNode && rng._sNode.parentNode;
      if (!span) return [];
      const full = String(rng._sNode.nodeValue || "");
      if (!full.length) return [];
      const sr = span.getBoundingClientRect();
      const charW = sr.width / full.length;
      const left = sr.left + rng._sOff * charW;
      const width = Math.max(1, (rng._eOff - rng._sOff) * charW);
      return [{{ left, top: sr.top, width, height: sr.height, right: left + width, bottom: sr.top + sr.height }}];
    }};
    return rng;
  }},
}};
document.body.appendChild(viewerContainer);

const pageEvents = [];
const mockBridge = {{
  onPageChanged(cur, tot) {{ pageEvents.push([cur, tot]); }},
  getUserState(_k, cb) {{ if (cb) cb({{}}); }},
  saveUserState() {{}},
  clearUserState() {{}},
  logEvent() {{}},
}};
function QWebChannel(_transport, cb) {{
  cb({{ objects: {{ annolidBridge: mockBridge }} }});
}}
const window = {{
  __annolidAssetBaseUrl: "http://127.0.0.1:9999",
  __annolidPdfUrl: "http://127.0.0.1:9999/pdf/mock",
  __annolidPdfBase64: "",
  __annolidPdfTitle: "demo.pdf",
  __annolidPdfKey: "",
  qt: {{ webChannelTransport: {{}} }},
  devicePixelRatio: 1,
  innerWidth: 800,
  innerHeight: 900,
  addEventListener(type, fn) {{ (winListeners[type] = winListeners[type] || []).push(fn); }},
  getSelection() {{ return {{ rangeCount: 0, isCollapsed: true, toString: () => "" }}; }},
}};
const localStorage = {{ getItem: () => null, setItem() {{}}, removeItem() {{}} }};
const navigator = {{ userAgent: "MockJSC" }};
const console = {{ log() {{}}, warn() {{}}, error() {{}} }};

// 3-page mock PDF.js library (initial render renders pages 1..2; page 3 is lazy!)
const pageTexts = {{
  1: [{{ str: "First, dopamine rises rapidly.", x: 40, y: 100, w: 300, h: 14 }}],
  2: [{{ str: "Page two observations continue.", x: 40, y: 100, w: 280, h: 14 }}],
  3: [{{ str: "Lazy page three measures freezing accurately.", x: 40, y: 120, w: 360, h: 14 }}],
}};
const pdfjsLib = {{
  GlobalWorkerOptions: {{}},
  getDocument() {{
    return {{
      promise: Promise.resolve({{
        numPages: 3,
        getPage(pageNum) {{
          const items = pageTexts[pageNum] || [];
          return Promise.resolve({{
            getViewport({{ scale, rotation }}) {{
              const s = scale || 1;
              return {{ width: 600 * s, height: 800 * s, scale: s, rotation: rotation || 0 }};
            }},
            render() {{ return {{ promise: Promise.resolve() }}; }},
            getAnnotations() {{ return Promise.resolve([]); }},
            getTextContent() {{
              return Promise.resolve({{
                items: items.map((it) => ({{
                  str: it.str,
                  width: it.w,
                  height: it.h,
                  transform: [1, 0, 0, it.h, it.x, 800 - it.y],
                }})),
              }});
            }},
          }});
        }},
      }}),
    }};
  }},
  renderTextLayer({{ textContent, container, viewport }}) {{
    const s = (viewport && viewport.scale) ? viewport.scale : 1;
    const items = (textContent && textContent.items) || [];
    for (let i = 0; i < items.length; i++) {{
      const it = items[i];
      const sp = document.createElement("span");
      sp.textContent = it.str;
      const x = (it.transform[4] || 40) * s;
      const y = (800 - (it.transform[5] || 700)) * s;
      const w = (it.width || 200) * s;
      const h = (it.height || 14) * s;
      sp._rect = {{ left: x, top: y, width: w, height: h, right: x + w, bottom: y + h }};
      container.appendChild(sp);
    }}
    return {{ promise: Promise.resolve() }};
  }},
}};

// Load BOTH real Annolid JS files!
load({json.dumps(str(_SPEECH_JS.resolve()))});
load({json.dumps(str(_VIEWER_JS.resolve()))});

async function runAssertions() {{
  // Trigger DOMContentLoaded to boot the real viewer runtime
  const handlers = docListeners["DOMContentLoaded"] || [];
  if (!handlers.length) throw new Error("DOMContentLoaded listener not registered");
  await handlers[0]();

  const errAttr = document.body.getAttribute("data-pdfjs-error");
  if (errAttr) throw new Error("Viewer bootstrap error: " + errAttr);
  if (!window.__annolidPdfLoaded) throw new Error("Expected __annolidPdfLoaded=true");
  if (window.__annolidRenderedPages !== 2) {{
    throw new Error("Expected initial 2 rendered pages, got " + window.__annolidRenderedPages);
  }}

  const page1Ctx = window.__annolidPages["1"].ttsCtx;
  const span0 = window.__annolidSpans[0];
  if (!span0 || span0.textContent.indexOf("dopamine") < 0) {{
    throw new Error("Unexpected page 1 span: " + (span0 && span0.textContent));
  }}

  // 1) Start audio-aligned speech highlight with 200ms leading silence and 300ms comma pause
  nowMs = 2000;
  page1Ctx.ops.length = 0;
  const started = await window.__annolidStartSpeechHighlight({{
    spanIndices: [0],
    pageNum: 1,
    sentenceText: "First, dopamine rises rapidly.",
    durationMs: 1500,
    speechSegments: [[200, 500], [800, 1300]],
    keywords: ["dopamine"],
    highlightKeywords: true,
    autoKeywords: false,
    sessionId: 10,
  }});
  if (!started) throw new Error("__annolidStartSpeechHighlight returned false");

  // At t=0ms (inside 200ms leading silence): sentence (yellow) + keyword (teal) drawn, NO active word yet!
  const fills0 = page1Ctx.ops.filter((o) => o.op === "fill");
  const hasYellow = fills0.some((o) => o.fillStyle === "rgba(255, 214, 79, 0.24)");
  const hasTeal = fills0.some((o) => o.fillStyle === "rgba(38, 198, 218, 0.34)");
  const hasActiveWord0 = fills0.some((o) => o.fillStyle === "rgba(255, 145, 0, 0.50)" || o.fillStyle === "rgba(255, 87, 34, 0.56)");
  if (!hasYellow || !hasTeal) throw new Error("Expected yellow sentence + teal keyword layers at t=0: " + JSON.stringify(fills0));
  if (hasActiveWord0) throw new Error("Active word must wait during 200ms leading silence!");

  // Sub-span geometry check: teal "dopamine" box must be strictly narrower than the full 300px span!
  const tealBox = fills0.find((o) => o.fillStyle === "rgba(38, 198, 218, 0.34)");
  if (!(tealBox.x > 80 && tealBox.w > 50 && tealBox.w < 120)) {{
    throw new Error("Expected exact sub-span Range rect for dopamine, got: " + JSON.stringify(tealBox));
  }}

  // 2) Advance to t=220ms (voice onset on "First"): vivid amber active word box!
  page1Ctx.ops.length = 0;
  flushRafAt(2220);
  const fills220 = page1Ctx.ops.filter((o) => o.op === "fill");
  const amberBox = fills220.find((o) => o.fillStyle === "rgba(255, 145, 0, 0.50)");
  if (!amberBox || !(amberBox.x < 60 && amberBox.w < 80)) {{
    throw new Error("Expected amber sub-span box for 'First' at t=220ms: " + JSON.stringify(fills220));
  }}

  // 3) Advance to t=650ms (inside 300ms pause between [200..500] and [800..1300]):
  // Highlight holds on "First" (does NOT jump early to "dopamine" coral spotlight!).
  page1Ctx.ops.length = 0;
  flushRafAt(2650);
  if (window.__annolidTts.activeIsKeyword !== false) {{
    throw new Error("Should still hold non-keyword 'First' during pause at t=650ms");
  }}

  // 4) Advance to t=820ms (voice resumes on keyword "dopamine"): coral-orange spotlight!
  page1Ctx.ops.length = 0;
  flushRafAt(2820);
  const fills820 = page1Ctx.ops.filter((o) => o.op === "fill");
  const coralBox = fills820.find((o) => o.fillStyle === "rgba(255, 87, 34, 0.56)");
  if (!coralBox || window.__annolidTts.activeIsKeyword !== true) {{
    throw new Error("Expected coral keyword spotlight for 'dopamine' at t=820ms: " + JSON.stringify(fills820));
  }}

  // 5) Live keyword config update & toggle off
  page1Ctx.ops.length = 0;
  window.__annolidSetKeywordHighlightConfig({{ keywords: ["rapidly"], enabled: true, autoKeywords: false }});
  const fillsPin = page1Ctx.ops.filter((o) => o.op === "fill");
  if (!fillsPin.some((o) => o.fillStyle === "rgba(0, 188, 212, 0.18)")) {{
    throw new Error("Expected persistent page keyword pin layer after config update");
  }}
  page1Ctx.ops.length = 0;
  window.__annolidSetKeywordHighlightConfig({{ keywords: ["rapidly"], enabled: false, autoKeywords: false }});
  const fillsOff = page1Ctx.ops.filter((o) => o.op === "fill");
  if (fillsOff.some((o) => o.fillStyle === "rgba(38, 198, 218, 0.34)" || o.fillStyle === "rgba(0, 188, 212, 0.18)")) {{
    throw new Error("Keyword layers should clear when keyword highlighting is disabled");
  }}

  // 6) Zoom re-render mid-sentence restores active sentence + keyword highlight at new scale
  window.__annolidSetKeywordHighlightConfig({{ keywords: ["dopamine"], enabled: true, autoKeywords: false }});
  zoomInBtn.dispatchEvent("click");
  for (let tick = 0; tick < 30; tick++) {{
    await Promise.resolve();
  }}
  const zoomedPage1 = window.__annolidPages["1"];
  if (!(zoomedPage1 && zoomedPage1.width > 650)) {{
    throw new Error("Expected zoomed page 1 width > 650, got " + (zoomedPage1 && zoomedPage1.width));
  }}
  if (!zoomedPage1.ttsCtx.ops.some((o) => o.op === "fill" && o.fillStyle === "rgba(255, 214, 79, 0.24)")) {{
    throw new Error("Expected active sentence highlight restored after zoom re-render");
  }}
  if (!pageEvents.some((ev) => ev[0] === 1 && ev[1] === 3)) {{
    throw new Error("Expected bridge.onPageChanged(1, 3) notification, got " + JSON.stringify(pageEvents));
  }}

  // 7) Unrendered future page 3 fallback lazily renders page 3 and highlights keyword "freezing"!
  const lazyOk = await window.__annolidStartSpeechHighlight({{
    spanIndices: [],
    pageNum: 3,
    sentenceText: "Lazy page three measures freezing accurately.",
    durationMs: 1000,
    speechSegments: [[0, 950]],
    keywords: ["freezing"],
    highlightKeywords: true,
    autoKeywords: false,
    sessionId: 11,
  }});
  if (!lazyOk) throw new Error("Lazy page 3 highlight failed");
  const page3State = window.__annolidPages["3"];
  if (!page3State || !page3State.ttsCtx) throw new Error("Page 3 was not lazily rendered");
  const p3Fills = page3State.ttsCtx.ops.filter((o) => o.op === "fill");
  if (!p3Fills.some((o) => o.fillStyle === "rgba(38, 198, 218, 0.34)")) {{
    throw new Error("Expected teal keyword highlight on lazily rendered page 3: " + JSON.stringify(p3Fills));
  }}

  print("VIEWER_DOM_CANVAS_RUNTIME_OK");
}}

let _done = false;
let _err = null;
runAssertions().then(() => {{ _done = true; }}).catch((e) => {{ _err = e; _done = true; }});
if (typeof drainMicrotasks === "function") drainMicrotasks();
if (_err) throw _err;
if (!_done) throw new Error("Async assertions did not settle");
"""
    proc = subprocess.run(
        [str(_JSC_PATH), "-e", script],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "VIEWER_DOM_CANVAS_RUNTIME_OK" in proc.stdout


def test_speak_text_task_emits_audio_aligned_speech_segments(monkeypatch) -> None:
    app = _app()
    sr = 8000
    n_total = int(sr * 1.5)
    pcm = [0.0] * n_total
    for s_ms, e_ms in ((200, 500), (800, 1300)):
        for i in range(int(sr * s_ms / 1000), int(sr * e_ms / 1000)):
            pcm[i] = 0.5 * math.sin(2.0 * math.pi * 220.0 * i / sr)

    class _Receiver(QtCore.QObject):
        def __init__(self) -> None:
            super().__init__()
            self.chunks: list[tuple[int, int, int, list[list[int]]]] = []
            self.finished_sessions: list[int] = []

        @QtCore.Slot(int, int, int, str)
        def _on_speak_chunk_aligned(
            self,
            session_id: int,
            index: int,
            duration_ms: int,
            segments_json: str,
        ) -> None:
            self.chunks.append(
                (session_id, index, duration_ms, json.loads(segments_json))
            )

        @QtCore.Slot(int)
        def _on_speak_finished_for_session(self, session_id: int) -> None:
            self.finished_sessions.append(session_id)

        @QtCore.Slot(int, str)
        def _on_speak_error(self, _session_id: int, message: str) -> None:
            raise AssertionError(f"Unexpected TTS error: {message}")

    receiver = _Receiver()
    monkeypatch.setattr(
        "annolid.agents.tts_router.synthesize_tts",
        lambda _chunk, _settings: (pcm, sr),
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.pdf_speech_controller.play_audio_buffer",
        lambda _samples, _sr, blocking=True: True,
    )

    token = _SpeakToken(session_id=7)
    task = _SpeakTextTask(
        receiver,  # type: ignore[arg-type]
        "First, dopamine rises.",
        {"engine": "kokoro"},
        chunks=["First, dopamine rises."],
        token=token,
    )
    task.run()
    app.processEvents()

    assert len(receiver.chunks) == 1
    sid, idx, dur_ms, segs = receiver.chunks[0]
    assert sid == 7
    assert idx == 0
    assert dur_ms == 1500
    assert len(segs) == 2
    assert 170 <= segs[0][0] <= 215
    assert 770 <= segs[1][0] <= 815
    assert receiver.finished_sessions == [7]


def test_fallback_text_view_respects_leading_silence_and_internal_pause() -> None:
    _app()
    widget = PdfViewerWidget.__new__(PdfViewerWidget)
    QtWidgets.QWidget.__init__(widget)
    widget.text_view = QtWidgets.QTextEdit(widget)
    text = "First, dopamine rises quickly."
    widget.text_view.setPlainText(text)
    widget._doc = None
    widget._pdf_path = None
    widget._web_view = None
    widget._web_profile = None
    widget._pdfjs_active = False
    widget._speaking = True
    widget._speak_session_id = 4
    widget._highlight_mode = "text-sentence"
    widget._text_sentence_spans = [(0, len(text))]
    widget._active_text_sentence_span = None
    widget._active_text_keyword_spans = []
    widget._highlight_keywords = ["dopamine"]
    widget._keyword_highlighting_enabled = True
    widget._auto_keyword_highlighting_enabled = False
    widget._word_highlight_timer = None
    widget._word_highlight_units = []
    widget._word_highlight_durations_ms = []
    widget._word_highlight_schedule = []
    widget._word_highlight_index = 0
    widget._word_highlight_active_idx = -1
    widget._word_highlight_started_at = 0.0

    # Audio has 220ms leading silence, "First," in [220..500], 300ms pause, rest in [800..1300].
    widget._on_speak_chunk_aligned(4, 0, 1500, json.dumps([[220, 500], [800, 1300]]))

    # At t=0ms (leading silence > 40ms): no active word spotlight yet!
    assert widget._word_highlight_active_idx == -1
    assert not any(
        sel.format.fontWeight() == QtGui.QFont.Bold
        for sel in widget.text_view.extraSelections()
    )

    # Simulate t=250ms -> "First" lights up.
    widget._word_highlight_started_at = time.monotonic() - 0.250
    widget._advance_word_highlight()
    assert widget._word_highlight_active_idx == 0
    assert widget.text_view.extraSelections()[-1].cursor.selectedText() == "First"

    # Simulate t=650ms (inside 300ms pause) -> holds on "First", does NOT jump to "dopamine".
    widget._word_highlight_started_at = time.monotonic() - 0.650
    widget._advance_word_highlight()
    assert widget._word_highlight_active_idx == 0

    # Simulate t=820ms (voice resumes) -> keyword "dopamine" lights up in bold!
    widget._word_highlight_started_at = time.monotonic() - 0.820
    widget._advance_word_highlight()
    assert widget._word_highlight_active_idx == 1
    active_sel = widget.text_view.extraSelections()[-1]
    assert active_sel.cursor.selectedText() == "dopamine"
    assert active_sel.format.fontWeight() == QtGui.QFont.Bold

    widget._stop_word_highlight()
    widget.close()


def _invoke_http_handler(
    method: str, path: str, *, req_headers: dict[str, str] | None = None
) -> tuple[int, dict[str, str], bytes]:
    """Execute _PdfJsHttpRequestHandler purely in-memory without opening sockets."""
    handler = _PdfJsHttpRequestHandler.__new__(_PdfJsHttpRequestHandler)
    handler.path = path
    handler.command = method
    handler.request_version = "HTTP/1.1"
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.client_address = ("127.0.0.1", 0)
    hdrs = Message()
    for k, v in (req_headers or {}).items():
        hdrs[k] = v
    handler.headers = hdrs
    handler.wfile = io.BytesIO()
    handler._headers_buffer = []

    recorded_status: list[int] = []
    recorded_headers: dict[str, str] = {}

    def _send_response(code: int, _message=None) -> None:
        recorded_status.append(int(code))

    def _send_header(key: str, value: str) -> None:
        recorded_headers[key.lower()] = str(value)

    def _end_headers() -> None:
        return

    def _send_error(code: int, _message=None, _explain=None) -> None:
        recorded_status.append(int(code))

    handler.send_response = _send_response  # type: ignore[method-assign]
    handler.send_header = _send_header  # type: ignore[method-assign]
    handler.end_headers = _end_headers  # type: ignore[method-assign]
    handler.send_error = _send_error  # type: ignore[method-assign]

    getattr(handler, f"do_{method}")()
    status = recorded_status[-1] if recorded_status else 0
    return status, recorded_headers, handler.wfile.getvalue()


def test_pdf_viewer_server_http_handler_serves_assets_ranges_and_empty_pdf_in_process(
    tmp_path: Path,
) -> None:
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(b"")
    sample_pdf = tmp_path / "sample.pdf"
    sample_bytes = b"%PDF-1.7\n0123456789\n%%EOF\n"
    sample_pdf.write_bytes(sample_bytes)

    saved_tokens = dict(_PDFJS_HTTP_TOKENS)
    saved_viewers = dict(_PDFJS_HTTP_VIEWERS)
    try:
        empty_tok = _store_pdfjs_http_token(empty_pdf)
        sample_tok = _store_pdfjs_http_token(sample_pdf)
        viewer_tok = _store_pdfjs_http_viewer_html("<!doctype html><html>ok</html>")

        # 1) Whitelisted JS & CSS assets + path traversal rejection
        st, hdrs, body = _invoke_http_handler("GET", "/pdfjs/annolid_pdf_speech.js")
        assert st == 200
        assert hdrs["content-type"] == "application/javascript"
        assert b"AnnolidPdfSpeech" in body

        st_css, hdrs_css, _ = _invoke_http_handler("GET", "/pdfjs/annolid_viewer.css")
        assert st_css == 200
        assert hdrs_css["content-type"] == "text/css"

        st_trav, _, _ = _invoke_http_handler("GET", "/pdfjs/../pyproject.toml")
        assert st_trav == 404

        # 2) Viewer HTML route
        st_v, hdrs_v, body_v = _invoke_http_handler("GET", f"/viewer/{viewer_tok}")
        assert st_v == 200
        assert hdrs_v["content-type"] == "text/html; charset=utf-8"
        assert body_v == b"<!doctype html><html>ok</html>"

        # 3) Empty 0-byte PDF returns 200 with Content-Length: 0
        st_e, hdrs_e, body_e = _invoke_http_handler("GET", f"/pdf/{empty_tok}")
        assert st_e == 200
        assert hdrs_e["content-length"] == "0"
        assert body_e == b""

        # 4) HEAD returns exact Content-Length with empty body
        st_h, hdrs_h, body_h = _invoke_http_handler("HEAD", f"/pdf/{sample_tok}")
        assert st_h == 200
        assert hdrs_h["content-length"] == str(len(sample_bytes))
        assert body_h == b""

        # 5) Partial Range request returns 206 + exact slice + Content-Range
        st_r, hdrs_r, body_r = _invoke_http_handler(
            "GET", f"/pdf/{sample_tok}", req_headers={"Range": "bytes=2-6"}
        )
        assert st_r == 206
        assert hdrs_r["content-range"] == f"bytes 2-6/{len(sample_bytes)}"
        assert hdrs_r["content-length"] == "5"
        assert body_r == sample_bytes[2:7]

        # 6) OPTIONS CORS preflight allows only loopback origins (never wildcard '*')
        st_opt, hdrs_opt, _ = _invoke_http_handler(
            "OPTIONS",
            f"/pdf/{sample_tok}",
            req_headers={"Origin": "http://127.0.0.1:8000"},
        )
        assert st_opt == 204
        assert hdrs_opt["access-control-allow-origin"] == "http://127.0.0.1:8000"

        st_evil, hdrs_evil, _ = _invoke_http_handler(
            "OPTIONS",
            f"/pdf/{sample_tok}",
            req_headers={"Origin": "https://evil.example"},
        )
        assert st_evil == 204
        assert "access-control-allow-origin" not in hdrs_evil

        st_404, _, _ = _invoke_http_handler("GET", "/pdf/nonexistent")
        assert st_404 == 404
    finally:
        _PDFJS_HTTP_TOKENS.clear()
        _PDFJS_HTTP_TOKENS.update(saved_tokens)
        _PDFJS_HTTP_VIEWERS.clear()
        _PDFJS_HTTP_VIEWERS.update(saved_viewers)
