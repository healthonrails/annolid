from __future__ import annotations

from typing import Optional

from qtpy import QtCore


class DictionaryLookupTask(QtCore.QRunnable):
    """Background task to fetch dictionary definitions for a single word."""

    def __init__(self, widget: QtCore.QObject, request_id: str, word: str) -> None:
        super().__init__()
        self.widget = widget
        self.request_id = str(request_id or "")
        self.word = str(word or "").strip()

    @staticmethod
    def _format_html(word: str, payload: object) -> str:
        from html import escape

        safe_word = escape(word)
        if not isinstance(payload, list):
            return f"<h2>{safe_word}</h2><pre>{escape(str(payload))}</pre>"

        parts: list[str] = [f"<h2>{safe_word}</h2>"]
        phonetic = ""
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            phonetic = str(entry.get("phonetic") or "").strip()
            if phonetic:
                break
            for ph in entry.get("phonetics") or []:
                if isinstance(ph, dict):
                    phonetic = str(ph.get("text") or "").strip()
                    if phonetic:
                        break
            if phonetic:
                break
        if phonetic:
            parts.append(f"<p><i>{escape(phonetic)}</i></p>")

        for entry in payload[:2]:
            if not isinstance(entry, dict):
                continue
            meanings = entry.get("meanings") or []
            if not isinstance(meanings, list):
                continue
            for meaning in meanings[:6]:
                if not isinstance(meaning, dict):
                    continue
                pos = str(meaning.get("partOfSpeech") or "").strip()
                if pos:
                    parts.append(f"<h3>{escape(pos)}</h3>")
                defs = meaning.get("definitions") or []
                if not isinstance(defs, list):
                    continue
                items: list[str] = []
                for d in defs[:6]:
                    if not isinstance(d, dict):
                        continue
                    definition = str(d.get("definition") or "").strip()
                    if not definition:
                        continue
                    example = str(d.get("example") or "").strip()
                    block = f"<li>{escape(definition)}"
                    if example:
                        block += f"<br/><span style='color:#555'><i>Example: {escape(example)}</i></span>"
                    block += "</li>"
                    items.append(block)
                if items:
                    parts.append("<ol>" + "".join(items) + "</ol>")
        parts.append(
            "<p style='color:#777;font-size:11px'>Source: dictionaryapi.dev</p>"
        )
        return "".join(parts)

    @staticmethod
    def _lookup_macos_dictionary(word: str) -> Optional[str]:
        import ctypes
        import ctypes.util

        cf_path = ctypes.util.find_library("CoreFoundation")
        if not cf_path:
            return None
        try:
            core_foundation = ctypes.cdll.LoadLibrary(cf_path)
        except Exception:
            return None

        dictionary_services = None
        for path in (
            "/System/Library/Frameworks/CoreServices.framework/Frameworks/DictionaryServices.framework/DictionaryServices",
            "/System/Library/Frameworks/DictionaryServices.framework/DictionaryServices",
        ):
            try:
                dictionary_services = ctypes.cdll.LoadLibrary(path)
                break
            except Exception:
                continue
        if dictionary_services is None:
            return None

        kCFStringEncodingUTF8 = 0x08000100

        class CFRange(ctypes.Structure):
            _fields_ = [("location", ctypes.c_long), ("length", ctypes.c_long)]

        core_foundation.CFStringCreateWithCString.restype = ctypes.c_void_p
        core_foundation.CFStringCreateWithCString.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int32,
        ]
        core_foundation.CFStringGetLength.restype = ctypes.c_long
        core_foundation.CFStringGetLength.argtypes = [ctypes.c_void_p]
        core_foundation.CFStringGetMaximumSizeForEncoding.restype = ctypes.c_long
        core_foundation.CFStringGetMaximumSizeForEncoding.argtypes = [
            ctypes.c_long,
            ctypes.c_int32,
        ]
        core_foundation.CFStringGetCString.restype = ctypes.c_bool
        core_foundation.CFStringGetCString.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_long,
            ctypes.c_int32,
        ]
        core_foundation.CFRelease.restype = None
        core_foundation.CFRelease.argtypes = [ctypes.c_void_p]

        dictionary_services.DCSCopyTextDefinition.restype = ctypes.c_void_p
        dictionary_services.DCSCopyTextDefinition.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            CFRange,
        ]

        cf_word = None
        cf_definition = None
        try:
            cf_word = core_foundation.CFStringCreateWithCString(
                None, word.encode("utf-8"), kCFStringEncodingUTF8
            )
            if not cf_word:
                return None
            cf_definition = dictionary_services.DCSCopyTextDefinition(
                None, cf_word, CFRange(0, len(word))
            )
            if not cf_definition:
                return None
            length = core_foundation.CFStringGetLength(cf_definition)
            max_size = (
                core_foundation.CFStringGetMaximumSizeForEncoding(
                    length, kCFStringEncodingUTF8
                )
                + 1
            )
            buffer = ctypes.create_string_buffer(max_size)
            ok = core_foundation.CFStringGetCString(
                cf_definition, buffer, max_size, kCFStringEncodingUTF8
            )
            if not ok:
                return None
            return buffer.value.decode("utf-8", errors="replace").strip()
        finally:
            try:
                if cf_definition:
                    core_foundation.CFRelease(cf_definition)
            except Exception:
                pass
            try:
                if cf_word:
                    core_foundation.CFRelease(cf_word)
            except Exception:
                pass

    def run(self) -> None:  # pragma: no cover - network + UI
        import sys

        word = (self.word or "").strip()
        html = ""
        error = ""
        try:
            if sys.platform == "darwin":
                definition = self._lookup_macos_dictionary(word)
                if definition:
                    from html import escape

                    html = (
                        f"<h2>{escape(word)}</h2>"
                        "<pre style='white-space:pre-wrap'>"
                        f"{escape(definition)}"
                        "</pre>"
                        "<p style='color:#777;font-size:11px'>Source: macOS Dictionary</p>"
                    )
            if not html:
                import requests
                from urllib.parse import quote

                url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{quote(word)}"
                response = requests.get(url, timeout=8)
                if response.status_code == 404:
                    error = f"No definition found for “{word}”."
                else:
                    response.raise_for_status()
                    html = self._format_html(word, response.json())
        except Exception as exc:
            error = f"Dictionary lookup failed: {exc}"

        try:
            QtCore.QMetaObject.invokeMethod(
                self.widget,
                "_on_dictionary_lookup_finished",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, str(self.request_id)),
                QtCore.Q_ARG(str, str(word)),
                QtCore.Q_ARG(str, str(html)),
                QtCore.Q_ARG(str, str(error)),
            )
        except Exception:
            pass
