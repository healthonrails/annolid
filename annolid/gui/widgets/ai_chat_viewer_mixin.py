"""Viewer navigation, web/PDF/Three.js slots, and citation helpers for AIChatWidget."""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, Optional
import webbrowser

from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.widgets.citation_manager_widget import CitationManagerDialog
from annolid.infrastructure.agent_workspace import get_agent_workspace_path
from annolid.services.citation_verify import (
    build_citation_verification_report,
    write_citation_verification_report,
)
from annolid.utils.citations import (
    BibEntry,
    load_bibtex,
    merge_validated_fields,
    save_bibtex,
    upsert_entry,
    validate_basic_citation_fields,
    validate_citation_metadata,
)


def _widget_module_attr(name: str, default: Any) -> Any:
    mod = sys.modules.get("annolid.gui.widgets.ai_chat_widget")
    if mod is not None and hasattr(mod, name):
        return getattr(mod, name)
    return default


class AIChatViewerCitationsMixin:
    """Mixin encapsulating citation workflows and viewer/web/PDF/3D bot slots."""

    def _default_citation_bib_path(self) -> Path:
        getter = _widget_module_attr(
            "get_agent_workspace_path", get_agent_workspace_path
        )
        return getter() / "citations.bib"

    @staticmethod
    def _extract_doi_for_citation(text: str) -> str:
        raw = str(text or "")
        if not raw:
            return ""
        match = re.search(
            r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b",
            raw,
            flags=re.IGNORECASE,
        )
        return str(match.group(1) or "").rstrip(").,;!?") if match else ""

    @staticmethod
    def _extract_year_for_citation(text: str) -> str:
        years = re.findall(r"\b(19\d{2}|20\d{2})\b", str(text or ""))
        return years[0] if years else ""

    @staticmethod
    def _normalize_citation_key_for_ui(
        title: str, year: str, fallback: str = "paper"
    ) -> str:
        text = str(title or fallback or "paper").strip().lower()
        tokens = re.findall(r"[a-z0-9]+", text)
        stem = "_".join(tokens[:3]) if tokens else "paper"
        yr = str(year or "").strip()
        key = f"{stem}_{yr}" if yr else stem
        key = re.sub(r"[^a-zA-Z0-9:_\-.]+", "_", key).strip("_")
        return key or "paper"

    def _citation_entry_from_active_pdf(self) -> Optional[BibEntry]:
        self.bot_pdf_get_state()
        state = self.get_bot_action_result("pdf_get_state") or dict(
            self._bot_action_result or {}
        )
        if not bool(state.get("ok")) or not bool(state.get("has_pdf")):
            return None
        path = str(state.get("path") or "").strip()
        title = str(state.get("title") or "").strip()
        if title.lower().endswith(".pdf"):
            title = title[:-4]
        self.bot_pdf_get_text(9000, 2)
        text_payload = self.get_bot_action_result("pdf_get_text") or dict(
            self._bot_action_result or {}
        )
        text = str(text_payload.get("text") or "")
        doi = self._extract_doi_for_citation(text)
        year = self._extract_year_for_citation(text)
        fields: Dict[str, str] = {
            "title": title or Path(path).stem.replace("_", " "),
            "note": "Saved from active Annolid PDF viewer.",
            "source_path": path,
        }
        if year:
            fields["year"] = year
        if doi:
            fields["doi"] = doi
            fields["url"] = f"https://doi.org/{doi}"
        arxiv_match = re.search(
            r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b",
            text,
            flags=re.IGNORECASE,
        )
        if arxiv_match:
            fields["archiveprefix"] = "arXiv"
            fields["eprint"] = str(arxiv_match.group(1) or "").strip()
        key = self._normalize_citation_key_for_ui(
            fields["title"], fields.get("year", "")
        )
        return BibEntry(entry_type="article", key=key, fields=fields)

    def _citation_entry_from_active_web(self) -> Optional[BibEntry]:
        self.bot_web_get_state()
        state = self.get_bot_action_result("web_get_state") or dict(
            self._bot_action_result or {}
        )
        if not bool(state.get("ok")) or not bool(state.get("has_page")):
            return None
        url = str(state.get("url") or "").strip()
        if not url:
            return None
        title = str(state.get("title") or "").strip()
        self.bot_web_get_dom_text(9000)
        text_payload = self.get_bot_action_result("web_get_dom_text") or dict(
            self._bot_action_result or {}
        )
        text = str(text_payload.get("text") or "")
        doi = self._extract_doi_for_citation(f"{url}\n{text}")
        year = self._extract_year_for_citation(text)
        fields: Dict[str, str] = {
            "title": title or "Web page citation",
            "url": url,
            "note": "Saved from active Annolid web viewer.",
        }
        if year:
            fields["year"] = year
        if doi:
            fields["doi"] = doi
        arxiv_match = re.search(
            r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)",
            url,
            flags=re.IGNORECASE,
        )
        if arxiv_match:
            fields["archiveprefix"] = "arXiv"
            fields["eprint"] = str(arxiv_match.group(1) or "").strip()
        key = self._normalize_citation_key_for_ui(
            fields["title"], fields.get("year", "")
        )
        return BibEntry(entry_type="article", key=key, fields=fields)

    def _save_citation_from_active_context(
        self,
        *,
        source: str = "auto",
        key: str = "",
        bib_path: Optional[Path] = None,
        validate_before_save: bool = True,
        strict_validation: bool = False,
        verify_after_save: bool = False,
    ) -> Dict[str, Any]:
        source_norm = str(source or "auto").strip().lower()
        if source_norm not in {"auto", "pdf", "web"}:
            source_norm = "auto"
        entry: Optional[BibEntry] = None
        used_source = source_norm
        if source_norm in {"auto", "pdf"}:
            entry = self._citation_entry_from_active_pdf()
            if entry:
                used_source = "pdf"
        if entry is None and source_norm in {"auto", "web"}:
            entry = self._citation_entry_from_active_web()
            if entry:
                used_source = "web"
        if entry is None:
            return {
                "ok": False,
                "error": "No active PDF/web context found to create citation.",
            }
        basic_errors = validate_basic_citation_fields(
            {
                "__key__": str(key or "").strip(),
                "year": str(entry.fields.get("year") or ""),
                "doi": str(entry.fields.get("doi") or ""),
            }
        )
        if basic_errors:
            return {"ok": False, "error": " ".join(basic_errors)}
        chosen_key = str(key or "").strip()
        if chosen_key:
            entry.key = re.sub(r"[^a-zA-Z0-9:_\-.]+", "_", chosen_key).strip("_")
        validation: Dict[str, Any] = {
            "checked": False,
            "verified": False,
            "provider": "",
            "score": 0.0,
            "message": "",
            "candidate": {},
        }
        if bool(validate_before_save):
            validator = _widget_module_attr(
                "validate_citation_metadata", validate_citation_metadata
            )
            validation = validator(entry.fields, timeout_s=1.8)
            entry.fields = merge_validated_fields(
                entry.fields, validation, replace_when_confident=True
            )
            if bool(strict_validation) and not bool(validation.get("verified")):
                return {
                    "ok": False,
                    "error": (
                        "Citation validation failed strict mode. "
                        + str(validation.get("message") or "")
                    ).strip(),
                    "validation": validation,
                }
            if not chosen_key:
                candidate_key = str(
                    dict(validation.get("candidate") or {}).get("__bibkey__") or ""
                ).strip()
                if candidate_key:
                    entry.key = re.sub(r"[^a-zA-Z0-9:_\-.]+", "_", candidate_key).strip(
                        "_"
                    )
                else:
                    entry.key = self._normalize_citation_key_for_ui(
                        str(entry.fields.get("title") or "").strip(),
                        str(entry.fields.get("year") or "").strip(),
                    )
        target_bib = bib_path or self._default_citation_bib_path()
        target_bib = Path(target_bib).expanduser()
        target_bib.parent.mkdir(parents=True, exist_ok=True)
        entries = load_bibtex(target_bib) if target_bib.exists() else []
        updated, created = upsert_entry(entries, entry)
        save_bibtex(target_bib, updated, sort_keys=True)
        payload: Dict[str, Any] = {
            "ok": True,
            "created": bool(created),
            "key": entry.key,
            "bib_file": str(target_bib),
            "source": used_source,
            "validation": validation,
        }
        if bool(verify_after_save):
            report = build_citation_verification_report(
                key=str(entry.key or ""),
                bib_file=str(target_bib),
                source=used_source,
                fields=dict(entry.fields),
                validation=validation,
            )
            payload["verification"] = dict(report.get("verification") or {})
            report_path = write_citation_verification_report(
                report,
                reports_dir=target_bib.parent
                / ".annolid_cache"
                / "citation_verification",
                report_stem=f"{target_bib.stem}_{entry.key}",
            )
            payload["verification_report"] = str(report_path)
        return payload

    def _open_citation_manager(self) -> None:
        dialog_cls = _widget_module_attr("CitationManagerDialog", CitationManagerDialog)
        dialog = dialog_cls(
            default_bib_path_getter=self._default_citation_bib_path,
            save_from_context=self._save_citation_from_active_context,
            parent=self,
        )
        dialog.exec_()

    def _canvas_pixmap_ready(self) -> bool:
        host = self.host_window_widget or self.window()
        canvas = getattr(host, "canvas", None)
        pixmap = getattr(canvas, "pixmap", None)
        return bool(
            pixmap is not None and hasattr(pixmap, "isNull") and not pixmap.isNull()
        )

    def _wait_for_canvas_pixmap(self, timeout_ms: int = 3000) -> bool:
        deadline = QtCore.QDeadlineTimer(int(max(1, timeout_ms)))
        while not deadline.hasExpired():
            if self._canvas_pixmap_ready():
                return True
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
        return self._canvas_pixmap_ready()

    @QtCore.Slot(str)
    def bot_open_video(self, video_path: str) -> None:
        path = str(video_path or "").strip()
        if not path:
            self.status_label.setText("Bot action failed: empty video path.")
            return
        host = self.host_window_widget or self.window()
        open_video = getattr(host, "openVideo", None)
        if not callable(open_video):
            self.status_label.setText("Bot action failed: video loader is unavailable.")
            return
        try:
            open_video(from_video_list=True, video_path=path, programmatic_call=True)
            self.status_label.setText(f"Opened video: {os.path.basename(path)}")
        except Exception as exc:
            self.status_label.setText(f"Bot action failed: {exc}")

    @QtCore.Slot(str)
    def bot_open_image(self, image_path: str) -> None:
        path_text = str(image_path or "").strip()
        if not path_text:
            self.status_label.setText("Bot action failed: empty image path.")
            return
        resolved = Path(path_text).expanduser()
        if not resolved.exists() or not resolved.is_file():
            self.status_label.setText(
                f"Bot action failed: image not found: {path_text}"
            )
            return
        host = self.host_window_widget or self.window()
        load_file = getattr(host, "loadFile", None)
        if not callable(load_file):
            self.status_label.setText("Bot action failed: image loader is unavailable.")
            return
        try:
            load_file(str(resolved))
            set_view = getattr(host, "_set_active_view", None)
            if callable(set_view):
                set_view("canvas")
            self.status_label.setText(f"Opened image on canvas: {resolved.name}")
        except Exception as exc:
            self.status_label.setText(f"Bot action failed: {exc}")

    def _set_bot_action_result(self, *args: Any, **kwargs: Any) -> None:
        """
        Backward-compatible bot action result setter.
        Supports both call styles:
        - _set_bot_action_result(payload_dict)
        - _set_bot_action_result(action_name, payload_dict)
        """
        action = ""
        payload: Dict[str, Any] = {}
        if len(args) >= 2:
            action = str(args[0] or "").strip()
            payload = dict(args[1] or {})
        elif len(args) == 1:
            value = args[0]
            if isinstance(value, dict):
                payload = dict(value or {})
                action = str(payload.get("action") or "").strip()
            else:
                action = str(value or "").strip()
                payload = dict(kwargs.get("result") or {})
        else:
            action = str(kwargs.get("action") or "").strip()
            payload = dict(kwargs.get("result") or kwargs.get("payload") or {})

        if action:
            self._bot_action_results[action] = dict(payload)
        self._bot_action_result = dict(payload)

    def _resolve_web_manager(self):
        host = self.host_window_widget or self.window()
        return getattr(host, "web_manager", None)

    def _resolve_pdf_manager(self):
        host = self.host_window_widget or self.window()
        return getattr(host, "pdf_manager", None)

    def _normalize_url_for_open(self, url: str) -> Optional[QtCore.QUrl]:
        value = str(url or "").strip()
        if not value:
            return None
        parsed = QtCore.QUrl(value)
        if parsed.isValid() and parsed.scheme().lower() in {"http", "https", "file"}:
            return parsed
        local_path = Path(value).expanduser()
        if local_path.exists() and local_path.is_file():
            return QtCore.QUrl.fromLocalFile(str(local_path))
        if "://" not in value:
            fallback = QtCore.QUrl(f"https://{value}")
            if fallback.isValid() and fallback.scheme().lower() in {"http", "https"}:
                return fallback
        return None

    def _open_url_in_web_viewer(self, parsed: QtCore.QUrl) -> bool:
        normalized = str(parsed.toString() or "").strip()
        if not normalized:
            return False
        host = self.host_window_widget or self.window()
        show_web = getattr(host, "show_web_in_viewer", None)
        if callable(show_web) and bool(show_web(normalized)):
            return True
        manager = self._resolve_web_manager()
        if manager is not None and hasattr(manager, "show_url_in_viewer"):
            try:
                return bool(manager.show_url_in_viewer(normalized))
            except Exception:
                return False
        return False

    def _open_chat_link_default(self, url: str) -> None:
        parsed = self._normalize_url_for_open(url)
        if parsed is None:
            self.status_label.setText("Invalid URL.")
            return
        normalized = str(parsed.toString() or "").strip()
        if self._open_url_in_web_viewer(parsed):
            self.status_label.setText(
                f"Opened URL in embedded web viewer: {normalized}"
            )
            return
        if QtGui.QDesktopServices.openUrl(parsed):
            self.status_label.setText(f"Opened URL in browser: {normalized}")
            return
        self.status_label.setText("Could not open URL.")

    def _open_chat_link_in_browser(self, url: str, mode: str = "default") -> None:
        parsed = self._normalize_url_for_open(url)
        if parsed is None:
            self.status_label.setText("Invalid URL.")
            return
        normalized = str(parsed.toString() or "").strip()
        request = str(mode or "default").strip().lower()
        opened = False
        wb = _widget_module_attr("webbrowser", webbrowser)
        if request == "new_tab":
            opened = bool(wb.open(normalized, new=2, autoraise=True))
        elif request == "new_window":
            opened = bool(wb.open(normalized, new=1, autoraise=True))
        else:
            opened = QtGui.QDesktopServices.openUrl(parsed)
        if not opened:
            opened = QtGui.QDesktopServices.openUrl(parsed)
        if opened:
            if request == "new_tab":
                self.status_label.setText(
                    f"Requested opening in browser new tab: {normalized}"
                )
            elif request == "new_window":
                self.status_label.setText(
                    f"Requested opening in browser new window: {normalized}"
                )
            else:
                self.status_label.setText(f"Opened URL in browser: {normalized}")
            return
        self.status_label.setText("Could not open URL in browser.")

    @QtCore.Slot(str)
    def bot_open_pdf(self, pdf_path: str = "") -> None:
        host = self.host_window_widget or self.window()
        pdf_import_widget = getattr(host, "pdf_import_widget", None)
        open_pdf = getattr(pdf_import_widget, "open_pdf", None)
        open_pdf_path = getattr(pdf_import_widget, "open_pdf_path", None)
        if not callable(open_pdf):
            self.status_label.setText("Bot action failed: PDF opener is unavailable.")
            return
        try:
            resolved = str(pdf_path or "").strip()
            if resolved and callable(open_pdf_path):
                ok = bool(open_pdf_path(resolved))
                if ok:
                    self.status_label.setText(
                        f"Opened PDF: {os.path.basename(resolved)}"
                    )
                else:
                    self.status_label.setText("Bot action failed: could not open PDF.")
                return
            open_pdf()
            self.status_label.setText("Opened PDF picker.")
        except Exception as exc:
            self.status_label.setText(f"Bot action failed: {exc}")

    @QtCore.Slot(str)
    def bot_open_url(self, url: str) -> None:
        parsed = self._normalize_url_for_open(url)
        if parsed is None:
            self.status_label.setText("Bot action failed: invalid URL or file path.")
            return
        normalized = str(parsed.toString() or "").strip()
        if self._open_url_in_web_viewer(parsed):
            self.status_label.setText(f"Opened URL in canvas: {normalized}")
            return
        opened = QtGui.QDesktopServices.openUrl(parsed)
        if opened:
            self.status_label.setText(f"Opened URL in browser: {normalized}")
            return
        self.status_label.setText("Bot action failed: could not open URL.")

    @QtCore.Slot(str)
    def bot_open_in_browser(self, url: str) -> None:
        parsed = self._normalize_url_for_open(url)
        if parsed is None:
            self.status_label.setText("Bot action failed: invalid URL.")
            return
        target_url = str(parsed.toString() or "").strip()
        opened = QtGui.QDesktopServices.openUrl(parsed)
        if opened:
            self.status_label.setText(f"Opened URL in browser: {target_url}")
            return
        self.status_label.setText("Bot action failed: could not open URL.")

    @staticmethod
    def _normalize_threejs_example_input(example: str) -> str:
        raw = str(example or "").strip().lower()
        if not raw:
            return "two_mice_html"
        mapping = {
            "helix": "helix_points_csv",
            "helix_points_csv": "helix_points_csv",
            "wave": "wave_surface_obj",
            "wave_surface_obj": "wave_surface_obj",
            "sphere": "sphere_points_ply",
            "sphere_points_ply": "sphere_points_ply",
            "brain": "brain_viewer_html",
            "brain_viewer_html": "brain_viewer_html",
            "two_mice": "two_mice_html",
            "two_mice_html": "two_mice_html",
            "fly_brain": "fly_brain_mouse_html",
            "fly_brain_mouse": "fly_brain_mouse_html",
            "fly_brain_mouse_html": "fly_brain_mouse_html",
            "rat": "rat_ncap_html",
            "rat_ncap": "rat_ncap_html",
            "rat_ncap_html": "rat_ncap_html",
            "rat_ncap_simulation_json": "rat_ncap_simulation_json",
        }
        normalized = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
        if normalized in mapping:
            return mapping[normalized]
        if "rat" in raw:
            return "rat_ncap_html"
        if "fly" in raw and ("brain" in raw or "mouse" in raw or "mice" in raw):
            return "fly_brain_mouse_html"
        if "two" in raw and "mice" in raw:
            return "two_mice_html"
        if "brain" in raw:
            return "brain_viewer_html"
        if "helix" in raw:
            return "helix_points_csv"
        if "wave" in raw:
            return "wave_surface_obj"
        if "sphere" in raw:
            return "sphere_points_ply"
        return "two_mice_html"

    @QtCore.Slot(str)
    def bot_open_threejs_example(self, example_id: str = "") -> None:
        host = self.host_window_widget or self.window()
        open_example = getattr(host, "open_threejs_example", None)
        if not callable(open_example):
            self.status_label.setText(
                "Bot action failed: Three.js examples unavailable."
            )
            return
        resolved = self._normalize_threejs_example_input(example_id)
        try:
            open_example(resolved)
            self.status_label.setText(f"Opened Three.js example: {resolved}")
        except Exception as exc:
            self.status_label.setText(f"Bot action failed: {exc}")

    @QtCore.Slot(str)
    def bot_open_threejs(self, path_or_url: str) -> None:
        target = str(path_or_url or "").strip()
        if not target:
            self.status_label.setText("Bot action failed: provide a 3D path or URL.")
            return
        host = self.host_window_widget or self.window()
        manager = getattr(host, "threejs_manager", None)
        if manager is None:
            ensure_threejs_manager = getattr(host, "ensure_threejs_manager", None)
            if callable(ensure_threejs_manager):
                try:
                    manager = ensure_threejs_manager()
                except Exception:
                    manager = None
        if manager is None:
            self.status_label.setText(
                "Bot action failed: Three.js viewer is unavailable."
            )
            return

        local_path = Path(target).expanduser()
        try:
            is_local_zarr = (
                local_path.exists()
                and local_path.is_dir()
                and local_path.suffix.lower() == ".zarr"
            )
            if local_path.exists() and (local_path.is_file() or is_local_zarr):
                suffix = local_path.suffix.lower()
                if suffix in {".html", ".htm", ".xhtml"}:
                    url = QtCore.QUrl.fromLocalFile(str(local_path)).toString()
                    if bool(manager.show_url_in_viewer(url)):
                        self.status_label.setText(
                            f"Opened Three.js URL: {local_path.name}"
                        )
                        return
                elif bool(manager.show_model_in_viewer(local_path)):
                    self.status_label.setText(
                        f"Opened Three.js model: {local_path.name}"
                    )
                    return
        except Exception:
            pass

        parsed = self._normalize_url_for_open(target)
        if parsed is not None:
            normalized = str(parsed.toString() or "").strip()
            scheme = str(parsed.scheme() or "").lower()
            if normalized and scheme in {"http", "https", "file"}:
                try:
                    if bool(manager.show_url_in_viewer(normalized)):
                        self.status_label.setText(f"Opened Three.js URL: {normalized}")
                        return
                except Exception:
                    pass

        self.status_label.setText("Bot action failed: could not open Three.js content.")

    @QtCore.Slot(int)
    def bot_web_get_dom_text(self, max_chars: int) -> None:
        manager = self._resolve_web_manager()
        if manager is None:
            payload = {"ok": False, "error": "Embedded web manager is unavailable."}
            self._set_bot_action_result("web_get_dom_text", payload)
            self.status_label.setText("Bot action failed: web manager unavailable.")
            return
        try:
            payload = manager.get_page_text(max_chars=int(max_chars))
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result("web_get_dom_text", payload)
        self.status_label.setText(
            "Captured page text." if payload.get("ok") else "Bot action failed."
        )

    @QtCore.Slot()
    def bot_web_get_state(self) -> None:
        manager = self._resolve_web_manager()
        if manager is None:
            payload = {"ok": False, "error": "Embedded web manager is unavailable."}
            self._set_bot_action_result("web_get_state", payload)
            self.status_label.setText("Bot action failed: web manager unavailable.")
            return
        try:
            payload = manager.get_web_state()
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result("web_get_state", payload)

    @QtCore.Slot(int)
    def bot_web_capture_screenshot(self, max_width: int) -> None:
        manager = self._resolve_web_manager()
        if manager is None:
            payload = {"ok": False, "error": "Embedded web manager is unavailable."}
            self._set_bot_action_result("web_capture_screenshot", payload)
            self.status_label.setText("Bot action failed: web manager unavailable.")
            return
        try:
            payload = manager.capture_screenshot(max_width=int(max_width))
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result("web_capture_screenshot", payload)
        self.status_label.setText(
            "Captured web screenshot." if payload.get("ok") else "Bot action failed."
        )

    @QtCore.Slot(int)
    def bot_web_describe_view(self, max_width: int) -> None:
        manager = self._resolve_web_manager()
        if manager is None:
            payload = {"ok": False, "error": "Embedded web manager is unavailable."}
            self._set_bot_action_result("web_describe_view", payload)
            self.status_label.setText("Bot action failed: web manager unavailable.")
            return
        try:
            payload = manager.describe_current_view(max_width=int(max_width))
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result("web_describe_view", payload)
        self.status_label.setText(
            "Submitted screenshot for description."
            if payload.get("ok")
            else "Bot action failed."
        )

    @QtCore.Slot()
    def bot_pdf_get_state(self) -> None:
        manager = self._resolve_pdf_manager()
        if manager is None:
            payload = {"ok": False, "error": "PDF manager is unavailable."}
            self._set_bot_action_result("pdf_get_state", payload)
            self.status_label.setText("Bot action failed: PDF manager unavailable.")
            return
        try:
            payload = manager.get_pdf_state()
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result("pdf_get_state", payload)

    @QtCore.Slot(int, int)
    @QtCore.Slot(int, int, int)
    def bot_pdf_get_text(self, max_chars: int, pages: int, start_page: int = 0) -> None:
        manager = self._resolve_pdf_manager()
        if manager is None:
            payload = {"ok": False, "error": "PDF manager is unavailable."}
            self._set_bot_action_result("pdf_get_text", payload)
            self.status_label.setText("Bot action failed: PDF manager unavailable.")
            return
        try:
            payload = manager.get_pdf_text(
                max_chars=int(max_chars),
                pages=int(pages),
                start_page=int(start_page or 0),
            )
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result("pdf_get_text", payload)
        self.status_label.setText(
            "Captured PDF text." if payload.get("ok") else "Bot action failed."
        )

    @QtCore.Slot(int, int)
    def bot_pdf_find_sections(self, max_sections: int, max_pages: int) -> None:
        manager = self._resolve_pdf_manager()
        if manager is None:
            payload = {"ok": False, "error": "PDF manager is unavailable."}
            self._set_bot_action_result(payload)
            self.status_label.setText("Bot action failed: PDF manager unavailable.")
            return
        try:
            payload = manager.get_pdf_sections(
                max_sections=int(max_sections),
                max_pages=int(max_pages),
            )
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result(payload)
        self.status_label.setText(
            "Detected PDF sections." if payload.get("ok") else "Bot action failed."
        )

    @QtCore.Slot(str)
    def bot_web_click(self, selector: str) -> None:
        manager = self._resolve_web_manager()
        if manager is None:
            payload = {"ok": False, "error": "Embedded web manager is unavailable."}
            self._set_bot_action_result(payload)
            self.status_label.setText("Bot action failed: web manager unavailable.")
            return
        try:
            payload = manager.click_selector(str(selector or ""))
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result(payload)
        self.status_label.setText(
            "Clicked selector." if payload.get("ok") else "Bot action failed."
        )

    @QtCore.Slot(str, str, bool)
    def bot_web_type(self, selector: str, text: str, submit: bool) -> None:
        manager = self._resolve_web_manager()
        if manager is None:
            payload = {"ok": False, "error": "Embedded web manager is unavailable."}
            self._set_bot_action_result(payload)
            self.status_label.setText("Bot action failed: web manager unavailable.")
            return
        try:
            payload = manager.type_selector(
                str(selector or ""), str(text or ""), submit=bool(submit)
            )
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result(payload)
        self.status_label.setText(
            "Typed into selector." if payload.get("ok") else "Bot action failed."
        )

    @QtCore.Slot(int)
    def bot_web_scroll(self, delta_y: int) -> None:
        manager = self._resolve_web_manager()
        if manager is None:
            payload = {"ok": False, "error": "Embedded web manager is unavailable."}
            self._set_bot_action_result(payload)
            self.status_label.setText("Bot action failed: web manager unavailable.")
            return
        try:
            payload = manager.scroll_by(delta_y=int(delta_y))
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result(payload)
        self.status_label.setText(
            "Scrolled page." if payload.get("ok") else "Bot action failed."
        )

    @QtCore.Slot()
    def bot_web_find_forms(self) -> None:
        manager = self._resolve_web_manager()
        if manager is None:
            payload = {"ok": False, "error": "Embedded web manager is unavailable."}
            self._set_bot_action_result(payload)
            self.status_label.setText("Bot action failed: web manager unavailable.")
            return
        try:
            payload = manager.find_forms()
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result(payload)
        self.status_label.setText(
            "Inspected forms." if payload.get("ok") else "Bot action failed."
        )

    @QtCore.Slot()
    def bot_web_save_current(self) -> None:
        manager = self._resolve_web_manager()
        if manager is None:
            payload = {"ok": False, "error": "Embedded web manager is unavailable."}
            self._set_bot_action_result("web_save_current", payload)
            self.status_label.setText("Bot action failed: web manager unavailable.")
            return
        try:
            payload = manager.save_current_page()
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result("web_save_current", payload)
        if payload.get("ok"):
            if payload.get("queued"):
                self.status_label.setText("Saving web content in background...")
            else:
                saved_path = str(payload.get("last_saved_path") or "").strip()
                self.status_label.setText(
                    f"Saved web content: {os.path.basename(saved_path)}"
                    if saved_path
                    else "Saved web content."
                )
            return
        self.status_label.setText("Bot action failed.")

    @QtCore.Slot(int)
    def bot_set_frame(self, frame_index: int) -> None:
        host = self.host_window_widget or self.window()
        setter = getattr(host, "set_frame_number", None)
        if not callable(setter):
            self.status_label.setText(
                "Bot action failed: frame navigation unavailable."
            )
            return
        try:
            setter(int(frame_index))
            self.status_label.setText(f"Moved to frame {int(frame_index)}")
        except Exception as exc:
            self.status_label.setText(f"Bot action failed: {exc}")

    @QtCore.Slot(str)
    def bot_set_chat_prompt(self, text: str) -> None:
        value = str(text or "").strip()
        self.prompt_text_edit.setPlainText(value)
        self.prompt_text_edit.setFocus()
        self.status_label.setText("Chat prompt updated by bot.")


__all__ = ["AIChatViewerCitationsMixin"]
