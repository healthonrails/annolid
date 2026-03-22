"""Background workers for research paper chat features."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

from qtpy import QtCore
from qtpy.QtCore import QRunnable, Signal, Slot

from annolid.services.literature_search import LiteratureResult, search_literature
from annolid.services.paper_writer import draft_section, generate_outline
from annolid.utils.llm_settings import provider_kind

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM bridge — synchronous completion for use in worker threads
# ---------------------------------------------------------------------------


def make_sync_llm_call(settings: Dict[str, Any]) -> Callable[[str, str], str]:
    """Return a synchronous ``(system, user) -> str`` callable.

    Uses the provider already configured in Annolid's LLM settings.
    Falls back to a informative error string if the provider is not set up.
    """
    from annolid.core.agent.providers import (
        run_gemini_chat,
        run_ollama_streaming_chat,
        run_openai_compat_chat,
    )

    provider = str(settings.get("provider") or "").strip() or "openai_compat"
    model = str(settings.get("model") or "").strip()
    kind = str(provider_kind(settings, provider) or "").strip()

    def _call(system_prompt: str, user_prompt: str) -> str:
        full_prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
        try:
            if kind in (
                "openai_compat",
                "openai",
                "openai_codex",
                "codex_cli",
                "nvidia",
            ):
                _, text = run_openai_compat_chat(
                    prompt=full_prompt,
                    image_path=None,
                    model=model,
                    provider_name=provider,
                    settings=settings,
                    load_history_messages=lambda: [],
                    max_tokens=2048,
                    timeout_s=120.0,
                )
                return str(text or "")
            if kind == "gemini":
                _, text = run_gemini_chat(
                    prompt=full_prompt,
                    image_path=None,
                    model=model,
                    provider_name=provider,
                    settings=settings,
                )
                return str(text or "")
            if kind == "ollama":
                # Ollama is streaming — accumulate chunks
                chunks: list[str] = []
                final: list[str] = []

                def _chunk(c: str) -> None:
                    chunks.append(c)

                def _final(msg: str, _is_err: bool) -> None:
                    final.append(msg)

                run_ollama_streaming_chat(
                    prompt=full_prompt,
                    image_path=None,
                    model=model,
                    settings=settings,
                    load_history_messages=lambda: [],
                    emit_chunk=_chunk,
                    emit_final=_final,
                    persist_turn=lambda _u, _a: None,
                )
                return "".join(final) or "".join(chunks)
            raise ValueError(
                f"Unsupported provider kind '{kind}' for research drafting."
            )
        except Exception as exc:
            raise RuntimeError(
                f"LLM call to provider '{provider}' failed: {exc}"
            ) from exc

    return _call


# ---------------------------------------------------------------------------
# Worker helpers
# ---------------------------------------------------------------------------


class LiteratureSearchWorker(QtCore.QObject):
    """Runs ``search_literature`` in a dedicated thread."""

    finished = Signal(list)  # list[LiteratureResult]
    error = Signal(str)

    def __init__(self, query: str, max_results: int = 8) -> None:
        super().__init__()
        self._query = query
        self._max_results = max_results

    @Slot()
    def run(self) -> None:
        try:
            payload = search_literature(self._query, max_results=self._max_results)
            raw = payload.get("results", [])
            results: list[LiteratureResult] = []
            for row in raw:
                if isinstance(row, dict):
                    results.append(
                        LiteratureResult(
                            source=str(row.get("source") or ""),
                            title=str(row.get("title") or ""),
                            summary=str(row.get("summary") or ""),
                            id_url=str(row.get("id_url") or ""),
                            abs_url=str(row.get("abs_url") or ""),
                            pdf_url=str(row.get("pdf_url") or ""),
                            arxiv_id=str(row.get("arxiv_id") or ""),
                            doi=str(row.get("doi") or ""),
                            year=row.get("year"),
                        )
                    )
            self.finished.emit(results)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Literature search failed: %s", exc)
            self.error.emit(str(exc))


class DraftWorkerSignals(QtCore.QObject):
    section_done = Signal(str, str)  # (title, content)
    progress = Signal(str)
    finished = Signal()
    error = Signal(str)


class DraftWorker(QRunnable):
    """Generates outline then drafts each section. Runs via QThreadPool."""

    def __init__(
        self,
        topic: str,
        papers: list[LiteratureResult],
        llm_call: Callable[[str, str], str],
    ) -> None:
        super().__init__()
        self.signals = DraftWorkerSignals()
        self._topic = topic
        self._papers = papers
        self._llm_call = llm_call
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            self.signals.progress.emit("Generating outline…")
            outline = generate_outline(self._topic, self._papers, self._llm_call)
            for item in outline:
                title = str(item.get("title") or "Section")
                guidance = str(item.get("guidance") or "")
                self.signals.progress.emit(f'Drafting "{title}"…')
                content = draft_section(
                    title, guidance, self._topic, self._papers, self._llm_call
                )
                self.signals.section_done.emit(title, content)
            self.signals.finished.emit()
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("DraftWorker failed")
            self.signals.error.emit(str(exc))
