from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, List, Optional


_STAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})\]\s+")
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{2,}")
_MANAGED_BLOCK_RE_TEMPLATE = r"(?ms)^## {title} \(managed\)\n.*?(?=^## |\Z)"
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "been",
    "were",
    "have",
    "has",
    "had",
    "will",
    "would",
    "could",
    "should",
    "about",
    "after",
    "before",
    "during",
    "while",
    "where",
    "when",
    "which",
    "there",
    "their",
    "them",
    "then",
    "than",
    "your",
    "ours",
    "they",
    "what",
    "why",
    "how",
    "who",
    "you",
    "its",
    "our",
    "not",
    "just",
    "very",
}
_DEEP_WEIGHTS = {
    "frequency": 0.24,
    "relevance": 0.30,
    "query_diversity": 0.15,
    "recency": 0.15,
    "consolidation": 0.10,
    "conceptual_richness": 0.06,
}
_MIN_DEEP_SCORE = 0.58
_MIN_RECALL_COUNT = 2
_MIN_UNIQUE_QUERIES = 1
_PHASE_SIGNAL_DECAY_DAYS = 14.0
_PHASE_SIGNAL_MAX_BOOST = 0.06


@dataclass(frozen=True)
class HistoryEntry:
    stamp: str
    day: str
    text: str


@dataclass(frozen=True)
class DreamRunResult:
    ok: bool
    did_work: bool
    status: str
    message: str
    run_id: str = ""
    cursor_start: int = 0
    cursor_end: int = 0
    processed_entries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "did_work": self.did_work,
            "status": self.status,
            "message": self.message,
            "run_id": self.run_id,
            "cursor_start": int(self.cursor_start),
            "cursor_end": int(self.cursor_end),
            "processed_entries": int(self.processed_entries),
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _parse_iso(value: str) -> Optional[datetime]:
    try:
        normalized = str(value or "").replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        if token in _STOPWORDS:
            continue
        tokens.append(token)
    return tokens


class DreamMemoryManager:
    """Phase-based Dream-style memory maintenance for Annolid workspaces."""

    def __init__(self, workspace: Path | str) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.memory_dir = self.workspace / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.history_file = self.memory_dir / "HISTORY.md"
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.dreams_file = self._resolve_dreams_file()

        self.cursor_file = self.memory_dir / ".dream_cursor"
        self.runs_file = self.memory_dir / "dream_runs.jsonl"
        self.snapshots_dir = self.memory_dir / ".dream_snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self.dream_state_dir = self.memory_dir / ".dreams"
        self.dream_state_dir.mkdir(parents=True, exist_ok=True)
        self.ingestion_file = self.dream_state_dir / "ingestion.json"
        self.recall_store_file = self.dream_state_dir / "recall-store.json"
        self.phase_signals_file = self.dream_state_dir / "phase-signals.json"
        self.lock_file = self.dream_state_dir / "lock"

        self.phase_report_root = self.memory_dir / "dreaming"

    def run(
        self,
        *,
        max_batch_entries: int = 50,
        initialize_cursor_to_end: bool = True,
    ) -> DreamRunResult:
        entries = self._read_history_entries()
        if not self.cursor_file.exists():
            initial = len(entries) if initialize_cursor_to_end else 0
            self._write_cursor(initial)
            self._write_ingestion_state(
                {
                    "cursor": initial,
                    "history_entries": len(entries),
                    "updated_at": _now_iso(),
                    "initialized": True,
                }
            )
            return DreamRunResult(
                ok=True,
                did_work=False,
                status="initialized",
                message=(
                    "Dream cursor initialized at end of history."
                    if initialize_cursor_to_end
                    else "Dream cursor initialized at beginning of history."
                ),
                cursor_start=initial,
                cursor_end=initial,
            )

        if not self._acquire_lock():
            return DreamRunResult(
                ok=False,
                did_work=False,
                status="locked",
                message="Dream run skipped: another dreaming run is active.",
            )

        try:
            cursor_start = self._read_cursor()
            cursor_start = max(0, min(cursor_start, len(entries)))
            limit = max(1, int(max_batch_entries))
            pending = entries[cursor_start : cursor_start + limit]
            if not pending:
                return DreamRunResult(
                    ok=True,
                    did_work=False,
                    status="noop",
                    message="Dream: nothing to process.",
                    cursor_start=cursor_start,
                    cursor_end=cursor_start,
                )

            cursor_end = cursor_start + len(pending)
            run_id = self._make_run_id(cursor_start=cursor_start, cursor_end=cursor_end)
            snapshot_dir = self._snapshot_current_state(run_id)

            recall_store = self._read_json(
                self.recall_store_file, default={"items": {}}
            )
            phase_signals = self._read_json(
                self.phase_signals_file,
                default={"light": {}, "rem": {}},
            )
            if not isinstance(recall_store.get("items"), dict):
                recall_store["items"] = {}
            if not isinstance(phase_signals.get("light"), dict):
                phase_signals["light"] = {}
            if not isinstance(phase_signals.get("rem"), dict):
                phase_signals["rem"] = {}

            light_result = self._run_light_phase(
                pending=pending,
                recall_store=recall_store,
                phase_signals=phase_signals,
                run_id=run_id,
            )
            rem_result = self._run_rem_phase(
                pending=pending,
                phase_signals=phase_signals,
                run_id=run_id,
            )
            deep_result = self._run_deep_phase(
                recall_store=recall_store,
                phase_signals=phase_signals,
                run_id=run_id,
            )

            self._write_json(self.recall_store_file, recall_store)
            self._write_json(self.phase_signals_file, phase_signals)

            summary = self._build_summary_entry(
                pending=pending,
                cursor_start=cursor_start,
                cursor_end=cursor_end,
                promoted=deep_result["promoted_count"],
            )
            self._append_history_line(summary)

            self._write_cursor(cursor_end)
            self._write_ingestion_state(
                {
                    "cursor": cursor_end,
                    "history_entries": len(entries),
                    "updated_at": _now_iso(),
                    "last_run_id": run_id,
                    "processed_entries": len(pending),
                }
            )

            row = {
                "run_id": run_id,
                "timestamp": _now_iso(),
                "status": "ok",
                "processed_entries": len(pending),
                "cursor_start": int(cursor_start),
                "cursor_end": int(cursor_end),
                "summary": summary,
                "snapshot_dir": str(snapshot_dir),
                "phases": {
                    "light": light_result,
                    "rem": rem_result,
                    "deep": deep_result,
                },
            }
            self._append_run_row(row)
            return DreamRunResult(
                ok=True,
                did_work=True,
                status="ok",
                message=(
                    f"Dream completed: processed {len(pending)} entries "
                    f"(cursor {cursor_start}->{cursor_end})."
                ),
                run_id=run_id,
                cursor_start=cursor_start,
                cursor_end=cursor_end,
                processed_entries=len(pending),
            )
        finally:
            self._release_lock()

    def format_status(self) -> str:
        cursor = self._read_cursor()
        ingestion = self._read_json(self.ingestion_file, default={})
        rows = self.list_runs(limit=1)
        latest = rows[0] if rows else {}
        deep = (
            latest.get("phases", {}).get("deep", {}) if isinstance(latest, dict) else {}
        )
        return (
            "## Dreaming Status\n\n"
            f"- Cursor: {cursor}\n"
            f"- Last run: `{latest.get('run_id', '')}`\n"
            f"- Last run timestamp: {latest.get('timestamp', '')}\n"
            f"- Last processed entries: {latest.get('processed_entries', 0)}\n"
            f"- Last promoted entries: {deep.get('promoted_count', 0)}\n"
            f"- Ingestion updated: {ingestion.get('updated_at', '')}"
        ).strip()

    @staticmethod
    def format_help() -> str:
        return (
            "## Dreaming Help\n\n"
            "- `/dream` or `/dreaming run`: execute one dreaming sweep.\n"
            "- `/dream-log [run_id]`: inspect latest or selected run metadata.\n"
            "- `/dream-restore [run_id]`: list runs or restore snapshot.\n"
            "- `/dreaming status`: show cursor/phase status."
        )

    def list_runs(self, *, limit: int = 20) -> List[Dict[str, Any]]:
        rows = self._read_run_rows()
        keep = max(1, int(limit))
        return rows[-keep:][::-1]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        target = str(run_id or "").strip()
        if not target:
            return None
        for row in reversed(self._read_run_rows()):
            if str(row.get("run_id") or "").strip() == target:
                return row
        return None

    def restore(self, run_id: str) -> DreamRunResult:
        row = self.get_run(run_id)
        target = str(run_id or "").strip()
        if row is None:
            return DreamRunResult(
                ok=False,
                did_work=False,
                status="not_found",
                message=f"Dream restore failed: run `{target}` not found.",
            )

        snapshot_dir = Path(str(row.get("snapshot_dir") or "")).expanduser()
        if not snapshot_dir.exists():
            return DreamRunResult(
                ok=False,
                did_work=False,
                status="missing_snapshot",
                message=f"Dream restore failed: snapshot missing for `{target}`.",
            )

        self._restore_file(snapshot_dir / "HISTORY.md", self.history_file)
        self._restore_file(snapshot_dir / "MEMORY.md", self.memory_file)
        self._restore_file(snapshot_dir / "DREAMS.md", self.dreams_file)
        self._restore_file(snapshot_dir / ".dream_cursor", self.cursor_file)
        self._restore_file(
            snapshot_dir / ".dreams" / "ingestion.json",
            self.ingestion_file,
        )
        self._restore_file(
            snapshot_dir / ".dreams" / "recall-store.json",
            self.recall_store_file,
        )
        self._restore_file(
            snapshot_dir / ".dreams" / "phase-signals.json",
            self.phase_signals_file,
        )

        restore_id = self._make_run_id(cursor_start=0, cursor_end=0)
        restore_row = {
            "run_id": restore_id,
            "timestamp": _now_iso(),
            "status": "restore",
            "processed_entries": 0,
            "cursor_start": int(self._read_cursor()),
            "cursor_end": int(self._read_cursor()),
            "summary": f"Restored memory snapshot from run {target}.",
            "snapshot_dir": str(snapshot_dir),
            "restored_from": target,
        }
        self._append_run_row(restore_row)
        return DreamRunResult(
            ok=True,
            did_work=True,
            status="restored",
            message=f"Dream restored memory state from run `{target}`.",
            run_id=restore_id,
        )

    def format_run_log(self, run_id: str = "") -> str:
        row = self.get_run(run_id) if str(run_id or "").strip() else None
        if row is None:
            rows = self.list_runs(limit=1)
            if not rows:
                return "Dream has no recorded runs yet."
            row = rows[0]

        phases = row.get("phases") if isinstance(row.get("phases"), dict) else {}
        light = phases.get("light", {}) if isinstance(phases, dict) else {}
        rem = phases.get("rem", {}) if isinstance(phases, dict) else {}
        deep = phases.get("deep", {}) if isinstance(phases, dict) else {}
        promoted = deep.get("promoted_count", 0)

        return (
            "## Dream Run\n\n"
            f"- Run: `{row.get('run_id', '')}`\n"
            f"- Timestamp: {row.get('timestamp', '')}\n"
            f"- Status: {row.get('status', '')}\n"
            f"- Processed entries: {row.get('processed_entries', 0)}\n"
            f"- Cursor: {row.get('cursor_start', 0)} -> {row.get('cursor_end', 0)}\n"
            f"- Snapshot: {row.get('snapshot_dir', '')}\n"
            f"- Light staged: {light.get('staged_count', 0)}\n"
            f"- REM themes: {rem.get('theme_count', 0)}\n"
            f"- Deep promoted: {promoted}\n\n"
            f"Summary:\n{row.get('summary', '')}"
        ).strip()

    def format_restore_list(self, *, limit: int = 10) -> str:
        rows = self.list_runs(limit=limit)
        if not rows:
            return "Dream has no recorded runs yet."
        lines = ["## Dream Restore", "", "Recent Dream runs (latest first):", ""]
        for row in rows:
            run_id = str(row.get("run_id") or "").strip()
            ts = str(row.get("timestamp") or "").strip()
            status = str(row.get("status") or "").strip()
            count = int(row.get("processed_entries") or 0)
            lines.append(f"- `{run_id}` {ts} status={status} processed={count}")
        lines.extend(
            [
                "",
                "Use `/dream-log <run_id>` to inspect a run.",
                "Use `/dream-restore <run_id>` to restore a snapshot.",
            ]
        )
        return "\n".join(lines)

    def _run_light_phase(
        self,
        *,
        pending: List[HistoryEntry],
        recall_store: Dict[str, Any],
        phase_signals: Dict[str, Any],
        run_id: str,
    ) -> Dict[str, Any]:
        items = recall_store["items"]
        staged: List[Dict[str, Any]] = []
        for entry in pending:
            normalized = entry.text.strip()
            if not normalized:
                continue
            candidate_id = hashlib.sha1(normalized.lower().encode("utf-8")).hexdigest()[
                :16
            ]
            tokens = _tokenize(normalized)
            record = items.get(candidate_id)
            if not isinstance(record, dict):
                record = {
                    "id": candidate_id,
                    "snippet": normalized,
                    "first_seen": entry.stamp,
                    "last_seen": entry.stamp,
                    "recall_count": 0,
                    "unique_queries": [],
                    "days": [],
                    "token_count": len(tokens),
                    "relevance_sum": 0.0,
                }
            record["last_seen"] = entry.stamp
            record["recall_count"] = int(record.get("recall_count", 0)) + 1

            unique_queries = record.get("unique_queries")
            if not isinstance(unique_queries, list):
                unique_queries = []
            if entry.day and entry.day not in unique_queries:
                unique_queries.append(entry.day)
            record["unique_queries"] = unique_queries[-50:]

            days = record.get("days")
            if not isinstance(days, list):
                days = []
            if entry.day and entry.day not in days:
                days.append(entry.day)
            record["days"] = sorted(days)[-90:]

            richness = self._concept_richness(normalized)
            relevance = min(1.0, 0.45 + 0.55 * richness)
            record["relevance_sum"] = (
                float(record.get("relevance_sum", 0.0)) + relevance
            )
            record["token_count"] = max(int(record.get("token_count", 0)), len(tokens))

            items[candidate_id] = record
            staged.append({"id": candidate_id, "snippet": normalized, "day": entry.day})

            phase_signals["light"][candidate_id] = {
                "last_seen": _now_iso(),
                "count": int(
                    phase_signals["light"].get(candidate_id, {}).get("count", 0)
                )
                + 1,
            }

        self._write_managed_block(
            self.dreams_file,
            "Light Sleep",
            self._render_light_block(run_id=run_id, staged=staged),
        )
        return {
            "staged_count": len(staged),
            "candidate_ids": [row["id"] for row in staged[:20]],
        }

    def _run_rem_phase(
        self,
        *,
        pending: List[HistoryEntry],
        phase_signals: Dict[str, Any],
        run_id: str,
    ) -> Dict[str, Any]:
        token_counts: Dict[str, int] = {}
        for entry in pending:
            for token in _tokenize(entry.text):
                token_counts[token] = token_counts.get(token, 0) + 1
        ranked = sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))
        themes = [token for token, _ in ranked[:8]]
        theme_ids: List[str] = []
        for theme in themes:
            theme_id = hashlib.sha1(theme.encode("utf-8")).hexdigest()[:16]
            theme_ids.append(theme_id)
        touched_candidates: set[str] = set()
        for entry in pending:
            candidate_id = hashlib.sha1(
                entry.text.strip().lower().encode("utf-8")
            ).hexdigest()[:16]
            touched_candidates.add(candidate_id)
        for candidate_id in touched_candidates:
            phase_signals["rem"][candidate_id] = {
                "last_seen": _now_iso(),
                "count": int(phase_signals["rem"].get(candidate_id, {}).get("count", 0))
                + 1,
            }

        self._write_managed_block(
            self.dreams_file,
            "REM Sleep",
            self._render_rem_block(run_id=run_id, themes=themes),
        )
        return {
            "theme_count": len(themes),
            "themes": themes,
            "theme_ids": theme_ids,
        }

    def _run_deep_phase(
        self,
        *,
        recall_store: Dict[str, Any],
        phase_signals: Dict[str, Any],
        run_id: str,
    ) -> Dict[str, Any]:
        items = recall_store.get("items") if isinstance(recall_store, dict) else {}
        if not isinstance(items, dict):
            items = {}

        metrics: List[Dict[str, Any]] = []
        max_recall = 1
        for value in items.values():
            if isinstance(value, dict):
                max_recall = max(max_recall, int(value.get("recall_count", 0)))

        for candidate_id, record in items.items():
            if not isinstance(record, dict):
                continue
            snippet = str(record.get("snippet") or "").strip()
            if not snippet:
                continue
            recall_count = max(0, int(record.get("recall_count", 0)))
            days = record.get("days") if isinstance(record.get("days"), list) else []
            unique_queries = (
                record.get("unique_queries")
                if isinstance(record.get("unique_queries"), list)
                else []
            )
            relevance_sum = float(record.get("relevance_sum", 0.0))

            frequency = min(1.0, recall_count / float(max_recall))
            relevance = min(1.0, relevance_sum / float(max(1, recall_count)))
            query_diversity = min(1.0, len(unique_queries) / 5.0)
            consolidation = min(1.0, len(days) / 4.0)
            recency = self._recency_score(str(record.get("last_seen") or ""))
            conceptual_richness = self._concept_richness(snippet)
            phase_boost = self._phase_boost(candidate_id, phase_signals)

            score = (
                _DEEP_WEIGHTS["frequency"] * frequency
                + _DEEP_WEIGHTS["relevance"] * relevance
                + _DEEP_WEIGHTS["query_diversity"] * query_diversity
                + _DEEP_WEIGHTS["recency"] * recency
                + _DEEP_WEIGHTS["consolidation"] * consolidation
                + _DEEP_WEIGHTS["conceptual_richness"] * conceptual_richness
                + phase_boost
            )

            metrics.append(
                {
                    "id": candidate_id,
                    "snippet": snippet,
                    "score": score,
                    "recall_count": recall_count,
                    "unique_query_count": len(unique_queries),
                    "frequency": frequency,
                    "relevance": relevance,
                    "query_diversity": query_diversity,
                    "recency": recency,
                    "consolidation": consolidation,
                    "conceptual_richness": conceptual_richness,
                    "phase_boost": phase_boost,
                }
            )

        ranked = sorted(metrics, key=lambda row: (-float(row["score"]), str(row["id"])))

        existing_memory = self._read_memory_lines()
        promoted: List[Dict[str, Any]] = []
        for row in ranked:
            if float(row["score"]) < _MIN_DEEP_SCORE:
                continue
            if int(row["recall_count"]) < _MIN_RECALL_COUNT:
                continue
            if int(row["unique_query_count"]) < _MIN_UNIQUE_QUERIES:
                continue
            if row["snippet"] in existing_memory:
                continue
            promoted.append(row)

        if promoted:
            self._append_memory_entries([row["snippet"] for row in promoted])
            self._write_phase_report(
                "deep", run_id=run_id, promoted=promoted, ranked=ranked
            )

        self._write_managed_block(
            self.dreams_file,
            "Deep Sleep",
            self._render_deep_block(
                run_id=run_id,
                promoted=promoted,
                top_ranked=ranked[:8],
            ),
        )
        return {
            "promoted_count": len(promoted),
            "candidate_count": len(ranked),
            "promoted_ids": [str(row["id"]) for row in promoted[:20]],
            "thresholds": {
                "min_score": _MIN_DEEP_SCORE,
                "min_recall_count": _MIN_RECALL_COUNT,
                "min_unique_queries": _MIN_UNIQUE_QUERIES,
            },
        }

    def _read_history_entries(self) -> List[HistoryEntry]:
        if not self.history_file.exists():
            return []
        try:
            lines = self.history_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

        entries: List[HistoryEntry] = []
        for line in lines:
            text = str(line or "").strip()
            if not text or text.startswith("#") or "[DREAM]" in text:
                continue
            match = _STAMP_RE.match(text)
            if match is None:
                continue
            day = str(match.group(1) or "")
            time_text = str(match.group(2) or "")
            stamp = f"{day}T{time_text}:00"
            body = _STAMP_RE.sub("", text, count=1).strip()
            if not body:
                continue
            entries.append(HistoryEntry(stamp=stamp, day=day, text=body))
        return entries

    def _read_cursor(self) -> int:
        if not self.cursor_file.exists():
            return 0
        try:
            return max(0, int(self.cursor_file.read_text(encoding="utf-8").strip()))
        except (OSError, ValueError):
            return 0

    def _write_cursor(self, value: int) -> None:
        self.cursor_file.write_text(str(max(0, int(value))), encoding="utf-8")

    def _read_run_rows(self) -> List[Dict[str, Any]]:
        if not self.runs_file.exists():
            return []
        rows: List[Dict[str, Any]] = []
        try:
            with self.runs_file.open("r", encoding="utf-8") as fh:
                for raw in fh:
                    text = str(raw or "").strip()
                    if not text:
                        continue
                    try:
                        row = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        rows.append(dict(row))
        except OSError:
            return []
        return rows

    def _append_run_row(self, row: Dict[str, Any]) -> None:
        with self.runs_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    def _append_history_line(self, text: str) -> None:
        line = str(text or "").strip()
        if not line:
            return
        if self.history_file.exists():
            current = self.history_file.read_text(encoding="utf-8").rstrip()
            merged = (current + "\n\n" + line).strip() + "\n"
            self.history_file.write_text(merged, encoding="utf-8")
            return
        self.history_file.write_text("# History\n\n" + line + "\n", encoding="utf-8")

    def _snapshot_current_state(self, run_id: str) -> Path:
        target = self.snapshots_dir / str(run_id)
        target.mkdir(parents=True, exist_ok=True)
        self._copy_if_exists(self.history_file, target / "HISTORY.md")
        self._copy_if_exists(self.memory_file, target / "MEMORY.md")
        self._copy_if_exists(self.dreams_file, target / "DREAMS.md")
        self._copy_if_exists(self.cursor_file, target / ".dream_cursor")
        self._copy_if_exists(
            self.ingestion_file,
            target / ".dreams" / "ingestion.json",
        )
        self._copy_if_exists(
            self.recall_store_file,
            target / ".dreams" / "recall-store.json",
        )
        self._copy_if_exists(
            self.phase_signals_file,
            target / ".dreams" / "phase-signals.json",
        )
        return target

    @staticmethod
    def _copy_if_exists(src: Path, dst: Path) -> None:
        if not src.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    @staticmethod
    def _restore_file(src: Path, dst: Path) -> None:
        if not src.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    @staticmethod
    def _make_run_id(*, cursor_start: int, cursor_end: int) -> str:
        seed = f"{_now_iso()}::{cursor_start}:{cursor_end}"
        return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]

    def _build_summary_entry(
        self,
        *,
        pending: List[HistoryEntry],
        cursor_start: int,
        cursor_end: int,
        promoted: int,
    ) -> str:
        keywords = self._top_keywords([entry.text for entry in pending], top_k=4)
        keyword_text = ", ".join(keywords) if keywords else "none"
        return (
            f"[{_now_stamp()}] [DREAM] consolidated {len(pending)} entries "
            f"(cursor {cursor_start}->{cursor_end}); promoted={promoted}; "
            f"keywords: {keyword_text}."
        )

    @staticmethod
    def _top_keywords(entries: List[str], *, top_k: int) -> List[str]:
        counts: Dict[str, int] = {}
        for line in entries:
            for token in _tokenize(str(line or "").strip()):
                counts[token] = counts.get(token, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [token for token, _ in ranked[: max(1, int(top_k))]]

    def _resolve_dreams_file(self) -> Path:
        lower = self.memory_dir / "dreams.md"
        upper = self.memory_dir / "DREAMS.md"
        if lower.exists() and not upper.exists():
            return lower
        return upper

    def _write_managed_block(self, path: Path, title: str, body: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        header = f"## {title} (managed)\n"
        managed = header + body.strip() + "\n"
        text = path.read_text(encoding="utf-8") if path.exists() else ""

        pattern = re.compile(_MANAGED_BLOCK_RE_TEMPLATE.format(title=re.escape(title)))
        if pattern.search(text):
            updated = pattern.sub(managed, text, count=1)
        else:
            prefix = text.rstrip()
            updated = (prefix + "\n\n" + managed).strip() + "\n"
        path.write_text(updated, encoding="utf-8")

    def _render_light_block(self, *, run_id: str, staged: List[Dict[str, Any]]) -> str:
        lines = [
            f"- run_id: `{run_id}`",
            f"- staged candidates: {len(staged)}",
            "- latest snippets:",
        ]
        for row in staged[:8]:
            snippet = str(row.get("snippet") or "").strip()
            if snippet:
                lines.append(f"  - {snippet}")
        if len(lines) == 3:
            lines.append("  - (none)")
        return "\n".join(lines)

    def _render_rem_block(self, *, run_id: str, themes: List[str]) -> str:
        lines = [
            f"- run_id: `{run_id}`",
            f"- themes: {len(themes)}",
            f"- top themes: {', '.join(themes) if themes else '(none)'}",
        ]
        return "\n".join(lines)

    def _render_deep_block(
        self,
        *,
        run_id: str,
        promoted: List[Dict[str, Any]],
        top_ranked: List[Dict[str, Any]],
    ) -> str:
        lines = [
            f"- run_id: `{run_id}`",
            f"- promoted: {len(promoted)}",
            "- promoted snippets:",
        ]
        for row in promoted[:8]:
            lines.append(f"  - {row['snippet']}")
        if not promoted:
            lines.append("  - (none)")
        lines.append("- top ranked:")
        for row in top_ranked[:8]:
            lines.append(
                f"  - score={float(row['score']):.3f} recall={int(row['recall_count'])} {row['snippet']}"
            )
        if not top_ranked:
            lines.append("  - (none)")
        return "\n".join(lines)

    def _write_phase_report(
        self,
        phase: str,
        *,
        run_id: str,
        promoted: List[Dict[str, Any]],
        ranked: List[Dict[str, Any]],
    ) -> None:
        day = datetime.now().strftime("%Y-%m-%d")
        target = self.phase_report_root / phase / f"{day}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Dreaming {phase.title()} Report",
            "",
            f"- date: {day}",
            f"- run_id: `{run_id}`",
            f"- promoted_count: {len(promoted)}",
            f"- candidate_count: {len(ranked)}",
            "",
            "## Promoted",
            "",
        ]
        for row in promoted:
            lines.append(f"- score={float(row['score']):.3f} {row['snippet']}")
        if not promoted:
            lines.append("- (none)")
        target.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    def _read_memory_lines(self) -> set[str]:
        if not self.memory_file.exists():
            return set()
        try:
            lines = self.memory_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            return set()
        out: set[str] = set()
        for line in lines:
            normalized = self._normalize_memory_line(str(line or ""))
            if not normalized:
                continue
            out.add(normalized)
        return out

    def _append_memory_entries(self, snippets: List[str]) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d")
        existing = self._read_memory_lines()
        unique_snippets: List[str] = []
        seen_batch: set[str] = set()
        for raw in snippets:
            normalized = self._normalize_memory_line(str(raw or ""))
            if not normalized:
                continue
            if normalized in existing or normalized in seen_batch:
                continue
            seen_batch.add(normalized)
            unique_snippets.append(normalized)
        lines = [f"- [{stamp}] {snippet}" for snippet in unique_snippets]
        if not lines:
            return
        if self.memory_file.exists():
            body = self.memory_file.read_text(encoding="utf-8").rstrip()
            merged = (body + "\n" + "\n".join(lines)).strip() + "\n"
            self.memory_file.write_text(merged, encoding="utf-8")
            return
        self.memory_file.write_text(
            "# Memory\n\n" + "\n".join(lines) + "\n", encoding="utf-8"
        )

    def _write_ingestion_state(self, state: Dict[str, Any]) -> None:
        payload = dict(state)
        payload["cursor"] = int(payload.get("cursor", self._read_cursor()))
        self._write_json(self.ingestion_file, payload)

    @staticmethod
    def _concept_richness(text: str) -> float:
        tokens = _tokenize(text)
        if not tokens:
            return 0.0
        unique = len(set(tokens))
        return min(1.0, unique / 10.0)

    @staticmethod
    def _recency_score(last_seen: str) -> float:
        dt = _parse_iso(last_seen)
        if dt is None:
            return 0.2
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_seconds = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
        age_days = age_seconds / 86400.0
        return math.exp(-age_days / 7.0)

    def _phase_boost(self, candidate_id: str, phase_signals: Dict[str, Any]) -> float:
        now = datetime.now(timezone.utc)

        def signal_component(phase: str) -> float:
            phase_data = phase_signals.get(phase, {})
            if not isinstance(phase_data, dict):
                return 0.0
            row = phase_data.get(candidate_id)
            if not isinstance(row, dict):
                return 0.0
            seen = _parse_iso(str(row.get("last_seen") or ""))
            if seen is None:
                return 0.0
            if seen.tzinfo is None:
                seen = seen.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (now - seen).total_seconds() / 86400.0)
            decay = math.exp(-age_days / _PHASE_SIGNAL_DECAY_DAYS)
            count = max(1, int(row.get("count", 1)))
            return min(1.0, math.log1p(count) / 3.0) * decay

        light_signal = signal_component("light")
        rem_signal = signal_component("rem")
        return min(
            _PHASE_SIGNAL_MAX_BOOST,
            _PHASE_SIGNAL_MAX_BOOST * (0.75 * light_signal + 0.25 * rem_signal),
        )

    def _read_json(self, path: Path, *, default: Dict[str, Any]) -> Dict[str, Any]:
        if not path.exists():
            return dict(default)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except (OSError, json.JSONDecodeError):
            pass
        return dict(default)

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def _acquire_lock(self) -> bool:
        now = datetime.now(timezone.utc)
        payload = {
            "pid": int(os.getpid()),
            "acquired_at": _now_iso(),
        }
        if self._try_create_lock(payload):
            return True

        if not self.lock_file.exists():
            return self._try_create_lock(payload)

        try:
            existing = json.loads(self.lock_file.read_text(encoding="utf-8"))
            ts = _parse_iso(str(existing.get("acquired_at") or ""))
            if ts is not None:
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age = (now - ts).total_seconds()
                if age < 3600:
                    return False
        except (OSError, json.JSONDecodeError):
            pass

        with contextlib.suppress(OSError):
            self.lock_file.unlink()
        return self._try_create_lock(payload)

    def _release_lock(self) -> None:
        if self.lock_file.exists():
            with contextlib.suppress(OSError):
                self.lock_file.unlink()

    @staticmethod
    def _normalize_memory_line(text: str) -> str:
        line = str(text or "").strip()
        if not line or line.startswith("#"):
            return ""
        if line.startswith("- "):
            line = line[2:].strip()
        if re.match(r"^\[\d{4}-\d{2}-\d{2}\]\s+", line):
            line = re.sub(r"^\[\d{4}-\d{2}-\d{2}\]\s+", "", line, count=1).strip()
        return line

    def _try_create_lock(self, payload: Dict[str, Any]) -> bool:
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            fd = os.open(self.lock_file, flags)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                return False
            raise
        try:
            os.write(
                fd,
                (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
                    "utf-8"
                ),
            )
        finally:
            os.close(fd)
        return True


__all__ = ["DreamMemoryManager", "DreamRunResult"]
