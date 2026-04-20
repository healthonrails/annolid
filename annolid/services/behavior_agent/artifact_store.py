"""Immutable artifact writer for typed behavior-agent analysis runs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from annolid.domain.behavior_agent import (
    AnalysisRun,
    BehaviorSegment,
    Episode,
    MemoryRecord,
    SCHEMA_VERSION,
    TaskPlan,
    TrackArtifact,
)


class BehaviorAgentArtifactStore:
    """Writes replayable run outputs as manifest + NDJSON/Parquet artifacts."""

    def __init__(self, root_dir: str | Path) -> None:
        self._root_dir = Path(root_dir).expanduser().resolve()

    def write_run(
        self,
        *,
        analysis_run: AnalysisRun,
        episode: Episode,
        task_plan: TaskPlan,
        track_artifacts: Iterable[TrackArtifact],
        behavior_segments: Iterable[BehaviorSegment],
        memory_records: Iterable[MemoryRecord],
        metrics_rows: Iterable[dict[str, Any]] | None = None,
        analysis_code: str | None = None,
        report_text: str | None = None,
        report_html: str | None = None,
        evidence_rows: Iterable[dict[str, Any]] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> Path:
        run_dir = self._root_dir / "analysis_runs" / analysis_run.run_id
        if run_dir.exists():
            raise FileExistsError(
                f"analysis run directory already exists (immutable): {run_dir}"
            )
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=False)

        track_path = artifacts_dir / "tracks.ndjson"
        segment_path = artifacts_dir / "behaviors.ndjson"
        memory_path = artifacts_dir / "memory.ndjson"

        self._write_ndjson(track_path, (record.to_dict() for record in track_artifacts))
        self._write_ndjson(
            segment_path, (record.to_dict() for record in behavior_segments)
        )
        self._write_ndjson(memory_path, (record.to_dict() for record in memory_records))

        metrics_paths: dict[str, str | None] = {"ndjson": None, "parquet": None}
        if metrics_rows is not None:
            metrics_paths = self._write_metrics_artifact(
                artifacts_dir / "metrics", list(metrics_rows)
            )

        analysis_code_rel: str | None = None
        if analysis_code:
            code_path = artifacts_dir / "analysis.py"
            code_path.write_text(str(analysis_code), encoding="utf-8")
            analysis_code_rel = str(code_path.relative_to(run_dir))

        report_rel: str | None = None
        if report_text:
            report_path = artifacts_dir / "report.md"
            report_path.write_text(str(report_text), encoding="utf-8")
            report_rel = str(report_path.relative_to(run_dir))

        report_html_rel: str | None = None
        if report_html:
            report_html_path = artifacts_dir / "report.html"
            report_html_path.write_text(str(report_html), encoding="utf-8")
            report_html_rel = str(report_html_path.relative_to(run_dir))

        evidence_rel: str | None = None
        if evidence_rows is not None:
            evidence_path = artifacts_dir / "evidence.ndjson"
            self._write_ndjson(evidence_path, evidence_rows)
            evidence_rel = str(evidence_path.relative_to(run_dir))

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "analysis_run": asdict(analysis_run),
            "episode": asdict(episode),
            "task_plan": asdict(task_plan),
            "artifacts": {
                "tracks": str(track_path.relative_to(run_dir)),
                "segments": str(segment_path.relative_to(run_dir)),
                "memory": str(memory_path.relative_to(run_dir)),
                "metrics": dict(metrics_paths),
                "analysis_code": analysis_code_rel,
                "report": report_rel,
                "report_html": report_html_rel,
                "evidence": evidence_rel,
            },
            "provenance": dict(provenance or {}),
        }

        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return manifest_path

    @staticmethod
    def _write_ndjson(path: Path, records: Iterable[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")

    @staticmethod
    def _write_metrics_artifact(
        base_path: Path, rows: list[dict[str, Any]]
    ) -> dict[str, str | None]:
        ndjson_path = base_path.with_suffix(".ndjson")
        with ndjson_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")

        # Parquet is optional. NDJSON remains the stable baseline contract.
        parquet_path = base_path.with_suffix(".parquet")
        try:
            import pandas as pd

            frame = pd.DataFrame(rows)
            frame.to_parquet(parquet_path, index=False)
            parquet_rel: str | None = str(parquet_path.name)
        except Exception:
            parquet_rel = None
        return {"ndjson": str(ndjson_path.name), "parquet": parquet_rel}


__all__ = ["BehaviorAgentArtifactStore"]
