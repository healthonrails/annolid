"""Analysis coding agent backed by Annolid Bot ACP generation when available."""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from annolid.core.agent.providers.background_chat import (
    dependency_error_for_kind,
    run_codex_cli_chat,
    run_gemini_chat,
    run_ollama_streaming_chat,
    run_openai_codex_chat,
    run_openai_compat_chat,
)
from annolid.domain.behavior_agent import BehaviorSegment, TaskPlan, TrackArtifact
from annolid.infrastructure.sandbox.execution import execute_generated_analysis
from annolid.services.behavior_agent.bout_scoring import (
    aggregate_aggression_bout_counts,
)
from annolid.utils.llm_settings import (
    ensure_provider_env,
    load_llm_settings,
    provider_kind,
    resolve_llm_config,
)

logger = logging.getLogger(__name__)
_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_ACP_TIMEOUT_SECONDS = 60.0
_SUPPORTED_PROVIDER_KINDS = {
    "codex_cli",
    "openai_compat",
    "openai_codex",
    "gemini",
    "ollama",
}


@dataclass(frozen=True)
class CodingResult:
    code: str
    execution_output: dict[str, Any]
    derived_metrics: list[dict[str, Any]]
    evidence: list[dict[str, object]] = field(default_factory=list)


class AnalysisCodingAgent:
    """Generate and run compact sandboxed analysis code."""

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        settings: dict[str, Any] | None = None,
        invoke_turn: Callable[..., tuple[str, str]] | None = None,
        invoke_openai_compat: Callable[..., tuple[str, str]] | None = None,
        invoke_openai_codex: Callable[..., tuple[str, str]] | None = None,
        invoke_gemini: Callable[..., tuple[str, str]] | None = None,
        invoke_ollama_streaming: Callable[..., None] | None = None,
    ) -> None:
        self._provider_override = str(provider or "").strip() or None
        self._model_override = str(model or "").strip() or None
        self._settings = dict(settings or {})
        self._invoke_codex_cli = invoke_turn or run_codex_cli_chat
        self._invoke_openai_compat = invoke_openai_compat or run_openai_compat_chat
        self._invoke_openai_codex = invoke_openai_codex or run_openai_codex_chat
        self._invoke_gemini = invoke_gemini or run_gemini_chat
        self._invoke_ollama_streaming = (
            invoke_ollama_streaming or run_ollama_streaming_chat
        )
        self._session_id = f"acp:analysis_coding:{uuid.uuid4().hex[:12]}"

    def run(
        self,
        *,
        plan: TaskPlan,
        artifacts: list[TrackArtifact],
        segments: list[BehaviorSegment],
    ) -> CodingResult:
        code, generation_meta = self._generate_code_with_meta(plan)
        inputs = {
            "assay_type": str(plan.assay_type),
            "artifact_count": int(len(artifacts)),
            "segment_count": int(len(segments)),
        }
        exec_output = execute_generated_analysis(code, inputs)
        metrics = self._derive_metrics(
            plan=plan, artifacts=artifacts, segments=segments
        )
        evidence_row: dict[str, object] = {
            "stage": "analysis_coding",
            "assay_type": str(plan.assay_type),
            "generation_mode": str(generation_meta["mode"]),
            "provider": str(generation_meta["provider"]),
            "model": str(generation_meta["model"]),
            "generated_code_chars": len(code),
            "execution_status": str(exec_output.get("status") or "ok"),
            "metric_count": len(metrics),
        }
        fallback_reason = str(generation_meta.get("fallback_reason") or "").strip()
        if fallback_reason:
            evidence_row["fallback_reason"] = fallback_reason
        return CodingResult(
            code=code,
            execution_output=exec_output,
            derived_metrics=metrics,
            evidence=[evidence_row],
        )

    def generate_code(self, plan: TaskPlan) -> str:
        code, _ = self._generate_code_with_meta(plan)
        return code

    def execute_code(self, code: str, inputs: dict[str, Any]) -> dict[str, Any]:
        return execute_generated_analysis(code, inputs)

    def _generate_code_with_meta(
        self,
        plan: TaskPlan,
    ) -> tuple[str, dict[str, str]]:
        config = resolve_llm_config(
            profile=None,
            provider=self._provider_override,
            model=self._model_override,
            persist=False,
        )
        settings = self._resolved_settings()
        kind = provider_kind(settings, config.provider)
        fallback_meta = {
            "mode": "deterministic_fallback",
            "provider": str(config.provider),
            "model": str(config.model),
            "fallback_reason": "",
        }
        if kind not in _SUPPORTED_PROVIDER_KINDS:
            fallback_meta["fallback_reason"] = f"provider_kind_not_supported:{kind}"
            return self._deterministic_code(plan), fallback_meta

        dependency_error = dependency_error_for_kind(kind)
        if dependency_error:
            fallback_meta["fallback_reason"] = dependency_error
            return self._deterministic_code(plan), fallback_meta

        ensure_provider_env(config)
        try:
            raw_response = self._generate_with_provider(
                provider_kind_name=kind,
                provider_name=str(config.provider),
                model_name=str(config.model),
                settings=settings,
                prompt=self._build_prompt(plan),
            )
            code = self._extract_python_module(raw_response)
            self._validate_generated_code(code)
            self._validate_sandbox_compatibility(code, plan)
            return code, {
                "mode": "annolid_bot_provider",
                "provider": str(config.provider),
                "model": str(config.model),
                "fallback_reason": "",
            }
        except Exception as exc:
            logger.warning("ACP analysis code generation failed: %s", exc)
            fallback_meta["fallback_reason"] = str(exc) or exc.__class__.__name__
            return self._deterministic_code(plan), fallback_meta

    def _resolved_settings(self) -> dict[str, Any]:
        if self._settings:
            return dict(self._settings)
        return load_llm_settings()

    def _generate_with_provider(
        self,
        *,
        provider_kind_name: str,
        provider_name: str,
        model_name: str,
        settings: dict[str, Any],
        prompt: str,
    ) -> str:
        history_loader = self._load_history_messages
        if provider_kind_name == "codex_cli":
            _, text = self._invoke_codex_cli(
                prompt=prompt,
                image_path="",
                model=model_name,
                provider_name=provider_name,
                settings=settings,
                load_history_messages=history_loader,
                session_id=self._session_id,
                runtime="acp",
                timeout_s=_ACP_TIMEOUT_SECONDS,
                max_tokens=2048,
            )
            return text
        if provider_kind_name == "openai_compat":
            _, text = self._invoke_openai_compat(
                prompt=prompt,
                image_path="",
                model=model_name,
                provider_name=provider_name,
                settings=settings,
                load_history_messages=history_loader,
                timeout_s=_ACP_TIMEOUT_SECONDS,
                max_tokens=2048,
            )
            return text
        if provider_kind_name == "openai_codex":
            _, text = self._invoke_openai_codex(
                prompt=prompt,
                image_path="",
                model=model_name,
                provider_name=provider_name,
                settings=settings,
                load_history_messages=history_loader,
                timeout_s=_ACP_TIMEOUT_SECONDS,
                max_tokens=2048,
            )
            return text
        if provider_kind_name == "gemini":
            _, text = self._invoke_gemini(
                prompt=prompt,
                image_path="",
                model=model_name,
                provider_name=provider_name,
                settings=settings,
            )
            return text
        if provider_kind_name == "ollama":
            return self._run_ollama_generation(
                prompt=prompt,
                model_name=model_name,
                settings=settings,
                load_history_messages=history_loader,
            )
        raise ValueError(f"Unsupported provider kind: {provider_kind_name}")

    def _run_ollama_generation(
        self,
        *,
        prompt: str,
        model_name: str,
        settings: dict[str, Any],
        load_history_messages: Callable[[], list[dict[str, Any]]],
    ) -> str:
        chunks: list[str] = []
        final_error = {"message": "", "is_error": False}
        persisted = {"assistant": ""}

        def _emit_chunk(chunk: str) -> None:
            chunks.append(str(chunk or ""))

        def _emit_final(message: str, is_error: bool) -> None:
            final_error["message"] = str(message or "")
            final_error["is_error"] = bool(is_error)

        def _persist_turn(_prompt: str, assistant: str) -> None:
            persisted["assistant"] = str(assistant or "")

        self._invoke_ollama_streaming(
            prompt=prompt,
            image_path="",
            model=model_name,
            settings=settings,
            load_history_messages=load_history_messages,
            emit_chunk=_emit_chunk,
            emit_final=_emit_final,
            persist_turn=_persist_turn,
        )
        if bool(final_error["is_error"]):
            raise RuntimeError(str(final_error["message"] or "ollama_stream_error"))
        text = str(persisted["assistant"] or "".join(chunks)).strip()
        if not text:
            raise RuntimeError("Ollama returned empty analysis code.")
        return text

    @staticmethod
    def _load_history_messages() -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are Annolid Bot running in ACP coding mode. "
                    "Return only valid Python source code. "
                    "Do not use imports. "
                    "Define exactly one function named run(inputs)."
                ),
            }
        ]

    @staticmethod
    def _build_prompt(plan: TaskPlan) -> str:
        objectives = [
            str(item).strip() for item in plan.objectives if str(item).strip()
        ]
        target_features = [
            str(item).strip() for item in plan.target_features if str(item).strip()
        ]
        lines = [
            "Write a compact Python module for Annolid sandbox execution.",
            "Constraints:",
            "- no imports",
            "- define `run(inputs)`",
            "- read keys `assay_type`, `artifact_count`, `segment_count`, and `bout_count`",
            "- return a dict with at least `status`, `assay_type`, `artifact_count`, and `segment_count`",
            "- do not include markdown fences or commentary",
            "",
            f"Assay type: {plan.assay_type}",
        ]
        if objectives:
            lines.append(f"Objectives: {', '.join(objectives)}")
        if target_features:
            lines.append(f"Target features: {', '.join(target_features)}")
        return "\n".join(lines)

    @staticmethod
    def _extract_python_module(raw_text: str) -> str:
        text = str(raw_text or "").strip()
        if not text:
            raise ValueError("ACP returned empty analysis code.")

        fenced_blocks = _CODE_FENCE_RE.findall(text)
        if fenced_blocks:
            for block in fenced_blocks:
                candidate = str(block or "").strip()
                if "def run" in candidate:
                    return candidate
            return str(fenced_blocks[0] or "").strip()
        return text

    @staticmethod
    def _validate_generated_code(code: str) -> None:
        normalized = str(code or "").strip()
        if "def run(" not in normalized:
            raise ValueError("ACP response did not define run(inputs).")
        if "import " in normalized or "from " in normalized:
            raise ValueError(
                "ACP response used imports, which sandbox execution forbids."
            )

    @staticmethod
    def _validate_sandbox_compatibility(code: str, plan: TaskPlan) -> None:
        result = execute_generated_analysis(
            str(code),
            {
                "assay_type": str(plan.assay_type),
                "artifact_count": 0,
                "segment_count": 0,
                "bout_count": 0,
            },
        )
        if not isinstance(result, dict):
            raise ValueError("Generated analysis must return a dict.")
        required = {"status", "assay_type", "artifact_count", "segment_count"}
        missing = sorted(required - set(result.keys()))
        if missing:
            raise ValueError(
                f"Generated analysis result missing required keys: {', '.join(missing)}"
            )
        if str(result.get("status") or "").strip().lower() != "ok":
            raise ValueError("Generated analysis result must set status='ok'.")

    @staticmethod
    def _deterministic_code(plan: TaskPlan) -> str:
        return (
            "def run(inputs):\n"
            f"    assay = str(inputs.get('assay_type', {plan.assay_type!r}))\n"
            "    artifacts = int(inputs.get('artifact_count', 0))\n"
            "    segments = int(inputs.get('segment_count', 0))\n"
            "    bouts = int(inputs.get('bout_count', 0))\n"
            "    return {\n"
            "        'status': 'ok',\n"
            "        'assay_type': assay,\n"
            "        'artifact_count': artifacts,\n"
            "        'segment_count': segments,\n"
            "        'bout_count': bouts,\n"
            "    }\n"
        )

    @staticmethod
    def _derive_metrics(
        *,
        plan: TaskPlan,
        artifacts: list[TrackArtifact],
        segments: list[BehaviorSegment],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = [
            {
                "metric": "artifact_count",
                "value": int(len(artifacts)),
            },
            {
                "metric": "segment_count",
                "value": int(len(segments)),
            },
        ]
        if str(plan.assay_type).strip().lower() == "aggression":
            for row in aggregate_aggression_bout_counts(segments):
                rows.append(
                    {
                        "metric": "aggression_bout",
                        **row.to_dict(),
                    }
                )
        return rows


__all__ = ["AnalysisCodingAgent", "CodingResult"]
