from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence, Tuple


EvidenceRows = List[Tuple[str, List[str]]]


def select_annolid_reference_paths(
    *,
    root: Path,
    topic: str,
    include_code_refs: bool,
) -> list[str]:
    lower_topic = str(topic or "").lower()
    refs: list[Path] = [
        root / "README.md",
        root / "docs" / "source" / "agent_tools.md",
    ]
    if any(k in lower_topic for k in ("realtime", "rtsp", "camera", "stream")):
        refs.extend(
            [
                root / "annolid" / "gui" / "realtime_launch.py",
                root / "annolid" / "gui" / "widgets" / "realtime_control_widget.py",
                root / "annolid" / "gui" / "widgets" / "realtime_manager.py",
            ]
        )
    if any(k in lower_topic for k in ("annotation", "segment", "track", "label")):
        refs.extend(
            [
                root / "annolid" / "engine" / "cli.py",
                root / "annolid" / "gui",
            ]
        )
    if include_code_refs:
        refs.extend(
            [
                root / "annolid" / "gui" / "widgets" / "ai_chat_backend.py",
                root / "annolid" / "core" / "agent" / "gui_backend" / "commands.py",
            ]
        )

    normalized: list[str] = []
    seen: set[str] = set()
    for path in refs:
        candidate = Path(path)
        if not candidate.exists():
            continue
        try:
            rel = str(candidate.resolve().relative_to(root.resolve()))
        except Exception:
            rel = str(candidate)
        key = rel.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(rel)
    return normalized[:10]


def build_tutorial_model_prompts(
    *,
    topic: str,
    level: str,
    evidence_rows: EvidenceRows,
    references: Sequence[str],
) -> tuple[str, str]:
    evidence_lines: list[str] = []
    for rel, findings in evidence_rows:
        evidence_lines.append(f"FILE: {rel}")
        for finding in findings:
            evidence_lines.append(f"- {finding}")
    if not evidence_lines:
        evidence_lines.append("- No structured evidence extracted.")
    evidence_block = "\n".join(evidence_lines)
    ref_lines = "\n".join(f"- {ref}" for ref in references)

    system_prompt = (
        "You are an Annolid expert technical writer. "
        "Write a high-quality, practical tutorial in Markdown. "
        "Use ONLY repository evidence provided by the user prompt. "
        "Do not invent APIs/files/settings. "
        "Cite concrete file paths from provided references."
    )
    user_prompt = (
        f"Topic: {topic}\n"
        f"Level: {level}\n\n"
        "Repository evidence:\n"
        f"{evidence_block}\n\n"
        "Reference files:\n"
        f"{ref_lines}\n\n"
        "Write Markdown with sections:\n"
        "1) Title\n"
        "2) Goal\n"
        "3) Prerequisites\n"
        "4) Step-by-step workflow\n"
        "5) Common failure modes and fixes\n"
        "6) Validation checklist\n"
        "7) Reference files\n"
        "Include concrete settings/actions where evidence supports them."
    )
    return system_prompt, user_prompt


def collect_tutorial_evidence(
    *,
    root: Path,
    topic: str,
    refs: list[str],
    max_files: int = 10,
) -> tuple[EvidenceRows, list[str]]:
    source_files = _expand_tutorial_reference_files(
        root=root,
        refs=refs,
        topic=topic,
        max_files=max_files,
    )
    evidence_rows: EvidenceRows = []
    used_refs: list[str] = []
    for file_path in source_files:
        try:
            rel = str(file_path.resolve().relative_to(root))
        except Exception:
            rel = str(file_path)
        try:
            raw = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        text = raw[:14000]
        if file_path.suffix.lower() == ".md":
            findings = _extract_markdown_evidence(text, topic=topic)
        else:
            findings = _extract_python_evidence(text, topic=topic)
        if not findings:
            continue
        evidence_rows.append((rel, findings[:4]))
        used_refs.append(rel)
        if len(evidence_rows) >= 8:
            break
    return evidence_rows, used_refs


def build_tutorial_workflow_steps(
    *,
    topic: str,
    level: str,
    evidence_rows: EvidenceRows,
) -> list[str]:
    lower_topic = str(topic or "").lower()
    steps: list[str] = []
    if any(k in lower_topic for k in ("realtime", "camera", "stream", "rtsp")):
        steps.extend(
            [
                "Configure source/model in Realtime Control, then validate stream connectivity before inference.",
                "Start realtime inference with a lightweight model first (for example `yolo11n`), then increase complexity.",
                "Use periodic snapshot/report actions only after detection confidence and event filters are stable.",
            ]
        )
    elif any(k in lower_topic for k in ("behavior", "segment", "track", "label")):
        steps.extend(
            [
                "Open a representative video and confirm frame navigation and annotation model selection.",
                "Run segmentation/tracking with a constrained prompt first, then expand labels after baseline quality is good.",
                "Export behavior segments and verify timestamps/labels before downstream analysis.",
            ]
        )
    else:
        steps.extend(
            [
                "Start with a minimal end-to-end run using one video or stream and default model settings.",
                "Record failures and bottlenecks from logs/UI state before changing multiple settings at once.",
                "Iterate with one controlled parameter change per run (model, threshold, or source).",
            ]
        )
    if evidence_rows:
        steps.append(
            f"Cross-check implementation details directly in `{evidence_rows[0][0]}`."
        )
    if level == "advanced":
        steps.append(
            "Map this workflow to extension points (tool handlers, router, and session/memory layers) before customization."
        )
    return steps[:5]


def build_tutorial_fallback_markdown(
    *,
    topic: str,
    level: str,
    evidence_rows: EvidenceRows,
    references: list[str],
) -> str:
    level_goals = {
        "beginner": "Focus on quick setup, first run, and common pitfalls.",
        "intermediate": "Focus on end-to-end workflow and model/tool tradeoffs.",
        "advanced": "Focus on architecture, extension points, and debugging.",
    }
    workflow_steps = build_tutorial_workflow_steps(
        topic=topic,
        level=level,
        evidence_rows=evidence_rows,
    )
    lines = [
        f"# Annolid Tutorial: {topic.title()}",
        "",
        f"Level: {level}",
        "",
        "## Goal",
        f"Learn how to use Annolid for: {topic}.",
        "",
        "## Guidance",
        level_goals.get(level, level_goals["intermediate"]),
        "",
        "## Repository Evidence",
    ]
    if evidence_rows:
        for rel, findings in evidence_rows:
            lines.append(f"### `{rel}`")
            for finding in findings:
                lines.append(f"- {finding}")
    else:
        lines.append(
            "- No high-signal references were found; fallback guidance applied."
        )

    lines.extend(
        [
            "",
            "## Workflow",
        ]
    )
    for idx, step in enumerate(workflow_steps, start=1):
        lines.append(f"{idx}. {step}")

    lines.extend(
        [
            "",
            "## Ask Annolid Bot",
            f"- Explain this topic in plain language: `{topic}`",
            f"- Generate a troubleshooting checklist for `{topic}`",
            "- Show which model/tool settings are most important and why.",
            "",
            "## Reference Files",
        ]
    )
    for ref in references:
        lines.append(f"- `{ref}`")
    return "\n".join(lines).strip()


def _topic_tokens(topic: str) -> list[str]:
    tokens = [
        token
        for token in re.split(r"[^a-zA-Z0-9]+", str(topic or "").lower())
        if len(token) >= 3
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered[:8]


def _expand_tutorial_reference_files(
    *,
    root: Path,
    refs: list[str],
    topic: str,
    max_files: int = 10,
) -> list[Path]:
    tokens = _topic_tokens(topic)
    selected: list[Path] = []
    seen: set[str] = set()
    for rel in refs:
        candidate = (root / rel).resolve()
        if candidate.is_file():
            key = str(candidate).lower()
            if key not in seen:
                seen.add(key)
                selected.append(candidate)
            continue
        if not candidate.is_dir():
            continue
        matches: list[Path] = []
        fallbacks: list[Path] = []
        for item in candidate.rglob("*"):
            if not item.is_file() or item.suffix.lower() not in {".py", ".md"}:
                continue
            rel_item = str(item.resolve().relative_to(root.resolve())).lower()
            if any(part in rel_item for part in {"/.venv/", "/tests/", "__pycache__"}):
                continue
            if any(token in rel_item for token in tokens):
                matches.append(item.resolve())
            else:
                fallbacks.append(item.resolve())
            if len(matches) >= 4:
                break
        for item in (matches + fallbacks[:2])[:4]:
            key = str(item).lower()
            if key in seen:
                continue
            seen.add(key)
            selected.append(item)
            if len(selected) >= max_files:
                return selected
        if len(selected) >= max_files:
            return selected
    return selected[:max_files]


def _extract_markdown_evidence(text: str, *, topic: str) -> list[str]:
    tokens = set(_topic_tokens(topic))
    findings: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if line.startswith("#") and len(line) <= 140:
            findings.append(f"section: {line.lstrip('#').strip()}")
        elif any(k in lower for k in {"annolid", "realtime", "camera", "track"}):
            if tokens and not any(t in lower for t in tokens):
                continue
            if len(line) <= 160:
                findings.append(f"note: {line}")
        if len(findings) >= 6:
            break
    return findings


def _extract_python_evidence(text: str, *, topic: str) -> list[str]:
    tokens = set(_topic_tokens(topic))
    findings: list[str] = []
    for match in re.finditer(
        r"^\s*(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)",
        text,
        flags=re.MULTILINE,
    ):
        symbol = str(match.group(2) or "").strip()
        symbol_lower = symbol.lower()
        if tokens and not any(token in symbol_lower for token in tokens):
            if len(findings) >= 3:
                continue
        findings.append(f"symbol: {symbol}")
        if len(findings) >= 6:
            break
    return findings
