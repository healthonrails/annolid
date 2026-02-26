from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional


@dataclass(frozen=True)
class PromptBuildInputs:
    workspace: Path
    prompt: str
    enable_web_tools: bool
    enable_ollama_fallback: bool
    allowed_read_roots: Optional[List[str]] = None
    allow_web_tools: Optional[bool] = None
    include_workspace_docs: bool = True
    now: Optional[datetime] = None


def _annolid_docs_index_preview(limit: int = 12) -> str:
    try:
        root = Path(__file__).resolve().parents[4]
    except Exception:
        return ""
    docs_dir = root / "docs" / "source"
    entries: list[str] = []
    for path in [root / "README.md", root / "RELEASING.md"]:
        if path.exists():
            entries.append(str(path.relative_to(root)))
    if docs_dir.exists():
        for md in sorted(docs_dir.glob("*.md")):
            entries.append(str(md.relative_to(root)))
    if not entries:
        return ""
    sliced = entries[:limit]
    suffix = "\n- ..." if len(entries) > limit else ""
    return "\n".join(f"- {item}" for item in sliced) + suffix


def build_compact_system_prompt(
    *,
    inputs: PromptBuildInputs,
    read_text_limited: Callable[[Path, int], str],
    list_skill_names: Callable[[Path], List[str]],
    should_attach_live_web_context: Callable[[str], bool],
    should_attach_live_pdf_context: Callable[[str], bool],
    build_live_web_context_prompt_block: Callable[..., str],
    build_live_pdf_context_prompt_block: Callable[..., str],
) -> str:
    prompt_text = str(inputs.prompt or "")
    short_prompt = len(prompt_text.strip()) <= 80
    web_tools_enabled = (
        inputs.enable_web_tools
        if inputs.allow_web_tools is None
        else bool(inputs.allow_web_tools)
    )
    local_now = (inputs.now or datetime.now()).astimezone()
    tz_name = local_now.tzname() or "local"
    tz_offset = local_now.strftime("%z")
    pretty_offset = (
        f"{tz_offset[:3]}:{tz_offset[3:]}" if len(tz_offset) == 5 else tz_offset
    )
    now_iso = local_now.isoformat(timespec="seconds")

    parts: List[str] = [
        "You are Annolid Bot. Be concise, practical, and return plain text answers."
    ]
    parts.append(
        "Use this local datetime as the source of truth for relative time "
        f"phrases (today/tomorrow/next week): {now_iso} ({tz_name}, UTC{pretty_offset}). "
        "Do not ask the user for today's date unless they explicitly ask for a different timezone/date reference."
    )
    parts.append(
        "CRITICAL INSTRUCTION ON AUTOMATION: You DO have native capabilities for both email and scheduling. "
        "If a user asks you to schedule an email, you MUST use the `cron` tool (action='add') and set its `message` "
        "to a prompt instructing yourself to send the email (e.g. 'Send an email to user@example...'). "
        "Do NOT claim you lack an email scheduling tool, and do NOT offer to create shell scripts or calendar exports."
    )
    parts.append(
        "For camera checks and snapshots, use `gui_check_stream_source` with `save_snapshot=true`. "
        "For sending results by email, use the `email` tool; it supports `attachment_paths` for local files "
        "(for example saved camera snapshots). Do not claim these capabilities are unavailable."
    )
    parts.append(
        "When the user asks to check a camera and email a snapshot in one step, prefer `gui_check_stream_source` "
        "with `save_snapshot=true` and `email_to=<recipient>` so the GUI flow can probe, save, and send together."
    )
    parts.append(
        "For Annolid usage, code, or docs questions: treat the local repository/docs as source-of-truth. "
        "Use `list_dir`/`read_file` to inspect relevant files before answering implementation details. "
        "When explaining code, cite concrete file paths and summarize behavior, inputs, outputs, and caveats."
    )
    parts.append(
        "You can run shell commands with `exec` and inspect Git/GitHub state using "
        "`git_status`, `git_diff`, `git_log`, `git_cli`, `github_pr_status`, `github_pr_checks`, and `gh_cli` "
        "(subject to runtime policy). Do not claim these tools are unavailable unless an actual tool call fails."
    )
    parts.append(
        "When users ask for how-to guidance or tutorials, produce structured on-demand tutorials with: "
        "goal, prerequisites, step-by-step workflow, verification checklist, and troubleshooting tips."
    )
    parts.append(
        "Direct command aliases are supported for automation scheduling and shell sessions. "
        "Use these forms when helpful: "
        "'schedule camera check every 5 minutes', "
        "'schedule periodic report every 10 minutes', "
        "'schedule email summary every 1 hour', "
        "'list automation tasks', "
        "'automation scheduler status', "
        "'run automation task <task_id>', "
        "'remove automation task <task_id>', "
        "'start shell session for <command>', "
        "'list sessions', "
        "'poll session <session_id>', "
        "'show session log <session_id>', "
        "'kill session <session_id>'."
    )
    parts.append(
        "If MCP browser tools are available and the request needs live web search "
        "(weather/news/prices/current events), use browser-first flow by default: "
        "navigate to a search engine, type the query, snapshot/parse results, then summarize with source URLs. "
        "If MCP browser is unavailable, use `gui_web_run_steps` for text extraction."
    )
    if web_tools_enabled:
        parts.append(
            "Web tools are enabled (`web_search`, `web_fetch`). "
            "Use them as fallback when MCP browser search flow is unavailable or fails. "
            "Do not claim you cannot browse."
        )
        parts.append(
            "Do not assume the currently open embedded page is relevant. "
            "Use it only when the user explicitly asks about the open/current page "
            "or references that page URL/topic."
        )
        live_web_context = build_live_web_context_prompt_block(
            include_snapshot=should_attach_live_web_context(prompt_text)
        )
        if live_web_context:
            parts.append(live_web_context)

    live_pdf_context = build_live_pdf_context_prompt_block(
        include_snapshot=should_attach_live_pdf_context(prompt_text)
    )
    if live_pdf_context:
        parts.append(live_pdf_context)

    parts.append(
        "Tools starting with `mcp_` are dynamically injected through an external MCP Server. "
        "CRITICAL INSTRUCTION: You MUST use native Annolid tools (`web_search`, `web_fetch`, `gui_web_run_steps`) FIRST for all web tasks. "
        "Only use MCP browser tools as a SECONDARY BACKUP if the native tools fail or are insufficient for the task. "
        "Because MCP tools can return massive, unchecked strings or DOM trees, their output is forcefully "
        "truncated at 50,000 characters. If querying an MCP browser (e.g. Playwright), always run evaluation scripts "
        "that map `.textContent` of exactly the nodes you need instead of returning full HTML structures."
    )

    roots = [
        str(r).strip() for r in (inputs.allowed_read_roots or []) if str(r).strip()
    ]
    if roots:
        parts.append(
            "Readable paths include workspace plus configured read roots. "
            "Do not claim a path is inaccessible before trying the relevant tool."
        )
        parts.append(
            "# Allowed Read Roots\n" + "\n".join(f"- {root}" for root in roots[:20])
        )

    if inputs.include_workspace_docs:
        agents_limit = 900 if short_prompt else 1600
        memory_limit = 500 if short_prompt else 900
    else:
        agents_limit = 320 if short_prompt else 640
        memory_limit = 180 if short_prompt else 320

    agents_md = read_text_limited(inputs.workspace / "AGENTS.md", agents_limit)
    if agents_md:
        parts.append(f"# Workspace Instructions\n{agents_md}")
    memory_md = read_text_limited(
        inputs.workspace / "memory" / "MEMORY.md", memory_limit
    )
    if memory_md:
        parts.append(f"# Long-term Memory\n{memory_md}")

    if inputs.include_workspace_docs:
        skills_dir = inputs.workspace / "skills"
        if skills_dir.exists():
            names = list_skill_names(skills_dir)
            if names:
                preview = ", ".join(names[:15])
                if len(names) > 15:
                    preview += ", ..."
                parts.append(
                    "Available skills exist in workspace. Use `read_file` to inspect a "
                    f"skill before using it. Skills: {preview}"
                )
                parts.append(
                    "When reading skills, use absolute paths under "
                    f"`{str((inputs.workspace / 'skills').resolve())}` "
                    "instead of relative `.annolid/...` paths."
                )
                if "weather" in {name.strip().lower() for name in names}:
                    parts.append(
                        "For weather/forecast requests, consult the `weather` skill first "
                        "before ad-hoc browsing."
                    )
        docs_preview = _annolid_docs_index_preview()
        if docs_preview:
            parts.append("# Annolid Docs Index\n" + docs_preview)

    return "\n\n".join(parts)
