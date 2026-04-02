from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class SlashCommandSpec:
    name: str
    aliases: Tuple[str, ...]
    display: str
    description: str
    insert: str
    kind: str = "command"
    action: str = ""
    examples: Tuple[str, ...] = ()
    prompt_examples: Tuple[str, ...] = ()
    required_tools: Tuple[str, ...] = ()
    parser: Callable[[str], Dict[str, Any]] = lambda _text: {}

    @property
    def primary_alias(self) -> str:
        return self.aliases[0] if self.aliases else self.name


_ROOT_SLASH_SELECTION_ENTRIES: Tuple[Dict[str, str], ...] = (
    {
        "display": "Select a skill",
        "search": "/skill",
        "insert": "/skill ",
        "kind": "command",
        "action": "",
        "description": "Choose one or more skills to guide the answer",
    },
    {
        "display": "Select skills",
        "search": "/skills",
        "insert": "/skills ",
        "kind": "command",
        "action": "",
        "description": "Choose one or more skills to guide the answer",
    },
    {
        "display": "Select a tool",
        "search": "/tool",
        "insert": "/tool ",
        "kind": "command",
        "action": "",
        "description": "Choose one or more tools to bias the runtime",
    },
    {
        "display": "Select tools",
        "search": "/tools",
        "insert": "/tools ",
        "kind": "command",
        "action": "",
        "description": "Choose one or more tools to bias the runtime",
    },
    {
        "display": "Open capabilities",
        "search": "/capabilities",
        "insert": "",
        "kind": "action",
        "action": "open_capabilities",
        "description": "Open the combined tools and skills panel",
    },
)


def _strip_wrapping_quotes(text: str) -> str:
    value = str(text or "").strip()
    if len(value) >= 2 and (
        (value[0] == "'" and value[-1] == "'")
        or (value[0] == '"' and value[-1] == '"')
        or (value[0] == "`" and value[-1] == "`")
    ):
        return value[1:-1].strip()
    return value


def _extract_command_identifier(text: str) -> str:
    match = re.search(r"\b([a-zA-Z0-9_-]{3,128})\b", str(text or ""))
    return str(match.group(1) if match else "").strip()


def _parse_slash_command_action_args(raw: str, slash: str) -> Dict[str, str]:
    trimmed = str(raw or "").strip()
    slash_lower = str(slash or "").strip().lower()
    if not slash_lower or not trimmed.lower().startswith(slash_lower):
        return {"kind": "no-match", "action": "", "args": ""}
    rest = trimmed[len(slash) :].strip()
    if not rest:
        return {"kind": "empty", "action": "", "args": ""}
    match = re.match(r"^(\S+)(?:\s+([\s\S]+))?$", rest)
    if not match:
        return {"kind": "invalid", "action": "", "args": ""}
    action = str(match.group(1) or "").strip().lower()
    args = str(match.group(2) or "").strip()
    return {"kind": "parsed", "action": action, "args": args}


def _parse_cron_command(text: str) -> Dict[str, Any]:
    cron = _parse_slash_command_action_args(text, "/cron")
    if cron["kind"] not in {"empty", "parsed"}:
        return {}
    action = "status"
    job_id = ""
    if cron["kind"] == "parsed":
        raw_action = cron["action"]
        args = cron["args"]
        if raw_action in {"list", "ls", "show"}:
            action = "list"
        elif raw_action in {"status"}:
            action = "status"
        elif raw_action in {"check"}:
            action = "check"
            job_id = _extract_command_identifier(args)
        elif raw_action in {"remove", "delete", "cancel"}:
            action = "remove"
            job_id = _extract_command_identifier(args)
        elif raw_action in {"run", "enable", "disable"}:
            action = raw_action
            job_id = _extract_command_identifier(args)
        else:
            return {}
    payload: Dict[str, Any] = {"action": action}
    if job_id:
        payload["job_id"] = job_id
    return {"name": "cron", "args": payload}


def _parse_automation_command(text: str) -> Dict[str, Any]:
    automation = _parse_slash_command_action_args(text, "/automation")
    if automation["kind"] not in {"empty", "parsed"}:
        return {}
    if automation["kind"] == "empty":
        return {"name": "automation_schedule", "args": {"action": "status"}}
    raw_action = automation["action"]
    args = automation["args"]
    if raw_action in {"list", "ls", "show"}:
        return {"name": "automation_schedule", "args": {"action": "list"}}
    if raw_action in {"status"}:
        return {"name": "automation_schedule", "args": {"action": "status"}}
    if raw_action in {"run", "remove", "delete", "cancel"}:
        task_id = _extract_command_identifier(args)
        if not task_id:
            return {}
        action = "run" if raw_action == "run" else "remove"
        return {
            "name": "automation_schedule",
            "args": {"action": action, "task_id": task_id},
        }
    return {}


def _parse_session_command(text: str) -> Dict[str, Any]:
    session = _parse_slash_command_action_args(text, "/session")
    if session["kind"] not in {"empty", "parsed"}:
        return {}
    if session["kind"] == "empty":
        return {"name": "exec_process", "args": {"action": "list"}}
    raw_action = session["action"]
    args = session["args"]
    session_id = _extract_command_identifier(args)
    if raw_action in {"list", "ls", "show"}:
        return {"name": "exec_process", "args": {"action": "list"}}
    if raw_action in {"poll", "status", "check"} and session_id:
        return {
            "name": "exec_process",
            "args": {"action": "poll", "session_id": session_id, "wait_ms": 1500},
        }
    if raw_action in {"log", "logs", "tail", "output"} and session_id:
        return {
            "name": "exec_process",
            "args": {"action": "log", "session_id": session_id, "tail_lines": 200},
        }
    if raw_action in {"kill", "stop", "terminate", "cancel"} and session_id:
        return {
            "name": "exec_process",
            "args": {"action": "kill", "session_id": session_id},
        }
    return {}


def _parse_git_command(text: str) -> Dict[str, Any]:
    git = _parse_slash_command_action_args(text, "/git")
    if git["kind"] not in {"empty", "parsed"}:
        return {}
    if git["kind"] == "empty" or git["action"] == "status":
        return {"name": "git_status", "args": {"short": True}}
    if git["action"] in {"diff"}:
        return {"name": "git_diff", "args": {}}
    if git["action"] in {"log"}:
        return {"name": "git_log", "args": {"max_count": 20, "oneline": True}}
    return {}


def _parse_github_command(text: str) -> Dict[str, Any]:
    gh = _parse_slash_command_action_args(text, "/gh")
    if gh["kind"] not in {"empty", "parsed"}:
        return {}
    if gh["kind"] == "empty":
        return {"name": "github_pr_status", "args": {}}
    if gh["action"] in {"status", "pr-status", "pr_status"}:
        return {"name": "github_pr_status", "args": {}}
    if gh["action"] in {"checks", "pr-checks", "pr_checks"}:
        return {"name": "github_pr_checks", "args": {}}
    return {}


def _parse_capabilities_command(text: str) -> Dict[str, Any]:
    for alias in ("/capabilities", "/caps"):
        capabilities = _parse_slash_command_action_args(text, alias)
        if capabilities["kind"] in {"empty", "parsed"}:
            return {"name": "open_agent_capabilities", "args": {}}
    return {}


DIRECT_SLASH_COMMAND_SPECS: Tuple[SlashCommandSpec, ...] = (
    SlashCommandSpec(
        name="cron",
        aliases=("cron",),
        display="/cron",
        description="Cron scheduler commands",
        insert="/cron ",
        examples=(
            "'/cron status'",
            "'/cron list'",
            "'/cron check <job_id>'",
        ),
        prompt_examples=(
            "'schedule camera check every 5 minutes'",
            "'list cron jobs'",
            "'check cron job <job_id>'",
        ),
        required_tools=("cron",),
        parser=_parse_cron_command,
    ),
    SlashCommandSpec(
        name="automation",
        aliases=("automation",),
        display="/automation",
        description="Automation scheduler commands",
        insert="/automation ",
        examples=(
            "'/automation status'",
            "'/automation list'",
            "'/automation run <task_id>'",
        ),
        prompt_examples=("'list automation tasks'",),
        required_tools=("automation_schedule",),
        parser=_parse_automation_command,
    ),
    SlashCommandSpec(
        name="session",
        aliases=("session",),
        display="/session",
        description="Shell session commands",
        insert="/session ",
        examples=(
            "'/session list'",
            "'/session poll <session_id>'",
            "'/session logs <session_id>'",
        ),
        required_tools=("exec_start", "exec_process"),
        parser=_parse_session_command,
    ),
    SlashCommandSpec(
        name="git",
        aliases=("git",),
        display="/git",
        description="Git status / diff / log",
        insert="/git ",
        examples=("'/git status'", "'/git diff'", "'/git log'"),
        parser=_parse_git_command,
    ),
    SlashCommandSpec(
        name="github",
        aliases=("gh",),
        display="/gh",
        description="GitHub PR commands",
        insert="/gh ",
        examples=("'/gh status'", "'/gh checks'"),
        parser=_parse_github_command,
    ),
    SlashCommandSpec(
        name="capabilities",
        aliases=("capabilities", "caps"),
        display="/capabilities",
        description="Open the combined tools and skills panel",
        insert="",
        kind="action",
        action="open_capabilities",
        parser=_parse_capabilities_command,
    ),
)


def iter_direct_slash_command_specs() -> Tuple[SlashCommandSpec, ...]:
    return DIRECT_SLASH_COMMAND_SPECS


def build_root_slash_completion_entries() -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = [dict(row) for row in _ROOT_SLASH_SELECTION_ENTRIES]
    for spec in DIRECT_SLASH_COMMAND_SPECS:
        entries.append(
            {
                "display": spec.display,
                "search": f"/{spec.primary_alias}",
                "insert": spec.insert,
                "kind": spec.kind,
                "action": spec.action,
                "description": spec.description,
            }
        )
    return entries


def matches_slash_completion_search(search: str, query: str) -> bool:
    normalized_search = str(search or "").strip().lower().lstrip("/")
    normalized_query = str(query or "").strip().lower().lstrip("/@")
    if not normalized_query:
        return True
    return normalized_search.startswith(normalized_query)


def parse_direct_slash_command(prompt: str) -> Dict[str, Any]:
    text = str(prompt or "").strip()
    if not text.startswith("/"):
        return {}
    for spec in DIRECT_SLASH_COMMAND_SPECS:
        command = spec.parser(text)
        if command:
            return command
    return {}


def build_direct_command_alias_line(tool_names: Sequence[str]) -> str:
    tool_set = {
        str(name or "").strip().lower()
        for name in tool_names
        if str(name or "").strip()
    }
    examples: List[str] = []
    for spec in DIRECT_SLASH_COMMAND_SPECS:
        required_tools = {name.lower() for name in spec.required_tools if name}
        if tool_set and required_tools and not (tool_set & required_tools):
            continue
        if tool_set and spec.name == "cron" and "cron" not in tool_set:
            continue
        examples.extend(spec.examples)
        examples.extend(spec.prompt_examples)
    if not examples:
        return ""
    return (
        "Direct command aliases are supported for automation scheduling and shell sessions. "
        "Use these forms when helpful: " + ", ".join(examples) + "."
    )
