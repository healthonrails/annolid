from __future__ import annotations

from dataclasses import dataclass
import shlex
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


def _looks_like_video_path_token(value: str) -> bool:
    token = str(value or "").strip().strip("\"'`")
    if not token:
        return False
    return bool(
        re.search(
            r"\.(?:mp4|avi|mov|mkv|m4v|wmv|flv)\b",
            token,
            flags=re.IGNORECASE,
        )
        or token.startswith(("~", "/", "./", "../"))
        or "/" in token
        or "\\" in token
    )


def _parse_track_command(text: str) -> Dict[str, Any]:
    """Parse the structured `/track` shortcut into the existing GUI workflow."""
    trimmed = str(text or "").strip()
    if not trimmed.lower().startswith("/track"):
        return {}

    payload: Dict[str, Any] = {
        "path": "",
        "text_prompt": "",
        "mode": "track",
        "use_countgd": False,
        "model_name": "",
        "to_frame": None,
    }
    raw_args = trimmed[len("/track") :].strip()
    if not raw_args:
        return {"name": "open_track_dialog", "args": {}}

    tokens = shlex.split(raw_args) if raw_args else []
    positional: list[str] = []
    key_values: dict[str, str] = {}
    for token in tokens:
        if "=" in token:
            key, value = token.split("=", 1)
            key_values[str(key).strip().lower()] = str(value).strip()
        else:
            positional.append(token)

    prompt_candidates: list[str] = []
    path_candidates: list[str] = []
    for token in positional:
        if not token:
            continue
        if _looks_like_video_path_token(token):
            path_candidates.append(token)
        else:
            prompt_candidates.append(token)

    payload["path"] = str(
        key_values.get("video")
        or key_values.get("video_path")
        or key_values.get("video_file")
        or key_values.get("path")
        or key_values.get("file")
        or key_values.get("source")
        or (path_candidates[0] if path_candidates else "")
    ).strip()

    payload["text_prompt"] = str(
        key_values.get("prompt")
        or key_values.get("text_prompt")
        or key_values.get("text")
        or key_values.get("query")
        or key_values.get("label")
        or (" ".join(prompt_candidates).strip())
    ).strip()

    payload["model_name"] = str(
        key_values.get("model")
        or key_values.get("model_name")
        or key_values.get("modelname")
        or ""
    ).strip()
    payload["mode"] = str(key_values.get("mode") or "track").strip().lower()
    if payload["mode"] not in {"track", "segment"}:
        payload["mode"] = "track"

    model_name_lower = payload["model_name"].strip().lower()
    if payload["mode"] == "track" and model_name_lower == "sam3":
        return {
            "name": "sam3_agent_video_track",
            "args": {
                "video_path": payload["path"],
                "agent_prompt": payload["text_prompt"],
            },
        }

    target_frame = key_values.get("to") or key_values.get("to_frame")
    if target_frame is None:
        target_frame = key_values.get("frame") or key_values.get("target_frame")
    if target_frame is not None:
        try:
            payload["to_frame"] = int(str(target_frame).strip())
        except Exception:
            payload["to_frame"] = None

    use_countgd_raw = (
        str(
            key_values.get("countgd")
            or key_values.get("use_countgd")
            or key_values.get("with_countgd")
            or ""
        )
        .strip()
        .lower()
    )
    if use_countgd_raw:
        payload["use_countgd"] = use_countgd_raw not in {"0", "false", "no", "off"}
    elif "countgd" in str(text).lower():
        payload["use_countgd"] = True

    return {"name": "segment_track_video", "args": payload}


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


def _parse_dream_command(text: str) -> Dict[str, Any]:
    dream = _parse_slash_command_action_args(text, "/dream")
    if dream["kind"] not in {"empty", "parsed"}:
        return {}
    return {"name": "dream_memory", "args": {"action": "run"}}


def _parse_dreaming_command(text: str) -> Dict[str, Any]:
    dreaming = _parse_slash_command_action_args(text, "/dreaming")
    if dreaming["kind"] not in {"empty", "parsed"}:
        return {}
    if dreaming["kind"] == "empty":
        return {"name": "dream_memory", "args": {"action": "status"}}
    action = str(dreaming["action"] or "").strip().lower()
    args_text = str(dreaming["args"] or "").strip()
    if action in {"status", "state"}:
        return {"name": "dream_memory", "args": {"action": "status"}}
    if action in {"help"}:
        return {"name": "dream_memory", "args": {"action": "help"}}
    if action in {"run", "start", "sweep"}:
        return {"name": "dream_memory", "args": {"action": "run"}}
    if action in {"log", "history"}:
        run_id = _extract_command_identifier(args_text)
        return {"name": "dream_memory", "args": {"action": "log", "run_id": run_id}}
    if action in {"restore", "rollback", "revert"}:
        run_id = _extract_command_identifier(args_text)
        return {"name": "dream_memory", "args": {"action": "restore", "run_id": run_id}}
    return {}


def _parse_dream_log_command(text: str) -> Dict[str, Any]:
    dream_log = _parse_slash_command_action_args(text, "/dream-log")
    if dream_log["kind"] not in {"empty", "parsed"}:
        return {}
    run_id = ""
    if dream_log["kind"] == "parsed":
        run_id = _extract_command_identifier(
            dream_log["action"] + " " + dream_log["args"]
        )
    return {"name": "dream_memory", "args": {"action": "log", "run_id": run_id}}


def _parse_dream_restore_command(text: str) -> Dict[str, Any]:
    dream_restore = _parse_slash_command_action_args(text, "/dream-restore")
    if dream_restore["kind"] not in {"empty", "parsed"}:
        return {}
    run_id = ""
    if dream_restore["kind"] == "parsed":
        run_id = _extract_command_identifier(
            dream_restore["action"] + " " + dream_restore["args"]
        )
    return {"name": "dream_memory", "args": {"action": "restore", "run_id": run_id}}


DIRECT_SLASH_COMMAND_SPECS: Tuple[SlashCommandSpec, ...] = (
    SlashCommandSpec(
        name="dreaming",
        aliases=("dreaming",),
        display="/dreaming",
        description="Check Dreaming status, help, or run a sweep",
        insert="/dreaming ",
        required_tools=("dream_memory",),
        examples=(
            "'/dreaming status'",
            "'/dreaming run'",
            "'/dreaming help'",
        ),
        parser=_parse_dreaming_command,
    ),
    SlashCommandSpec(
        name="dream-log",
        aliases=("dream-log",),
        display="/dream-log",
        description="Show latest Dream run or a specific run",
        insert="/dream-log ",
        required_tools=("dream_memory",),
        examples=(
            "'/dream-log'",
            "'/dream-log <run_id>'",
        ),
        parser=_parse_dream_log_command,
    ),
    SlashCommandSpec(
        name="dream-restore",
        aliases=("dream-restore",),
        display="/dream-restore",
        description="List Dream runs or restore one snapshot",
        insert="/dream-restore ",
        required_tools=("dream_memory",),
        examples=(
            "'/dream-restore'",
            "'/dream-restore <run_id>'",
        ),
        parser=_parse_dream_restore_command,
    ),
    SlashCommandSpec(
        name="dream",
        aliases=("dream",),
        display="/dream",
        description="Run Dream memory consolidation now",
        insert="/dream",
        required_tools=("dream_memory",),
        examples=("'/dream'",),
        parser=_parse_dream_command,
    ),
    SlashCommandSpec(
        name="track",
        aliases=("track",),
        display="/track",
        description="Open the guided video tracking form",
        insert="/track ",
        examples=(
            "'/track video=/path/to/video.mp4 prompt=\"mouse\" model=Cutie'",
            "'/track video=/path/to/video.mp4 text_prompt=\"mouse\" model=Cutie to_frame=400'",
            "'/track /path/to/video.mp4 mouse model=Cutie'",
        ),
        prompt_examples=(
            "'track the mouse in /path/to/video.mp4 using model=Cutie'",
            "'track /path/to/video.mp4 mouse'",
        ),
        required_tools=("segment_track_video",),
        kind="action",
        action="open_track_dialog",
        parser=_parse_track_command,
    ),
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
        "Direct command aliases are supported for video workflows, automation scheduling, and shell sessions. "
        "Use these forms when helpful: " + ", ".join(examples) + "."
    )
