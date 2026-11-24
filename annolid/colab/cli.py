from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, TypeVar

from .auth import ColabAuthManager, ColabAuthRequiredError
from .models import (
    ColabPredictRequest,
    ColabTrainRequest,
    SUPPORTED_ACCELERATORS,
    VALID_AUTH_METHODS,
    VALID_BACKENDS,
)
from .service import ColabWorkflowService

_T = TypeVar("_T")


def _make_service(args: argparse.Namespace) -> ColabWorkflowService:
    auth = ColabAuthManager(
        credentials_file=getattr(args, "credentials", None),
        token_file=getattr(args, "token", None),
        allow_interactive_auth=bool(getattr(args, "interactive_auth", False)),
        colab_cli_auth_mode=str(getattr(args, "cli_auth", "auto") or "auto"),
    )
    home = getattr(args, "colab_home", None)
    return ColabWorkflowService(
        auth_manager=auth,
        home_dir=Path(home).expanduser() if home else None,
    )


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _log_stdout(msg: str) -> None:
    print(msg, flush=True)


def _can_prompt_interactive(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "json", False)) or bool(getattr(args, "no_prompt", False)):
        return False
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except Exception:
        return False


def _prompt_interactive_login(
    svc: ColabWorkflowService,
    *,
    reason: str,
    allow_notebook_fallback: bool = True,
) -> Optional[str]:
    """Prompt the user in an interactive terminal to sign in to Google Colab.

    Returns the chosen backend (``"cli"``, ``"api"``, or ``"notebook"``) or ``None`` if cancelled.
    """
    status = svc.auth_status()
    cli_installed = bool(status["colab_cli"]["installed"])
    has_oauth_client = bool(status["oauth_colab"].get("has_credentials_file"))
    has_gcloud = bool(status["adc"].get("gcloud_installed"))

    print("\n=== Google Colab Sign-In Required ===", flush=True)
    if reason:
        print(f"Reason: {reason}", flush=True)
    print("\nChoose how to continue:", flush=True)
    cli_note = (
        "installed — opens browser sign-in"
        if cli_installed
        else "personal Google account; installs `google-colab-cli` automatically"
    )
    print(
        f"  [1] Sign in with Official Google Colab CLI ({cli_note}) [recommended]",
        flush=True,
    )
    oauth_note = (
        "OAuth client JSON detected"
        if has_oauth_client
        else f"requires OAuth client JSON at {status['credentials_file']}"
    )
    print(
        f"  [2] Sign in with Google OAuth2 — Colab REST API + Drive ({oauth_note})",
        flush=True,
    )
    adc_note = "gcloud installed" if has_gcloud else "requires `gcloud` CLI"
    print(
        f"  [3] Sign in with Google Cloud ADC — gcloud ({adc_note})",
        flush=True,
    )
    if allow_notebook_fallback:
        print(
            "  [4] Continue in .ipynb Notebook mode (generate notebook without direct runtime auth)",
            flush=True,
        )
        print("  [5] Cancel", flush=True)
    else:
        print("  [4] Cancel", flush=True)

    try:
        raw = input("\nSelect an option [1]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nSign-in cancelled.", flush=True)
        return None

    choice = raw or "1"
    if choice in {"1", "cli", "colab-cli"}:
        res = svc.login(
            method="colab-cli",
            open_browser=True,
            install_cli_if_missing=True,
            log_callback=_log_stdout,
        )
        print(f"Signed in via {res['method']}. Resuming...", flush=True)
        return "cli"
    if choice in {"2", "oauth", "oauth2"}:
        res = svc.login(
            method="oauth2",
            include_drive=True,
            open_browser=True,
            log_callback=_log_stdout,
        )
        print(f"Signed in via {res['method']}. Resuming...", flush=True)
        return "api"
    if choice in {"3", "adc", "gcloud"}:
        res = svc.login(
            method="adc",
            open_browser=True,
            log_callback=_log_stdout,
        )
        print(f"Signed in via {res['method']}. Resuming...", flush=True)
        return "cli" if svc.auth.find_colab_cli() else "api"
    if allow_notebook_fallback and choice in {"4", "notebook", "nb"}:
        print("Switching to .ipynb Notebook mode...", flush=True)
        return "notebook"
    print("Cancelled.", flush=True)
    return None


def _run_with_auth_prompt(
    svc: ColabWorkflowService,
    args: argparse.Namespace,
    *,
    requested_backend: str,
    allow_notebook_fallback: bool,
    runner: Callable[[str], _T],
) -> Optional[_T]:
    """Run a backend-aware operation, prompting interactively for login if needed."""
    backend = str(requested_backend or "auto").strip().lower()
    status = svc.auth_status()
    needs_preflight_prompt = (
        (backend == "auto" and not (status["cli_ready"] or status["api_ready"]))
        or (backend == "cli" and not status["cli_ready"])
        or (backend == "api" and not status["api_ready"])
    )
    if needs_preflight_prompt and _can_prompt_interactive(args):
        chosen = _prompt_interactive_login(
            svc,
            reason=status["summary"],
            allow_notebook_fallback=allow_notebook_fallback,
        )
        if chosen is None:
            return None
        backend = chosen

    try:
        return runner(backend)
    except ColabAuthRequiredError as exc:
        if not _can_prompt_interactive(args):
            raise
        chosen = _prompt_interactive_login(
            svc,
            reason=str(exc),
            allow_notebook_fallback=allow_notebook_fallback,
        )
        if chosen is None:
            return None
        return runner(chosen)


def _cmd_status(args: argparse.Namespace) -> int:
    svc = _make_service(args)
    status = svc.auth_status()
    payload: dict[str, Any] = {
        "auth": status,
        "jobs_count": len(svc.list_jobs(limit=100)),
    }
    if bool(getattr(args, "json", False)):
        _print_json(payload)
        return 0
    print(f"Recommended backend: {status['recommended_backend']}")
    print(f"Summary: {status['summary']}")
    cli_info = status["colab_cli"]
    print(
        f"  Colab CLI (`colab`): {'installed' if cli_info['installed'] else 'not installed'}"
        + (f" ({cli_info['version']})" if cli_info.get("version") else "")
    )
    oc = status["oauth_colab"]
    od = status["oauth_drive"]
    print(
        f"  Google OAuth (Colab API): {'ready' if oc['ready'] else 'not ready'}"
        + (f" — {oc['reason']}" if oc.get("reason") else "")
    )
    print(
        f"  Google OAuth (Drive sync): {'ready' if od['ready'] else 'not ready'}"
        + (f" — {od['reason']}" if od.get("reason") else "")
    )
    adc = status["adc"]
    print(
        f"  Google Cloud ADC: {'ready' if adc['ready'] else 'not ready'} "
        f"(gcloud={'installed' if adc.get('gcloud_installed') else 'missing'})"
    )
    print(f"  Recorded Colab jobs: {payload['jobs_count']}")
    return 0


def _cmd_auth_status(args: argparse.Namespace) -> int:
    svc = _make_service(args)
    status = svc.auth_status()
    if bool(getattr(args, "json", False)):
        _print_json(status)
        return 0
    print(f"Recommended backend: {status['recommended_backend']}")
    print(status["summary"])
    print(f"Credentials file: {status['credentials_file']}")
    print(f"Token file:       {status['token_file']}")
    if status["oauth_colab"].get("granted_scopes"):
        print("Granted scopes:   " + ", ".join(status["oauth_colab"]["granted_scopes"]))
    if not status["cli_ready"]:
        print(
            f"Tip (personal Google account): {status['colab_cli']['install_command']}"
        )
    if not status["adc"]["ready"]:
        print(f"Tip (gcloud ADC): {status['adc']['setup_command']}")
    return 0


def _cmd_auth_login(args: argparse.Namespace) -> int:
    svc = _make_service(args)
    result = svc.login(
        method=str(getattr(args, "method", "auto") or "auto"),
        include_drive=not bool(getattr(args, "no_drive", False)),
        open_browser=not bool(getattr(args, "no_browser", False)),
        install_cli_if_missing=bool(getattr(args, "install_cli", False)),
        log_callback=None if bool(getattr(args, "json", False)) else _log_stdout,
    )
    if bool(getattr(args, "json", False)):
        _print_json(result)
        return 0
    print(
        f"Colab authentication succeeded via {result['method']}. "
        f"Recommended backend: {result['status']['recommended_backend']}"
    )
    return 0


def _cmd_auth_whoami(args: argparse.Namespace) -> int:
    svc = _make_service(args)
    info = svc.get_account_info()
    if bool(getattr(args, "json", False)):
        _print_json(info)
        return 0
    if info.get("signed_in"):
        print(f"Account:         {info.get('email') or '(authenticated)'}")
        print(f"Provider:        {info.get('provider') or '-'}")
        if info.get("expires_in"):
            print(f"Token expires:   {info['expires_in']}")
        if info.get("compute_balance"):
            print(f"Compute balance: {info['compute_balance']}")
        if info.get("usage_rate"):
            print(f"Usage rate:      {info['usage_rate']}")
        if info.get("active_assignments") is not None:
            print(f"Active runtimes: {info['active_assignments']}")
    else:
        print(info.get("detail") or "Not signed in to Google Colab.")
    return 0


def _cmd_auth_logout(args: argparse.Namespace) -> int:
    svc = _make_service(args)
    clear_all = bool(getattr(args, "all", False))
    result = svc.logout(
        include_colab_cli=clear_all,
        clear_adc=clear_all,
    )
    if bool(getattr(args, "json", False)):
        _print_json(result)
        return 0
    if result["removed_token"]:
        print(f"Removed cached Google OAuth token: {result['token_file']}")
    else:
        print(f"No cached Google OAuth token found at: {result['token_file']}")
    if clear_all:
        print(
            f"Cleared Colab CLI keyring: {result['cleared_colab_cli']} | "
            f"Cleared gcloud ADC: {result['cleared_adc']}"
        )
    return 0


def _cmd_specs(args: argparse.Namespace) -> int:
    svc = _make_service(args)
    specs = _run_with_auth_prompt(
        svc,
        args,
        requested_backend="api",
        allow_notebook_fallback=False,
        runner=lambda _backend: svc.list_runtime_specs(),
    )
    if specs is None:
        return 1
    rows = [s.to_dict() for s in specs]
    if bool(getattr(args, "json", False)):
        _print_json(rows)
        return 0
    if not rows:
        print("No Colab runtime specs returned.")
        return 0
    for row in rows:
        flag = "eligible" if row["eligible"] else "ineligible"
        print(
            f"  {row['variant']:<14} {row['accelerator']:<10} "
            f"{row['shape']:<16} [{flag}]"
        )
    return 0


def _cmd_sessions(args: argparse.Namespace) -> int:
    svc = _make_service(args)
    sessions = _run_with_auth_prompt(
        svc,
        args,
        requested_backend=str(getattr(args, "backend", "auto") or "auto"),
        allow_notebook_fallback=False,
        runner=lambda backend: svc.list_sessions(backend=backend),
    )
    if sessions is None:
        return 1
    rows = [s.to_public_dict() for s in sessions]
    if bool(getattr(args, "json", False)):
        _print_json(rows)
        return 0
    if not rows:
        print("No active Colab sessions found.")
        return 0
    for row in rows:
        print(
            f"  {row['name'] or row['session_name']:<28} "
            f"backend={row['backend']} accel={row['accelerator'] or '-'} "
            f"state={row['state']}"
        )
    return 0


def _cmd_new(args: argparse.Namespace) -> int:
    svc = _make_service(args)
    accel = str(getattr(args, "tpu", None) or getattr(args, "gpu", None) or "T4")
    if getattr(args, "tpu", None) and not accel.upper().startswith("TPU_"):
        accel = f"TPU_{accel.upper()}"
    rt = _run_with_auth_prompt(
        svc,
        args,
        requested_backend=str(getattr(args, "backend", "auto") or "auto"),
        allow_notebook_fallback=False,
        runner=lambda backend: svc.create_session(
            session_name=getattr(args, "session", None),
            accelerator=accel,
            high_mem=bool(getattr(args, "high_mem", False)),
            backend=backend,
            log_callback=None if bool(getattr(args, "json", False)) else _log_stdout,
        ),
    )
    if rt is None:
        return 1
    if bool(getattr(args, "json", False)):
        _print_json(rt.to_public_dict())
        return 0
    print(f"Created Colab runtime: {rt.name} ({rt.accelerator}, backend={rt.backend})")
    if rt.browser_url:
        print(f"Browser URL: {rt.browser_url}")
    return 0


def _cmd_stop(args: argparse.Namespace) -> int:
    svc = _make_service(args)
    svc.stop_session(
        str(args.session),
        backend=str(getattr(args, "backend", "auto") or "auto"),
    )
    if bool(getattr(args, "json", False)):
        _print_json({"ok": True, "stopped": str(args.session)})
        return 0
    print(f"Stopped Colab session/runtime: {args.session}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    svc = _make_service(args)
    req = ColabTrainRequest(
        model=str(args.model),
        dataset_path=str(args.data),
        weights=getattr(args, "weights", None),
        epochs=getattr(args, "epochs", None),
        max_iterations=getattr(args, "max_iterations", None),
        batch=getattr(args, "batch", None),
        imgsz=getattr(args, "imgsz", None),
        preset=getattr(args, "preset", None),
        data_format=getattr(args, "data_format", None),
        plots=not bool(getattr(args, "no_plots", False)),
        resume=bool(getattr(args, "resume", False)),
        extra_args=tuple(getattr(args, "extra_arg", None) or ()),
        accelerator=str(getattr(args, "gpu", "T4") or "T4"),
        high_mem=bool(getattr(args, "high_mem", False)),
        backend=str(getattr(args, "backend", "auto") or "auto"),
        session_name=getattr(args, "session", None),
        keep_runtime=bool(getattr(args, "keep_runtime", False)),
        output_dir=getattr(args, "output", None),
        drive_folder=str(
            getattr(args, "drive_folder", "Annolid/colab") or "Annolid/colab"
        ),
        upload_to_drive=not bool(getattr(args, "no_drive_upload", False)),
        open_browser=bool(getattr(args, "open_browser", False)),
        annolid_install_spec=str(getattr(args, "annolid_spec", "annolid") or "annolid"),
        job_name=getattr(args, "name", None),
    )
    log_cb = None if bool(getattr(args, "json", False)) else _log_stdout
    record = _run_with_auth_prompt(
        svc,
        args,
        requested_backend=req.backend,
        allow_notebook_fallback=True,
        runner=lambda backend: svc.run_train(
            dataclasses.replace(req, backend=backend),
            log_callback=log_cb,
        ),
    )
    if record is None:
        return 1
    if bool(getattr(args, "json", False)):
        _print_json(record.to_dict())
        return 0
    print(f"\nJob ID:   {record.job_id}")
    print(f"Status:   {record.status}")
    print(f"Backend:  {record.backend}")
    if record.colab_url:
        print(f"Colab URL:{record.colab_url}")
    if record.notebook_path:
        print(f"Notebook: {record.notebook_path}")
    if record.downloaded_files:
        print(
            f"Outputs:  {record.local_output_dir} ({len(record.downloaded_files)} files)"
        )
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    svc = _make_service(args)
    req = ColabPredictRequest(
        model=str(args.model),
        source_path=str(args.source),
        weights=getattr(args, "weights", None),
        dataset_dir=getattr(args, "dataset_dir", None),
        conf=getattr(args, "conf", None),
        iou=getattr(args, "iou", None),
        imgsz=getattr(args, "imgsz", None),
        batch=getattr(args, "batch", None),
        save_csv=not bool(getattr(args, "no_csv", False)),
        save_video=bool(getattr(args, "save_video", False)),
        no_tracking=bool(getattr(args, "no_tracking", False)),
        classes=getattr(args, "classes", None),
        extra_args=tuple(getattr(args, "extra_arg", None) or ()),
        accelerator=str(getattr(args, "gpu", "T4") or "T4"),
        high_mem=bool(getattr(args, "high_mem", False)),
        backend=str(getattr(args, "backend", "auto") or "auto"),
        session_name=getattr(args, "session", None),
        keep_runtime=bool(getattr(args, "keep_runtime", False)),
        output_dir=getattr(args, "output", None),
        drive_folder=str(
            getattr(args, "drive_folder", "Annolid/colab") or "Annolid/colab"
        ),
        upload_to_drive=not bool(getattr(args, "no_drive_upload", False)),
        open_browser=bool(getattr(args, "open_browser", False)),
        annolid_install_spec=str(getattr(args, "annolid_spec", "annolid") or "annolid"),
        job_name=getattr(args, "name", None),
    )
    log_cb = None if bool(getattr(args, "json", False)) else _log_stdout
    record = _run_with_auth_prompt(
        svc,
        args,
        requested_backend=req.backend,
        allow_notebook_fallback=True,
        runner=lambda backend: svc.run_predict(
            dataclasses.replace(req, backend=backend),
            log_callback=log_cb,
        ),
    )
    if record is None:
        return 1
    if bool(getattr(args, "json", False)):
        _print_json(record.to_dict())
        return 0
    print(f"\nJob ID:   {record.job_id}")
    print(f"Status:   {record.status}")
    print(f"Backend:  {record.backend}")
    if record.colab_url:
        print(f"Colab URL:{record.colab_url}")
    if record.notebook_path:
        print(f"Notebook: {record.notebook_path}")
    if record.downloaded_files:
        print(
            f"Outputs:  {record.local_output_dir} ({len(record.downloaded_files)} files)"
        )
    return 0


def _cmd_notebook(args: argparse.Namespace) -> int:
    svc = _make_service(args)
    mode = str(args.mode).strip().lower()
    if mode == "train":
        if not getattr(args, "data", None):
            raise SystemExit("--data is required when --mode train.")
        req = ColabTrainRequest(
            model=str(args.model),
            dataset_path=str(args.data),
            weights=getattr(args, "weights", None),
            epochs=getattr(args, "epochs", None),
            batch=getattr(args, "batch", None),
            imgsz=getattr(args, "imgsz", None),
            preset=getattr(args, "preset", None),
            accelerator=str(getattr(args, "gpu", "T4") or "T4"),
            backend="notebook",
            output_dir=getattr(args, "output", None),
            drive_folder=str(
                getattr(args, "drive_folder", "Annolid/colab") or "Annolid/colab"
            ),
            upload_to_drive=bool(getattr(args, "upload_drive", False)),
            open_browser=bool(getattr(args, "open_browser", False)),
            annolid_install_spec=str(
                getattr(args, "annolid_spec", "annolid") or "annolid"
            ),
        )
        record = svc.run_train(
            req,
            log_callback=None if bool(getattr(args, "json", False)) else _log_stdout,
        )
    else:
        if not getattr(args, "source", None):
            raise SystemExit("--source is required when --mode predict.")
        req_p = ColabPredictRequest(
            model=str(args.model),
            source_path=str(args.source),
            weights=getattr(args, "weights", None),
            conf=getattr(args, "conf", None),
            save_csv=not bool(getattr(args, "no_csv", False)),
            save_video=bool(getattr(args, "save_video", False)),
            accelerator=str(getattr(args, "gpu", "T4") or "T4"),
            backend="notebook",
            output_dir=getattr(args, "output", None),
            drive_folder=str(
                getattr(args, "drive_folder", "Annolid/colab") or "Annolid/colab"
            ),
            upload_to_drive=bool(getattr(args, "upload_drive", False)),
            open_browser=bool(getattr(args, "open_browser", False)),
            annolid_install_spec=str(
                getattr(args, "annolid_spec", "annolid") or "annolid"
            ),
        )
        record = svc.run_predict(
            req_p,
            log_callback=None if bool(getattr(args, "json", False)) else _log_stdout,
        )
    if bool(getattr(args, "json", False)):
        _print_json(record.to_dict())
        return 0
    print(f"Generated notebook: {record.notebook_path}")
    if record.colab_url:
        print(f"Colab URL:          {record.colab_url}")
    return 0


def _cmd_pull(args: argparse.Namespace) -> int:
    svc = _make_service(args)
    record = svc.pull_results(
        str(args.job_id),
        output_dir=getattr(args, "output", None),
        log_callback=None if bool(getattr(args, "json", False)) else _log_stdout,
    )
    if bool(getattr(args, "json", False)):
        _print_json(record.to_dict())
        return 0
    print(f"Pulled {len(record.downloaded_files)} file(s) to {record.local_output_dir}")
    return 0


def _cmd_jobs(args: argparse.Namespace) -> int:
    svc = _make_service(args)
    jobs = svc.list_jobs(limit=int(getattr(args, "limit", 25) or 25))
    rows = [j.to_dict() for j in jobs]
    if bool(getattr(args, "json", False)):
        _print_json(rows)
        return 0
    if not rows:
        print("No Colab jobs recorded yet.")
        return 0
    for row in rows:
        print(
            f"  {row['job_id']:<34} {row['kind']:<7} {row['model']:<18} "
            f"{row['backend']:<8} {row['status']:<14} {row['updated_at']}"
        )
    return 0


def _cmd_templates(args: argparse.Namespace) -> int:
    svc = _make_service(args)
    templates = svc.list_templates()
    if bool(getattr(args, "json", False)):
        _print_json(templates)
        return 0
    for tpl in templates:
        print(f"  {tpl['key']:<24} [{tpl['category']}] {tpl['title']}")
        print(f"    {tpl['colab_url']}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    svc = _make_service(args)
    report = svc.run_diagnostics(live=bool(getattr(args, "live", False)))
    if bool(getattr(args, "json", False)):
        _print_json(report)
        return 0 if report["ok"] else 1
    print(f"Recommended backend: {report['recommended_backend']}")
    print(f"Summary: {report['summary']}")
    for check in report["checks"]:
        badge = "OK" if check["ok"] else check["severity"].upper()
        print(f"  [{badge:<5}] {check['name']}: {check['detail']}")
    return 0 if report["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="annolid-run colab",
        description=(
            "Authenticate with Google Colab and run Annolid training or inference "
            "remotely via Colab CLI, Colab REST API, or Drive-synced notebooks."
        ),
    )
    parser.add_argument(
        "--credentials",
        default=None,
        help="Optional override path to Google OAuth client secrets JSON.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional override path to cached Google OAuth token JSON.",
    )
    parser.add_argument(
        "--colab-home",
        default=None,
        help="Optional override directory for Colab job state & staging (~/.annolid/colab).",
    )
    parser.add_argument(
        "--cli-auth",
        choices=("auto", "oauth2", "adc"),
        default="auto",
        help="Authentication mode passed to `google-colab-cli` (default: auto).",
    )
    parser.add_argument(
        "--interactive-auth",
        action="store_true",
        help="Allow interactive OAuth browser prompt if token needs refresh/consent.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Disable interactive terminal sign-in prompts when Colab auth is missing or expired.",
    )

    sub = parser.add_subparsers(dest="subcommand", required=True)

    # status
    p_status = sub.add_parser(
        "status", help="Show Colab auth, backend readiness, and job summary."
    )
    p_status.add_argument("--json", action="store_true", help="Output JSON.")
    p_status.set_defaults(_handler=_cmd_status)

    # auth
    p_auth = sub.add_parser("auth", help="Manage Google Colab & Drive authentication.")
    auth_sub = p_auth.add_subparsers(dest="auth_command", required=True)

    p_auth_status = auth_sub.add_parser(
        "status", help="Inspect Colab & Drive authentication status."
    )
    p_auth_status.add_argument("--json", action="store_true", help="Output JSON.")
    p_auth_status.set_defaults(_handler=_cmd_auth_status)

    p_auth_login = auth_sub.add_parser(
        "login", help="Sign in to Google Colab and/or Drive."
    )
    p_auth_login.add_argument(
        "--method",
        choices=VALID_AUTH_METHODS,
        default="auto",
        help="Authentication method: auto, personal, colab-cli, oauth2, drive, or adc.",
    )
    p_auth_login.add_argument(
        "--no-drive",
        action="store_true",
        help="Do not request Google Drive scope when logging in via oauth2.",
    )
    p_auth_login.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open a web browser during OAuth/ADC login.",
    )
    p_auth_login.add_argument(
        "--install-cli",
        action="store_true",
        help="Automatically install `google-colab-cli` via `uv` or `pip` if missing.",
    )
    p_auth_login.add_argument("--json", action="store_true", help="Output JSON.")
    p_auth_login.set_defaults(_handler=_cmd_auth_login)

    p_auth_whoami = auth_sub.add_parser(
        "whoami",
        help="Show the active personal Google Colab account email and compute-unit usage.",
    )
    p_auth_whoami.add_argument("--json", action="store_true", help="Output JSON.")
    p_auth_whoami.set_defaults(_handler=_cmd_auth_whoami)

    p_auth_logout = auth_sub.add_parser(
        "logout",
        help="Remove cached Annolid Google OAuth token (use --all to also sign out Colab CLI / ADC).",
    )
    p_auth_logout.add_argument(
        "--all",
        action="store_true",
        help="Also clear saved Colab CLI keyring and gcloud ADC credentials so you can switch Google accounts.",
    )
    p_auth_logout.add_argument("--json", action="store_true", help="Output JSON.")
    p_auth_logout.set_defaults(_handler=_cmd_auth_logout)

    # specs
    p_specs = sub.add_parser(
        "specs", help="List eligible Colab GPU/TPU/CPU runtime specs via API."
    )
    p_specs.add_argument("--json", action="store_true", help="Output JSON.")
    p_specs.set_defaults(_handler=_cmd_specs)

    # sessions
    p_sessions = sub.add_parser(
        "sessions", help="List active Google Colab runtimes/sessions."
    )
    p_sessions.add_argument("--backend", choices=VALID_BACKENDS, default="auto")
    p_sessions.add_argument("--json", action="store_true", help="Output JSON.")
    p_sessions.set_defaults(_handler=_cmd_sessions)

    # new
    p_new = sub.add_parser("new", help="Provision a new Google Colab runtime/session.")
    p_new.add_argument("-s", "--session", default=None, help="Optional session name.")
    p_new.add_argument(
        "--gpu",
        default="T4",
        help=f"Accelerator ({', '.join(SUPPORTED_ACCELERATORS)}).",
    )
    p_new.add_argument(
        "--tpu", default=None, help="Optional TPU type (e.g. v5e1, v6e1)."
    )
    p_new.add_argument(
        "--high-mem", action="store_true", help="Request high-RAM runtime."
    )
    p_new.add_argument("--backend", choices=VALID_BACKENDS, default="auto")
    p_new.add_argument("--json", action="store_true", help="Output JSON.")
    p_new.set_defaults(_handler=_cmd_new)

    # stop
    p_stop = sub.add_parser(
        "stop", help="Stop and release a Google Colab session/runtime."
    )
    p_stop.add_argument(
        "session", help="Session name or runtime resource name (runtimes/...)."
    )
    p_stop.add_argument("--backend", choices=VALID_BACKENDS, default="auto")
    p_stop.add_argument("--json", action="store_true", help="Output JSON.")
    p_stop.set_defaults(_handler=_cmd_stop)

    # train
    p_train = sub.add_parser(
        "train", help="Run Annolid model training on Google Colab."
    )
    p_train.add_argument(
        "model",
        help="Annolid model plugin (e.g. yolo, dino_kpseg, maskrcnn_detectron2, keypoint_rcnn).",
    )
    p_train.add_argument(
        "--data", required=True, help="Local dataset YAML, folder, or .zip archive."
    )
    p_train.add_argument(
        "--weights",
        default=None,
        help="Initial weights (preset name or local .pt/.pth file).",
    )
    p_train.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    p_train.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Max iterations (Mask/Keypoint R-CNN).",
    )
    p_train.add_argument("--batch", type=int, default=None, help="Batch size.")
    p_train.add_argument(
        "--imgsz", type=int, default=None, help="Image size in pixels."
    )
    p_train.add_argument(
        "--preset", default=None, help="Optional YOLO training preset."
    )
    p_train.add_argument(
        "--data-format", default=None, help="Optional dataset format (dino_kpseg)."
    )
    p_train.add_argument(
        "--no-plots", action="store_true", help="Disable training plots."
    )
    p_train.add_argument(
        "--resume", action="store_true", help="Resume from checkpoint."
    )
    p_train.add_argument(
        "--gpu",
        default="T4",
        help="Colab accelerator (T4, L4, A100, H100, CPU, TPU_V5E1, TPU_V6E1).",
    )
    p_train.add_argument(
        "--high-mem", action="store_true", help="Request high-RAM runtime."
    )
    p_train.add_argument("--backend", choices=VALID_BACKENDS, default="auto")
    p_train.add_argument(
        "--session",
        default=None,
        help="Optional existing Colab session/runtime to reuse.",
    )
    p_train.add_argument(
        "--keep-runtime",
        action="store_true",
        help="Keep provisioned runtime alive after job finishes.",
    )
    p_train.add_argument(
        "--output",
        default=None,
        help="Local directory where trained outputs will be saved.",
    )
    p_train.add_argument(
        "--drive-folder",
        default="Annolid/colab",
        help="Google Drive folder path for notebook mode.",
    )
    p_train.add_argument(
        "--no-drive-upload",
        action="store_true",
        help="Do not upload bundle to Google Drive in notebook mode.",
    )
    p_train.add_argument(
        "--open-browser", action="store_true", help="Open Colab URL in default browser."
    )
    p_train.add_argument(
        "--annolid-spec",
        default="annolid",
        help="Pip package or git URL to install on Colab.",
    )
    p_train.add_argument("--name", default=None, help="Optional job name prefix.")
    p_train.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra flag passed through to `annolid-run train`.",
    )
    p_train.add_argument("--json", action="store_true", help="Output JSON.")
    p_train.set_defaults(_handler=_cmd_train)

    # predict
    p_pred = sub.add_parser(
        "predict", help="Run Annolid model inference on Google Colab."
    )
    p_pred.add_argument(
        "model",
        help="Annolid model plugin (e.g. yolo_labelme, yolo, dino_kpseg, keypoint_rcnn, maskrcnn_detectron2).",
    )
    p_pred.add_argument(
        "--source",
        required=True,
        help="Local video file, image file, or folder of images.",
    )
    p_pred.add_argument(
        "--weights", default=None, help="Trained checkpoint (.pt/.pth) or preset name."
    )
    p_pred.add_argument(
        "--dataset-dir",
        default=None,
        help="Optional COCO dataset dir (needed for maskrcnn_detectron2).",
    )
    p_pred.add_argument(
        "--conf", type=float, default=None, help="Confidence/score threshold."
    )
    p_pred.add_argument("--iou", type=float, default=None, help="NMS IoU threshold.")
    p_pred.add_argument("--imgsz", type=int, default=None, help="Inference image size.")
    p_pred.add_argument("--batch", type=int, default=None, help="Inference batch size.")
    p_pred.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable automatic tracking CSV export (yolo_labelme).",
    )
    p_pred.add_argument(
        "--save-video", action="store_true", help="Save annotated preview video."
    )
    p_pred.add_argument(
        "--no-tracking", action="store_true", help="Disable multi-object tracking."
    )
    p_pred.add_argument(
        "--classes", default=None, help="Optional class prompt/filter string."
    )
    p_pred.add_argument(
        "--gpu", default="T4", help="Colab accelerator (T4, L4, A100, H100, CPU)."
    )
    p_pred.add_argument(
        "--high-mem", action="store_true", help="Request high-RAM runtime."
    )
    p_pred.add_argument("--backend", choices=VALID_BACKENDS, default="auto")
    p_pred.add_argument(
        "--session",
        default=None,
        help="Optional existing Colab session/runtime to reuse.",
    )
    p_pred.add_argument(
        "--keep-runtime",
        action="store_true",
        help="Keep provisioned runtime alive after job finishes.",
    )
    p_pred.add_argument(
        "--output",
        default=None,
        help="Local directory where prediction outputs will be saved.",
    )
    p_pred.add_argument(
        "--drive-folder",
        default="Annolid/colab",
        help="Google Drive folder path for notebook mode.",
    )
    p_pred.add_argument(
        "--no-drive-upload",
        action="store_true",
        help="Do not upload bundle to Google Drive in notebook mode.",
    )
    p_pred.add_argument(
        "--open-browser", action="store_true", help="Open Colab URL in default browser."
    )
    p_pred.add_argument(
        "--annolid-spec",
        default="annolid",
        help="Pip package or git URL to install on Colab.",
    )
    p_pred.add_argument("--name", default=None, help="Optional job name prefix.")
    p_pred.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra flag passed through to `annolid-run predict`.",
    )
    p_pred.add_argument("--json", action="store_true", help="Output JSON.")
    p_pred.set_defaults(_handler=_cmd_predict)

    # notebook
    p_nb = sub.add_parser(
        "notebook",
        help="Generate a pre-configured Colab .ipynb notebook for train or predict.",
    )
    p_nb.add_argument("--mode", choices=("train", "predict"), required=True)
    p_nb.add_argument("--model", required=True, help="Annolid model plugin name.")
    p_nb.add_argument("--data", default=None, help="Dataset path (for --mode train).")
    p_nb.add_argument(
        "--source", default=None, help="Video/image source path (for --mode predict)."
    )
    p_nb.add_argument(
        "--weights", default=None, help="Weights preset or local checkpoint path."
    )
    p_nb.add_argument("--epochs", type=int, default=None)
    p_nb.add_argument("--batch", type=int, default=None)
    p_nb.add_argument("--imgsz", type=int, default=None)
    p_nb.add_argument("--preset", default=None)
    p_nb.add_argument("--conf", type=float, default=None)
    p_nb.add_argument("--no-csv", action="store_true")
    p_nb.add_argument("--save-video", action="store_true")
    p_nb.add_argument("--gpu", default="T4")
    p_nb.add_argument("--output", default=None, help="Output directory for results.")
    p_nb.add_argument("--drive-folder", default="Annolid/colab")
    p_nb.add_argument(
        "--upload-drive",
        action="store_true",
        help="Upload notebook + input archive to Google Drive.",
    )
    p_nb.add_argument(
        "--open-browser",
        action="store_true",
        help="Open generated Colab URL in browser.",
    )
    p_nb.add_argument("--annolid-spec", default="annolid")
    p_nb.add_argument("--json", action="store_true", help="Output JSON.")
    p_nb.set_defaults(_handler=_cmd_notebook)

    # pull
    p_pull = sub.add_parser(
        "pull", help="Pull finished Colab job outputs from Google Drive."
    )
    p_pull.add_argument("job_id", help="Colab job ID or Google Drive folder path.")
    p_pull.add_argument(
        "--output", default=None, help="Optional local destination directory."
    )
    p_pull.add_argument("--json", action="store_true", help="Output JSON.")
    p_pull.set_defaults(_handler=_cmd_pull)

    # jobs
    p_jobs = sub.add_parser("jobs", help="List recorded Annolid Colab jobs.")
    p_jobs.add_argument("--limit", type=int, default=25)
    p_jobs.add_argument("--json", action="store_true", help="Output JSON.")
    p_jobs.set_defaults(_handler=_cmd_jobs)

    # templates
    p_tpl = sub.add_parser(
        "templates", help="List curated official Annolid Colab notebooks."
    )
    p_tpl.add_argument("--json", action="store_true", help="Output JSON.")
    p_tpl.set_defaults(_handler=_cmd_templates)

    # doctor
    p_doc = sub.add_parser(
        "doctor",
        help="Run Colab/Drive preflight diagnostics (use --live for read-only API probes).",
    )
    p_doc.add_argument(
        "--live",
        action="store_true",
        help="Run read-only live network probes against Colab API, Drive API, and Colab CLI.",
    )
    p_doc.add_argument("--json", action="store_true", help="Output JSON.")
    p_doc.set_defaults(_handler=_cmd_doctor)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    handler = getattr(args, "_handler", None)
    if not callable(handler):
        parser.print_help()
        return 1
    try:
        return int(handler(args))
    except ColabAuthRequiredError as exc:
        if bool(getattr(args, "json", False)):
            _print_json(
                {
                    "ok": False,
                    "auth_required": True,
                    "backend": exc.backend,
                    "error": str(exc),
                }
            )
        else:
            print(f"[annolid-run colab] Sign-in required: {exc}", file=sys.stderr)
            print(
                "Sign in with one of:\n"
                "  annolid-run colab auth login --method colab-cli --install-cli   (personal Google account)\n"
                "  annolid-run colab auth login --method oauth2                    (Google OAuth2 client)\n"
                "  annolid-run colab auth login --method adc                       (gcloud ADC)",
                file=sys.stderr,
            )
        return 1
    except Exception as exc:
        if bool(getattr(args, "json", False)):
            _print_json({"ok": False, "error": str(exc)})
        else:
            print(f"[annolid-run colab] Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
