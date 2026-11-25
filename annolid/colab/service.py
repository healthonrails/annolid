from __future__ import annotations

import importlib.util
import json
import os
import stat
import threading
import webbrowser
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

from .auth import ColabAuthManager, ColabAuthRequiredError
from .backends import (
    ColabApiBackend,
    ColabCliBackend,
    ColabDriveNotebookBackend,
    LogCallback,
)
from .models import (
    ColabJobRecord,
    ColabJobStore,
    ColabPredictRequest,
    ColabRuntimeInfo,
    ColabRuntimeSpec,
    ColabTrainRequest,
    colab_home_dir,
    new_job_id,
    normalize_accelerator,
)
from .notebooks import (
    generate_predict_notebook,
    generate_train_notebook,
    list_notebook_templates,
    write_notebook,
)
from .packaging import (
    StagedInputBundle,
    build_remote_predict_command,
    build_remote_runner_script,
    build_remote_train_command,
    package_inference_inputs,
    package_training_dataset,
    safe_extract_archive,
)

ProgressCallback = Optional[Callable[[int, str], None]]


def _emit_log(cb: LogCallback, msg: str) -> None:
    if cb is not None:
        try:
            cb(msg)
        except Exception:
            pass


def _emit_progress(cb: ProgressCallback, pct: int, label: str) -> None:
    if cb is not None:
        try:
            cb(max(0, min(100, int(pct))), label)
        except Exception:
            pass


def _parse_result_json_from_logs(log_text: str) -> dict[str, Any]:
    marker = "ANNOLID_COLAB_RESULT_JSON="
    for line in reversed(str(log_text or "").splitlines()):
        idx = line.find(marker)
        if idx != -1:
            raw = line[idx + len(marker) :].strip()
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
    return {}


class ColabWorkflowService:
    """High-level orchestration service for Google Colab auth, runtimes, train & predict."""

    def __init__(
        self,
        *,
        auth_manager: Optional[ColabAuthManager] = None,
        home_dir: Optional[Path] = None,
        cli_backend: Optional[ColabCliBackend] = None,
        api_backend: Optional[ColabApiBackend] = None,
        notebook_backend: Optional[ColabDriveNotebookBackend] = None,
        allowed_read_roots: Optional[Sequence[Path | str]] = None,
    ) -> None:
        self.home_dir = Path(home_dir) if home_dir is not None else colab_home_dir()
        self.auth = auth_manager or ColabAuthManager()
        self.job_store = ColabJobStore(self.home_dir)
        self.cli_backend = cli_backend or ColabCliBackend(self.auth)
        self.api_backend = api_backend or ColabApiBackend(self.auth)
        self.notebook_backend = notebook_backend or ColabDriveNotebookBackend(self.auth)
        self._allowed_read_roots = (
            tuple(Path(p).expanduser().resolve() for p in allowed_read_roots)
            if allowed_read_roots
            else None
        )

    # ------------------------------------------------------------------ #
    # Auth & runtime discovery
    # ------------------------------------------------------------------ #

    def auth_status(self) -> dict[str, Any]:
        return self.auth.get_auth_status()

    def run_diagnostics(self, *, live: bool = False) -> dict[str, Any]:
        """Run offline (and optional live read-only) Colab/Drive preflight diagnostics."""
        status = self.auth.get_auth_status()
        checks: list[dict[str, Any]] = []

        def _add(name: str, ok: bool, detail: str, *, severity: str = "info") -> None:
            checks.append(
                {
                    "name": name,
                    "ok": bool(ok),
                    "severity": "ok" if ok else severity,
                    "detail": detail,
                }
            )

        # 1. Python dependencies
        for mod in (
            "google.auth",
            "google_auth_oauthlib",
            "googleapiclient",
            "websockets",
        ):
            present = False
            try:
                present = importlib.util.find_spec(mod) is not None
            except Exception:
                present = False
            _add(
                f"python_dep:{mod}",
                present,
                "installed"
                if present
                else 'missing (install with `pip install "annolid[annolid_bot]"`)',
                severity="warn",
            )

        # 2. Local Colab home directory writability
        try:
            self.home_dir.mkdir(parents=True, exist_ok=True)
            probe = self.home_dir / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            _add("colab_home_writable", True, str(self.home_dir))
        except Exception as exc:
            _add(
                "colab_home_writable",
                False,
                f"{self.home_dir}: {exc}",
                severity="error",
            )

        # 3. Official google-colab-cli
        cli_info = status["colab_cli"]
        _add(
            "colab_cli",
            bool(cli_info["installed"]),
            (
                f"{cli_info['binary']} ({cli_info['version'] or 'version unknown'})"
                if cli_info["installed"]
                else f"not installed (install with `{cli_info['install_command']}`)"
            ),
            severity="warn",
        )

        # 4. OAuth client secrets schema check
        creds_path = Path(status["credentials_file"])
        if creds_path.is_file():
            try:
                raw = json.loads(creds_path.read_text(encoding="utf-8"))
                client_block = (
                    raw.get("installed") or raw.get("web")
                    if isinstance(raw, dict)
                    else None
                )
                if isinstance(client_block, dict) and client_block.get("client_id"):
                    _add(
                        "oauth_client_json",
                        True,
                        f"valid OAuth client JSON at {creds_path}",
                    )
                elif isinstance(raw, dict) and raw.get("type") == "service_account":
                    _add(
                        "oauth_client_json",
                        False,
                        f"{creds_path} is a service_account key; Desktop OAuth client JSON (`installed`) is required.",
                        severity="error",
                    )
                else:
                    _add(
                        "oauth_client_json",
                        False,
                        f"{creds_path} is missing expected `installed`/`web` OAuth client keys.",
                        severity="error",
                    )
            except Exception as exc:
                _add(
                    "oauth_client_json",
                    False,
                    f"failed to parse {creds_path}: {exc}",
                    severity="error",
                )
        else:
            _add(
                "oauth_client_json",
                False,
                f"not found at {creds_path} (optional if using google-colab-cli or gcloud ADC)",
                severity="warn",
            )

        # 5. Token file permissions & scope readiness
        token_path = Path(status["token_file"])
        if token_path.is_file():
            if os.name != "nt":
                mode = stat.S_IMODE(token_path.stat().st_mode)
                perms_ok = (mode & 0o077) == 0
                _add(
                    "oauth_token_permissions",
                    perms_ok,
                    f"mode {oct(mode)}"
                    if perms_ok
                    else f"mode {oct(mode)} is too permissive; expected 0o600",
                    severity="warn",
                )
            _add(
                "oauth_colab_scopes",
                bool(status["oauth_colab"]["ready"]),
                "ready"
                if status["oauth_colab"]["ready"]
                else str(status["oauth_colab"]["reason"]),
                severity="warn",
            )
            _add(
                "oauth_drive_scopes",
                bool(status["oauth_drive"]["ready"]),
                "ready"
                if status["oauth_drive"]["ready"]
                else str(status["oauth_drive"]["reason"]),
                severity="warn",
            )
        else:
            _add(
                "oauth_token_file",
                False,
                f"no cached token at {token_path} (run `annolid-run colab auth login`)",
                severity="warn",
            )

        # 6. ADC status
        adc = status["adc"]
        _add(
            "gcloud_adc",
            bool(adc["ready"]),
            f"ADC ready ({adc['adc_file']})"
            if adc["ready"]
            else f"not configured (run `{adc['setup_command']}`)",
            severity="warn",
        )

        # 7. Optional live read-only network probes
        live_probes: dict[str, Any] = {}
        if live:
            if status["cli_ready"]:
                try:
                    sessions = self.cli_backend.list_sessions()
                    live_probes["colab_cli"] = {
                        "ok": True,
                        "active_sessions": len(sessions),
                    }
                    _add(
                        "live:colab_cli",
                        True,
                        f"authenticated ({len(sessions)} active session(s))",
                    )
                except Exception as exc:
                    live_probes["colab_cli"] = {"ok": False, "error": str(exc)}
                    _add("live:colab_cli", False, str(exc), severity="error")
            else:
                live_probes["colab_cli"] = {
                    "ok": False,
                    "skipped": True,
                    "reason": "google-colab-cli not installed",
                }

            cli_live_ok = bool((live_probes.get("colab_cli") or {}).get("ok") is True)
            secondary_sev = "warn" if cli_live_ok else "error"

            if status["api_ready"]:
                try:
                    specs = self.list_runtime_specs()
                    eligible = [s for s in specs if s.eligible]
                    live_probes["colab_api"] = {
                        "ok": True,
                        "total_specs": len(specs),
                        "eligible_accelerators": sorted(
                            {f"{s.variant}:{s.accelerator}" for s in eligible}
                        ),
                    }
                    _add(
                        "live:colab_api",
                        True,
                        f"reachable ({len(eligible)}/{len(specs)} eligible specs)",
                    )
                except Exception as exc:
                    live_probes["colab_api"] = {"ok": False, "error": str(exc)}
                    _add("live:colab_api", False, str(exc), severity=secondary_sev)
            else:
                live_probes["colab_api"] = {
                    "ok": False,
                    "skipped": True,
                    "reason": "Colab API OAuth/ADC not configured",
                }

            if status["drive_ready"]:
                try:
                    acct = self.notebook_backend.probe_drive_account()
                    live_probes["google_drive"] = {"ok": True, **acct}
                    _add(
                        "live:google_drive",
                        True,
                        f"authenticated as {acct.get('email') or acct.get('display_name') or 'Google user'}",
                    )
                except Exception as exc:
                    live_probes["google_drive"] = {"ok": False, "error": str(exc)}
                    _add("live:google_drive", False, str(exc), severity=secondary_sev)
            else:
                live_probes["google_drive"] = {
                    "ok": False,
                    "skipped": True,
                    "reason": "Google Drive OAuth not configured",
                }

        has_errors = any(c["severity"] == "error" for c in checks)
        return {
            "ok": not has_errors,
            "live": bool(live),
            "recommended_backend": status["recommended_backend"],
            "summary": status["summary"],
            "checks": checks,
            "live_probes": live_probes,
        }

    def login(
        self,
        *,
        method: str = "auto",
        include_drive: bool = True,
        open_browser: bool = True,
        install_cli_if_missing: bool = False,
        log_callback: LogCallback = None,
        auth_code_callback: Optional[Callable[[str], Optional[str]]] = None,
    ) -> dict[str, Any]:
        return self.auth.login(
            method=method,
            include_drive=include_drive,
            open_browser=open_browser,
            install_cli_if_missing=install_cli_if_missing,
            log_callback=log_callback,
            auth_code_callback=auth_code_callback,
        )

    def install_colab_cli(self, *, log_callback: LogCallback = None) -> str:
        return self.auth.install_colab_cli(log_callback=log_callback)

    def get_account_info(self) -> dict[str, Any]:
        return self.auth.get_account_info()

    def logout(
        self,
        *,
        include_colab_cli: bool = False,
        clear_adc: bool = False,
    ) -> dict[str, Any]:
        return self.auth.logout(
            include_colab_cli=include_colab_cli,
            clear_adc=clear_adc,
        )

    def resolve_backend(self, requested: str = "auto") -> str:
        choice = str(requested or "auto").strip().lower()
        if choice in {"cli", "api", "notebook"}:
            return choice
        status = self.auth.get_auth_status()
        return str(status.get("recommended_backend") or "notebook")

    def list_runtime_specs(self) -> list[ColabRuntimeSpec]:
        return self.api_backend.list_runtime_specs()

    def list_sessions(self, *, backend: str = "auto") -> list[ColabRuntimeInfo]:
        resolved = self.resolve_backend(backend)
        if resolved == "cli":
            return self.cli_backend.list_sessions()
        if resolved == "api":
            return self.api_backend.list_runtimes()
        # Try both if notebook was recommended (nothing configured yet)
        out: list[ColabRuntimeInfo] = []
        if self.cli_backend.is_available():
            try:
                out.extend(self.cli_backend.list_sessions())
            except Exception:
                pass
        return out

    def create_session(
        self,
        *,
        session_name: Optional[str] = None,
        accelerator: str = "T4",
        high_mem: bool = False,
        backend: str = "auto",
        log_callback: LogCallback = None,
    ) -> ColabRuntimeInfo:
        resolved = self.resolve_backend(backend)
        if resolved == "notebook":
            raise ColabAuthRequiredError(
                "Direct runtime provisioning requires `cli` (`google-colab-cli`) or "
                "`api` (Google Colab OAuth2 / gcloud ADC). Run `annolid-run colab auth login` first.",
                backend=backend,
                auth_status=self.auth.get_auth_status(),
            )
        if resolved == "cli":
            name = str(session_name or new_job_id("annolid"))
            return self.cli_backend.create_runtime(
                session_name=name,
                accelerator=accelerator,
                high_mem=high_mem,
                log_callback=log_callback,
            )
        return self.api_backend.create_runtime(
            accelerator=accelerator,
            high_mem=high_mem,
            log_callback=log_callback,
        )

    def stop_session(
        self,
        session_or_runtime: str,
        *,
        backend: str = "auto",
    ) -> None:
        target = str(session_or_runtime or "").strip()
        if not target:
            raise ValueError("A session or runtime name is required.")
        resolved = self.resolve_backend(backend)
        if resolved == "api" or target.startswith("runtimes/"):
            self.api_backend.delete_runtime(target)
            return
        self.cli_backend.stop_runtime(target)

    def list_jobs(self, *, limit: int = 50) -> list[ColabJobRecord]:
        return self.job_store.list_jobs(limit=limit)

    def get_job(self, job_id: str) -> Optional[ColabJobRecord]:
        return self.job_store.load(job_id)

    def list_templates(self) -> list[dict[str, Any]]:
        return list_notebook_templates()

    # ------------------------------------------------------------------ #
    # Train / Predict / Notebook / Pull
    # ------------------------------------------------------------------ #

    def _default_output_dir(self, job_id: str, explicit: Optional[str]) -> Path:
        if explicit and str(explicit).strip():
            return Path(str(explicit).strip()).expanduser().resolve()
        return (self.home_dir / "outputs" / job_id).resolve()

    def _stage_job(
        self,
        request: Union[ColabTrainRequest, ColabPredictRequest],
    ) -> tuple[
        str,
        str,
        Path,
        Path,
        str,
        StagedInputBundle,
        list[str],
        dict[str, Any],
    ]:
        is_train = isinstance(request, ColabTrainRequest)
        kind = "train" if is_train else "predict"
        prefix = request.job_name or f"{kind}_{request.model}"
        job_id = new_job_id(prefix)
        staging_dir = (self.home_dir / "staging" / job_id).resolve()
        staging_dir.mkdir(parents=True, exist_ok=True)
        local_output_dir = self._default_output_dir(job_id, request.output_dir)
        remote_job_root = f"/content/annolid_jobs/{job_id}"
        remote_input_root = f"{remote_job_root}/input"
        remote_output_root = f"{remote_job_root}/output"

        if is_train:
            bundle = package_training_dataset(
                request.dataset_path,
                model=request.model,
                staging_dir=staging_dir,
                weights=request.weights,
                allowed_read_roots=self._allowed_read_roots,
            )
            remote_cmd = build_remote_train_command(
                request,
                remote_input_root=remote_input_root,
                remote_output_root=remote_output_root,
                dataset_entry=bundle.dataset_entry,
                weights_entry=bundle.weights_entry,
            )
            drive_job_rel = f"{request.drive_folder.strip('/')}/{job_id}"
            nb_payload = generate_train_notebook(
                request,
                job_id=job_id,
                remote_job_root=remote_job_root,
                input_archive_name=bundle.archive_path.name,
                remote_command=remote_cmd,
                drive_job_rel_path=drive_job_rel,
                use_drive=bool(request.upload_to_drive),
            )
        else:
            bundle = package_inference_inputs(
                request.source_path,
                staging_dir=staging_dir,
                weights=request.weights,
                dataset_dir=request.dataset_dir,
                allowed_read_roots=self._allowed_read_roots,
            )
            remote_cmd = build_remote_predict_command(
                request,
                remote_input_root=remote_input_root,
                remote_output_root=remote_output_root,
                source_entry=bundle.dataset_entry,
                weights_entry=bundle.weights_entry,
                extra_dataset_entry=bundle.extra_dataset_entry,
            )
            drive_job_rel = f"{request.drive_folder.strip('/')}/{job_id}"
            nb_payload = generate_predict_notebook(
                request,
                job_id=job_id,
                remote_job_root=remote_job_root,
                input_archive_name=bundle.archive_path.name,
                remote_command=remote_cmd,
                drive_job_rel_path=drive_job_rel,
                use_drive=bool(request.upload_to_drive),
            )

        return (
            job_id,
            kind,
            staging_dir,
            local_output_dir,
            remote_job_root,
            bundle,
            remote_cmd,
            nb_payload,
        )

    def run_train(
        self,
        request: ColabTrainRequest,
        *,
        log_callback: LogCallback = None,
        progress_callback: ProgressCallback = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> ColabJobRecord:
        return self._run_job(
            request,
            log_callback=log_callback,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

    def run_predict(
        self,
        request: ColabPredictRequest,
        *,
        log_callback: LogCallback = None,
        progress_callback: ProgressCallback = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> ColabJobRecord:
        return self._run_job(
            request,
            log_callback=log_callback,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

    def _run_job(
        self,
        request: Union[ColabTrainRequest, ColabPredictRequest],
        *,
        log_callback: LogCallback = None,
        progress_callback: ProgressCallback = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> ColabJobRecord:
        backend = self.resolve_backend(request.backend)
        acc = normalize_accelerator(request.accelerator)

        # Fail fast on missing/expired credentials before zipping potentially large datasets.
        if backend == "api":
            _emit_progress(progress_callback, 2, "Verifying Colab API credentials...")
            self.auth.get_colab_api_access_token()
        elif backend == "cli" and not self.cli_backend.is_available():
            raise ColabAuthRequiredError(
                "Official `google-colab-cli` (`colab`) is not installed. "
                "Run `annolid-run colab auth login --method colab-cli --install-cli`.",
                backend="cli",
                auth_status=self.auth.get_auth_status(),
            )

        _emit_progress(progress_callback, 5, "Packaging local inputs...")
        _emit_log(
            log_callback,
            f"Packaging inputs for {request.model} (backend={backend}, accelerator={acc})...",
        )
        (
            job_id,
            kind,
            staging_dir,
            local_output_dir,
            remote_job_root,
            bundle,
            remote_cmd,
            nb_payload,
        ) = self._stage_job(request)

        notebook_path = write_notebook(nb_payload, staging_dir / f"{job_id}.ipynb")
        drive_job_rel = f"{request.drive_folder.strip('/')}/{job_id}"
        record = ColabJobRecord(
            job_id=job_id,
            kind=kind,
            model=request.model,
            backend=backend,
            accelerator=acc,
            status="running" if backend in {"cli", "api"} else "notebook_ready",
            remote_command=list(remote_cmd),
            notebook_path=str(notebook_path),
            local_staging_dir=str(staging_dir),
            local_output_dir=str(local_output_dir),
            drive_folder=drive_job_rel,
        )
        self.job_store.save(record)
        _emit_log(
            log_callback,
            f"Packaged {bundle.file_count} file(s) ({bundle.archive_bytes / (1024 * 1024):.2f} MB) -> {bundle.archive_path}",
        )
        for note in bundle.notes:
            _emit_log(log_callback, f"[WARN] {note}")

        if backend == "notebook":
            return self._execute_notebook_backend(
                request=request,
                record=record,
                notebook_path=notebook_path,
                bundle=bundle,
                log_callback=log_callback,
                progress_callback=progress_callback,
            )

        # Stage input zip directly inside /content (always exists on fresh Colab VMs)
        # so `colab upload` never fails due to a non-existent remote subdirectory.
        remote_input_archive = f"/content/{job_id}_{bundle.archive_path.name}"
        runner_script = build_remote_runner_script(
            job_id=job_id,
            kind=kind,
            model=request.model,
            install_spec=request.annolid_install_spec,
            remote_job_root=remote_job_root,
            input_archive_remote_path=remote_input_archive,
            remote_command=remote_cmd,
        )
        runner_path = staging_dir / f"run_{job_id}.py"
        runner_path.write_text(runner_script, encoding="utf-8")

        if backend == "cli":
            return self._execute_cli_backend(
                request=request,
                record=record,
                bundle=bundle,
                runner_path=runner_path,
                remote_job_root=remote_job_root,
                remote_input_archive=remote_input_archive,
                local_output_dir=local_output_dir,
                log_callback=log_callback,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )

        return self._execute_api_backend(
            request=request,
            record=record,
            bundle=bundle,
            runner_script=runner_script,
            remote_job_root=remote_job_root,
            remote_input_archive=remote_input_archive,
            local_output_dir=local_output_dir,
            log_callback=log_callback,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

    def _execute_notebook_backend(
        self,
        *,
        request: Union[ColabTrainRequest, ColabPredictRequest],
        record: ColabJobRecord,
        notebook_path: Path,
        bundle: StagedInputBundle,
        log_callback: LogCallback,
        progress_callback: ProgressCallback,
    ) -> ColabJobRecord:
        if request.upload_to_drive and self.notebook_backend.is_drive_ready():
            _emit_progress(
                progress_callback, 45, "Uploading bundle & notebook to Google Drive..."
            )
            try:
                uploaded = self.notebook_backend.upload_job_bundle(
                    drive_job_rel_path=record.drive_folder,
                    notebook_path=notebook_path,
                    input_archive_path=bundle.archive_path,
                    log_callback=log_callback,
                )
                record.drive_folder_id = str(uploaded.get("drive_folder_id") or "")
                record.drive_notebook_id = str(uploaded.get("drive_notebook_id") or "")
                record.colab_url = str(uploaded.get("colab_url") or "")
                record.status = "notebook_ready"
                self.job_store.save(record)
                _emit_log(
                    log_callback,
                    f"Uploaded to Google Drive. Open in Colab: {record.colab_url}",
                )
            except Exception as exc:
                record.status = "failed"
                record.error = str(exc)
                self.job_store.save(record)
                raise
        else:
            record.colab_url = "https://colab.research.google.com/#create=true"
            record.status = "notebook_ready"
            self.job_store.save(record)
            _emit_log(
                log_callback,
                f"Generated local Colab notebook: {notebook_path}\n"
                f"Staged input archive: {bundle.archive_path}\n"
                "Tip: sign in with Google Drive OAuth (`annolid-run colab auth login --method drive`) "
                "to upload datasets & open notebooks in Colab automatically.",
            )

        if request.open_browser and record.colab_url:
            try:
                webbrowser.open(record.colab_url)
            except Exception:
                pass
        _emit_progress(progress_callback, 100, "Colab notebook ready")
        return record

    def _execute_cli_backend(
        self,
        *,
        request: Union[ColabTrainRequest, ColabPredictRequest],
        record: ColabJobRecord,
        bundle: StagedInputBundle,
        runner_path: Path,
        remote_job_root: str,
        remote_input_archive: str,
        local_output_dir: Path,
        log_callback: LogCallback,
        progress_callback: ProgressCallback,
        cancel_event: Optional[threading.Event],
    ) -> ColabJobRecord:
        session_name = str(request.session_name or f"annolid-{record.job_id[-10:]}")
        reused_session = False
        if request.session_name:
            try:
                active = {s.session_name for s in self.cli_backend.list_sessions()}
                reused_session = session_name in active
            except Exception:
                reused_session = False
        created_new = False
        try:
            if not reused_session:
                _emit_progress(
                    progress_callback, 15, "Provisioning Colab GPU runtime..."
                )
                rt = self.cli_backend.create_runtime(
                    session_name=session_name,
                    accelerator=request.accelerator,
                    high_mem=request.high_mem,
                    log_callback=log_callback,
                )
                created_new = True
                record.session_name = rt.session_name
                record.runtime_name = rt.name
                record.colab_url = rt.browser_url
                if rt.browser_url:
                    _emit_log(log_callback, f"Colab URL: {rt.browser_url}")
            else:
                record.session_name = session_name
                record.runtime_name = session_name
                try:
                    url_proc = self.cli_backend._run_capture(
                        ["url", "-s", session_name], timeout=20
                    )
                    for tok in (url_proc.stdout or "").split():
                        if tok.startswith("https://"):
                            record.colab_url = tok
                            _emit_log(log_callback, f"Colab URL: {tok}")
                            break
                except Exception:
                    pass
            self.job_store.save(record)

            _emit_progress(progress_callback, 30, "Uploading packaged inputs...")
            self.cli_backend.upload_file(
                session_name=session_name,
                local_path=bundle.archive_path,
                remote_path=remote_input_archive,
                log_callback=log_callback,
            )

            _emit_progress(progress_callback, 45, "Running Annolid on Colab...")
            logs = self.cli_backend.exec_script(
                session_name=session_name,
                script_path=runner_path,
                log_callback=log_callback,
                cancel_event=cancel_event,
            )

            _emit_progress(progress_callback, 85, "Downloading outputs...")
            local_output_dir.mkdir(parents=True, exist_ok=True)
            local_archive = local_output_dir / "output_bundle.zip"
            self.cli_backend.download_file(
                session_name=session_name,
                remote_path=f"{remote_job_root}/output_bundle.zip",
                local_path=local_archive,
                log_callback=log_callback,
            )
            extracted = safe_extract_archive(local_archive, local_output_dir)
            record.downloaded_files = [str(p) for p in extracted]
            record.result_summary = _parse_result_json_from_logs(logs)
            record.status = "completed"
            self.job_store.save(record)
            _emit_progress(progress_callback, 100, "Completed")
            _emit_log(
                log_callback,
                f"Downloaded {len(extracted)} output file(s) to {local_output_dir}",
            )
            return record
        except InterruptedError as exc:
            record.status = "cancelled"
            record.error = str(exc)
            self.job_store.save(record)
            raise
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
            self.job_store.save(record)
            raise
        finally:
            if created_new and not request.keep_runtime:
                try:
                    _emit_log(
                        log_callback, f"Stopping Colab session '{session_name}'..."
                    )
                    self.cli_backend.stop_runtime(session_name)
                except Exception as exc:
                    _emit_log(
                        log_callback,
                        f"[WARN] Could not stop Colab session '{session_name}': {exc}",
                    )

    def _execute_api_backend(
        self,
        *,
        request: Union[ColabTrainRequest, ColabPredictRequest],
        record: ColabJobRecord,
        bundle: StagedInputBundle,
        runner_script: str,
        remote_job_root: str,
        remote_input_archive: str,
        local_output_dir: Path,
        log_callback: LogCallback,
        progress_callback: ProgressCallback,
        cancel_event: Optional[threading.Event],
    ) -> ColabJobRecord:
        runtime: Optional[ColabRuntimeInfo] = None
        created_new = False
        try:
            _emit_progress(progress_callback, 15, "Provisioning Colab API runtime...")
            if request.session_name:
                runtime = self.api_backend.get_runtime(request.session_name)
            else:
                runtime = self.api_backend.create_runtime(
                    accelerator=request.accelerator,
                    high_mem=request.high_mem,
                    log_callback=log_callback,
                )
                created_new = True
            record.runtime_name = runtime.name
            record.session_name = runtime.session_name
            self.job_store.save(record)

            _emit_progress(progress_callback, 30, "Uploading packaged inputs...")
            runtime = self.api_backend.upload_file(
                runtime=runtime,
                local_path=bundle.archive_path,
                remote_path=remote_input_archive,
                log_callback=log_callback,
                cancel_event=cancel_event,
            )

            _emit_progress(progress_callback, 45, "Running Annolid on Colab...")
            logs = self.api_backend.execute_python(
                runtime=runtime,
                code=runner_script,
                log_callback=log_callback,
                cancel_event=cancel_event,
            )

            _emit_progress(progress_callback, 85, "Downloading outputs...")
            local_output_dir.mkdir(parents=True, exist_ok=True)
            local_archive = local_output_dir / "output_bundle.zip"
            runtime = self.api_backend.download_file(
                runtime=runtime,
                remote_path=f"{remote_job_root}/output_bundle.zip",
                local_path=local_archive,
                log_callback=log_callback,
            )
            extracted = safe_extract_archive(local_archive, local_output_dir)
            record.downloaded_files = [str(p) for p in extracted]
            record.result_summary = _parse_result_json_from_logs(logs)
            record.status = "completed"
            self.job_store.save(record)
            _emit_progress(progress_callback, 100, "Completed")
            _emit_log(
                log_callback,
                f"Downloaded {len(extracted)} output file(s) to {local_output_dir}",
            )
            return record
        except InterruptedError as exc:
            record.status = "cancelled"
            record.error = str(exc)
            self.job_store.save(record)
            raise
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
            self.job_store.save(record)
            raise
        finally:
            if created_new and runtime is not None and not request.keep_runtime:
                try:
                    _emit_log(
                        log_callback, f"Deleting Colab runtime '{runtime.name}'..."
                    )
                    self.api_backend.delete_runtime(runtime.name)
                except Exception as exc:
                    _emit_log(
                        log_callback,
                        f"[WARN] Could not delete Colab runtime '{runtime.name}': {exc}",
                    )

    def pull_results(
        self,
        job_id_or_folder: str,
        *,
        output_dir: Optional[str | Path] = None,
        log_callback: LogCallback = None,
    ) -> ColabJobRecord:
        """Pull outputs from Google Drive for a previously submitted Colab notebook job."""
        record = self.job_store.load(job_id_or_folder)
        if record is None:
            record = ColabJobRecord(
                job_id=new_job_id("pull"),
                kind="pull",
                model="",
                backend="notebook",
                accelerator="",
                status="running",
                drive_folder=str(job_id_or_folder).strip(),
            )
        dest_dir = (
            Path(output_dir).expanduser().resolve()
            if output_dir
            else (
                Path(record.local_output_dir).expanduser().resolve()
                if record.local_output_dir
                else self._default_output_dir(record.job_id, None)
            )
        )
        pulled = self.notebook_backend.pull_job_outputs(
            drive_folder_id=record.drive_folder_id,
            drive_folder_path=record.drive_folder,
            dest_dir=dest_dir,
            log_callback=log_callback,
        )
        extracted_files: list[str] = []
        for path in list(pulled.get("downloaded") or []):
            p = Path(path)
            if p.suffix.lower() == ".zip" and p.is_file():
                extracted = safe_extract_archive(p, dest_dir)
                extracted_files.extend(str(item) for item in extracted)
            else:
                extracted_files.append(str(p))

        record.local_output_dir = str(dest_dir)
        record.drive_folder_id = str(
            pulled.get("drive_folder_id") or record.drive_folder_id
        )
        record.downloaded_files = extracted_files
        if pulled.get("result_summary"):
            record.result_summary = dict(pulled["result_summary"])
        record.status = "completed" if extracted_files else "notebook_ready"
        self.job_store.save(record)
        return record
