from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import urlopen

from annolid.core.agent.observability import emit_governance_event

from .auto_update import AutoUpdatePolicy
from .manager import SignedUpdateManager, SignedUpdatePlan


@dataclass(frozen=True)
class UpdateArtifact:
    path: str
    sha256: str
    bytes_downloaded: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "bytes_downloaded": self.bytes_downloaded,
        }


class UpdateManagerService:
    """Safe update transaction service around SignedUpdateManager."""

    def __init__(
        self,
        *,
        project: str = "annolid",
        manager: Optional[SignedUpdateManager] = None,
        stage_dir: Optional[Path] = None,
    ) -> None:
        self.project = str(project or "annolid").strip() or "annolid"
        self.manager = manager or SignedUpdateManager(project=self.project)
        default_stage = Path.home() / ".annolid" / "updates" / "stage"
        self.stage_dir = Path(stage_dir or default_stage).expanduser().resolve()
        self.stage_dir.mkdir(parents=True, exist_ok=True)

    def check(
        self,
        *,
        channel: str = "stable",
        timeout_s: float = 4.0,
        require_signature: bool = False,
    ) -> SignedUpdatePlan:
        return self.manager.stage(
            channel=channel,
            timeout_s=timeout_s,
            require_signature=bool(require_signature),
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def download_and_verify_artifact(
        self,
        plan: SignedUpdatePlan,
        *,
        timeout_s: float = 20.0,
        execute: bool = False,
    ) -> Dict[str, Any]:
        artifact_name = os.path.basename(
            str(plan.manifest.artifact_url or "").strip()
        ) or (f"{plan.project}-{plan.manifest.version}.artifact")
        out_path = self.stage_dir / artifact_name
        payload: Dict[str, Any] = {
            "phase": "stage_artifact",
            "ok": True,
            "executed": bool(execute),
            "artifact_url": str(plan.manifest.artifact_url or ""),
            "path": str(out_path),
        }
        if not execute:
            payload["dry_run"] = True
            return payload
        try:
            with urlopen(
                plan.manifest.artifact_url, timeout=max(1.0, float(timeout_s))
            ) as resp:  # noqa: S310
                data = resp.read()
            out_path.write_bytes(data)
            digest = self._sha256(out_path)
            expected = str(plan.manifest.artifact_sha256 or "").strip().lower()
            payload["sha256"] = digest
            payload["bytes_downloaded"] = len(data)
            if expected and digest.lower() != expected:
                payload["ok"] = False
                payload["reason"] = "artifact_checksum_mismatch"
        except Exception as exc:
            payload["ok"] = False
            payload["reason"] = "artifact_download_failed"
            payload["error"] = str(exc)

        emit_governance_event(
            event_type="update",
            action="stage_artifact",
            outcome="ok" if bool(payload.get("ok", False)) else "failed",
            actor="operator" if execute else "system",
            details=dict(payload),
        )
        return payload

    def run_transaction(
        self,
        *,
        channel: str = "stable",
        timeout_s: float = 4.0,
        require_signature: bool = False,
        execute: bool = False,
        run_post_check: bool = True,
    ) -> Dict[str, Any]:
        plan = self.check(
            channel=channel,
            timeout_s=timeout_s,
            require_signature=require_signature,
        )
        transaction = {
            "project": self.project,
            "channel": channel,
            "executed": bool(execute),
            "steps": [],
            "status": "ok",
        }
        transaction["steps"].append(
            {"phase": "preflight", "ok": True, "executed": execute}
        )
        transaction["steps"].append(
            {
                "phase": "stage",
                "ok": bool(plan.update_available),
                "executed": bool(execute),
                "verification": plan.verification.to_dict(),
            }
        )
        artifact_step = self.download_and_verify_artifact(plan, execute=execute)
        transaction["steps"].append(artifact_step)
        if execute and not bool(artifact_step.get("ok", False)):
            transaction["status"] = "failed_stage_artifact"
            return transaction

        run_payload = self.manager.run(
            plan,
            execute=execute,
            run_post_check=run_post_check,
        )
        pipeline = run_payload.get("pipeline")
        if isinstance(pipeline, list):
            transaction["steps"].extend(list(pipeline))
        transaction["status"] = str(run_payload.get("status") or "ok")
        for key in ("rollback", "rollback_plan", "verification", "canary"):
            if key in run_payload:
                transaction[key] = run_payload.get(key)
        return transaction

    def auto_update_tick(
        self,
        *,
        policy: Optional[AutoUpdatePolicy] = None,
        last_check_epoch_s: Optional[float] = None,
        now_epoch_s: Optional[float] = None,
        execute: bool = False,
    ) -> Dict[str, Any]:
        pol = policy or AutoUpdatePolicy.from_config_and_env()
        if not pol.enabled:
            return {"status": "disabled", "policy": pol.to_dict()}
        due = pol.is_due(
            last_check_epoch_s=last_check_epoch_s,
            now_epoch_s=now_epoch_s,
        )
        if not due["due"]:
            return {"status": "not_due", "policy": pol.to_dict(), "schedule": due}
        tx = self.run_transaction(
            channel=pol.channel,
            timeout_s=pol.timeout_s,
            require_signature=pol.require_signature,
            execute=execute,
            run_post_check=True,
        )
        return {
            "status": str(tx.get("status") or "ok"),
            "policy": pol.to_dict(),
            "schedule": due,
            "transaction": tx,
        }
