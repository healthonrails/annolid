from __future__ import annotations

import hashlib
from pathlib import Path

from annolid.core.agent.update_manager.auto_update import AutoUpdatePolicy
from annolid.core.agent.update_manager.manager import SignedUpdatePlan
from annolid.core.agent.update_manager.manifest import UpdateManifest
from annolid.core.agent.update_manager.service import UpdateManagerService
from annolid.core.agent.update_manager.verify import VerificationResult


class _FakeManager:
    def __init__(self, plan: SignedUpdatePlan) -> None:
        self._plan = plan

    def stage(self, **kwargs) -> SignedUpdatePlan:
        del kwargs
        return self._plan

    def run(self, plan: SignedUpdatePlan, **kwargs):  # noqa: ANN001
        del plan, kwargs
        return {"status": "staged", "pipeline": [{"phase": "restart", "ok": True}]}


def _plan(url: str, sha: str) -> SignedUpdatePlan:
    manifest = UpdateManifest(
        project="annolid",
        channel="stable",
        version="9.9.9",
        release_date="",
        artifact_url=url,
        artifact_sha256=sha,
        signature="",
        signature_alg="none",
        source="https://example.invalid/manifest.json",
    )
    verify = VerificationResult(ok=True, reason="ok", details={})
    return SignedUpdatePlan(
        channel="stable",
        project="annolid",
        current_version="1.0.0",
        manifest=manifest,
        install_mode="package",
        source_root=".",
        commands=[],
        verification=verify,
    )


def test_update_manager_service_download_and_verify(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.whl"
    artifact.write_bytes(b"abc")
    sha = hashlib.sha256(b"abc").hexdigest()
    plan = _plan(artifact.as_uri(), sha)
    service = UpdateManagerService(
        project="annolid",
        manager=_FakeManager(plan),  # type: ignore[arg-type]
        stage_dir=tmp_path / "stage",
    )
    payload = service.download_and_verify_artifact(plan, execute=True)
    assert payload["ok"] is True
    assert int(payload["bytes_downloaded"]) == 3


def test_update_manager_service_transaction_includes_steps(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.whl"
    artifact.write_bytes(b"abc")
    sha = hashlib.sha256(b"abc").hexdigest()
    plan = _plan(artifact.as_uri(), sha)
    service = UpdateManagerService(
        project="annolid",
        manager=_FakeManager(plan),  # type: ignore[arg-type]
        stage_dir=tmp_path / "stage",
    )
    payload = service.run_transaction(execute=False)
    assert payload["status"] == "staged"
    phases = [str(step.get("phase") or "") for step in payload["steps"]]
    assert "preflight" in phases
    assert "stage" in phases
    assert "stage_artifact" in phases


def test_auto_update_policy_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ANNOLID_AUTO_UPDATE_ENABLED", raising=False)
    monkeypatch.delenv("ANNOLID_AUTO_UPDATE_CHANNEL", raising=False)
    monkeypatch.delenv("ANNOLID_AUTO_UPDATE_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv("ANNOLID_AUTO_UPDATE_JITTER_SECONDS", raising=False)
    policy = AutoUpdatePolicy.from_env()
    assert policy.enabled is False


def test_auto_update_policy_due_with_interval_and_jitter() -> None:
    policy = AutoUpdatePolicy(
        enabled=True,
        channel="stable",
        interval_seconds=600,
        jitter_seconds=60,
    )
    result = policy.is_due(last_check_epoch_s=0.0, now_epoch_s=1000.0)
    assert isinstance(result["due"], bool)
    assert "next_due_epoch_s" in result
