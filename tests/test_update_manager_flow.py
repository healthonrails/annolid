from __future__ import annotations

from annolid.core.agent.update_manager.manager import SignedUpdateManager
from annolid.core.agent.update_manager.manifest import UpdateManifest


def _manifest(version: str = "999.0.0") -> UpdateManifest:
    return UpdateManifest(
        project="annolid",
        channel="stable",
        version=version,
        release_date="2026-02-25T00:00:00Z",
        artifact_url="https://example.invalid/annolid.whl",
        artifact_sha256="deadbeef",
        signature="",
        signature_alg="none",
        source="https://example.invalid/manifest.json",
    )


def test_update_manager_stage_and_dry_run_pipeline(monkeypatch) -> None:
    import annolid.core.agent.update_manager.manager as manager_mod

    monkeypatch.setattr(
        manager_mod, "fetch_channel_manifest", lambda **kwargs: _manifest()
    )
    monkeypatch.setattr(
        manager_mod,
        "_build_apply_commands",
        lambda **kwargs: [["python", "-V"]],
    )

    manager = SignedUpdateManager(project="annolid")
    plan = manager.stage(channel="stable", timeout_s=0.1, require_signature=False)
    assert plan.update_available is True
    assert plan.verification.ok is True

    payload = manager.run(plan, execute=False, run_post_check=True)
    assert payload["status"] == "staged"
    assert payload["executed"] is False
    assert any(step.get("phase") == "restart" for step in payload["pipeline"])


def test_update_manager_blocks_when_signature_required(monkeypatch) -> None:
    import annolid.core.agent.update_manager.manager as manager_mod

    monkeypatch.setattr(
        manager_mod, "fetch_channel_manifest", lambda **kwargs: _manifest()
    )
    manager = SignedUpdateManager(project="annolid")
    plan = manager.stage(channel="stable", timeout_s=0.1, require_signature=True)
    assert plan.verification.ok is False
    assert plan.verification.reason == "signature_required_missing"

    payload = manager.run(plan, execute=False, run_post_check=False)
    assert payload["status"] == "blocked"
