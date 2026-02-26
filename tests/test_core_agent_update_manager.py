from __future__ import annotations

from annolid.core.agent.update_manager.manager import (
    SignedUpdateManager,
    SignedUpdatePlan,
)
from annolid.core.agent.update_manager.canary import CanaryPolicy
from annolid.core.agent.update_manager.manifest import UpdateManifest
from annolid.core.agent.update_manager.verify import VerificationResult


def _plan(
    *, commands: list[list[str]], install_mode: str = "package"
) -> SignedUpdatePlan:
    manifest = UpdateManifest(
        project="annolid",
        channel="stable",
        version="9.9.9",
        release_date="",
        artifact_url="https://example.invalid/a.whl",
        artifact_sha256="abc123",
        signature="",
        signature_alg="none",
        source="https://pypi.org/pypi/annolid/json",
    )
    verify = VerificationResult(ok=True, reason="ok", details={})
    return SignedUpdatePlan(
        channel="stable",
        project="annolid",
        current_version="1.0.0",
        manifest=manifest,
        install_mode=install_mode,
        source_root=".",
        commands=commands,
        verification=verify,
    )


def test_signed_update_manager_dry_run_pipeline() -> None:
    manager = SignedUpdateManager(project="annolid")
    plan = _plan(commands=[["python", "-V"]])
    payload = manager.run(plan, execute=False, run_post_check=True)
    assert payload["status"] == "staged"
    assert payload["executed"] is False
    assert payload["rollback_plan"]["manual_required"] is False
    assert any(step.get("phase") == "restart" for step in payload["pipeline"])


def test_signed_update_manager_rolls_back_on_apply_failure() -> None:
    manager = SignedUpdateManager(project="annolid")
    plan = _plan(
        commands=[["python", "-c", "import sys; sys.exit(3)"]],
        install_mode="source",
    )
    payload = manager.run(plan, execute=True, run_post_check=False)
    assert payload["status"] == "failed"
    assert isinstance(payload.get("rollback"), dict)


def test_signed_update_manager_rolls_back_on_canary_threshold() -> None:
    manager = SignedUpdateManager(project="annolid")
    plan = _plan(commands=[], install_mode="source")
    payload = manager.run(
        plan,
        execute=True,
        run_post_check=False,
        canary_metrics={"sample_count": 100, "failure_count": 20, "regressions": 0},
        canary_policy=CanaryPolicy(
            min_samples=20, max_failure_rate=0.05, max_regressions=0
        ),
    )
    assert payload["status"] == "failed_canary"
    assert isinstance(payload.get("rollback"), dict)
