from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_bash_installer_gates_sam_hq_on_selected_extras() -> None:
    script = (ROOT / "install.sh").read_text(encoding="utf-8")

    sam_install = "segment-anything @ git+https://github.com/SysCV/sam-hq.git"
    assert sam_install in script
    assert 'if extra_selected "sam3"' in script
    assert script.index('if extra_selected "sam3"') < script.index(sam_install)
    assert "Skipping optional SAM-HQ install" in script


def test_bash_installer_defines_named_profiles() -> None:
    script = (ROOT / "install.sh").read_text(encoding="utf-8")

    assert "--profile PROFILE" in script
    assert "minimal,gui,workstation,full" in script
    assert 'PROFILE="gui"' in script
    assert "profile_extras()" in script
    assert "workstation)" in script
    assert "sam3,yolo,training" in script
    assert "full)" in script
    assert "all" in script
    assert 'merge_extras "gui"' in script


def test_powershell_installer_gates_sam_hq_on_selected_extras() -> None:
    script = (ROOT / "install.ps1").read_text(encoding="utf-8")

    sam_install = "segment-anything @ git+https://github.com/SysCV/sam-hq.git"
    assert sam_install in script
    assert "Test-ExtraSelected" in script
    assert script.index("Test-ExtraSelected") < script.index(sam_install)
    assert "Skipping optional SAM-HQ install" in script


def test_powershell_installer_defines_named_profiles() -> None:
    script = (ROOT / "install.ps1").read_text(encoding="utf-8")

    assert '[ValidateSet("minimal", "gui", "workstation", "full")]' in script
    assert '[string]$Profile = "gui"' in script
    assert "Get-ProfileExtras" in script
    assert '"workstation" { return @("sam3", "yolo", "training") }' in script
    assert '"full" { return @("all") }' in script
    assert "Merge-Extras" in script


def test_release_workflow_uses_shared_bundle_artifact_guard() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "scripts/check_distribution_artifacts.py --kind bundle dist" in workflow
    assert "forbidden_suffixes" not in workflow
    assert "forbidden_names" not in workflow
