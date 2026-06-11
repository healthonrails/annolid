from __future__ import annotations

import tomllib
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
    assert "gui)" in script
    assert "ml,tracking,cutie" in script
    assert "workstation)" in script
    assert "tracking,sam3,training" in script
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
    assert '"gui" { return @("ml", "tracking", "cutie") }' in script
    assert '"workstation" { return @("tracking", "sam3", "training") }' in script
    assert '"full" { return @("all") }' in script
    assert "Merge-Extras" in script


def test_installers_write_machine_readable_reports() -> None:
    bash_script = (ROOT / "install.sh").read_text(encoding="utf-8")
    powershell_script = (ROOT / "install.ps1").read_text(encoding="utf-8")

    for script in (bash_script, powershell_script):
        assert "annolid-install-report.json" in script
        assert "onnx_providers" in script
        assert "failed_optional_steps" in script
        assert "sam_hq_status" in script


def test_powershell_installer_does_not_require_torch_for_onnx_repair() -> None:
    script = (ROOT / "install.ps1").read_text(encoding="utf-8")
    repair_block = script[
        script.index("function Repair-OnnxRuntime") : script.index(
            "function Test-Installation"
        )
    ]

    assert "import torch" not in repair_block
    assert "import onnxruntime as ort" in repair_block


def test_powershell_installer_skips_ai_polygon_check_without_pycocotools() -> None:
    script = (ROOT / "install.ps1").read_text(encoding="utf-8")
    validation_block = script[
        script.index("function Test-Installation") : script.index(
            "function Write-InstallReport"
        )
    ]

    assert "Checking AI polygon model path" in validation_block
    assert "find_spec('pycocotools')" in validation_block
    assert "find_spec('pycocotools.mask')" in validation_block
    assert (
        "Skipping AI polygon check; optional pycocotools is not installed"
        in validation_block
    )
    assert "AI polygon validation failed" in validation_block


def test_powershell_install_report_handles_empty_python_version_output() -> None:
    script = (ROOT / "install.ps1").read_text(encoding="utf-8")
    report_block = script[
        script.index("function Write-InstallReport") : script.index(
            "function Write-Summary"
        )
    ]

    assert "$pythonVersionOutput =" in report_block
    assert "else { $script:PythonVersion }" in report_block
    assert '"unknown"' in report_block
    assert ".Trim()" in report_block
    assert "$pythonVersion = (& python" not in report_block
    assert "Convert-CommandOutputToString" in script
    assert "OSArchitecture.ToString()" not in report_block
    assert "UtcNow.ToString" not in report_block
    assert "Install report could not be written" in report_block


def test_installer_gpu_decision_uses_cuda12_provider_validation() -> None:
    bash_script = (ROOT / "install.sh").read_text(encoding="utf-8")
    powershell_script = (ROOT / "install.ps1").read_text(encoding="utf-8")

    for script in (bash_script, powershell_script):
        assert "onnxruntime-cuda-12/pypi/simple/" in script
        assert "CUDAExecutionProvider" in script
        assert "HasCuda12" in script or "HAS_CUDA12" in script
        assert ">= 12" in script or "-ge 12" in script


def test_installer_python_future_version_falls_back_through_uv() -> None:
    bash_script = (ROOT / "install.sh").read_text(encoding="utf-8")
    powershell_script = (ROOT / "install.ps1").read_text(encoding="utf-8")

    assert "Python 3.15+" in bash_script
    assert "USE_UV_PYTHON=true" in bash_script
    assert "Will use uv to create venv with Python 3.11" in bash_script
    assert "Python 3.15+" in powershell_script
    assert "$script:UseUvPython = $true" in powershell_script
    assert '$script:UvPythonVersion = "3.12"' in powershell_script


def test_pyproject_install_tiers_keep_heavy_ml_out_of_base() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = data["project"]["dependencies"]
    base_names = {
        dependency.split(">=")[0].split("[")[0] for dependency in dependencies
    }

    assert "torch" not in base_names
    assert "torchvision" not in base_names
    assert "transformers" not in base_names
    assert "huggingface-hub" not in base_names
    assert "onnxruntime" not in base_names
    assert "pycocotools" in base_names


def test_pyproject_defines_professional_optional_tiers() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]

    assert {"gui", "ml", "tracking", "training", "bot", "all"}.issubset(extras)
    assert any(requirement.startswith("transformers") for requirement in extras["ml"])
    assert any(
        requirement.startswith("onnxruntime") for requirement in extras["tracking"]
    )
    assert any(requirement.startswith("mcp") for requirement in extras["bot"])
    assert any(requirement.startswith("onnxruntime") for requirement in extras["all"])


def test_pyproject_test_extra_preserves_full_collection_import_contract() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    test_requirements = data["project"]["optional-dependencies"]["test"]
    test_names = {
        requirement.split(">=")[0].split("[")[0] for requirement in test_requirements
    }

    assert {
        "torch",
        "torchvision",
        "transformers",
        "onnxruntime",
        "omegaconf",
        "hydra-core",
        "pycocotools",
        "einops",
    }.issubset(test_names)


def test_release_workflow_uses_shared_bundle_artifact_guard() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "scripts/check_distribution_artifacts.py --kind bundle dist" in workflow
    assert "forbidden_suffixes" not in workflow
    assert "forbidden_names" not in workflow


def test_release_workflow_uploads_checksums_and_manifest() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "Create release manifest and checksum" in workflow
    assert ".sha256" in workflow
    assert "-manifest.json" in workflow
    assert "unsigned-ci-build" in workflow
