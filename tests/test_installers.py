from __future__ import annotations

import os
import stat
import subprocess
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
    assert "minimal,gui,tracking,workstation,full" in script
    assert 'PROFILE="gui"' in script
    assert "profile_extras()" in script
    assert "gui)" in script
    assert "tracking)" in script
    assert "ml,tracking,cutie" in script
    assert "workstation)" in script
    assert "ml,tracking,cutie,sam3,training,webengine,pdf" in script
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

    assert (
        '[ValidateSet("minimal", "gui", "tracking", "workstation", "full")]' in script
    )
    assert '[string]$Profile = "gui"' in script
    assert "Get-ProfileExtras" in script
    assert '"gui" { return @() }' in script
    assert '"tracking" { return @("ml", "tracking", "cutie") }' in script
    assert (
        '"workstation" { return @("ml", "tracking", "cutie", "sam3", "training", "webengine", "pdf") }'
        in script
    )
    assert '"full" { return @("all") }' in script
    assert "Merge-Extras" in script


def test_installers_gate_pytorch_and_onnx_on_selected_extras() -> None:
    bash_script = (ROOT / "install.sh").read_text(encoding="utf-8")
    powershell_script = (ROOT / "install.ps1").read_text(encoding="utf-8")
    shortcuts_script = (ROOT / "scripts/create_desktop_shortcuts.py").read_text(
        encoding="utf-8"
    )

    assert "needs_pytorch_runtime()" in bash_script
    assert "if needs_pytorch_runtime; then" in bash_script
    assert "needs_onnx_runtime()" in bash_script
    assert "if needs_onnx_runtime; then" in bash_script
    assert "Skipping upfront PyTorch download for lean profile" in bash_script
    assert "Skipping upfront ONNX Runtime install for lean profile" in bash_script

    assert "function Test-PytorchRequested" in powershell_script
    assert "if (Test-PytorchRequested $installExtras)" in powershell_script
    assert "function Test-OnnxRequested" in powershell_script
    assert "if (Test-OnnxRequested $installExtras)" in powershell_script
    assert "if (Test-OnnxRequested $script:InstallExtras)" in powershell_script
    assert "Skipping upfront PyTorch download for lean profile" in powershell_script
    assert "Skipping upfront ONNX Runtime install for lean profile" in powershell_script

    assert '("minimal", "gui", "tracking", "workstation", "full")' in shortcuts_script


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

    assert "Python 3.16+" in bash_script
    assert "USE_UV_PYTHON=true" in bash_script
    assert "Will use uv to create venv with Python 3.11" in bash_script
    assert "Python 3.16+" in powershell_script
    assert "$script:UseUvPython = $true" in powershell_script
    assert '$script:UvPythonVersion = "3.12"' in powershell_script


def test_installers_support_python_314_and_315_as_preferred_versions() -> None:
    bash_script = (ROOT / "install.sh").read_text(encoding="utf-8")
    powershell_script = (ROOT / "install.ps1").read_text(encoding="utf-8")

    assert (
        'PREFERRED_VERSIONS=("3.11" "3.12" "3.13" "3.14" "3.15" "3.10")' in bash_script
    )
    assert '"$PYTHON_MINOR" -eq 14 || "$PYTHON_MINOR" -eq 15' in bash_script

    assert (
        '@{ Command = "py"; Args = @("-3.14"); Version = "3.14" }' in powershell_script
    )
    assert (
        '@{ Command = "py"; Args = @("-3.15"); Version = "3.15" }' in powershell_script
    )
    assert (
        '@{ Command = "python3.14"; Args = @(); Version = "3.14" }' in powershell_script
    )
    assert (
        '@{ Command = "python3.15"; Args = @(); Version = "3.15" }' in powershell_script
    )
    assert 'Version = ""' not in powershell_script
    assert "$minor -eq 14 -or $minor -eq 15" in powershell_script
    assert "--python $script:PythonVersion" in powershell_script


def _write_fake_python(bin_dir: Path, name: str, major: int, minor: int) -> None:
    exe = bin_dir / name
    exe.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "-c" ]]; then\n'
        '  case "$2" in\n'
        f'    *major*minor*) echo "{major}.{minor}" ;;\n'
        f'    *major*) echo "{major}" ;;\n'
        f'    *minor*) echo "{minor}" ;;\n'
        "  esac\n"
        "  exit 0\n"
        "fi\n"
        f'echo "Python {major}.{minor}.0"\n',
        encoding="utf-8",
    )
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)


def _run_bash_check_python(
    tmp_path: Path,
    *,
    executables: dict[str, tuple[int, int]],
    use_uv: bool = True,
) -> subprocess.CompletedProcess[str]:
    script_text = (ROOT / "install.sh").read_text(encoding="utf-8")
    start = script_text.index("check_python() {")
    end = script_text.index(
        "# =============================================================================\n# Check FFmpeg"
    )
    check_python_fn = script_text[start:end]

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    # Symlink minimal POSIX utilities needed by check_python so system pythons are hidden.
    for util in ("bash", "env"):
        for search_dir in ("/bin", "/usr/bin"):
            cand = Path(search_dir) / util
            if cand.exists() and not (bin_dir / util).exists():
                (bin_dir / util).symlink_to(cand)

    for exe_name, (major, minor) in executables.items():
        _write_fake_python(bin_dir, exe_name, major, minor)

    harness = tmp_path / "run_check_python.sh"
    harness.write_text(
        "set -e\n"
        'PYTHON_MIN_VERSION="3.10"\n'
        f"USE_UV={'true' if use_uv else 'false'}\n"
        "USE_CONDA=false\n"
        "USE_UV_PYTHON=false\n"
        "NO_INTERACTIVE=true\n"
        'OS_TYPE="macos"\n'
        'DISTRO="macos"\n'
        'print_step() { echo "STEP:$1"; }\n'
        'print_info() { echo "INFO:$1"; }\n'
        'print_warning() { echo "WARN:$1"; }\n'
        'print_error() { echo "ERR:$1"; }\n'
        'print_success() { echo "OK:$1"; }\n'
        'prompt_yes_no() { [[ "$2" == "y" ]]; }\n'
        f"{check_python_fn}\n"
        "check_python\n"
        'echo "RESULT_VERSION=$PYTHON_VERSION"\n'
        'echo "RESULT_UV_PYTHON=$USE_UV_PYTHON"\n',
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PATH"] = str(bin_dir)
    return subprocess.run(
        ["/bin/bash", str(harness)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_bash_check_python_accepts_315_and_falls_back_on_316(tmp_path: Path) -> None:
    res_315 = _run_bash_check_python(
        tmp_path / "py315",
        executables={"python3.15": (3, 15)},
    )
    assert res_315.returncode == 0, res_315.stderr
    assert "Found preferred: python3.15 (version 3.15)" in res_315.stdout
    assert (
        "Annolid supports the default GUI/core install on Python 3.15."
        in res_315.stdout
    )
    assert "RESULT_VERSION=3.15" in res_315.stdout
    assert "RESULT_UV_PYTHON=false" in res_315.stdout

    res_316_uv = _run_bash_check_python(
        tmp_path / "py316_uv",
        executables={"python3": (3, 16)},
        use_uv=True,
    )
    assert res_316_uv.returncode == 0, res_316_uv.stderr
    assert "Annolid is not yet validated on Python 3.16+." in res_316_uv.stdout
    assert "RESULT_VERSION=3.11" in res_316_uv.stdout
    assert "RESULT_UV_PYTHON=true" in res_316_uv.stdout

    res_316_no_uv = _run_bash_check_python(
        tmp_path / "py316_nouv",
        executables={"python3": (3, 16)},
        use_uv=False,
    )
    assert res_316_no_uv.returncode == 1


def test_pyproject_install_tiers_keep_heavy_ml_out_of_base() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = data["project"]["dependencies"]
    extras = data["project"]["optional-dependencies"]
    base_names = {
        dependency.split(">=")[0].split("[")[0].split(";")[0].strip()
        for dependency in dependencies
    }

    assert "torch" not in base_names
    assert "torchvision" not in base_names
    assert "transformers" not in base_names
    assert "huggingface-hub" not in base_names
    assert "onnxruntime" not in base_names
    assert "numba" not in base_names
    assert "scipy" not in base_names
    assert "scikit-learn" not in base_names
    assert "scikit-image" not in base_names
    assert "h5py" not in base_names
    assert "gdown" not in base_names
    assert "pycocotools" in base_names
    assert extras["gui"] == ["PySide6-Essentials>=6.7.0"]
    assert extras["webengine"] == ["PySide6-Addons>=6.7.0"]
    assert {"onnx", "scientific", "pdf", "model_downloads"}.issubset(extras)


def test_pyproject_python_314_315_compatibility_contracts() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    classifiers = set(project["classifiers"])
    extras = project["optional-dependencies"]

    assert project["requires-python"] == ">=3.10"
    assert "Programming Language :: Python :: 3.14" in classifiers
    assert "Programming Language :: Python :: 3.15" in classifiers

    for req_list in (
        project["dependencies"],
        extras["cutie"],
        extras["tracking"],
        extras["all"],
    ):
        pillow_specs = [r for r in req_list if r.lower().startswith("pillow")]
        assert pillow_specs == ["Pillow>=9.3.0"]

    assert extras["remote_video"] == ["ffpyplayer>=4.5.0"]
    assert extras["sleap"] == ["h5py>=3.15.1"]
    assert 'ffpyplayer>=4.5.0; python_version < "3.14"' in extras["all"]
    assert 'h5py>=3.15.1; python_version < "3.15"' in extras["test"]
    assert 'h5py>=3.15.1; python_version < "3.15"' in extras["all"]
    assert 'scikit-image>=0.20.0; python_version < "3.15"' in extras["scientific"]
    assert 'scikit-image>=0.20.0; python_version < "3.15"' in extras["test"]
    assert 'scikit-image>=0.20.0; python_version < "3.15"' in extras["all"]
    for audio_extra in ("audio", "text_to_speech", "all"):
        assert "numba>=0.60.0" in extras[audio_extra]


def test_pyproject_defines_professional_optional_tiers() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]

    assert {
        "gui",
        "webengine",
        "onnx",
        "scientific",
        "pdf",
        "model_downloads",
        "ml",
        "tracking",
        "training",
        "bot",
        "all",
    }.issubset(extras)
    assert any(requirement.startswith("transformers") for requirement in extras["ml"])
    assert any(requirement.startswith("torch") for requirement in extras["sam3"])
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


def test_release_workflow_supports_guarded_binary_recovery() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "release_tag:" in workflow
    assert "Validate recovery source matches release tag" in workflow
    assert (
        'git diff --exit-code "${RELEASE_TAG}" -- annolid annolid.spec pyproject.toml'
        in workflow
    )
    assert "inputs.release_tag || github.ref_name" in workflow


def test_release_workflow_builds_unix_desktop_archives_only() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "macos-latest" in workflow
    assert "ubuntu-latest" in workflow
    assert "windows-latest" not in workflow
    assert "Annolid-Windows.zip" not in workflow
    assert "download.pytorch.org/whl/cpu" not in workflow
    assert "torch torchvision" not in workflow


def test_release_workflow_uploads_checksums_and_manifest() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "Create release manifest and checksum" in workflow
    assert ".sha256" in workflow
    assert "-manifest.json" in workflow
    assert "unsigned-ci-build" in workflow


def test_release_workflow_does_not_duplicate_macos_bundle_in_archive() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert 'python -m zipfile -c "${{ matrix.archive-name }}" Annolid.app' in workflow
    assert (
        'python -m zipfile -c "${{ matrix.archive-name }}" annolid Annolid.app'
        not in workflow
    )


def test_installers_support_offline_wheelhouse_mode() -> None:
    bash_script = (ROOT / "install.sh").read_text(encoding="utf-8")
    powershell_script = (ROOT / "install.ps1").read_text(encoding="utf-8")

    assert "--wheelhouse DIR" in bash_script
    assert "--offline" in bash_script
    assert "ANNOLID_WHEELHOUSE" in bash_script
    assert "ANNOLID_OFFLINE" in bash_script
    assert "build_pip_index_args()" in bash_script
    assert "PIP_INDEX_ARGS+=(--no-index)" in bash_script
    assert 'PIP_INDEX_ARGS+=(--find-links "$WHEELHOUSE")' in bash_script
    assert '"offline":' in bash_script
    assert '"wheelhouse":' in bash_script

    assert "-Wheelhouse DIR" in powershell_script
    assert "-Offline" in powershell_script
    assert "Initialize-OfflineConfig" in powershell_script
    assert "Get-PipIndexArgs" in powershell_script
    assert '"--no-index"' in powershell_script
    assert '"--find-links"' in powershell_script
    assert "offline = [bool]$script:OfflineActive" in powershell_script
    assert "wheelhouse = $script:WheelhouseDir" in powershell_script

    # Running install.sh --offline without --wheelhouse fails fast before network/git steps
    env = dict(os.environ)
    env.pop("ANNOLID_WHEELHOUSE", None)
    env.pop("ANNOLID_OFFLINE", None)
    completed = subprocess.run(
        ["/bin/bash", str(ROOT / "install.sh"), "--offline"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "requires --wheelhouse" in completed.stdout
