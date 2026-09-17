from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "desktop_shortcuts", ROOT / "scripts/create_desktop_shortcuts.py"
)
shortcuts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shortcuts)


def installation(tmp_path, windows=False):
    root = tmp_path / "Annolid lab's 小鼠 $data `safe` %test%"
    root.mkdir()
    venv = root / "custom env"
    entry = venv / ("Scripts/annolid.exe" if windows else "bin/annolid")
    entry.parent.mkdir(parents=True)
    entry.write_text('#!/bin/bash\nprintf "%s\\n" "$PWD" "$@"\n')
    entry.chmod(0o755)
    (entry.parent / ("Activate.ps1" if windows else "activate")).touch()
    (root / ("install.ps1" if windows else "install.sh")).write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@"\n'
    )
    return root, venv


@pytest.mark.parametrize("platform,suffix", [("darwin", ".command"), ("linux", ".sh")])
def test_shortcuts_launch_and_update_with_literal_paths(tmp_path, platform, suffix):
    root, venv = installation(tmp_path)
    paths = shortcuts.create_shortcuts(
        root,
        venv,
        profile="minimal",
        extras="gui,audio",
        no_gpu=True,
        platform=platform,
    )
    assert all(path.exists() for path in paths)
    launch = subprocess.run(
        [str(root / ("Launch Annolid" + suffix)), "video with spaces.mp4"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert launch.returncode == 0
    assert launch.stdout.splitlines() == [str(root), "video with spaces.mp4"]
    update = subprocess.run(
        [str(root / ("Update Annolid" + suffix))],
        input="\n",
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert update.returncode == 0
    assert update.stdout.splitlines()[1:] == [
        "--install-dir",
        str(root),
        "--venv-dir",
        str(venv),
        "--profile",
        "minimal",
        "--no-interactive",
        "--extras",
        "gui,audio",
        "--no-gpu",
    ]
    assert not (tmp_path / "safe").exists()


def test_shortcut_preserves_failed_process_status(tmp_path):
    root, venv = installation(tmp_path)
    (venv / "bin/annolid").write_text("#!/bin/bash\nexit 7\n")
    shortcuts.create_shortcuts(root, venv, profile="gui", extras="", platform="darwin")
    result = subprocess.run(
        [str(root / "Launch Annolid.command")],
        input="\n",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 7
    assert "failed" in result.stdout


def test_windows_shortcuts_keep_user_paths_out_of_cmd(tmp_path):
    root, venv = installation(tmp_path, windows=True)
    shortcuts.create_shortcuts(
        root,
        venv,
        profile="workstation",
        extras="gui,sam3",
        no_gpu=True,
        platform="win32",
    )
    cmd = (root / "Launch Annolid.cmd").read_text()
    assert str(root) not in cmd
    assert '"%~dp0Launch Annolid.ps1"' in cmd
    assert (root / "Update Annolid.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    update = (root / "Update Annolid.ps1").read_text()
    assert "lab''s" in update
    assert "-Profile 'workstation' -Extras 'gui,sam3' -NoInteractive -NoGpu" in update


def test_shortcuts_require_successful_installation(tmp_path):
    with pytest.raises(FileNotFoundError):
        shortcuts.create_shortcuts(
            tmp_path, tmp_path / "missing", profile="gui", extras=""
        )
    assert not list(tmp_path.iterdir())


def run_bash_functions(tmp_path, body, env=None):
    script = (ROOT / "install.sh").read_text().rsplit('main "$@"', 1)[0]
    harness = tmp_path / "harness.sh"
    harness.write_text(script + "\n" + body)
    return subprocess.run(
        ["bash", str(harness)], cwd=tmp_path, capture_output=True, text=True, env=env
    )


def test_rerun_preserves_existing_environment(tmp_path):
    venv = tmp_path / "existing env"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").touch()
    (venv / "bin/activate").touch()
    (venv / "bin/python").write_text("#!/bin/bash\nexit 0\n")
    (venv / "bin/python").chmod(0o755)
    marker = venv / "keep-me"
    marker.write_text("installed packages")
    env = dict(os.environ, TEST_VENV=str(venv))
    result = run_bash_functions(
        tmp_path,
        'VENV_DIR="$TEST_VENV"\nUSE_UV=true\nUV_CMD=/does/not/exist\ncreate_venv\n',
        env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Reusing" in result.stdout
    assert marker.read_text() == "installed packages"


def test_invalid_environment_is_not_overwritten(tmp_path):
    venv = tmp_path / "broken env"
    venv.mkdir()
    marker = venv / "keep-me"
    marker.touch()
    result = run_bash_functions(
        tmp_path,
        'VENV_DIR="$TEST_VENV"\ncreate_venv\n',
        dict(os.environ, TEST_VENV=str(venv)),
    )
    assert result.returncode != 0
    assert "invalid" in result.stdout
    assert marker.exists()


def test_existing_non_repository_is_never_deleted(tmp_path):
    target = tmp_path / "user data"
    target.mkdir()
    (target / "annotations.json").write_text("{}")
    result = run_bash_functions(
        tmp_path,
        'INSTALL_DIR="$TEST_INSTALL"\nNO_INTERACTIVE=true\nclone_repo\n',
        dict(os.environ, TEST_INSTALL=str(target)),
    )
    assert result.returncode != 0
    assert (target / "annotations.json").read_text() == "{}"


@pytest.mark.parametrize("dirty", [False, True])
def test_failed_or_dirty_git_update_stops_installer(tmp_path, dirty):
    target = tmp_path / "repo"
    (target / ".git").mkdir(parents=True)
    (target / "annolid").mkdir()
    (target / "annolid/__init__.py").touch()
    (target / "pyproject.toml").touch()
    body = """
git() {
    if [[ "$1" == status ]]; then
        if [[ "$TEST_DIRTY" == yes ]]; then echo ' M tracked.py'; fi
        return 0
    fi
    echo "git $*" >&2
    return 1
}
INSTALL_DIR="$TEST_INSTALL"
clone_repo
echo SHOULD_NOT_CONTINUE
"""
    result = run_bash_functions(
        tmp_path,
        body,
        dict(os.environ, TEST_INSTALL=str(target), TEST_DIRTY="yes" if dirty else "no"),
    )
    assert result.returncode != 0
    assert "SHOULD_NOT_CONTINUE" not in result.stdout
    if dirty:
        assert "Local tracked changes" in result.stdout
        assert "pull" not in result.stderr
    else:
        assert "pull --ff-only --recurse-submodules" in result.stderr
        assert "Update failed" in result.stdout


def test_noninteractive_installs_do_not_launch_gui():
    bash = (ROOT / "install.sh").read_text()
    powershell = (ROOT / "install.ps1").read_text()
    assert '[[ "$NO_INTERACTIVE" == false ]] && prompt_yes_no "Launch' in bash
    assert 'if (!$NoInteractive -and (Prompt-YesNo "Launch' in powershell
