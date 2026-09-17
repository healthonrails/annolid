"""Create launch/update shortcuts for a source installation (stdlib only).

Shortcuts stay in the install directory so multiple installations do not overwrite
one another's desktop entries. No shell activation or global PATH edits are needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys


def _write(path: Path, text: str, *, executable: bool = False) -> Path:
    # Replace atomically so interrupted regeneration leaves a working shortcut.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        text, encoding="utf-8-sig" if path.suffix == ".ps1" else "utf-8"
    )
    if executable:
        temporary.chmod(0o755)
    temporary.replace(path)
    return path


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _desktop_quote(value: str) -> str:
    # Desktop Entry Exec has its own quoting rules, separate from shell syntax.
    value = value.replace("%", "%%")
    for character in ("\\", '"', "`", "$"):
        value = value.replace(character, "\\" + character)
    return '"' + value.replace("\\", "\\\\") + '"'


def create_shortcuts(
    install_dir: Path,
    venv_path: Path,
    *,
    profile: str,
    extras: str,
    no_gpu: bool = False,
    platform: str = sys.platform,
) -> list[Path]:
    install_dir = install_dir.resolve()
    venv_path = venv_path.resolve()
    if any(c in str(install_dir) + str(venv_path) + extras for c in "\r\n\0"):
        raise ValueError(
            "Shortcut paths and extras must not contain line breaks or NUL"
        )
    windows = platform == "win32"
    entry = venv_path / ("Scripts/annolid.exe" if windows else "bin/annolid")
    installer = install_dir / ("install.ps1" if windows else "install.sh")
    if not entry.is_file() or not installer.is_file():
        raise FileNotFoundError(
            "Install Annolid successfully before creating shortcuts"
        )

    paths: list[Path] = []
    if windows:
        # Keep user-controlled values in PowerShell literals, never cmd syntax.
        launch = (
            ". "
            + _ps_quote(str(venv_path / "Scripts/Activate.ps1"))
            + "\n"
            + "& "
            + _ps_quote(str(entry))
            + " @args\nexit $LASTEXITCODE\n"
        )
        update = "& " + _ps_quote(str(installer))
        for flag, value in (
            ("InstallDir", str(install_dir)),
            ("VenvDir", str(venv_path)),
            ("Profile", profile),
            ("Extras", extras),
        ):
            update += f" -{flag} {_ps_quote(value)}"
        update += " -NoInteractive" + (" -NoGpu" if no_gpu else "") + "\n"
        for name, body in (("Launch Annolid", launch), ("Update Annolid", update)):
            ps_path = install_dir / (name + ".ps1")
            preamble = (
                "$ErrorActionPreference = 'Stop'\nSet-Location -LiteralPath "
                + _ps_quote(str(install_dir))
                + "\n"
            )
            if name.startswith("Update"):
                preamble += "Write-Host 'Close Annolid before updating.'\nRead-Host 'Press Enter to update, or Ctrl+C to cancel'\n"
            paths.append(_write(ps_path, preamble + body))
            paths.append(
                _write(
                    install_dir / (name + ".cmd"),
                    "@echo off\r\nsetlocal DisableDelayedExpansion\r\n"
                    f'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0{name}.ps1"\r\n'
                    "if errorlevel 1 (echo Annolid failed. Review the error above. & pause & exit /b 1)\r\n",
                )
            )
        return paths

    launch_command = (
        "source "
        + shlex.quote(str(venv_path / "bin/activate"))
        + " || exit 1\n"
        + shlex.quote(str(entry))
        + ' "$@"'
    )
    arguments = [
        "bash",
        str(installer),
        "--install-dir",
        str(install_dir),
        "--venv-dir",
        str(venv_path),
        "--profile",
        profile,
        "--no-interactive",
    ]
    if extras:
        arguments += ["--extras", extras]
    if no_gpu:
        arguments.append("--no-gpu")
    suffix = ".command" if platform == "darwin" else ".sh"
    for name, command in (
        ("Launch Annolid", launch_command),
        ("Update Annolid", shlex.join(arguments)),
    ):
        preamble = (
            "#!/usr/bin/env bash\ncd -- "
            + shlex.quote(str(install_dir))
            + " || exit 1\n"
        )
        if name.startswith("Update"):
            preamble += "echo 'Close Annolid before updating.'\nread -r -p 'Press Enter to update, or Ctrl+C to cancel: ' || exit 1\n"
        body = (
            preamble + command + '\nstatus=$?\nif [ "$status" -ne 0 ]; then\n'
            "  echo 'Annolid failed. Review the error above.'\n"
            "  read -r -p 'Press Enter to close: '\nfi\nexit \"$status\"\n"
        )
        script = _write(install_dir / (name + suffix), body, executable=True)
        paths.append(script)
        if platform.startswith("linux"):
            paths.append(
                _write(
                    install_dir / (name + ".desktop"),
                    "[Desktop Entry]\nType=Application\n"
                    + f"Name={name}\n"
                    + f"Exec={_desktop_quote(str(script))}\nTerminal=true\n"
                    + "Icon=applications-science\nCategories=Education;Science;\n",
                    executable=True,
                )
            )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-dir", type=Path, required=True)
    parser.add_argument("--venv-path", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=("minimal", "gui", "workstation", "full"), required=True
    )
    parser.add_argument("--extras", default="")
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()
    for path in create_shortcuts(**vars(args)):
        print(path)


if __name__ == "__main__":
    main()
