from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from annolid.utils.logger import logger
from .shell import ExecTool


class SandboxedExecTool(ExecTool):
    """
    ExecTool that attempts to run commands within a Docker container sandbox.
    Falls back to normal ExecTool if Docker is not available.
    """

    def __init__(
        self,
        timeout: int = 60,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        restrict_to_workspace: bool = False,
        container_image: str = "ubuntu:24.04",
        docker_network_none: bool = True,
        docker_drop_all_caps: bool = True,
        docker_no_new_privileges: bool = True,
        docker_run_as_host_user: bool = True,
        docker_pids_limit: int = 256,
        docker_tmpfs_tmp: bool = True,
        docker_host_mount_read_only: bool = True,
    ):
        super().__init__(
            timeout=timeout,
            working_dir=working_dir,
            deny_patterns=deny_patterns,
            allow_patterns=allow_patterns,
            restrict_to_workspace=restrict_to_workspace,
        )
        self.container_image = container_image
        self.docker_network_none = docker_network_none
        self.docker_drop_all_caps = docker_drop_all_caps
        self.docker_no_new_privileges = docker_no_new_privileges
        self.docker_run_as_host_user = docker_run_as_host_user
        self.docker_pids_limit = docker_pids_limit
        self.docker_tmpfs_tmp = docker_tmpfs_tmp
        self.docker_host_mount_read_only = docker_host_mount_read_only
        self._has_docker: bool | None = None

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command (sandboxed if possible) and return stdout/stderr."
        )

    async def _check_docker(self) -> bool:
        if self._has_docker is not None:
            return self._has_docker
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            self._has_docker = proc.returncode == 0
        except Exception:
            self._has_docker = False
        return self._has_docker

    async def execute(
        self, command: str, working_dir: str | None = None, **kwargs: Any
    ) -> str:
        cwd = working_dir or self.working_dir or os.getcwd()
        guard_error = self._guard_command(command, cwd)
        if guard_error:
            return guard_error

        # Determine if we can run in docker
        use_docker = await self._check_docker()

        if not use_docker:
            # Fallback to standard ExecTool execution
            logger.info(
                "SandboxedExecTool: Docker not found. Falling back to host execution."
            )
            return await super().execute(command, working_dir, **kwargs)

        cwd_path = Path(cwd).resolve()
        docker_cmd = self._build_docker_command(command=command, cwd_path=cwd_path)

        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                return f"Error: Command timed out after {self.timeout} seconds"

            parts: list[str] = []
            if stdout:
                parts.append(stdout.decode("utf-8", errors="replace"))
            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    parts.append(f"STDERR:\n{stderr_text}")
            if proc.returncode != 0:
                parts.append(f"\nExit code: {proc.returncode}")
            result = "\n".join(parts) if parts else "(no output)"
            if len(result) > 10000:
                result = (
                    result[:10000]
                    + f"\n... (truncated, {len(result) - 10000} more chars)"
                )
            return result
        except Exception as exc:
            return f"Error executing sandboxed command: {exc}"

    def _build_docker_command(self, *, command: str, cwd_path: Path) -> list[str]:
        mount_spec = f"{cwd_path}:{cwd_path}"
        if self.docker_host_mount_read_only:
            mount_spec += ":ro"
        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            mount_spec,
            "-w",
            str(cwd_path),
        ]
        if self.docker_network_none:
            docker_cmd.extend(["--network", "none"])
        if self.docker_drop_all_caps:
            docker_cmd.extend(["--cap-drop", "ALL"])
        if self.docker_no_new_privileges:
            docker_cmd.extend(["--security-opt", "no-new-privileges"])
        if self.docker_pids_limit > 0:
            docker_cmd.extend(["--pids-limit", str(self.docker_pids_limit)])
        if self.docker_tmpfs_tmp:
            docker_cmd.extend(["--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=128m"])
        if (
            self.docker_run_as_host_user
            and hasattr(os, "getuid")
            and hasattr(os, "getgid")
        ):
            docker_cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        docker_cmd.extend([self.container_image, "bash", "-c", command])
        return docker_cmd


__all__ = ["SandboxedExecTool"]
