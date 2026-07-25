from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from annolid.core.agent.security_policy import (
    DEFAULT_SANDBOX_CONTAINER_IMAGE,
    is_digest_pinned_container_image,
)
from annolid.utils.logger import logger

from .shell import ExecTool


class SandboxedExecTool(ExecTool):
    """
    ExecTool that runs commands within a hardened Docker container.

    Host execution is disabled by default. Callers that deliberately accept the
    weaker boundary must opt in with ``allow_host_fallback=True``.
    """

    def __init__(
        self,
        timeout: int = 60,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        restrict_to_workspace: bool = True,
        container_image: str = DEFAULT_SANDBOX_CONTAINER_IMAGE,
        docker_network_none: bool = True,
        docker_drop_all_caps: bool = True,
        docker_no_new_privileges: bool = True,
        docker_run_as_host_user: bool = True,
        docker_pids_limit: int = 256,
        docker_tmpfs_tmp: bool = True,
        docker_host_mount_read_only: bool = True,
        allow_host_fallback: bool = False,
    ):
        super().__init__(
            timeout=timeout,
            working_dir=working_dir,
            deny_patterns=deny_patterns,
            allow_patterns=allow_patterns,
            restrict_to_workspace=restrict_to_workspace,
        )
        self.container_image = str(container_image or "").strip()
        self.docker_network_none = docker_network_none
        self.docker_drop_all_caps = docker_drop_all_caps
        self.docker_no_new_privileges = docker_no_new_privileges
        self.docker_run_as_host_user = docker_run_as_host_user
        self.docker_pids_limit = docker_pids_limit
        self.docker_tmpfs_tmp = docker_tmpfs_tmp
        self.docker_host_mount_read_only = docker_host_mount_read_only
        self.allow_host_fallback = allow_host_fallback
        self._has_docker: bool | None = None

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "Execute a shell command inside a hardened Docker sandbox."

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
        if not is_digest_pinned_container_image(self.container_image):
            return (
                "Error: Sandbox image must be pinned with an immutable "
                "@sha256:<64-hex-digest> reference."
            )

        # Determine if we can run in docker
        use_docker = await self._check_docker()

        if not use_docker:
            if not self.allow_host_fallback:
                logger.warning(
                    "SandboxedExecTool: Docker is unavailable; refusing host execution."
                )
                return (
                    "Error: Sandbox unavailable. Docker is required and host "
                    "execution was refused."
                )
            logger.warning(
                "SandboxedExecTool: Docker is unavailable; using explicitly "
                "enabled host fallback."
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
        if not is_digest_pinned_container_image(self.container_image):
            raise ValueError(
                "Sandbox image must be pinned with an immutable SHA-256 digest."
            )
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
            docker_cmd.extend(
                [
                    "--tmpfs",
                    "/tmp:rw,noexec,nosuid,nodev,size=128m",  # noqa: S108
                ]
            )
        if (
            self.docker_run_as_host_user
            and hasattr(os, "getuid")
            and hasattr(os, "getgid")
        ):
            docker_cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        docker_cmd.extend([self.container_image, "bash", "-c", command])
        return docker_cmd


__all__ = ["DEFAULT_SANDBOX_CONTAINER_IMAGE", "SandboxedExecTool"]
