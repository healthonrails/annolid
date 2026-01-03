from __future__ import annotations

import textwrap


class ImageEditingError(RuntimeError):
    """Base exception for image editing failures."""


class InvalidRequestError(ImageEditingError):
    """Raised when an ImageEditRequest is inconsistent or missing required data."""


class BackendNotAvailableError(ImageEditingError):
    """Raised when an optional backend dependency is missing."""


class ExternalCommandError(ImageEditingError):
    """Raised when an external command (e.g., sd-cli) fails."""

    def __init__(
        self,
        message: str,
        *,
        command: list[str] | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
        returncode: int | None = None,
    ) -> None:
        super().__init__(message)
        self.command = command
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode

    @staticmethod
    def _tail(text: str, *, max_lines: int = 40, max_chars: int = 6000) -> str:
        text = (text or "").strip("\n")
        if not text:
            return ""
        lines = text.splitlines()
        if len(lines) > max_lines:
            lines = ["…"] + lines[-max_lines:]
        out = "\n".join(lines)
        if len(out) > max_chars:
            out = "…" + out[-max_chars:]
        return out

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.returncode is not None:
            parts.append(f"returncode: {self.returncode}")
        if self.command:
            parts.append("command:\n" + textwrap.indent(" ".join(self.command), "  "))
        stderr_tail = self._tail(self.stderr or "")
        if stderr_tail:
            parts.append("stderr (tail):\n" + textwrap.indent(stderr_tail, "  "))
        stdout_tail = self._tail(self.stdout or "")
        if stdout_tail:
            parts.append("stdout (tail):\n" + textwrap.indent(stdout_tail, "  "))
        return "\n".join(parts)
