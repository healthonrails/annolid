from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from annolid.utils.logger import logger
from .sam3.utils import select_device


class Sam3SessionRuntime:
    """
    Session-lifecycle runtime for SAM3 predictor usage.

    This isolates predictor construction, device selection, and
    start/reset/cancel/close operations behind a narrow boundary.
    """

    def __init__(
        self,
        *,
        default_device: Optional[str],
        offload_video_to_cpu: bool,
        initialize_predictor: Callable[[torch.device], Any],
        activate_global_match_session: Callable[[Optional[str]], None],
    ) -> None:
        self.default_device = default_device
        self.offload_video_to_cpu = bool(offload_video_to_cpu)
        self._initialize_predictor = initialize_predictor
        self._activate_global_match_session = activate_global_match_session
        self.predictor: Optional[Any] = None
        self.predictor_device: Optional[torch.device] = None
        self.session_id: Optional[str] = None

    def hydrate(
        self,
        *,
        predictor: Optional[Any],
        predictor_device: Optional[torch.device],
        session_id: Optional[str],
    ) -> None:
        self.predictor = predictor
        self.predictor_device = predictor_device
        self.session_id = session_id

    def resolve_runtime_device(
        self,
        target_device: Optional[torch.device | str],
    ) -> torch.device:
        resolved = select_device(target_device or self.default_device)
        if resolved.type == "mps":
            logger.warning(
                "SAM3.1 multiplex runtime on MPS is unstable in this environment; "
                "falling back to CPU."
            )
            resolved = torch.device("cpu")
        return resolved

    def ensure_predictor(
        self,
        resolved_device: torch.device,
    ) -> None:
        if self.predictor is None or self.predictor_device != resolved_device:
            self.predictor = self._initialize_predictor(resolved_device)
            self.predictor_device = resolved_device

    def start_session(
        self,
        *,
        target_device: Optional[torch.device | str],
        resource_path: str,
        session_id: Optional[str] = None,
    ) -> str:
        resolved_device = self.resolve_runtime_device(target_device)
        self.ensure_predictor(resolved_device)
        assert self.predictor is not None
        try:
            session = self.predictor.start_session(
                resource_path=resource_path,
                session_id=session_id,
                offload_video_to_cpu=self.offload_video_to_cpu,
            )
        except TypeError:
            session = self.predictor.start_session(
                resource_path=resource_path,
                offload_video_to_cpu=self.offload_video_to_cpu,
            )
        self.session_id = str(session["session_id"])
        self._activate_global_match_session(self.session_id)
        return self.session_id

    def close_session(self) -> None:
        if self.predictor is not None and self.session_id:
            try:
                self.predictor.close_session(self.session_id)
            except Exception:
                pass
        self.session_id = None
        self._activate_global_match_session(None)

    def reset_session_state(self) -> None:
        try:
            if self.predictor is not None and self.session_id:
                self.predictor.reset_session(self.session_id)
        except Exception as exc:
            logger.debug("Unable to reset SAM3 session state: %s", exc)
        self._activate_global_match_session(self.session_id)

    def cancel_propagation(self) -> None:
        if not self.session_id or self.predictor is None:
            return
        try:
            self.predictor.cancel_propagation(session_id=self.session_id)
        except Exception:
            logger.debug("SAM3 cancel_propagation request failed.", exc_info=True)
