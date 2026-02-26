from __future__ import annotations

import hashlib
import os
import socket
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from annolid.core.agent.config import load_config


@dataclass(frozen=True)
class AutoUpdatePolicy:
    enabled: bool = False
    channel: str = "stable"
    interval_seconds: int = 24 * 3600
    jitter_seconds: int = 15 * 60
    timeout_s: float = 4.0
    require_signature: bool = False

    @classmethod
    def from_env(cls) -> "AutoUpdatePolicy":
        enabled_raw = str(os.getenv("ANNOLID_AUTO_UPDATE_ENABLED", "0")).strip().lower()
        enabled = enabled_raw in {"1", "true", "yes", "on"}
        channel = (
            str(os.getenv("ANNOLID_AUTO_UPDATE_CHANNEL", "stable")).strip().lower()
        )
        if channel not in {"stable", "beta", "dev"}:
            channel = "stable"
        try:
            interval_seconds = max(
                300,
                int(
                    str(
                        os.getenv("ANNOLID_AUTO_UPDATE_INTERVAL_SECONDS", "86400")
                    ).strip()
                ),
            )
        except Exception:
            interval_seconds = 86400
        try:
            jitter_seconds = max(
                0,
                int(
                    str(os.getenv("ANNOLID_AUTO_UPDATE_JITTER_SECONDS", "900")).strip()
                ),
            )
        except Exception:
            jitter_seconds = 900
        try:
            timeout_s = max(
                1.0,
                float(str(os.getenv("ANNOLID_AUTO_UPDATE_TIMEOUT_S", "4.0")).strip()),
            )
        except Exception:
            timeout_s = 4.0
        require_sig_raw = (
            str(os.getenv("ANNOLID_AUTO_UPDATE_REQUIRE_SIGNATURE", "0")).strip().lower()
        )
        require_signature = require_sig_raw in {"1", "true", "yes", "on"}
        return cls(
            enabled=enabled,
            channel=channel,
            interval_seconds=interval_seconds,
            jitter_seconds=jitter_seconds,
            timeout_s=timeout_s,
            require_signature=require_signature,
        )

    @classmethod
    def from_config_and_env(cls) -> "AutoUpdatePolicy":
        cfg_auto = None
        try:
            cfg = load_config()
            cfg_auto = getattr(getattr(cfg, "update", None), "auto", None)
        except Exception:
            cfg_auto = None

        base = cls(
            enabled=bool(getattr(cfg_auto, "enabled", False)),
            channel=str(getattr(cfg_auto, "channel", "stable") or "stable")
            .strip()
            .lower(),
            interval_seconds=max(
                300, int(getattr(cfg_auto, "interval_seconds", 24 * 3600) or 24 * 3600)
            ),
            jitter_seconds=max(
                0, int(getattr(cfg_auto, "jitter_seconds", 15 * 60) or 15 * 60)
            ),
            timeout_s=max(1.0, float(getattr(cfg_auto, "timeout_s", 4.0) or 4.0)),
            require_signature=bool(getattr(cfg_auto, "require_signature", False)),
        )
        if base.channel not in {"stable", "beta", "dev"}:
            base = cls(
                enabled=base.enabled,
                channel="stable",
                interval_seconds=base.interval_seconds,
                jitter_seconds=base.jitter_seconds,
                timeout_s=base.timeout_s,
                require_signature=base.require_signature,
            )
        env = cls.from_env()
        return cls(
            enabled=env.enabled
            if os.getenv("ANNOLID_AUTO_UPDATE_ENABLED") is not None
            else base.enabled,
            channel=env.channel
            if os.getenv("ANNOLID_AUTO_UPDATE_CHANNEL") is not None
            else base.channel,
            interval_seconds=env.interval_seconds
            if os.getenv("ANNOLID_AUTO_UPDATE_INTERVAL_SECONDS") is not None
            else base.interval_seconds,
            jitter_seconds=env.jitter_seconds
            if os.getenv("ANNOLID_AUTO_UPDATE_JITTER_SECONDS") is not None
            else base.jitter_seconds,
            timeout_s=env.timeout_s
            if os.getenv("ANNOLID_AUTO_UPDATE_TIMEOUT_S") is not None
            else base.timeout_s,
            require_signature=env.require_signature
            if os.getenv("ANNOLID_AUTO_UPDATE_REQUIRE_SIGNATURE") is not None
            else base.require_signature,
        )

    @staticmethod
    def _host_seed() -> int:
        hostname = socket.gethostname()
        digest = hashlib.sha256(hostname.encode("utf-8")).hexdigest()[:8]
        return int(digest, 16)

    def compute_jitter(self) -> int:
        if self.jitter_seconds <= 0:
            return 0
        seed = self._host_seed()
        return int(seed % int(self.jitter_seconds + 1))

    def next_due_epoch_s(self, *, last_check_epoch_s: float) -> float:
        base = float(last_check_epoch_s) + float(self.interval_seconds)
        return base + float(self.compute_jitter())

    def is_due(
        self,
        *,
        last_check_epoch_s: Optional[float],
        now_epoch_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        now = float(now_epoch_s if now_epoch_s is not None else time.time())
        if last_check_epoch_s is None:
            return {
                "due": True,
                "reason": "first_run",
                "next_due_epoch_s": now + float(self.interval_seconds),
            }
        next_due = self.next_due_epoch_s(last_check_epoch_s=float(last_check_epoch_s))
        return {
            "due": bool(now >= next_due),
            "reason": "interval_elapsed" if now >= next_due else "not_due",
            "next_due_epoch_s": next_due,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "channel": self.channel,
            "interval_seconds": int(self.interval_seconds),
            "jitter_seconds": int(self.jitter_seconds),
            "timeout_s": float(self.timeout_s),
            "require_signature": bool(self.require_signature),
        }
