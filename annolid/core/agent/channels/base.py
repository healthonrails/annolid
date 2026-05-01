from annolid.utils.logger import logger
from abc import ABC, abstractmethod
import re
from pathlib import Path
from typing import Any, Optional

from annolid.core.agent.bus import InboundMessage, MessageBus, OutboundMessage


class BaseChannel(ABC):
    """Base channel interface for bus-driven channel adapters."""

    name: str = "base"
    _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp", ".m4v"}
    _MAX_MEDIA_ITEMS = 16
    _MAX_VIDEOS_PER_MESSAGE = 4
    _MAX_MEDIA_REF_CHARS = 4096
    _MAX_IMAGE_BYTES = 8 * 1024 * 1024
    _MAX_VIDEO_BYTES = 20 * 1024 * 1024
    _ALLOWED_MIME_PREFIXES = (
        "image/",
        "video/",
        "audio/",
        "text/",
    )
    _ALLOWED_MIME_EXACT = {
        "application/pdf",
        "application/json",
        "application/octet-stream",
    }
    _DATA_URL_MIME_RE = re.compile(r"^data:([^;]+);base64,", re.DOTALL)

    def __init__(self, config: Any, bus: MessageBus):
        self.config = config
        self.bus = bus
        self._running = False

    @abstractmethod
    async def start(self) -> None:
        """Start receiving channel events."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop channel resources."""

    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        """Send an outbound bus message via channel."""

    def is_allowed(self, sender_id: str) -> bool:
        if isinstance(self.config, dict):
            allow_list = self.config.get("allow_from")
        else:
            allow_list = getattr(self.config, "allow_from", None)
        if not allow_list:
            return True
        sender = str(sender_id).lower()
        for allowed in allow_list:
            if str(allowed).lower() in sender:
                return True
        return False

    async def _handle_message(
        self,
        *,
        sender_id: str,
        chat_id: str,
        content: str,
        media: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        if not self.is_allowed(sender_id):
            logger.warning(
                "Channel %s: Message from %s blocked by allow_from list.",
                self.name,
                sender_id,
            )
            return False
        normalized_media, normalized_content, normalized_meta = (
            self._normalize_media_and_content(
                media=media,
                content=content,
                metadata=metadata,
            )
        )
        normalized_meta = self._normalize_session_metadata(
            sender_id=sender_id,
            chat_id=chat_id,
            metadata=normalized_meta,
        )
        msg = InboundMessage(
            channel=self.name,
            sender_id=str(sender_id),
            chat_id=str(chat_id),
            content=normalized_content,
            media=normalized_media,
            metadata=normalized_meta,
        )
        await self.bus.publish_inbound(msg)
        return True

    @classmethod
    def _looks_like_video_ref(cls, ref: str, mime: str = "", kind: str = "") -> bool:
        lowered_mime = str(mime or "").strip().lower()
        if lowered_mime.startswith("video/"):
            return True
        lowered_kind = str(kind or "").strip().lower()
        if lowered_kind == "video":
            return True
        lowered_ref = str(ref or "").strip().lower()
        if lowered_ref.startswith("data:video/"):
            return True
        if lowered_ref.startswith("video:"):
            return True
        suffix = Path(lowered_ref.split("?", 1)[0].split("#", 1)[0]).suffix
        return suffix in cls._VIDEO_EXTS

    @classmethod
    def _extract_data_url_mime(cls, ref: str) -> str:
        text = str(ref or "").strip()
        if not text:
            return ""
        match = cls._DATA_URL_MIME_RE.match(text)
        if not match:
            return ""
        return str(match.group(1) or "").strip().lower()

    @classmethod
    def _estimate_data_url_bytes(cls, ref: str) -> Optional[int]:
        text = str(ref or "").strip()
        match = cls._DATA_URL_MIME_RE.match(text)
        if not match:
            return None
        prefix = match.group(0)
        b64 = text[len(prefix) :].strip()
        if not b64:
            return 0
        # Strict base64 alphabet check to avoid accepting malformed payloads.
        for ch in b64:
            if (
                ch
                not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\r\n\t "
            ):
                return None
        compact = "".join(ch for ch in b64 if ch not in "\r\n\t ")
        if len(compact) % 4 != 0:
            return None
        pad = len(compact) - len(compact.rstrip("="))
        if pad > 2:
            return None
        return ((len(compact) * 3) // 4) - pad

    @classmethod
    def _allowed_data_url_size(cls, ref: str, *, is_video: bool) -> bool:
        size = cls._estimate_data_url_bytes(ref)
        if size is None:
            return False
        max_bytes = cls._MAX_VIDEO_BYTES if is_video else cls._MAX_IMAGE_BYTES
        return int(size) <= int(max_bytes)

    @classmethod
    def _extract_media_refs_from_metadata(
        cls,
        metadata: Optional[dict[str, Any]],
    ) -> tuple[list[str], bool, int]:
        meta = metadata if isinstance(metadata, dict) else {}
        refs: list[str] = []
        has_video = False
        dropped_count = 0
        video_count = 0
        raw_items: list[Any] = []
        for key in ("media", "media_urls", "attachments", "files"):
            value = meta.get(key)
            if isinstance(value, list):
                raw_items.extend(value)
            elif isinstance(value, str) and value.strip():
                raw_items.append(value.strip())
        for item in raw_items:
            if len(refs) >= cls._MAX_MEDIA_ITEMS:
                dropped_count += 1
                continue
            ref = ""
            mime = ""
            kind = ""
            if isinstance(item, str):
                ref = item.strip()
            elif isinstance(item, dict):
                ref = str(
                    item.get("url")
                    or item.get("path")
                    or item.get("file_path")
                    or item.get("data_url")
                    or item.get("id")
                    or ""
                ).strip()
                mime = str(
                    item.get("mime")
                    or item.get("mimetype")
                    or item.get("mime_type")
                    or ""
                )
                kind = str(item.get("kind") or item.get("type") or "")
                lowered_mime = mime.strip().lower()
                if lowered_mime and not (
                    lowered_mime.startswith(cls._ALLOWED_MIME_PREFIXES)
                    or lowered_mime in cls._ALLOWED_MIME_EXACT
                ):
                    dropped_count += 1
                    continue
            if not ref:
                continue
            is_data_url = ref.startswith("data:")
            if is_data_url and not mime:
                mime = cls._extract_data_url_mime(ref)
            if is_data_url and not mime:
                dropped_count += 1
                continue

            trimmed_ref = ref if is_data_url else ref[: cls._MAX_MEDIA_REF_CHARS]
            is_video = cls._looks_like_video_ref(trimmed_ref, mime=mime, kind=kind)
            if is_data_url and not cls._allowed_data_url_size(
                trimmed_ref, is_video=is_video
            ):
                dropped_count += 1
                continue
            if is_video:
                if video_count >= cls._MAX_VIDEOS_PER_MESSAGE:
                    dropped_count += 1
                    continue
                video_count += 1
                has_video = True
            refs.append(trimmed_ref)
        return refs, has_video, dropped_count

    @classmethod
    def _normalize_media_and_content(
        cls,
        *,
        media: Optional[list[str]],
        content: str,
        metadata: Optional[dict[str, Any]],
    ) -> tuple[list[str], str, dict[str, Any]]:
        normalized_meta = dict(metadata or {})
        raw_refs = [str(item).strip() for item in (media or []) if str(item).strip()]
        refs = []
        video_count = 0
        dropped_count = 0
        for item in raw_refs:
            if len(refs) >= cls._MAX_MEDIA_ITEMS:
                dropped_count += 1
                continue
            is_data_url = item.startswith("data:")
            trimmed_ref = item if is_data_url else item[: cls._MAX_MEDIA_REF_CHARS]
            is_video = cls._looks_like_video_ref(trimmed_ref)
            if is_data_url and not cls._allowed_data_url_size(
                trimmed_ref, is_video=is_video
            ):
                dropped_count += 1
                continue
            if is_video:
                if video_count >= cls._MAX_VIDEOS_PER_MESSAGE:
                    dropped_count += 1
                    continue
                video_count += 1
            refs.append(trimmed_ref)
        has_video = any(cls._looks_like_video_ref(item) for item in refs)
        if not refs:
            extracted, extracted_has_video, extracted_dropped = (
                cls._extract_media_refs_from_metadata(normalized_meta)
            )
            dropped_count += extracted_dropped
            refs = extracted
            has_video = extracted_has_video
        if dropped_count:
            normalized_meta.setdefault("media_dropped_count", dropped_count)

        if not refs:
            # Ensure we never emit stale media metadata when no refs survived.
            normalized_meta.pop("has_media", None)
            if normalized_meta.get("media_type") == "video":
                normalized_meta.pop("media_type", None)
        else:
            normalized_meta.setdefault("has_media", True)
            if has_video:
                normalized_meta.setdefault("media_type", "video")

        text = str(content or "").strip()
        if refs and not text:
            text = "[video message]" if has_video else "[media message]"
        return refs, text, normalized_meta

    def _normalize_session_metadata(
        self,
        *,
        sender_id: str,
        chat_id: str,
        metadata: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        merged = dict(metadata or {})
        sender = str(sender_id or "")
        chat = str(chat_id or "")
        merged.setdefault("peer_id", sender or chat)
        merged.setdefault("channel_key", chat or sender)

        conversation_type = str(
            merged.get("conversation_type") or merged.get("chat_type") or ""
        ).strip()
        if conversation_type:
            merged.setdefault("conversation_type", conversation_type.lower())
        is_dm_raw = merged.get("is_dm")

        if "is_dm" not in merged:
            lowered = conversation_type.lower()
            if lowered in {"dm", "direct", "direct_message", "private"}:
                merged["is_dm"] = True
            elif lowered in {"group", "channel", "room", "thread"}:
                merged["is_dm"] = False
            else:
                merged["is_dm"] = bool(sender and chat and sender == chat)
        is_dm = bool(merged.get("is_dm"))
        if not conversation_type:
            merged["conversation_type"] = "dm" if is_dm else "channel"
        elif is_dm and conversation_type.lower() not in {
            "dm",
            "direct",
            "direct_message",
            "private",
        }:
            merged["conversation_type"] = "dm"

        if is_dm_raw is None:
            merged["is_dm"] = is_dm
        return merged

    def normalize_outbound_message(self, msg: OutboundMessage) -> OutboundMessage:
        refs, content, meta = self._normalize_media_and_content(
            media=list(msg.media or []),
            content=str(msg.content or ""),
            metadata=dict(msg.metadata or {}),
        )
        if (
            refs == list(msg.media or [])
            and content == str(msg.content or "")
            and meta == dict(msg.metadata or {})
        ):
            return msg
        return OutboundMessage(
            channel=str(msg.channel or ""),
            chat_id=str(msg.chat_id or ""),
            content=content,
            reply_to=msg.reply_to,
            media=refs,
            metadata=meta,
        )

    @property
    def is_running(self) -> bool:
        return bool(self._running)
