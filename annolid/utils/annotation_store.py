import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Union

from annolid.utils.logger import logger


class AnnotationStoreError(Exception):
    """Raised when annotation store operations fail."""


_ORIGINAL_JSON_LOAD = json.load


class AnnotationStore:
    """Append-only store for per-frame annotations with lazy indexing."""

    STORE_SUFFIX = "_annotations.ndjson"
    STUB_VERSION = 1
    _CACHE: Dict[Path, Dict[str, Any]] = {}

    def __init__(self, store_path: Path):
        self.store_path = store_path

    def _discover_manual_seed_frames(self) -> Set[int]:
        seeds: Set[int] = set()
        folder = self.store_path.parent
        if not folder.exists() or not folder.is_dir():
            return seeds

        stem_prefix = f"{folder.name}_".lower()
        for png_path in folder.glob("*.png"):
            stem = png_path.stem.lower()
            if not stem.startswith(stem_prefix):
                continue
            suffix = stem[len(stem_prefix) :]
            if len(suffix) != 9 or not suffix.isdigit():
                continue
            if png_path.with_suffix(".json").exists():
                seeds.add(int(suffix))
        return seeds

    def _legacy_frame_for_index(self, record_index: int) -> Optional[int]:
        try:
            target_index = int(record_index)
        except (TypeError, ValueError):
            return None
        if target_index < 0:
            return None

        manual_seeds = self._discover_manual_seed_frames()
        if not manual_seeds:
            return target_index

        # Legacy stores without explicit frame metadata only contain predicted
        # frames. Map row N onto the Nth non-manual frame so manual seed frames
        # are never mistaken for store-backed predictions.
        seen = -1
        frame = -1
        while seen < target_index:
            frame += 1
            if frame in manual_seeds:
                continue
            seen += 1
        return frame

    @staticmethod
    def _explicit_frame_for_record(record: Dict[str, Any]) -> Optional[int]:
        frame_value = record.get("frame")
        try:
            return int(frame_value)
        except (TypeError, ValueError):
            return None

    def _frame_key_for_record(
        self, record: Dict[str, Any], fallback_index: Optional[int] = None
    ) -> Optional[int]:
        explicit_frame = self._explicit_frame_for_record(record)
        if explicit_frame is not None:
            return explicit_frame

        image_path = record.get("imagePath")
        if image_path:
            inferred = AnnotationStore.frame_number_from_path(image_path)
            if inferred is not None:
                return inferred

        if fallback_index is None:
            return None
        return self._legacy_frame_for_index(fallback_index)

    def _ensure_explicit_frame_metadata(self) -> None:
        if not self.store_path.exists():
            return

        try:
            raw_lines = self.store_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise AnnotationStoreError(
                f"Failed to read annotation store {self.store_path}: {exc}"
            ) from exc

        rewritten_lines = []
        needs_rewrite = False
        record_index = 0
        for raw_line in raw_lines:
            stripped = raw_line.strip()
            if not stripped:
                rewritten_lines.append(raw_line)
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                rewritten_lines.append(raw_line)
                continue

            explicit_frame = self._explicit_frame_for_record(record)
            if explicit_frame is None:
                inferred_frame = self._frame_key_for_record(record, record_index)
                if inferred_frame is None:
                    raise AnnotationStoreError(
                        f"Legacy annotation store row {record_index} in {self.store_path} is missing frame metadata and cannot be inferred."
                    )
                record = dict(record)
                record["frame"] = int(inferred_frame)
                needs_rewrite = True
            rewritten_lines.append(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            )
            record_index += 1

        if not needs_rewrite:
            return

        temp_path = self.store_path.with_suffix(self.store_path.suffix + ".tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as fh:
                for line in rewritten_lines:
                    fh.write(line)
                    fh.write("\n")
            temp_path.replace(self.store_path)
        except OSError as exc:
            raise AnnotationStoreError(
                f"Failed to migrate annotation store {self.store_path}: {exc}"
            ) from exc

        AnnotationStore._CACHE.pop(self.store_path, None)

    @classmethod
    def for_frame_path(
        cls, frame_path: Union[str, Path], store_name: Optional[str] = None
    ) -> "AnnotationStore":
        frame_path = Path(frame_path)
        store_path = cls._derive_store_path(frame_path, store_name)
        return cls(store_path)

    @staticmethod
    def _derive_store_path(frame_path: Path, store_name: Optional[str] = None) -> Path:
        directory = frame_path.parent
        if store_name:
            return directory / store_name
        return directory / f"{directory.name}{AnnotationStore.STORE_SUFFIX}"

    @staticmethod
    def frame_number_from_path(frame_path: Union[str, Path]) -> Optional[int]:
        frame_path = Path(frame_path)
        parts = frame_path.stem.split("_")
        if not parts:
            return None
        candidate = parts[-1]
        if candidate.isdigit():
            return int(candidate)
        return None

    def append_frame(
        self,
        record: Dict[str, Any],
    ) -> None:
        """Append or update a frame record."""
        frame = record.get("frame")
        if frame is None:
            raise AnnotationStoreError("Record must include a 'frame' key.")

        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with self.store_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")))
            fh.write("\n")

        # Keep the cache in sync without re-reading the entire file (O(1) append).
        cached = AnnotationStore._CACHE.get(self.store_path)
        if cached:
            try:
                frame_key = int(frame)
            except (TypeError, ValueError):
                frame_key = frame
            records = dict(cached["records"])
            records[frame_key] = record
            try:
                stat = self.store_path.stat()
            except OSError:
                AnnotationStore._CACHE.pop(self.store_path, None)
            else:
                AnnotationStore._CACHE[self.store_path] = {
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                    "records": records,
                }

    def update_frame(
        self,
        frame: int,
        record: Dict[str, Any],
    ) -> None:
        """Replace the stored record for a frame, preserving line order."""
        if frame is None:
            raise AnnotationStoreError("Frame is required to update a record.")
        if not self.store_path.exists():
            raise AnnotationStoreError(f"Annotation store not found: {self.store_path}")

        try:
            frame_key = int(frame)
        except (TypeError, ValueError) as exc:
            raise AnnotationStoreError(f"Invalid frame number: {frame!r}") from exc

        lines_to_keep = []
        replaced = False
        try:
            self._ensure_explicit_frame_metadata()
            with self.store_path.open("r", encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.rstrip("\n")
                    stripped = line.strip()
                    if not stripped:
                        lines_to_keep.append(
                            raw_line if raw_line.endswith("\n") else f"{raw_line}\n"
                        )
                        continue

                    try:
                        payload = json.loads(stripped)
                    except json.JSONDecodeError:
                        lines_to_keep.append(
                            raw_line if raw_line.endswith("\n") else f"{raw_line}\n"
                        )
                        continue

                    payload_frame = self._explicit_frame_for_record(payload)
                    if payload_frame is None:
                        raise AnnotationStoreError(
                            f"Frame metadata missing while updating {self.store_path}."
                        )

                    if payload_frame == frame_key:
                        lines_to_keep.append(
                            json.dumps(
                                record, ensure_ascii=False, separators=(",", ":")
                            )
                            + "\n"
                        )
                        replaced = True
                    else:
                        lines_to_keep.append(
                            raw_line if raw_line.endswith("\n") else f"{raw_line}\n"
                        )
        except OSError as exc:
            raise AnnotationStoreError(
                f"Failed to read annotation store {self.store_path}: {exc}"
            ) from exc

        if not replaced:
            raise AnnotationStoreError(
                f"Frame {frame_key} not present in store {self.store_path}"
            )

        temp_path = self.store_path.with_suffix(self.store_path.suffix + ".tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as fh:
                fh.writelines(lines_to_keep)
            temp_path.replace(self.store_path)
        except OSError as exc:
            raise AnnotationStoreError(
                f"Failed to rewrite annotation store {self.store_path}: {exc}"
            ) from exc

        AnnotationStore._CACHE.pop(self.store_path, None)
        self._load_records(force_reload=True)

    def get_frame(self, frame: int) -> Optional[Dict[str, Any]]:
        """Return the latest record for a frame if present."""
        records = self._load_records()
        try:
            frame_key = int(frame)
        except (TypeError, ValueError):
            frame_key = frame
        return records.get(frame_key)

    def iter_frames(self) -> Iterable[int]:
        records = self._load_records()
        return records.keys()

    def _load_records(self, force_reload: bool = False) -> Dict[int, Dict[str, Any]]:
        if not self.store_path.exists():
            return {}

        self._ensure_explicit_frame_metadata()
        stat = self.store_path.stat()
        cached = AnnotationStore._CACHE.get(self.store_path)
        if (
            not force_reload
            and cached
            and cached["mtime"] == stat.st_mtime
            and cached["size"] == stat.st_size
        ):
            return cached["records"]

        records: Dict[int, Dict[str, Any]] = {}
        with self.store_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping invalid annotation store line in %s", self.store_path
                    )
                    continue
                frame = self._explicit_frame_for_record(data)
                if frame is None:
                    logger.warning(
                        "Skipping annotation store row without explicit frame in %s",
                        self.store_path,
                    )
                    continue
                records[int(frame)] = data

        AnnotationStore._CACHE[self.store_path] = {
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "records": records,
        }
        return records

    def remove_frames_after(
        self,
        frame_threshold: int,
        protected_frames: Optional[Iterable[int]] = None,
    ) -> int:
        """Remove frames greater than the threshold from the store unless protected.

        Args:
            frame_threshold: Highest frame index to retain. Frames with a value
                greater than this number are removed.
            protected_frames: Optional iterable of frame numbers that should never
                be removed, even if they are greater than ``frame_threshold``.

        Returns:
            The number of store records that were removed.
        """
        if not self.store_path.exists():
            return 0

        try:
            protected: Set[int] = {
                int(frame) for frame in (protected_frames or []) if frame is not None
            }
        except Exception:
            protected = set()

        lines_to_keep = []
        removed = 0

        try:
            self._ensure_explicit_frame_metadata()
            with self.store_path.open("r", encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.rstrip("\n")
                    stripped = line.strip()
                    if not stripped:
                        # Preserve blank lines to avoid altering formatting unexpectedly.
                        lines_to_keep.append(raw_line)
                        continue

                    try:
                        payload = json.loads(stripped)
                    except json.JSONDecodeError:
                        # If the line is malformed, keep it to avoid data loss.
                        lines_to_keep.append(raw_line)
                        continue

                    frame_value = payload.get("frame")
                    try:
                        frame_number = int(frame_value)
                    except (TypeError, ValueError):
                        lines_to_keep.append(raw_line)
                        continue

                    if frame_number > frame_threshold and frame_number not in protected:
                        removed += 1
                        continue

                    # Ensure the newline is preserved when rewriting.
                    lines_to_keep.append(
                        raw_line if raw_line.endswith("\n") else f"{raw_line}\n"
                    )
        except OSError as exc:
            logger.error(
                "Unable to read annotation store %s for pruning: %s",
                self.store_path,
                exc,
            )
            return 0

        if removed == 0:
            return 0

        temp_path = self.store_path.with_suffix(self.store_path.suffix + ".tmp")

        try:
            with temp_path.open("w", encoding="utf-8") as fh:
                fh.writelines(lines_to_keep)
            temp_path.replace(self.store_path)
        except OSError as exc:
            logger.error(
                "Failed to rewrite annotation store %s: %s", self.store_path, exc
            )
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            return 0

        AnnotationStore._CACHE.pop(self.store_path, None)
        # Refresh cache after pruning so subsequent reads see the updated state.
        self._load_records(force_reload=True)

        return removed

    def remove_frames_in_range(
        self,
        start_frame: int,
        end_frame: Optional[int],
        protected_frames: Optional[Iterable[int]] = None,
    ) -> int:
        """Remove frames in a specific inclusive range unless protected.

        Args:
            start_frame: Lowest frame index to remove.
            end_frame: Highest frame index to remove. When None, remove frames
                greater than or equal to ``start_frame``.
            protected_frames: Optional iterable of frame numbers that should never
                be removed, even if they are in range.

        Returns:
            The number of store records that were removed.
        """
        if not self.store_path.exists():
            return 0

        try:
            protected: Set[int] = {
                int(frame) for frame in (protected_frames or []) if frame is not None
            }
        except Exception:
            protected = set()

        lines_to_keep = []
        removed = 0

        try:
            self._ensure_explicit_frame_metadata()
            with self.store_path.open("r", encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.rstrip("\n")
                    stripped = line.strip()
                    if not stripped:
                        lines_to_keep.append(raw_line)
                        continue

                    try:
                        payload = json.loads(stripped)
                    except json.JSONDecodeError:
                        lines_to_keep.append(raw_line)
                        continue

                    frame_value = payload.get("frame")
                    try:
                        frame_number = int(frame_value)
                    except (TypeError, ValueError):
                        lines_to_keep.append(raw_line)
                        continue

                    if frame_number in protected:
                        lines_to_keep.append(
                            raw_line if raw_line.endswith("\n") else f"{raw_line}\n"
                        )
                        continue

                    in_range = frame_number >= start_frame
                    if end_frame is not None:
                        in_range = start_frame <= frame_number <= end_frame

                    if in_range:
                        removed += 1
                        continue

                    lines_to_keep.append(
                        raw_line if raw_line.endswith("\n") else f"{raw_line}\n"
                    )
        except OSError as exc:
            logger.error(
                "Unable to read annotation store %s for pruning: %s",
                self.store_path,
                exc,
            )
            return 0

        if removed == 0:
            return 0

        temp_path = self.store_path.with_suffix(self.store_path.suffix + ".tmp")

        try:
            with temp_path.open("w", encoding="utf-8") as fh:
                fh.writelines(lines_to_keep)
            temp_path.replace(self.store_path)
        except OSError as exc:
            logger.error(
                "Failed to rewrite annotation store %s: %s", self.store_path, exc
            )
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            return 0

        AnnotationStore._CACHE.pop(self.store_path, None)
        self._load_records(force_reload=True)

        return removed

    def write_stub(
        self,
        frame_file: Union[str, Path],
        frame: int,
        record: Dict[str, Any],
    ) -> None:
        frame_file = Path(frame_file)
        stub = {
            "annotation_store": self.store_path.name,
            "frame": frame,
            "version": AnnotationStore.STUB_VERSION,
            "imagePath": record.get("imagePath"),
            "imageHeight": record.get("imageHeight"),
            "imageWidth": record.get("imageWidth"),
        }
        if record.get("caption") is not None:
            stub["caption"] = record["caption"]
        if record.get("flags"):
            stub["flags"] = record["flags"]
        with frame_file.open("w", encoding="utf-8") as fh:
            json.dump(stub, fh, separators=(",", ":"))


def _record_to_labelme_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "version": record.get("version") or "annolid",
        "flags": record.get("flags", {}),
        "shapes": record.get("shapes", []),
        "imagePath": record.get("imagePath"),
        "imageData": record.get("imageData"),
        "imageHeight": record.get("imageHeight"),
        "imageWidth": record.get("imageWidth"),
        "caption": record.get("caption"),
    }

    for key, value in (record.get("otherData") or {}).items():
        payload[key] = value

    return payload


def _load_from_store(path: Path) -> Dict[str, Any]:
    frame = AnnotationStore.frame_number_from_path(path)
    if frame is None:
        raise AnnotationStoreError(f"Cannot infer frame number from path {path}")
    store = AnnotationStore.for_frame_path(path)
    record = store.get_frame(frame)
    if record is None:
        raise AnnotationStoreError(
            f"Frame {frame} not present in store {store.store_path}"
        )
    return _record_to_labelme_payload(record)


def _resolve_stub(data: Dict[str, Any], source: Union[str, Path]) -> Dict[str, Any]:
    annotation_store = data.get("annotation_store")
    if not annotation_store:
        return data

    frame = data.get("frame")
    if frame is None:
        raise AnnotationStoreError("Annotation stub missing frame identifier.")

    store = AnnotationStore.for_frame_path(source, annotation_store)
    record = store.get_frame(frame)
    if record is None:
        raise AnnotationStoreError(
            f"Frame {frame} not present in store {store.store_path}"
        )

    return _record_to_labelme_payload(record)


def load_labelme_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a labelme-style JSON file with annotation store support."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = _ORIGINAL_JSON_LOAD(fh)
    except FileNotFoundError:
        return _load_from_store(path)

    if not isinstance(data, dict) or "annotation_store" not in data:
        return data

    return _resolve_stub(data, path)


def _patched_json_load(fp, *args, **kwargs):
    result = _ORIGINAL_JSON_LOAD(fp, *args, **kwargs)
    if not isinstance(result, dict) or "annotation_store" not in result:
        return result

    source_name = getattr(fp, "name", None)
    if not source_name:
        return result

    try:
        return _resolve_stub(result, Path(source_name))
    except AnnotationStoreError as exc:
        logger.warning(
            "Failed to resolve annotation store reference for %s: %s", source_name, exc
        )
        return result


if not getattr(json, "_annolid_store_patched", False):
    json._annolid_store_patched = True
    json.load = _patched_json_load
