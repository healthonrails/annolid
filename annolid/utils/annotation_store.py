import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

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

    @classmethod
    def for_frame_path(cls, frame_path: Union[str, Path], store_name: Optional[str] = None) -> "AnnotationStore":
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

        # Refresh cache after appending.
        self._load_records(force_reload=True)

    def get_frame(self, frame: int) -> Optional[Dict[str, Any]]:
        """Return the latest record for a frame if present."""
        records = self._load_records()
        return records.get(frame)

    def iter_frames(self) -> Iterable[int]:
        records = self._load_records()
        return records.keys()

    def _load_records(self, force_reload: bool = False) -> Dict[int, Dict[str, Any]]:
        if not self.store_path.exists():
            return {}

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
                        "Skipping invalid annotation store line in %s", self.store_path)
                    continue
                frame = data.get("frame")
                if frame is None:
                    continue
                records[frame] = data

        AnnotationStore._CACHE[self.store_path] = {
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "records": records,
        }
        return records

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
        raise AnnotationStoreError(
            f"Cannot infer frame number from path {path}")
    store = AnnotationStore.for_frame_path(path)
    record = store.get_frame(frame)
    if record is None:
        raise AnnotationStoreError(
            f"Frame {frame} not present in store {store.store_path}")
    return _record_to_labelme_payload(record)


def _resolve_stub(data: Dict[str, Any], source: Union[str, Path]) -> Dict[str, Any]:
    annotation_store = data.get("annotation_store")
    if not annotation_store:
        return data

    frame = data.get("frame")
    if frame is None:
        raise AnnotationStoreError(
            "Annotation stub missing frame identifier.")

    store = AnnotationStore.for_frame_path(source, annotation_store)
    record = store.get_frame(frame)
    if record is None:
        raise AnnotationStoreError(
            f"Frame {frame} not present in store {store.store_path}")

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
            "Failed to resolve annotation store reference for %s: %s", source_name, exc)
        return result


if not getattr(json, "_annolid_store_patched", False):
    json._annolid_store_patched = True
    json.load = _patched_json_load
