"""Canonical intermediate representation for pose/keypoint dataset interoperability.

Bridges Annolid (LabelMe JSON, AnnotationStore, PoseSchema, Tracking CSV),
DeepLabCut (single/multi-animal analysis & training CSV/H5 + config.yaml),
Lightning Pose (DLC-compatible CSV), SLEAP (.slp, .pkg.slp, .analysis.h5),
COCO keypoints, and downstream behavior toolkits (SimBA, B-SOiD, VAME, movement).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from annolid.annotation.keypoint_visibility import KeypointVisibility
from annolid.annotation.pose_schema import (
    PoseSchema,
    _MOUSE_CANONICAL_EDGES as _MOUSE_PRESET_EDGES,  # noqa: F401 - backward-compatible alias
    dedup_undirected_pairs,
    infer_pose_skeleton_edges,
)
from annolid.utils.body_part_shapes import infer_intra_part_skeleton_edges

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
ZIP_MAGIC = b"PK\x03\x04"


def is_finite_number(value: Any) -> bool:
    """Return True when ``value`` can be converted to a finite float."""
    if value is None or isinstance(value, bool):
        return False
    try:
        num = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(num)


def sanitize_filename_token(name: str, fallback: str = "video") -> str:
    """Sanitize an arbitrary identifier into a safe filename prefix."""
    text = str(name or "").strip()
    if not text:
        return fallback
    chars: List[str] = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            chars.append(ch)
        elif ch.isspace():
            chars.append("_")
    cleaned = "".join(chars).strip("._-")
    return cleaned or fallback


def dedup_preserve_order(items: Iterable[str]) -> List[str]:
    """Deduplicate non-empty stripped strings while preserving first-seen order."""
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def dedup_pairs(pairs: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Deduplicate directed/undirected string pairs while preserving order."""
    seen: set[Tuple[str, str]] = set()
    out: List[Tuple[str, str]] = []
    for a_raw, b_raw in pairs:
        a = str(a_raw or "").strip()
        b = str(b_raw or "").strip()
        if not a or not b or a == b:
            continue
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def infer_edges_for_keypoints(
    keypoints: Sequence[str],
    *,
    preset: Optional[str] = "auto",
) -> List[Tuple[str, str]]:
    """Infer skeleton edges from keypoint names.

    Supported presets:
    - ``"auto"``: combines species-agnostic anatomical inference with mouse preset edges.
    - ``"mouse"``: mouse-specific preset edges, falling back to generic anatomical edges.
    - ``"none"`` / ``""`` / ``None``: returns no inferred edges.
    """
    kp_list = dedup_preserve_order(keypoints)
    if len(kp_list) < 2:
        return []

    mode = str(preset or "").strip().lower()
    if mode in ("", "none", "off", "false"):
        return []

    kp_set = set(kp_list)
    inferred = infer_pose_skeleton_edges(kp_list, preset=mode)
    if mode == "mouse":
        return dedup_undirected_pairs(
            inferred or infer_intra_part_skeleton_edges(kp_list)
        )

    # Default / "auto": full-body anatomical inference plus any extra intra-part chains.
    generic = infer_intra_part_skeleton_edges(kp_list)
    return dedup_undirected_pairs(
        (a, b)
        for a, b in list(inferred) + list(generic)
        if a in kp_set and b in kp_set and a != b
    )


@dataclass(frozen=True)
class PosePoint:
    """A single 2D keypoint observation."""

    x: float
    y: float
    score: Optional[float] = None
    visibility: int = int(KeypointVisibility.VISIBLE)

    @property
    def is_valid(self) -> bool:
        return (
            math.isfinite(float(self.x))
            and math.isfinite(float(self.y))
            and int(self.visibility) > int(KeypointVisibility.MISSING)
        )


@dataclass
class PoseInstance:
    """Keypoint pose of a single animal/instance on a single frame."""

    instance_name: str
    keypoints: Dict[str, PosePoint] = field(default_factory=dict)
    track_id: Optional[int] = None
    skeleton_name: Optional[str] = None
    score: Optional[float] = None
    is_predicted: bool = False


@dataclass
class PoseFrame:
    """All pose instances observed on a single video frame or image."""

    frame_index: int
    instances: List[PoseInstance] = field(default_factory=list)
    image_path: str = ""
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    video_index: int = 0
    video_name: str = ""
    source_json_path: Optional[str] = None
    source_image_path: Optional[str] = None


@dataclass
class PoseDataset:
    """Format-agnostic collection of pose frames and skeleton metadata."""

    frames: List[PoseFrame] = field(default_factory=list)
    keypoints: List[str] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    symmetry_pairs: List[Tuple[str, str]] = field(default_factory=list)
    instances: List[str] = field(default_factory=list)
    scorer: str = "annolid"
    multi_animal: bool = False
    source_format: str = ""
    video_paths: Dict[int, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def resolved_keypoints(self) -> List[str]:
        """Return ordered keypoint names (declared + any observed in frames)."""
        ordered = list(self.keypoints)
        seen = set(ordered)
        for frame in self.frames:
            for inst in frame.instances:
                for kp_name in inst.keypoints:
                    if kp_name and kp_name not in seen:
                        seen.add(kp_name)
                        ordered.append(kp_name)
        return ordered

    def resolved_instances(self) -> List[str]:
        """Return ordered instance/individual names."""
        ordered = list(self.instances)
        seen = set(ordered)
        for frame in self.frames:
            for inst in frame.instances:
                name = str(inst.instance_name or "").strip()
                if name and name not in seen:
                    seen.add(name)
                    ordered.append(name)
        return ordered

    def is_multi_animal(self) -> bool:
        if self.multi_animal:
            return True
        if len(self.resolved_instances()) > 1:
            return True
        return any(len(frame.instances) > 1 for frame in self.frames)

    def to_pose_schema(
        self,
        *,
        preset: Optional[str] = "auto",
        instance_separator: str = "_",
        include_single_instance: bool = False,
    ) -> PoseSchema:
        """Build an Annolid ``PoseSchema`` from this dataset."""
        keypoints = self.resolved_keypoints()
        edges = (
            dedup_pairs(self.edges)
            if self.edges
            else infer_edges_for_keypoints(keypoints, preset=preset)
        )
        sym = (
            dedup_pairs(self.symmetry_pairs)
            if self.symmetry_pairs
            else PoseSchema.infer_symmetry_pairs(keypoints)
        )
        all_instances = self.resolved_instances()
        if len(all_instances) > 1 or include_single_instance:
            schema_instances = all_instances
        else:
            schema_instances = []
        return PoseSchema(
            version="1.0",
            keypoints=keypoints,
            edges=edges,
            symmetry_pairs=sym,
            flip_idx=None,
            instances=schema_instances,
            instance_separator=str(instance_separator or "_"),
        )


def _read_magic_bytes(path: Path, n: int = 8) -> bytes:
    with path.open("rb") as fh:
        return fh.read(n)


def detect_pose_source_format(path: str | Path) -> str:
    """Auto-detect the pose/tracking format of a file or directory.

    Returns one of:
    - ``"deeplabcut_training"`` (DLC/Lightning Pose project dir or labeled-data dir)
    - ``"deeplabcut_csv"`` (DLC/Lightning Pose 3- or 4-level multi-index CSV)
    - ``"deeplabcut_h5"`` (DLC HDF5 table)
    - ``"sleap_slp"`` (SLEAP `.slp` or `.pkg.slp` HDF5)
    - ``"sleap_analysis_h5"`` (SLEAP Analysis HDF5 with `/tracks` and `/node_names`)
    - ``"coco_json"`` (COCO annotations JSON)
    - ``"annolid_tracking_csv"`` (Annolid `*_tracking.csv` / `*_tracked.csv`)
    - ``"labelme_dir"`` (Directory containing LabelMe JSONs or AnnotationStore)
    - ``"unknown"``
    """
    target = Path(path).expanduser()
    if not target.exists():
        raise FileNotFoundError(f"Pose source not found: {target}")

    if target.is_dir():
        labeled_data = target / "labeled-data"
        if labeled_data.is_dir() and (
            any(labeled_data.rglob("CollectedData_*.csv"))
            or any(labeled_data.rglob("CollectedData_*.h5"))
        ):
            return "deeplabcut_training"
        if any(target.rglob("CollectedData_*.csv")) or any(
            target.rglob("CollectedData_*.h5")
        ):
            return "deeplabcut_training"
        if any(target.glob("*_annotations.ndjson")) or any(target.glob("*.json")):
            return "labelme_dir"
        return "unknown"

    suffix = target.suffix.lower()
    if suffix == ".csv":
        try:
            with target.open("r", encoding="utf-8-sig", errors="replace") as fh:
                head = [fh.readline() for _ in range(5)]
        except OSError:
            return "unknown"
        first_cells = [line.split(",", 1)[0].strip().lower() for line in head]
        if (
            len(first_cells) >= 3
            and first_cells[0] == "scorer"
            and (
                (first_cells[1] == "bodyparts" and first_cells[2] == "coords")
                or (
                    len(first_cells) >= 4
                    and first_cells[1] in ("individuals", "animals", "animal")
                    and first_cells[2] == "bodyparts"
                    and first_cells[3] == "coords"
                )
            )
        ):
            return "deeplabcut_csv"
        header_tokens = {
            tok.strip().lower() for tok in (head[0].split(",") if head else [])
        }
        if {"frame_number", "instance_name", "cx", "cy"}.issubset(header_tokens):
            return "annolid_tracking_csv"
        return "unknown"

    if suffix == ".json":
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            return "unknown"
        if isinstance(payload, dict):
            if {"images", "annotations", "categories"}.issubset(payload.keys()):
                return "coco_json"
            if "shapes" in payload and "imagePath" in payload:
                return "labelme_json"
        return "unknown"

    # Check HDF5 container (.slp, .pkg.slp, .h5, .hdf5)
    try:
        magic = _read_magic_bytes(target, 8)
    except OSError:
        return "unknown"

    if magic.startswith(ZIP_MAGIC):
        import zipfile

        try:
            with zipfile.ZipFile(target, "r") as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    lname = info.filename.lower()
                    if lname.endswith((".slp", ".h5", ".hdf5")):
                        with zf.open(info, "r") as member_fh:
                            if member_fh.read(8).startswith(HDF5_MAGIC):
                                return "sleap_slp"
                    if lname.endswith(".json") and info.file_size <= 256 * 1024 * 1024:
                        with zf.open(info, "r") as member_fh:
                            sample = member_fh.read(4096)
                        if (
                            b"labeled_frames" in sample
                            or b"skeletons" in sample
                            or (suffix == ".slp" and b"nodes" in sample)
                        ):
                            return "sleap_slp"
        except Exception:
            pass
        if suffix == ".slp" or target.name.lower().endswith(".pkg.slp"):
            return "sleap_slp"
        return "unknown"

    if magic.startswith(HDF5_MAGIC):
        try:
            import h5py

            with h5py.File(target, "r") as h5:
                keys = set(h5.keys())
                has_pandas_group = any(
                    isinstance(h5[k], h5py.Group)
                    and (
                        "axis0" in h5[k] or "axis0_level0" in h5[k] or "table" in h5[k]
                    )
                    for k in keys
                )
        except Exception:
            if suffix == ".slp" or target.name.lower().endswith(".pkg.slp"):
                return "sleap_slp"
            return "unknown"

        if {"frames", "instances", "points"}.issubset(keys):
            return "sleap_slp"
        if "tracks" in keys and "node_names" in keys:
            return "sleap_analysis_h5"
        if has_pandas_group or "df_with_missing" in keys:
            return "deeplabcut_h5"
        if suffix == ".slp" or target.name.lower().endswith(".pkg.slp"):
            return "sleap_slp"
        return "unknown"

    return "unknown"
