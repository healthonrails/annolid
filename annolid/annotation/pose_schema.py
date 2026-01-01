from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _clean_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    name = str(value).strip()
    return name or None


def _normalize_pairs(pairs: Any) -> List[Tuple[str, str]]:
    if not pairs:
        return []
    normalized: List[Tuple[str, str]] = []
    for entry in pairs:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        left = _clean_name(entry[0])
        right = _clean_name(entry[1])
        if not left or not right or left == right:
            continue
        normalized.append((left, right))
    return normalized


@dataclass
class PoseSchema:
    """Pose schema for keypoint order, symmetry, and edges.

    Stored as JSON (or YAML) and can be used to generate Ultralytics `flip_idx`.
    """

    version: str = "1.0"
    keypoints: List[str] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    symmetry_pairs: List[Tuple[str, str]] = field(default_factory=list)
    flip_idx: Optional[List[int]] = None
    instances: List[str] = field(default_factory=list)
    instance_separator: str = "_"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PoseSchema":
        keypoints_raw = data.get("keypoints") or []
        keypoints = []
        for item in keypoints_raw:
            name = _clean_name(item)
            if name and name not in keypoints:
                keypoints.append(name)

        edges = _normalize_pairs(data.get("edges"))
        symmetry_pairs = _normalize_pairs(data.get("symmetry_pairs"))

        flip_idx = data.get("flip_idx")
        if flip_idx is not None and not isinstance(flip_idx, list):
            flip_idx = None
        if isinstance(flip_idx, list):
            try:
                flip_idx = [int(v) for v in flip_idx]
            except Exception:
                flip_idx = None

        instances_raw = data.get("instances") or []
        instances = []
        for item in instances_raw:
            name = _clean_name(item)
            if name and name not in instances:
                instances.append(name.rstrip("_"))

        instance_separator = _clean_name(data.get("instance_separator")) or "_"

        return cls(
            version=str(data.get("version") or "1.0"),
            keypoints=keypoints,
            edges=edges,
            symmetry_pairs=symmetry_pairs,
            flip_idx=flip_idx,
            instances=instances,
            instance_separator=instance_separator,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "keypoints": list(self.keypoints),
            "edges": [[a, b] for a, b in self.edges],
            "flip_idx": list(self.flip_idx) if self.flip_idx is not None else None,
            "symmetry_pairs": [[a, b] for a, b in self.symmetry_pairs],
            "instances": list(self.instances),
            "instance_separator": self.instance_separator,
        }

    def instance_prefix(self, instance: str) -> str:
        sep = self.instance_separator or "_"
        inst = str(instance or "").strip().rstrip("_")
        return f"{inst}{sep}" if inst else ""

    def expand_keypoints(self) -> List[str]:
        """Return keypoint names expanded with instance prefixes when configured."""
        if not self.instances:
            return list(self.keypoints)
        if not self.keypoints:
            return []
        sep = self.instance_separator or "_"
        expanded: List[str] = []
        for inst in self.instances:
            inst = str(inst or "").strip().rstrip("_")
            if not inst:
                continue
            prefix = f"{inst}{sep}"
            for kp in self.keypoints:
                kp = str(kp or "").strip()
                if kp:
                    expanded.append(f"{prefix}{kp}")
        return expanded

    def strip_instance_prefix(self, label: str) -> Tuple[Optional[str], str]:
        """Return (instance, base_keypoint) if the label matches an instance prefix."""
        sep = self.instance_separator or "_"
        text = str(label or "").strip()
        if not text:
            return None, ""
        if self.instances:
            for inst in self.instances:
                prefix = self.instance_prefix(inst)
                if prefix and text.lower().startswith(prefix.lower()):
                    return inst, text[len(prefix):].lstrip(sep).strip()
            # Instances are configured but no known prefix matched: treat as base label.
            return None, text
        return None, text

    @classmethod
    def load(cls, path: str | Path) -> "PoseSchema":
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(str(p))
        suffix = p.suffix.lower()
        if suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "PyYAML is required to load pose schema YAML files.") from exc
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            if not isinstance(data, dict):
                raise ValueError(f"Invalid pose schema in {p}")
            return cls.from_dict(data)

        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid pose schema in {p}")
        return cls.from_dict(data)

    def save(self, path: str | Path) -> Path:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        suffix = p.suffix.lower()
        if suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "PyYAML is required to save pose schema YAML files.") from exc
            p.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False),
                         encoding="utf-8")
            return p
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    def compute_flip_idx(self, keypoint_order: Optional[Sequence[str]] = None) -> Optional[List[int]]:
        order = list(keypoint_order or self.keypoints)
        if not order:
            return None
        if self.flip_idx is not None and len(self.flip_idx) == len(order):
            return list(self.flip_idx)

        index_by_name = {name: idx for idx, name in enumerate(order)}
        flip = list(range(len(order)))
        applied_any = False

        def apply_pair(left: str, right: str) -> None:
            nonlocal applied_any
            li = index_by_name.get(left)
            ri = index_by_name.get(right)
            if li is None or ri is None:
                return
            flip[li] = ri
            flip[ri] = li
            applied_any = True

        # Direct match (schemas that store fully-qualified names).
        for left, right in self.symmetry_pairs:
            apply_pair(left, right)

        # If no direct matches happened, try expanding base symmetry pairs per instance.
        if not applied_any and self.instances:
            sep = self.instance_separator or "_"
            for inst in self.instances:
                inst = str(inst or "").strip().rstrip("_")
                if not inst:
                    continue
                prefix = f"{inst}{sep}"
                for left, right in self.symmetry_pairs:
                    apply_pair(f"{prefix}{left}", f"{prefix}{right}")

        return flip

    def normalize_prefixed_keypoints(self) -> None:
        """Convert prefixed keypoints/edges into base+instances when possible."""
        sep = self.instance_separator or "_"
        if not self.keypoints:
            return

        inferred_instances: List[str] = []
        base_keypoints: List[str] = []
        seen_base = set()
        for name in self.keypoints:
            text = str(name or "").strip()
            if sep not in text:
                continue
            inst, base = text.split(sep, 1)
            inst = inst.strip().rstrip("_")
            base = base.strip()
            if inst and inst not in inferred_instances:
                inferred_instances.append(inst)
            if base and base not in seen_base:
                seen_base.add(base)
                base_keypoints.append(base)

        if not inferred_instances or not base_keypoints:
            return

        # Only normalize when at least one keypoint appears to be prefixed.
        if not any(sep in kp for kp in self.keypoints):
            return

        self.instances = inferred_instances
        self.keypoints = base_keypoints

        def strip_pair(pair: Tuple[str, str]) -> Optional[Tuple[str, str]]:
            a_text = str(pair[0] or "").strip()
            b_text = str(pair[1] or "").strip()
            if sep in a_text:
                _, a = a_text.split(sep, 1)
                a = a.strip()
            else:
                a = a_text
            if sep in b_text:
                _, b = b_text.split(sep, 1)
                b = b.strip()
            else:
                b = b_text
            if not a or not b or a == b:
                return None
            return a, b

        new_edges: List[Tuple[str, str]] = []
        for edge in self.edges:
            stripped = strip_pair(edge)
            if stripped:
                new_edges.append(stripped)
        self.edges = new_edges

        new_pairs: List[Tuple[str, str]] = []
        for pair in self.symmetry_pairs:
            stripped = strip_pair(pair)
            if stripped:
                new_pairs.append(stripped)
        self.symmetry_pairs = new_pairs

    @staticmethod
    def infer_symmetry_pairs(keypoints: Iterable[str]) -> List[Tuple[str, str]]:
        """Best-effort pairing based on common left/right naming patterns."""
        names = [str(k).strip() for k in keypoints if str(k).strip()]
        lower_to_original = {n.lower(): n for n in names}
        visited: set[str] = set()
        pairs: List[Tuple[str, str]] = []

        def swap(token: str, a: str, b: str) -> Optional[str]:
            if a in token:
                return token.replace(a, b)
            return None

        patterns = [
            ("left", "right"),
            ("right", "left"),
            ("l_", "r_"),
            ("r_", "l_"),
            ("l-", "r-"),
            ("r-", "l-"),
        ]

        for name in names:
            key = name.lower()
            if key in visited:
                continue
            other_key = None
            for a, b in patterns:
                other_key = swap(key, a, b)
                if other_key and other_key in lower_to_original:
                    break
                other_key = None
            if other_key and other_key not in visited:
                pairs.append((name, lower_to_original[other_key]))
                visited.add(key)
                visited.add(other_key)
        return pairs
