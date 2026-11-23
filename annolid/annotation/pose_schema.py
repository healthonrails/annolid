from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

POSE_SCHEMA_FILENAMES: Tuple[str, ...] = (
    "pose_schema.json",
    "pose_schema.yaml",
    "pose_schema.yml",
)

# Common mouse skeleton edges used by presets and auto edge inference.
_MOUSE_CANONICAL_EDGES: Tuple[Tuple[str, str], ...] = (
    ("nose", "neck"),
    ("neck", "back"),
    ("back", "root_tail"),
    ("root_tail", "mid_tail"),
    ("mid_tail", "tip_tail"),
    ("left_ear", "nose"),
    ("right_ear", "nose"),
    ("left_front_limb", "neck"),
    ("right_front_limb", "neck"),
    ("left_hind_limb", "back"),
    ("right_hind_limb", "back"),
    ("left_front_claw", "left_front_limb"),
    ("right_front_claw", "right_front_limb"),
    ("left_hind_claw", "left_hind_limb"),
    ("right_hind_claw", "right_hind_limb"),
)

POSE_SCHEMA_PRESETS: Dict[str, Dict[str, Any]] = {
    "Mouse (5 keypoints)": {
        "keypoints": [
            "nose",
            "left_ear",
            "right_ear",
            "centroid",
            "tail_base",
        ],
        "edges": [
            ("nose", "left_ear"),
            ("nose", "right_ear"),
            ("left_ear", "right_ear"),
            ("nose", "centroid"),
            ("centroid", "tail_base"),
        ],
        "symmetry_pairs": [
            ("left_ear", "right_ear"),
        ],
    },
    "Mouse CalMS21 / MARS (7 keypoints)": {
        "keypoints": [
            "nose",
            "left_ear",
            "right_ear",
            "neck",
            "left_hip",
            "right_hip",
            "tail_base",
        ],
        "edges": [
            ("nose", "left_ear"),
            ("nose", "right_ear"),
            ("left_ear", "neck"),
            ("right_ear", "neck"),
            ("nose", "neck"),
            ("neck", "left_hip"),
            ("neck", "right_hip"),
            ("left_hip", "tail_base"),
            ("right_hip", "tail_base"),
            ("neck", "tail_base"),
        ],
        "symmetry_pairs": [
            ("left_ear", "right_ear"),
            ("left_hip", "right_hip"),
        ],
    },
    "Mouse (9 keypoints)": {
        "keypoints": [
            "nose",
            "left_ear",
            "right_ear",
            "left_forepaw",
            "right_forepaw",
            "left_hindpaw",
            "right_hindpaw",
            "tail_base",
            "tail_tip",
        ],
        "edges": [
            ("nose", "left_ear"),
            ("nose", "right_ear"),
            ("nose", "left_forepaw"),
            ("nose", "right_forepaw"),
            ("left_forepaw", "left_hindpaw"),
            ("right_forepaw", "right_hindpaw"),
            ("left_hindpaw", "tail_base"),
            ("right_hindpaw", "tail_base"),
            ("tail_base", "tail_tip"),
        ],
        "symmetry_pairs": [
            ("left_ear", "right_ear"),
            ("left_forepaw", "right_forepaw"),
            ("left_hindpaw", "right_hindpaw"),
        ],
    },
    "Mouse (11 keypoints)": {
        "keypoints": [
            "nose",
            "left_ear",
            "right_ear",
            "neck",
            "left_forepaw",
            "right_forepaw",
            "spine",
            "left_hindpaw",
            "right_hindpaw",
            "tail_base",
            "tail_tip",
        ],
        "edges": [
            ("nose", "left_ear"),
            ("nose", "right_ear"),
            ("nose", "neck"),
            ("neck", "left_forepaw"),
            ("neck", "right_forepaw"),
            ("neck", "spine"),
            ("spine", "left_hindpaw"),
            ("spine", "right_hindpaw"),
            ("spine", "tail_base"),
            ("tail_base", "tail_tip"),
        ],
        "symmetry_pairs": [
            ("left_ear", "right_ear"),
            ("left_forepaw", "right_forepaw"),
            ("left_hindpaw", "right_hindpaw"),
        ],
    },
    "Mouse MaBE2022 (12 keypoints)": {
        "keypoints": [
            "nose",
            "left_ear",
            "right_ear",
            "neck",
            "left_forepaw",
            "right_forepaw",
            "center_back",
            "left_hindpaw",
            "right_hindpaw",
            "tail_base",
            "tail_middle",
            "tail_tip",
        ],
        "edges": [
            ("nose", "left_ear"),
            ("nose", "right_ear"),
            ("nose", "neck"),
            ("neck", "left_forepaw"),
            ("neck", "right_forepaw"),
            ("neck", "center_back"),
            ("center_back", "left_hindpaw"),
            ("center_back", "right_hindpaw"),
            ("center_back", "tail_base"),
            ("tail_base", "tail_middle"),
            ("tail_middle", "tail_tip"),
        ],
        "symmetry_pairs": [
            ("left_ear", "right_ear"),
            ("left_forepaw", "right_forepaw"),
            ("left_hindpaw", "right_hindpaw"),
        ],
    },
    "Rat (8 keypoints)": {
        "keypoints": [
            "nose",
            "left_ear",
            "right_ear",
            "neck",
            "mid_back",
            "left_hip",
            "right_hip",
            "tail_base",
        ],
        "edges": [
            ("nose", "left_ear"),
            ("nose", "right_ear"),
            ("nose", "neck"),
            ("neck", "mid_back"),
            ("mid_back", "left_hip"),
            ("mid_back", "right_hip"),
            ("mid_back", "tail_base"),
            ("left_hip", "tail_base"),
            ("right_hip", "tail_base"),
        ],
        "symmetry_pairs": [
            ("left_ear", "right_ear"),
            ("left_hip", "right_hip"),
        ],
    },
    "Fly (5 keypoints)": {
        "keypoints": [
            "head",
            "thorax",
            "abdomen",
            "left_wing",
            "right_wing",
        ],
        "edges": [
            ("head", "thorax"),
            ("thorax", "abdomen"),
            ("thorax", "left_wing"),
            ("thorax", "right_wing"),
        ],
        "symmetry_pairs": [
            ("left_wing", "right_wing"),
        ],
    },
    "Fly Full-Body (12 keypoints)": {
        "keypoints": [
            "head",
            "left_eye",
            "right_eye",
            "thorax",
            "abdomen",
            "left_wing",
            "right_wing",
            "left_front_leg",
            "right_front_leg",
            "left_mid_leg",
            "right_mid_leg",
            "tail_tip",
        ],
        "edges": [
            ("head", "left_eye"),
            ("head", "right_eye"),
            ("head", "thorax"),
            ("thorax", "abdomen"),
            ("abdomen", "tail_tip"),
            ("thorax", "left_wing"),
            ("thorax", "right_wing"),
            ("thorax", "left_front_leg"),
            ("thorax", "right_front_leg"),
            ("thorax", "left_mid_leg"),
            ("thorax", "right_mid_leg"),
        ],
        "symmetry_pairs": [
            ("left_eye", "right_eye"),
            ("left_wing", "right_wing"),
            ("left_front_leg", "right_front_leg"),
            ("left_mid_leg", "right_mid_leg"),
        ],
    },
    "Fish (5 keypoints)": {
        "keypoints": [
            "head",
            "dorsal_fin",
            "tail",
            "left_pectoral",
            "right_pectoral",
        ],
        "edges": [
            ("head", "dorsal_fin"),
            ("dorsal_fin", "tail"),
            ("head", "left_pectoral"),
            ("head", "right_pectoral"),
        ],
        "symmetry_pairs": [
            ("left_pectoral", "right_pectoral"),
        ],
    },
    "Zebrafish Spine (5 keypoints)": {
        "keypoints": [
            "head",
            "spine_1",
            "spine_2",
            "spine_3",
            "tail_tip",
        ],
        "edges": [
            ("head", "spine_1"),
            ("spine_1", "spine_2"),
            ("spine_2", "spine_3"),
            ("spine_3", "tail_tip"),
        ],
        "symmetry_pairs": [],
    },
    "Human COCO (17 keypoints)": {
        "keypoints": [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ],
        "edges": [
            ("nose", "left_eye"),
            ("nose", "right_eye"),
            ("left_eye", "left_ear"),
            ("right_eye", "right_ear"),
            ("left_ear", "left_shoulder"),
            ("right_ear", "right_shoulder"),
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"),
            ("right_knee", "right_ankle"),
        ],
        "symmetry_pairs": [
            ("left_eye", "right_eye"),
            ("left_ear", "right_ear"),
            ("left_shoulder", "right_shoulder"),
            ("left_elbow", "right_elbow"),
            ("left_wrist", "right_wrist"),
            ("left_hip", "right_hip"),
            ("left_knee", "right_knee"),
            ("left_ankle", "right_ankle"),
        ],
    },
}


def _clean_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    name = str(value).strip()
    return name or None


def dedup_undirected_pairs(
    pairs: Iterable[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """Deduplicate undirected string pairs while preserving first-seen orientation."""
    out: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for a_raw, b_raw in pairs:
        left = _clean_name(a_raw)
        right = _clean_name(b_raw)
        if not left or not right or left == right:
            continue
        ukey = (left, right) if left < right else (right, left)
        if ukey in seen:
            continue
        seen.add(ukey)
        out.append((left, right))
    return out


def _normalize_pairs(
    pairs: Any,
    *,
    keypoints: Optional[Sequence[str]] = None,
) -> List[Tuple[str, str]]:
    if not pairs or not isinstance(pairs, (list, tuple)):
        return []
    kp_list = list(keypoints or [])
    normalized: List[Tuple[str, str]] = []
    for entry in pairs:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        raw_a, raw_b = entry[0], entry[1]
        if (
            kp_list
            and isinstance(raw_a, int)
            and not isinstance(raw_a, bool)
            and isinstance(raw_b, int)
            and not isinstance(raw_b, bool)
        ):
            ia, ib = raw_a - 1, raw_b - 1
            if 0 <= ia < len(kp_list) and 0 <= ib < len(kp_list) and ia != ib:
                normalized.append((kp_list[ia], kp_list[ib]))
            continue
        left = _clean_name(raw_a)
        right = _clean_name(raw_b)
        if not left or not right or left == right:
            continue
        normalized.append((left, right))
    return dedup_undirected_pairs(normalized)


def find_pose_schema_in_dir(
    directory: str | Path,
    *,
    search_parent: bool = False,
) -> Optional[Path]:
    """Locate ``pose_schema.{json,yaml,yml}`` in ``directory`` (and optionally parent)."""
    try:
        base = Path(directory).expanduser()
    except Exception:
        return None
    if base.is_file():
        base = base.parent
    if not base.is_dir():
        return None
    dirs = [base]
    if search_parent and base.parent != base:
        dirs.append(base.parent)
    for folder in dirs:
        for name in POSE_SCHEMA_FILENAMES:
            candidate = folder / name
            if candidate.is_file():
                return candidate
    return None


def list_pose_schema_presets() -> List[str]:
    """Return ordered display names of built-in skeleton presets."""
    return list(POSE_SCHEMA_PRESETS.keys())


def get_pose_schema_preset(name: str) -> "PoseSchema":
    """Return a fresh ``PoseSchema`` copy for a built-in preset name."""
    key = str(name or "").strip()
    spec = POSE_SCHEMA_PRESETS.get(key)
    if spec is None:
        lower = key.lower()
        for preset_name, preset_spec in POSE_SCHEMA_PRESETS.items():
            if preset_name.lower() == lower:
                spec = preset_spec
                break
    if spec is None:
        raise KeyError(
            f"Unknown pose schema preset {name!r}. "
            f"Available: {', '.join(POSE_SCHEMA_PRESETS)}"
        )
    return PoseSchema(
        version="1.0",
        keypoints=list(spec["keypoints"]),
        edges=[tuple(pair) for pair in spec.get("edges", ())],
        symmetry_pairs=[tuple(pair) for pair in spec.get("symmetry_pairs", ())],
        flip_idx=None,
        instances=[],
        instance_separator="_",
    )


def _normalize_kp_tokens(raw_name: str) -> Tuple[str, set[str]]:
    leaf = raw_name.rsplit(":", 1)[-1].strip() if ":" in raw_name else raw_name.strip()
    # Split camelCase before normalizing so leftEar -> left_ear.
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", leaf)
    norm = re.sub(r"[^a-z0-9]+", "_", spaced.lower()).strip("_")
    toks = {tok for tok in norm.split("_") if tok}
    return norm, toks


def _kp_side(norm: str, toks: set[str]) -> str:
    if "left" in toks or "lt" in toks or norm.startswith("l_") or norm.endswith("_l"):
        return "left"
    if "right" in toks or "rt" in toks or norm.startswith("r_") or norm.endswith("_r"):
        return "right"
    return ""


def _kp_fore_hind(norm: str, toks: set[str]) -> str:
    if toks & {
        "fore",
        "front",
        "anterior",
        "upper",
        "shoulder",
        "elbow",
        "wrist",
        "forearm",
        "forepaw",
        "frontpaw",
        "forefoot",
        "hand",
        "manus",
        "wing",
        "pectoral",
        "t1",
        "arm",
    } or any(t.startswith(("fore", "front")) for t in toks):
        return "fore"
    if toks & {
        "hind",
        "back",
        "rear",
        "posterior",
        "lower",
        "hip",
        "knee",
        "ankle",
        "hock",
        "thigh",
        "shank",
        "hindpaw",
        "backpaw",
        "rearpaw",
        "hindfoot",
        "foot",
        "pes",
        "t3",
        "leg",
    } or any(t.startswith(("hind", "rear")) for t in toks):
        return "hind"
    if toks & {"mid", "middle", "t2"} and (toks & {"leg", "tarsus", "claw", "limb"}):
        return "mid"
    return ""


def _limb_joint_rank(toks: set[str]) -> Optional[float]:
    if toks & {"shoulder", "hip", "coxa"}:
        return 0.0
    if toks & {"upperarm", "thigh", "femur"}:
        return 1.0
    if toks & {"elbow", "knee", "stifle", "limb"}:
        return 2.0
    if toks & {"forearm", "shank", "tibia"}:
        return 3.0
    if toks & {"wrist", "ankle", "hock", "carpus", "tarsus"}:
        return 4.0
    if toks & {
        "paw",
        "forepaw",
        "hindpaw",
        "frontpaw",
        "backpaw",
        "rearpaw",
        "foot",
        "forefoot",
        "hindfoot",
        "hand",
        "manus",
        "pes",
        "toe",
        "claw",
        "wing",
        "pectoral",
    }:
        return 5.0
    return None


def _tail_rank(norm: str, toks: set[str]) -> Optional[float]:
    if not ("tail" in toks or any(t.startswith("tail") for t in toks)):
        return None
    if toks & {"base", "root", "start", "origin"} or norm in {
        "tailbase",
        "tailroot",
        "root_tail",
    }:
        return 0.0
    if toks & {"prox", "proximal", "upper"}:
        return 1.0
    if toks & {"mid", "middle", "center"} or norm in {"tailmid", "mid_tail"}:
        return 2.5
    if toks & {"dist", "distal", "lower"}:
        return 3.5
    if toks & {"tip", "end"} or norm in {"tailtip", "tip_tail"}:
        return 5.0
    for tok in toks:
        if tok.isdigit():
            return 0.5 + float(int(tok)) * 0.8
        m = re.match(r"^tail(\d+)$", tok)
        if m:
            return 0.5 + float(int(m.group(1))) * 0.8
    return 4.0


def _axial_rank(norm: str, toks: set[str]) -> Optional[float]:
    if _kp_side(norm, toks):
        return None
    if _tail_rank(norm, toks) is not None:
        return None
    if toks & {"nose", "snout", "rostrum", "beak", "bill", "proboscis"}:
        return 0.0
    if (
        toks & {"head", "skull", "crown", "forehead", "head_site"}
        or norm == "head_site"
    ):
        return 0.5
    if toks & {"neck", "nape", "throat", "cervical", "occiput"}:
        return 1.0
    if toks & {"withers", "thorax", "chest", "thorax_site"} or norm in {
        "upper_back",
        "thorax_site",
    }:
        return 2.0
    if "dorsal" in toks and "fin" in toks:
        return 2.5
    if "spine" in toks or "vertebra" in toks or norm == "back" or "back" in toks:
        for tok in toks:
            if tok.isdigit():
                return 2.0 + float(int(tok)) * 0.6
        if "mid" in toks or "center" in toks:
            return 3.0
        if "lower" in toks:
            return 3.6
        return 3.0
    if toks & {"center", "centroid", "midbody", "body_center", "torso"} or norm in {
        "body_center",
        "bodycenter",
    }:
        return 3.0
    if toks & {"abdomen", "pelvis", "sacrum", "rump"} or norm == "abdomen_tip":
        return 4.0
    return None


def infer_pose_skeleton_edges(
    keypoints: Iterable[str],
    *,
    preset: Optional[str] = "auto",
) -> List[Tuple[str, str]]:
    """Infer full-body anatomical pose skeleton edges from keypoint names.

    Supported presets:
    - ``"auto"``: anatomical head/axial/tail/limb inference plus canonical mouse pairs.
    - ``"mouse"``: mouse-focused preset pairs, falling back to generic inference.
    - ``"none"`` / ``""`` / ``None``: returns no inferred edges.
    """
    kp_list: List[str] = []
    seen_kps: set[str] = set()
    for raw in keypoints:
        cleaned = _clean_name(raw)
        if cleaned and cleaned not in seen_kps:
            seen_kps.add(cleaned)
            kp_list.append(cleaned)
    if len(kp_list) < 2:
        return []

    mode = str(preset or "").strip().lower()
    if mode in ("", "none", "off", "false"):
        return []

    kp_set = set(kp_list)
    mouse_edges = [
        (a, b) for a, b in _MOUSE_CANONICAL_EDGES if a in kp_set and b in kp_set
    ]
    if mode == "mouse" and mouse_edges:
        return dedup_undirected_pairs(mouse_edges)

    parsed: List[Tuple[str, str, set[str]]] = [
        (raw, *_normalize_kp_tokens(raw)) for raw in kp_list
    ]

    edges: List[Tuple[str, str]] = []
    seen_edges: set[Tuple[str, str]] = set()

    def _add(a: str, b: str) -> None:
        if not a or not b or a == b:
            return
        ukey = (a, b) if a < b else (b, a)
        if ukey not in seen_edges:
            seen_edges.add(ukey)
            edges.append((a, b))

    for a, b in mouse_edges:
        _add(a, b)

    # 1. Cranial / head features (rostral/head -> eyes -> ears, head -> antennae).
    rostral = [
        raw
        for raw, _n, toks in parsed
        if toks & {"nose", "snout", "rostrum", "beak", "bill", "proboscis"}
    ]
    heads = [
        raw
        for raw, norm, toks in parsed
        if (toks & {"head", "skull", "crown"} or norm == "head_site")
        and not _kp_side(norm, toks)
    ]
    eyes_by_side: Dict[str, List[str]] = {"left": [], "right": [], "": []}
    ears_by_side: Dict[str, List[str]] = {"left": [], "right": [], "": []}
    antennae_by_side: Dict[str, List[str]] = {"left": [], "right": [], "": []}
    for raw, norm, toks in parsed:
        side = _kp_side(norm, toks)
        if toks & {"eye", "pupil"} or any(t.endswith("eye") for t in toks):
            eyes_by_side[side].append(raw)
        elif toks & {"ear"} or any(t.endswith("ear") and t != "rear" for t in toks):
            ears_by_side[side].append(raw)
        elif toks & {"antenna", "antennae"} or any(
            t.startswith("antenna") for t in toks
        ):
            antennae_by_side[side].append(raw)

    head_anchor = rostral[0] if rostral else (heads[0] if heads else None)
    if head_anchor is not None:
        for side in ("left", "right", ""):
            if eyes_by_side[side]:
                e0 = eyes_by_side[side][0]
                _add(head_anchor, e0)
                for ear_k in ears_by_side[side]:
                    _add(e0, ear_k)
            else:
                for ear_k in ears_by_side[side]:
                    _add(head_anchor, ear_k)
            for ant_k in antennae_by_side[side]:
                _add(head_anchor, ant_k)

    # 2. Axial midline chain (rostral/head -> neck -> thorax -> spine/centroid -> abdomen/pelvis).
    axial_items: List[Tuple[float, str]] = []
    for raw, norm, toks in parsed:
        rank = _axial_rank(norm, toks)
        if rank is not None:
            axial_items.append((rank, raw))
    axial_items.sort(key=lambda item: (item[0], item[1]))
    for idx in range(len(axial_items) - 1):
        _add(axial_items[idx][1], axial_items[idx + 1][1])

    # 3. Tail chain (tail_base -> tail_1 -> tail_mid -> tail_tip), connected to last axial node.
    tail_items: List[Tuple[float, str]] = []
    for raw, norm, toks in parsed:
        rank = _tail_rank(norm, toks)
        if rank is not None:
            tail_items.append((rank, raw))
    tail_items.sort(key=lambda item: (item[0], item[1]))
    if tail_items and axial_items:
        _add(axial_items[-1][1], tail_items[0][1])
    for idx in range(len(tail_items) - 1):
        _add(tail_items[idx][1], tail_items[idx + 1][1])

    # 4. Same-limb joint chains + torso/girdle attachments.
    limb_groups: Dict[Tuple[str, str], List[Tuple[float, str]]] = {}
    shoulders: Dict[str, str] = {}
    hips: Dict[str, str] = {}
    for raw, norm, toks in parsed:
        side = _kp_side(norm, toks)
        fh = _kp_fore_hind(norm, toks)
        jr = _limb_joint_rank(toks)
        if jr is not None and (side or fh):
            limb_groups.setdefault((side, fh), []).append((jr, raw))
        if side and "shoulder" in toks and side not in shoulders:
            shoulders[side] = raw
        if side and "hip" in toks and side not in hips:
            hips[side] = raw

    for grp in limb_groups.values():
        grp.sort(key=lambda item: (item[0], item[1]))
        for idx in range(len(grp) - 1):
            if grp[idx][0] < grp[idx + 1][0]:
                _add(grp[idx][1], grp[idx + 1][1])

    # Human / bipedal torso box (shoulders <-> hips) when bilateral shoulders/hips exist.
    if "left" in shoulders and "right" in shoulders:
        _add(shoulders["left"], shoulders["right"])
    if "left" in hips and "right" in hips:
        _add(hips["left"], hips["right"])
    for side in ("left", "right"):
        if side in shoulders and side in hips:
            _add(shoulders[side], hips[side])
        if (
            side in shoulders
            and ears_by_side[side]
            and not any(1.0 <= r < 2.0 for r, _ in axial_items)
        ):
            _add(ears_by_side[side][0], shoulders[side])

    # Connect proximal limbs/wings/fins to axial midline anchors when present.
    fore_anchor = next(
        (raw for r, raw in axial_items if 1.0 <= r <= 2.5),
        axial_items[0][1] if axial_items else None,
    )
    hind_anchor = next(
        (raw for r, raw in reversed(axial_items) if r >= 2.0),
        tail_items[0][1] if tail_items else fore_anchor,
    )
    for (side, fh), grp in sorted(limb_groups.items()):
        proximal = grp[0][1]
        # Skip connecting shoulders/hips to midline if a bilateral torso box already connects them.
        if proximal in shoulders.values() and len(shoulders) == 2 and hips:
            continue
        if proximal in hips.values() and len(hips) == 2 and shoulders:
            continue
        if fh == "fore" and fore_anchor:
            _add(fore_anchor, proximal)
        elif fh == "hind" and hind_anchor:
            _add(hind_anchor, proximal)
        elif fh == "mid" and fore_anchor:
            _add(fore_anchor, proximal)

    # Connect bilateral forepaws <-> hindpaws when there is no intermediate axial spine node.
    has_mid_axial = any(1.0 <= r <= 4.0 for r, _ in axial_items)
    if not has_mid_axial:
        for side in ("left", "right"):
            fore_grp = limb_groups.get((side, "fore"))
            hind_grp = limb_groups.get((side, "hind"))
            if fore_grp and hind_grp:
                _add(fore_grp[0][1], hind_grp[0][1])

    return dedup_undirected_pairs(edges)


@dataclass
class PoseSchema:
    """Pose schema for keypoint order, symmetry, and skeleton edges.

    Stored as JSON (or YAML) and can be used for canvas skeleton overlays,
    COCO/YOLO/SLEAP/DLC conversion, anatomical QC, and Ultralytics ``flip_idx``.
    """

    version: str = "1.0"
    keypoints: List[str] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    symmetry_pairs: List[Tuple[str, str]] = field(default_factory=list)
    flip_idx: Optional[List[int]] = None
    instances: List[str] = field(default_factory=list)
    instance_separator: str = "_"
    sequencer: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PoseSchema":
        keypoints_raw = data.get("keypoints") or []
        keypoints: List[str] = []
        for item in keypoints_raw:
            name = _clean_name(item)
            if name and name not in keypoints:
                keypoints.append(name)

        raw_edges = (
            data.get("edges") if data.get("edges") is not None else data.get("skeleton")
        )
        edges = _normalize_pairs(raw_edges, keypoints=keypoints)
        symmetry_pairs = _normalize_pairs(
            data.get("symmetry_pairs"), keypoints=keypoints
        )

        flip_idx = data.get("flip_idx")
        if flip_idx is not None and not isinstance(flip_idx, list):
            flip_idx = None
        if isinstance(flip_idx, list):
            try:
                flip_idx = [int(v) for v in flip_idx]
            except Exception:
                flip_idx = None

        instance_separator = _clean_name(data.get("instance_separator")) or "_"
        instances_raw = data.get("instances") or []
        instances: List[str] = []
        for item in instances_raw:
            name = _clean_name(item)
            if not name:
                continue
            cleaned_inst = name.rstrip(instance_separator).rstrip("_")
            if cleaned_inst and cleaned_inst not in instances:
                instances.append(cleaned_inst)

        sequencer_raw = data.get("sequencer")
        sequencer = dict(sequencer_raw) if isinstance(sequencer_raw, dict) else {}

        return cls(
            version=str(data.get("version") or "1.0"),
            keypoints=keypoints,
            edges=edges,
            symmetry_pairs=symmetry_pairs,
            flip_idx=flip_idx,
            instances=instances,
            instance_separator=instance_separator,
            sequencer=sequencer,
        )

    @classmethod
    def from_coco_category(
        cls,
        category: Mapping[str, Any],
        *,
        infer_symmetry: bool = True,
    ) -> "PoseSchema":
        """Build a ``PoseSchema`` from a COCO category dictionary."""
        raw_kps = category.get("keypoints")
        keypoints: List[str] = []
        if isinstance(raw_kps, list):
            for item in raw_kps:
                name = _clean_name(item)
                if name and name not in keypoints:
                    keypoints.append(name)
        edges = _normalize_pairs(category.get("skeleton"), keypoints=keypoints)
        sym = (
            cls.infer_symmetry_pairs(keypoints) if infer_symmetry and keypoints else []
        )
        return cls(
            version="1.0",
            keypoints=keypoints,
            edges=edges,
            symmetry_pairs=sym,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "version": self.version,
            "keypoints": list(self.keypoints),
            "edges": [[a, b] for a, b in self.edges],
            "flip_idx": list(self.flip_idx) if self.flip_idx is not None else None,
            "symmetry_pairs": [[a, b] for a, b in self.symmetry_pairs],
            "instances": list(self.instances),
            "instance_separator": self.instance_separator,
        }
        if isinstance(self.sequencer, dict) and self.sequencer:
            payload["sequencer"] = dict(self.sequencer)
        return payload

    def instance_prefix(self, instance: str) -> str:
        sep = self.instance_separator or "_"
        inst = str(instance or "").strip().rstrip(sep).rstrip("_")
        return f"{inst}{sep}" if inst else ""

    def expand_keypoints(self) -> List[str]:
        """Return keypoint names expanded with instance prefixes when configured."""
        if not self.instances:
            return list(self.keypoints)
        if not self.keypoints:
            return []
        expanded: List[str] = []
        for inst in self.instances:
            prefix = self.instance_prefix(inst)
            if not prefix:
                continue
            for kp in self.keypoints:
                kp_clean = str(kp or "").strip()
                if kp_clean:
                    expanded.append(f"{prefix}{kp_clean}")
        return expanded

    def strip_instance_prefix(self, label: str) -> Tuple[Optional[str], str]:
        """Return (instance, base_keypoint) if the label matches a configured instance prefix.

        Checks longest configured instance prefix first so prefixes such as
        ``"mouse_1"`` take precedence over ``"mouse"``.
        """
        sep = self.instance_separator or "_"
        text = str(label or "").strip()
        if not text:
            return None, ""
        if self.instances:
            ordered_instances = sorted(
                (
                    str(inst or "").strip()
                    for inst in self.instances
                    if str(inst or "").strip()
                ),
                key=lambda item: len(self.instance_prefix(item)),
                reverse=True,
            )
            lower_text = text.lower()
            for inst in ordered_instances:
                prefix = self.instance_prefix(inst)
                if prefix and lower_text.startswith(prefix.lower()):
                    remainder = text[len(prefix) :].lstrip(sep).strip()
                    if remainder:
                        return inst, remainder
            # Instances are configured but no known prefix matched: treat as base label.
            return None, text
        return None, text

    def base_edges(self) -> List[Tuple[str, str]]:
        """Return deduplicated undirected edges stripped to base keypoint names."""
        stripped: List[Tuple[str, str]] = []
        for raw_a, raw_b in list(self.edges or []):
            _inst_a, a = self.strip_instance_prefix(str(raw_a or ""))
            _inst_b, b = self.strip_instance_prefix(str(raw_b or ""))
            if a and b and a != b:
                stripped.append((a, b))
        return dedup_undirected_pairs(stripped)

    def base_symmetry_pairs(self) -> List[Tuple[str, str]]:
        """Return deduplicated symmetry pairs stripped to base keypoint names."""
        stripped: List[Tuple[str, str]] = []
        for raw_a, raw_b in list(self.symmetry_pairs or []):
            _inst_a, a = self.strip_instance_prefix(str(raw_a or ""))
            _inst_b, b = self.strip_instance_prefix(str(raw_b or ""))
            if a and b and a != b:
                stripped.append((a, b))
        return dedup_undirected_pairs(stripped)

    def to_coco_skeleton(
        self,
        keypoint_order: Optional[Sequence[str]] = None,
    ) -> List[List[int]]:
        """Convert schema edges to COCO 1-indexed ``[[i, j], ...]`` bone pairs."""
        order = [
            str(name).strip()
            for name in (
                keypoint_order if keypoint_order is not None else self.keypoints
            )
            if str(name).strip()
        ]
        if not order or not self.edges:
            return []
        idx_by_name = {name: i + 1 for i, name in enumerate(order)}
        idx_by_lower = {name.lower(): i + 1 for i, name in enumerate(order)}

        def _lookup(raw: str) -> Optional[int]:
            text = str(raw or "").strip()
            if not text:
                return None
            if text in idx_by_name:
                return idx_by_name[text]
            _inst, base = self.strip_instance_prefix(text)
            if base in idx_by_name:
                return idx_by_name[base]
            if text.lower() in idx_by_lower:
                return idx_by_lower[text.lower()]
            if base.lower() in idx_by_lower:
                return idx_by_lower[base.lower()]
            return None

        skeleton: List[List[int]] = []
        seen: set[Tuple[int, int]] = set()
        for raw_a, raw_b in self.edges:
            ia = _lookup(raw_a)
            ib = _lookup(raw_b)
            if ia is None or ib is None or ia == ib:
                continue
            ukey = (ia, ib) if ia < ib else (ib, ia)
            if ukey in seen:
                continue
            seen.add(ukey)
            skeleton.append([ia, ib])
        return skeleton

    def validate(self) -> List[str]:
        """Return a list of schema consistency warnings (empty when fully valid)."""
        issues: List[str] = []
        seen_kps: set[str] = set()
        for kp in self.keypoints:
            name = _clean_name(kp)
            if not name:
                issues.append("Schema contains an empty keypoint name.")
                continue
            if name in seen_kps:
                issues.append(f"Duplicate keypoint name: {name!r}.")
            seen_kps.add(name)

        valid_kps = set(seen_kps)
        seen_edges: set[Tuple[str, str]] = set()
        for raw_a, raw_b in self.edges:
            _ia, a = self.strip_instance_prefix(str(raw_a or ""))
            _ib, b = self.strip_instance_prefix(str(raw_b or ""))
            if not a or not b:
                issues.append(f"Edge has empty endpoint: ({raw_a!r}, {raw_b!r}).")
                continue
            if a == b:
                issues.append(f"Self-loop edge detected on keypoint {a!r}.")
                continue
            if valid_kps:
                if a not in valid_kps:
                    issues.append(
                        f"Edge ({raw_a!r}, {raw_b!r}) references unknown keypoint {a!r}."
                    )
                if b not in valid_kps:
                    issues.append(
                        f"Edge ({raw_a!r}, {raw_b!r}) references unknown keypoint {b!r}."
                    )
            ukey = (a, b) if a < b else (b, a)
            if ukey in seen_edges:
                issues.append(f"Duplicate skeleton edge: ({a!r}, {b!r}).")
            seen_edges.add(ukey)

        for raw_a, raw_b in self.symmetry_pairs:
            _ia, a = self.strip_instance_prefix(str(raw_a or ""))
            _ib, b = self.strip_instance_prefix(str(raw_b or ""))
            if not a or not b or a == b:
                issues.append(f"Invalid symmetry pair: ({raw_a!r}, {raw_b!r}).")
                continue
            if valid_kps:
                if a not in valid_kps:
                    issues.append(
                        f"Symmetry pair ({raw_a!r}, {raw_b!r}) references unknown keypoint {a!r}."
                    )
                if b not in valid_kps:
                    issues.append(
                        f"Symmetry pair ({raw_a!r}, {raw_b!r}) references unknown keypoint {b!r}."
                    )

        if self.flip_idx is not None:
            expected_lengths = {len(self.keypoints)}
            expanded = self.expand_keypoints()
            if expanded:
                expected_lengths.add(len(expanded))
            n = len(self.flip_idx)
            if n not in expected_lengths:
                issues.append(
                    f"flip_idx length ({n}) does not match keypoints count ({len(self.keypoints)})."
                )
            elif any(not isinstance(v, int) or v < 0 or v >= n for v in self.flip_idx):
                issues.append("flip_idx contains out-of-range index values.")
            elif any(self.flip_idx[self.flip_idx[i]] != i for i in range(n)):
                issues.append("flip_idx is not symmetric (flip_idx[flip_idx[i]] != i).")

        return issues

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
                    "PyYAML is required to load pose schema YAML files."
                ) from exc
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
                    "PyYAML is required to save pose schema YAML files."
                ) from exc
            p.write_text(
                yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8"
            )
            return p
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    def compute_flip_idx(
        self, keypoint_order: Optional[Sequence[str]] = None
    ) -> Optional[List[int]]:
        order = list(keypoint_order or self.keypoints)
        if not order:
            return None
        if self.flip_idx is not None and len(self.flip_idx) == len(order):
            if keypoint_order is None or order == list(self.keypoints):
                return list(self.flip_idx)
            # Remap explicit flip_idx when keypoint_order is a permutation of self.keypoints.
            if len(self.keypoints) == len(order) and set(order) == set(self.keypoints):
                self_index = {name: idx for idx, name in enumerate(self.keypoints)}
                order_index = {name: idx for idx, name in enumerate(order)}
                remapped: List[int] = []
                valid = True
                for name in order:
                    src_i = self_index[name]
                    target_i = self.flip_idx[src_i]
                    if not (0 <= target_i < len(self.keypoints)):
                        valid = False
                        break
                    partner_name = self.keypoints[target_i]
                    remapped.append(order_index[partner_name])
                if valid:
                    return remapped

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
            for inst in self.instances:
                prefix = self.instance_prefix(inst)
                if not prefix:
                    continue
                for left, right in self.symmetry_pairs:
                    apply_pair(f"{prefix}{left}", f"{prefix}{right}")

        return flip

    def normalize_prefixed_keypoints(self) -> None:
        """Convert prefixed keypoints/edges into base+instances when possible."""
        sep = self.instance_separator or "_"
        if not self.keypoints:
            return

        anatomical_prefix_tokens = {
            "left",
            "right",
            "l",
            "r",
            "lt",
            "rt",
            "fore",
            "hind",
            "front",
            "back",
            "rear",
            "upper",
            "lower",
            "mid",
            "middle",
            "top",
            "bottom",
            "prox",
            "dist",
            "proximal",
            "distal",
            "anterior",
            "posterior",
            "dorsal",
            "ventral",
        }

        original_keypoints = list(self.keypoints)
        original_flip = list(self.flip_idx) if isinstance(self.flip_idx, list) else None
        inferred_instances: List[str] = []
        base_keypoints: List[str] = []
        seen_base: set[str] = set()
        for name in self.keypoints:
            text = str(name or "").strip()
            if sep not in text:
                continue
            inst, base = text.split(sep, 1)
            inst = inst.strip().rstrip(sep).rstrip("_")
            base = base.strip()
            if inst and inst not in inferred_instances:
                inferred_instances.append(inst)
            if base and base not in seen_base:
                seen_base.add(base)
                base_keypoints.append(base)

        if not inferred_instances or not base_keypoints:
            return

        # Guard against splitting unprefixed bilateral anatomical schemas such as
        # ["left_ear", "right_ear", "left_hip", "right_hip"] into instances=["left", "right"].
        if all(inst.lower() in anatomical_prefix_tokens for inst in inferred_instances):
            return

        # Only normalize when the keypoint list *looks like* a fully-expanded
        # instance-prefixed schema (e.g. intruder_nose/resident_nose/...).
        # This avoids breaking base keypoint names that legitimately contain
        # underscores (e.g. tail_base, left_ear).
        if not all(sep in str(kp or "") for kp in original_keypoints):
            return
        inst_count = len(inferred_instances)
        base_count = len(base_keypoints)
        if inst_count <= 0 or base_count <= 0:
            return
        if inst_count * base_count != len(original_keypoints):
            return

        self.instances = inferred_instances
        self.keypoints = base_keypoints

        # Preserve a usable flip_idx when the original was provided for an expanded order.
        if original_flip and len(original_flip) == len(original_keypoints):
            if (
                inst_count > 0
                and base_count > 0
                and inst_count * base_count == len(original_keypoints)
            ):
                segments: List[List[int]] = []
                valid = True
                for seg_idx in range(inst_count):
                    start = seg_idx * base_count
                    end = start + base_count
                    segment = original_flip[start:end]
                    if len(segment) != base_count:
                        valid = False
                        break
                    # Only accept when all mappings stay within the segment.
                    rel: List[int] = []
                    for v in segment:
                        if not isinstance(v, int):
                            valid = False
                            break
                        if v < start or v >= end:
                            valid = False
                            break
                        rel.append(v - start)
                    if not valid:
                        break
                    segments.append(rel)
                if (
                    valid
                    and segments
                    and all(seg == segments[0] for seg in segments[1:])
                ):
                    self.flip_idx = segments[0]
                else:
                    # Drop inconsistent flip indices so compute_flip_idx can fall back to symmetry_pairs.
                    self.flip_idx = None

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
        self.edges = dedup_undirected_pairs(new_edges)

        new_pairs: List[Tuple[str, str]] = []
        for pair in self.symmetry_pairs:
            stripped = strip_pair(pair)
            if stripped:
                new_pairs.append(stripped)
        self.symmetry_pairs = dedup_undirected_pairs(new_pairs)

    @staticmethod
    def infer_edges(
        keypoints: Iterable[str],
        *,
        preset: Optional[str] = "auto",
    ) -> List[Tuple[str, str]]:
        """Infer anatomical skeleton edges from keypoint names."""
        return infer_pose_skeleton_edges(keypoints, preset=preset)

    @staticmethod
    def infer_symmetry_pairs(keypoints: Iterable[str]) -> List[Tuple[str, str]]:
        """Best-effort pairing based on boundary-aware left/right naming patterns."""
        names = [str(k).strip() for k in keypoints if str(k).strip()]
        lower_to_original: Dict[str, str] = {}
        for name in names:
            lower_to_original.setdefault(name.lower(), name)
        visited: set[str] = set()
        pairs: List[Tuple[str, str]] = []

        regex_swaps: Tuple[Tuple[re.Pattern[str], str], ...] = (
            (re.compile(r"left"), "right"),
            (re.compile(r"right"), "left"),
            (re.compile(r"(^|[\s_:\-.])lt(?=[\s_:\-.]|$)"), r"\1rt"),
            (re.compile(r"(^|[\s_:\-.])rt(?=[\s_:\-.]|$)"), r"\1lt"),
            (re.compile(r"(^|[\s_:\-.])l(?=[\s_:\-.]|$)"), r"\1r"),
            (re.compile(r"(^|[\s_:\-.])r(?=[\s_:\-.]|$)"), r"\1l"),
        )

        for name in names:
            key = name.lower()
            if key in visited:
                continue
            other_key: Optional[str] = None
            for pattern, repl in regex_swaps:
                if not pattern.search(key):
                    continue
                candidate = pattern.sub(repl, key)
                if candidate != key and candidate in lower_to_original:
                    other_key = candidate
                    break
            # Handle camelCase single-letter prefix such as LEar / REar.
            if other_key is None and len(name) >= 2 and name[1].isupper():
                if name[0] == "L":
                    cand = ("R" + name[1:]).lower()
                    if cand in lower_to_original:
                        other_key = cand
                elif name[0] == "R":
                    cand = ("L" + name[1:]).lower()
                    if cand in lower_to_original:
                        other_key = cand

            if other_key and other_key not in visited:
                pairs.append((name, lower_to_original[other_key]))
                visited.add(key)
                visited.add(other_key)
        return pairs
