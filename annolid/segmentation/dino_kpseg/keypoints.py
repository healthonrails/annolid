from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


def infer_flip_idx_from_names(
    names: Sequence[str], *, kpt_count: int
) -> Optional[List[int]]:
    """Infer YOLO-style flip_idx from keypoint names.

    Returns a permutation list of length kpt_count mapping index -> its mirrored counterpart.
    """
    if not names or int(kpt_count) <= 0:
        return None
    if len(names) != int(kpt_count):
        return None

    lowered = [str(n).strip().lower() for n in names]
    index_by_name = {n: i for i, n in enumerate(lowered) if n}
    if not index_by_name:
        return None

    mapping = list(range(int(kpt_count)))
    swapped_any = False

    def counterpart(name: str) -> Optional[str]:
        if "left" in name:
            return name.replace("left", "right", 1)
        if "right" in name:
            return name.replace("right", "left", 1)
        if name.startswith("l_"):
            return "r_" + name[2:]
        if name.startswith("r_"):
            return "l_" + name[2:]
        if name.startswith("l-"):
            return "r-" + name[2:]
        if name.startswith("r-"):
            return "l-" + name[2:]
        return None

    for i, name in enumerate(lowered):
        other = counterpart(name)
        if not other:
            continue
        j = index_by_name.get(other)
        if j is None or j == i:
            continue
        mapping[i] = j
        swapped_any = True

    if not swapped_any:
        return None
    if sorted(mapping) != list(range(int(kpt_count))):
        return None
    return mapping


def symmetric_pairs_from_flip_idx(flip_idx: Sequence[int]) -> List[Tuple[int, int]]:
    """Return unique symmetric (i, j) pairs where flip_idx[i] == j and flip_idx[j] == i."""
    pairs: List[Tuple[int, int]] = []
    used = set()
    for i, j in enumerate(list(flip_idx)):
        try:
            j = int(j)
        except Exception:
            continue
        if j == i:
            continue
        if j < 0 or j >= len(flip_idx):
            continue
        if int(flip_idx[j]) != i:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in used:
            continue
        used.add((a, b))
        pairs.append((a, b))
    return pairs


def infer_left_right_pairs(
    keypoint_names: Sequence[str],
    *,
    flip_idx: Sequence[int],
) -> List[Tuple[int, int]]:
    """Infer directed (left_idx, right_idx) pairs using keypoint names + flip_idx."""
    names = [str(n or "").strip().lower() for n in keypoint_names]
    pairs = symmetric_pairs_from_flip_idx(flip_idx)
    out: List[Tuple[int, int]] = []

    def is_left(name: str) -> bool:
        return (
            "left" in name
            or name.startswith("l_")
            or name.startswith("l-")
            or name.startswith("l.")
            or name.startswith("lf")
        )

    def is_right(name: str) -> bool:
        return (
            "right" in name
            or name.startswith("r_")
            or name.startswith("r-")
            or name.startswith("r.")
            or name.startswith("rt")
        )

    for i, j in pairs:
        if i >= len(names) or j >= len(names):
            continue
        ni, nj = names[i], names[j]
        if is_left(ni) and is_right(nj):
            out.append((i, j))
        elif is_left(nj) and is_right(ni):
            out.append((j, i))
    return out


def infer_orientation_anchor_indices(
    keypoint_names: Sequence[str],
    *,
    max_anchors: int = 4,
) -> List[int]:
    """Pick asymmetric keypoints that define body orientation (e.g., nose, head, tailbase).

    These anchors can provide a stable orientation cue to help disambiguate left/right parts.
    """
    names = [str(n or "").strip().lower() for n in keypoint_names]
    if not names:
        return []

    # Prefer strong, asymmetric anchors first.
    priority_tokens = (
        "nose",
        "snout",
        "head",
        "tailbase",
        "tail_base",
        "tail-base",
        "tailroot",
        "tail_root",
        "tail-root",
        "rump",
        "spine",
        "center",
        "midline",
        "body",
        "base",
    )

    def is_symmetric(name: str) -> bool:
        return (
            "left" in name
            or "right" in name
            or name.startswith(("l_", "r_", "l-", "r-", "l.", "r."))
        )

    anchors: List[int] = []
    for token in priority_tokens:
        for idx, name in enumerate(names):
            if idx in anchors:
                continue
            if not name or is_symmetric(name):
                continue
            if token in name:
                anchors.append(idx)
                if len(anchors) >= int(max_anchors):
                    return anchors
    return anchors


@dataclass
class LRStabilizeConfig:
    """Heuristic left/right identity stabilizer for symmetric keypoints."""

    enabled: bool = True
    min_improvement_px: float = 1.0
    min_score: float = 0.0


def stabilize_symmetric_keypoints_xy(
    prev_xy: Sequence[Tuple[float, float]],
    curr_xy: List[Tuple[float, float]],
    *,
    pairs: Iterable[Tuple[int, int]],
    prev_scores: Optional[Sequence[float]] = None,
    curr_scores: Optional[Sequence[float]] = None,
    cfg: Optional[LRStabilizeConfig] = None,
) -> List[Tuple[float, float]]:
    """Swap symmetric keypoint pairs to minimize frame-to-frame displacement.

    This is useful when a model occasionally swaps left/right identities (e.g., ears).
    """
    config = cfg or LRStabilizeConfig()
    if not config.enabled:
        return curr_xy
    if len(prev_xy) != len(curr_xy):
        return curr_xy

    out = list(curr_xy)

    def score_ok(i: int) -> bool:
        if config.min_score <= 0:
            return True
        if prev_scores is not None:
            try:
                if float(prev_scores[i]) < float(config.min_score):
                    return False
            except Exception:
                pass
        if curr_scores is not None:
            try:
                if float(curr_scores[i]) < float(config.min_score):
                    return False
            except Exception:
                pass
        return True

    for i, j in pairs:
        if i >= len(out) or j >= len(out):
            continue
        if not (score_ok(i) and score_ok(j)):
            continue

        pi, pj = prev_xy[i], prev_xy[j]
        ci, cj = out[i], out[j]

        d_same = (
            (ci[0] - pi[0]) ** 2
            + (ci[1] - pi[1]) ** 2
            + (cj[0] - pj[0]) ** 2
            + (cj[1] - pj[1]) ** 2
        )
        d_swap = (
            (cj[0] - pi[0]) ** 2
            + (cj[1] - pi[1]) ** 2
            + (ci[0] - pj[0]) ** 2
            + (ci[1] - pj[1]) ** 2
        )

        if d_swap + float(config.min_improvement_px) ** 2 < d_same:
            out[i], out[j] = out[j], out[i]

    return out
