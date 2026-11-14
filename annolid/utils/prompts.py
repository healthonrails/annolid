"""Text-prompt parsing, label alias mapping, and candidate box filtering utilities."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


# Split multi-phrase prompts on commas, semicolons, newlines, or phrase-separating
# periods while preserving decimal numbers such as ``0.25``.
_PROMPT_PHRASE_SPLIT_RE = re.compile(r"[,;\n]+|(?<!\d)\.(?!\d)")
_WHITESPACE_RE = re.compile(r"\s+")
_WORD_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_ALIAS_ARROW_RE = re.compile(r"\s*->\s*")
_EVERY_TOKEN_RE = re.compile(
    r"(?:^|\b)every(?:\s*[=:]?\s*(\d+))?(?:\b|$)|(?:^|\b)(\d+)\s+every(?:\b|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PromptPhrase:
    """One visual text query and its target annotation label."""

    query: str
    label: str


@dataclass(frozen=True)
class ParsedTextPrompt:
    """Normalized representation of a user-entered segmentation text prompt."""

    raw: str
    phrases: tuple[PromptPhrase, ...]

    @property
    def is_empty(self) -> bool:
        return len(self.phrases) == 0

    @property
    def queries(self) -> tuple[str, ...]:
        return tuple(phrase.query for phrase in self.phrases)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(phrase.label for phrase in self.phrases)


@dataclass(frozen=True)
class DetectionBoxCandidate:
    """Sanitized bounding-box detection candidate prior to mask decoding."""

    bbox: tuple[float, float, float, float]
    label: str
    score: float
    source: str = "detector"


def _collapse_whitespace(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "")).strip()


def extract_number_and_remove_digits(input_string: str) -> tuple[int, str]:
    """Extract concatenated digits and return ``(number, non_digit_text)``.

    Preserved for backward compatibility with callers importing this helper
    directly. New code should prefer :func:`parse_every_prompt`, which preserves
    digits inside target label names (e.g. ``every 32 mouse_1``).
    """
    extracted_number = 0
    result_without_digits = ""

    for char in str(input_string or ""):
        if char.isdigit():
            extracted_number = extracted_number * 10 + int(char)
        else:
            result_without_digits += char

    return extracted_number, result_without_digits


def parse_every_prompt(
    prompt: str | None,
    *,
    default_points_per_side: int = 32,
    min_points_per_side: int = 1,
    max_points_per_side: int = 100,
) -> tuple[int, str] | None:
    """Parse a segment-everything prompt such as ``every 32 cell``.

    Returns ``(points_per_side, target_label)`` when the prompt invokes the
    ``every`` directive, or ``None`` for standard object prompts. Digits inside
    the target label (such as ``every 48 mouse_1``) are preserved intact.
    """
    text = _collapse_whitespace(str(prompt or ""))
    if not text:
        return None

    match = _EVERY_TOKEN_RE.search(text)
    if match is None:
        return None

    number_token = match.group(1) or match.group(2)
    remainder = (text[: match.start()] + " " + text[match.end() :]).strip(" ,;.\t\n")
    remainder = _collapse_whitespace(remainder)

    # Support alias mapping on ``every`` prompts as well (e.g. ``every 32 = cell``).
    if remainder.startswith(("=", "->")):
        remainder = remainder.lstrip("=-> ").strip()
    elif "->" in remainder or "=" in remainder:
        parsed_remainder = parse_text_prompt(remainder)
        if parsed_remainder.phrases:
            remainder = parsed_remainder.phrases[0].label

    points_per_side = int(default_points_per_side)
    if number_token is not None:
        try:
            candidate = int(number_token)
            if min_points_per_side <= candidate <= max_points_per_side:
                points_per_side = candidate
        except ValueError:
            points_per_side = int(default_points_per_side)

    label = remainder or "object"
    return points_per_side, label


def _parse_single_phrase(segment: str) -> PromptPhrase | None:
    cleaned = _collapse_whitespace(segment)
    if not cleaned:
        return None

    query = cleaned
    label = cleaned

    if "->" in cleaned:
        parts = _ALIAS_ARROW_RE.split(cleaned, maxsplit=1)
        lhs = _collapse_whitespace(parts[0])
        rhs = _collapse_whitespace(parts[1]) if len(parts) > 1 else ""
        if lhs and rhs:
            query, label = lhs, rhs
        elif lhs or rhs:
            query = label = lhs or rhs
    elif "=" in cleaned:
        lhs_raw, rhs_raw = cleaned.split("=", 1)
        lhs = _collapse_whitespace(lhs_raw)
        rhs = _collapse_whitespace(rhs_raw)
        if lhs and rhs:
            query, label = lhs, rhs
        elif lhs or rhs:
            query = label = lhs or rhs

    if not query:
        return None
    return PromptPhrase(query=query, label=label or query)


def parse_text_prompt(text_prompt: str | None) -> ParsedTextPrompt:
    """Parse a raw prompt string into deduplicated visual queries and target labels.

    Supports:
    - Multi-class prompts separated by ``,``, ``;``, ``.``, or newlines
      (e.g. ``"black mouse, white mouse; water spout"``).
    - Canonical label aliases via ``query=label`` or ``query -> label``
      (e.g. ``"black mouse -> mouse_black, white mouse -> mouse_white"``).
    - Case-insensitive query deduplication while preserving first-seen casing.
    """
    raw = str(text_prompt or "")
    if not raw.strip():
        return ParsedTextPrompt(raw=raw, phrases=())

    seen_queries: set[str] = set()
    phrases: list[PromptPhrase] = []
    for chunk in _PROMPT_PHRASE_SPLIT_RE.split(raw):
        phrase = _parse_single_phrase(chunk)
        if phrase is None:
            continue
        key = phrase.query.casefold()
        if key in seen_queries:
            continue
        seen_queries.add(key)
        phrases.append(phrase)

    return ParsedTextPrompt(raw=raw, phrases=tuple(phrases))


def format_grounding_dino_caption(
    text_prompt: str | ParsedTextPrompt | None,
) -> str:
    """Format a prompt into GroundingDINO's period-delimited caption convention.

    GroundingDINO isolates phrase self-attention masks on BERT's ``.`` token
    (token id 1012), not commas. Joining distinct phrases as ``"a . b ."``
    prevents cross-phrase attention leakage for comma-separated prompts.
    """
    parsed = (
        text_prompt
        if isinstance(text_prompt, ParsedTextPrompt)
        else parse_text_prompt(text_prompt)
    )
    if parsed.is_empty:
        fallback = _collapse_whitespace(str(text_prompt or "")).lower()
        if not fallback:
            return ""
        return fallback if fallback.endswith(".") else f"{fallback}."

    queries = [phrase.query.lower().strip(" .") for phrase in parsed.phrases]
    queries = [q for q in queries if q]
    if not queries:
        return ""
    return " . ".join(queries) + " ."


def _word_tokens(text: str) -> list[str]:
    return _WORD_TOKEN_RE.findall(str(text or "").casefold())


def match_predicted_phrase_to_label(
    predicted_phrase: str | None,
    parsed: ParsedTextPrompt | str | None,
    *,
    fallback_label: str = "object",
) -> str:
    """Map a detector's decoded token phrase back to the target annotation label."""
    parsed_prompt = (
        parsed if isinstance(parsed, ParsedTextPrompt) else parse_text_prompt(parsed)
    )
    cleaned_pred = _collapse_whitespace(str(predicted_phrase or "")).strip(" .,;:")

    if parsed_prompt.is_empty:
        return cleaned_pred or fallback_label

    if len(parsed_prompt.phrases) == 1:
        return parsed_prompt.phrases[0].label

    if not cleaned_pred:
        return parsed_prompt.phrases[0].label

    pred_folded = cleaned_pred.casefold()
    for phrase in parsed_prompt.phrases:
        if (
            pred_folded == phrase.query.casefold()
            or pred_folded == phrase.label.casefold()
        ):
            return phrase.label

    pred_tokens = _word_tokens(cleaned_pred)
    pred_token_set = set(pred_tokens)
    if pred_token_set:
        best_label: str | None = None
        best_rank: tuple[float, float, int, int] | None = None
        for index, phrase in enumerate(parsed_prompt.phrases):
            query_tokens = _word_tokens(phrase.query)
            if not query_tokens:
                continue
            query_set = set(query_tokens)
            shared = len(pred_token_set & query_set)
            if shared <= 0:
                continue
            precision = shared / len(pred_token_set)
            recall = shared / len(query_set)
            f1 = (
                (2.0 * precision * recall / (precision + recall))
                if (precision + recall) > 0.0
                else 0.0
            )
            exact_substring = int(
                phrase.query.casefold() in pred_folded
                or pred_folded in phrase.query.casefold()
            )
            # Prefer higher token F1, exact phrase substring match, more specific
            # query match, and earlier user prompt order.
            rank = (f1, float(exact_substring), shared, -index)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_label = phrase.label
        if best_label is not None:
            return best_label

    return cleaned_pred or parsed_prompt.phrases[0].label


def box_iou_xyxy(
    box_a: Sequence[float],
    box_b: Sequence[float],
) -> float:
    """Compute 2D IoU between two ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = (
        float(box_a[0]),
        float(box_a[1]),
        float(box_a[2]),
        float(box_a[3]),
    )
    bx1, by1, bx2, by2 = (
        float(box_b[0]),
        float(box_b[1]),
        float(box_b[2]),
        float(box_b[3]),
    )

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def sanitize_and_nms_box_candidates(
    candidates: Sequence[DetectionBoxCandidate],
    *,
    image_shape: tuple[int, int] | None = None,
    min_box_size: float = 2.0,
    min_score: float | None = None,
    same_label_iou_threshold: float = 0.50,
    cross_label_iou_threshold: float | None = 0.88,
    max_detections: int | None = None,
) -> list[DetectionBoxCandidate]:
    """Clamp, validate, score-filter, and NMS-deduplicate candidate boxes.

    Deterministic ordering is guaranteed: candidates are ranked by descending
    confidence score with stable input order breaking ties, then returned in
    priority order so downstream SAM decoding and label numbering are stable.
    """
    if not candidates:
        return []

    height: float | None = None
    width: float | None = None
    if image_shape is not None and len(image_shape) >= 2:
        height = max(0.0, float(image_shape[0]))
        width = max(0.0, float(image_shape[1]))

    valid: list[tuple[int, DetectionBoxCandidate]] = []
    for idx, cand in enumerate(candidates):
        try:
            x1, y1, x2, y2 = (
                float(cand.bbox[0]),
                float(cand.bbox[1]),
                float(cand.bbox[2]),
                float(cand.bbox[3]),
            )
            score = float(cand.score)
        except (IndexError, TypeError, ValueError):
            continue

        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            continue
        if not math.isfinite(score):
            score = 0.0
        if min_score is not None and score < float(min_score):
            continue

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        if width is not None and height is not None:
            x1 = min(max(0.0, x1), width)
            x2 = min(max(0.0, x2), width)
            y1 = min(max(0.0, y1), height)
            y2 = min(max(0.0, y2), height)

        if (x2 - x1) < float(min_box_size) or (y2 - y1) < float(min_box_size):
            continue

        label = _collapse_whitespace(cand.label) or "object"
        valid.append(
            (
                idx,
                DetectionBoxCandidate(
                    bbox=(x1, y1, x2, y2),
                    label=label,
                    score=score,
                    source=str(cand.source or "detector"),
                ),
            )
        )

    if not valid:
        return []

    valid.sort(key=lambda item: (-item[1].score, item[0]))

    same_iou = float(same_label_iou_threshold)
    cross_iou = (
        float(cross_label_iou_threshold)
        if cross_label_iou_threshold is not None
        else None
    )

    kept: list[DetectionBoxCandidate] = []
    for _orig_idx, candidate in valid:
        cand_label_key = candidate.label.casefold()
        suppressed = False
        for existing in kept:
            iou = box_iou_xyxy(candidate.bbox, existing.bbox)
            if existing.label.casefold() == cand_label_key:
                if iou >= same_iou:
                    suppressed = True
                    break
            elif cross_iou is not None and iou >= cross_iou:
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)
            if max_detections is not None and len(kept) >= int(max_detections):
                break

    return kept


def assign_instance_labels(
    base_labels: Sequence[str],
    *,
    use_group_id: bool = False,
) -> list[tuple[str, int | None]]:
    """Assign deterministic per-class instance labels or ``group_id`` values.

    When ``use_group_id=False`` (the legacy canvas default), each class receives
    independent ``0``-based suffixes: ``("mouse_0", None), ("mouse_1", None),
    ("pup_0", None)``.
    When ``use_group_id=True``, base class names are preserved and ``1``-based
    per-class ``group_id`` integers are assigned: ``("mouse", 1), ("mouse", 2),
    ("pup", 1)``.
    """
    counts: dict[str, int] = {}
    assigned: list[tuple[str, int | None]] = []
    for raw_label in base_labels:
        clean = _collapse_whitespace(str(raw_label or "")) or "object"
        key = clean.casefold()
        idx = counts.get(key, 0)
        counts[key] = idx + 1
        if use_group_id:
            assigned.append((clean, idx + 1))
        else:
            assigned.append((f"{clean}_{idx}", None))
    return assigned


def is_sliver_mask_remnant(
    exclusive_mask: np.ndarray,
    original_mask: np.ndarray | None = None,
    *,
    min_pixels: int = 9,
    min_retained_fraction: float = 0.12,
) -> bool:
    """Return True when pixel-exclusivity reduced a duplicate mask to a tiny sliver."""
    excl_bool = np.asarray(exclusive_mask, dtype=bool)
    excl_area = int(np.count_nonzero(excl_bool))
    if excl_area <= 0:
        return True
    if original_mask is None:
        return False

    orig_area = int(np.count_nonzero(np.asarray(original_mask, dtype=bool)))
    if orig_area <= 0:
        return True
    # Genuinely tiny objects that were not heavily clipped by a higher-priority
    # mask are kept; only masks that BOTH shrank drastically AND left < min_pixels
    # (or tiny retained fraction on larger masks) are rejected as duplicate rims.
    retained_ratio = excl_area / float(orig_area)
    if orig_area >= max(min_pixels * 2, 16) and retained_ratio < float(
        min_retained_fraction
    ):
        return True
    if excl_area < int(min_pixels) and retained_ratio < 0.5:
        return True
    return False
