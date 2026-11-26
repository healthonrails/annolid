"""Framework-agnostic speech segmentation, keyword matching, and highlight timing.

Shared by PDF.js mode, PyMuPDF fallback reading mode, and future web/document
speech readers. Contains zero Qt or optional ML/PDF dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

_SENTENCE_END_CHARS = frozenset(".!?。！？")
_QUOTE_CLOSER_CHARS = frozenset("”’\"'")
_CLOSER_CHARS = frozenset("”’\"')]）】》」』〉")

# Common scientific, bibliographic, and prose abbreviations whose trailing dot
# must not trigger a sentence break.
_NON_BREAKING_ABBREVS = frozenset(
    {
        "al",
        "approx",
        "art",
        "ca",
        "cf",
        "ch",
        "chap",
        "co",
        "corp",
        "dept",
        "dr",
        "ed",
        "eds",
        "eg",
        "eq",
        "eqs",
        "etc",
        "fig",
        "figs",
        "ie",
        "inc",
        "jr",
        "ltd",
        "mr",
        "mrs",
        "ms",
        "no",
        "nos",
        "p",
        "pp",
        "prof",
        "ref",
        "refs",
        "rev",
        "sec",
        "secs",
        "sr",
        "st",
        "supp",
        "tab",
        "vol",
        "vols",
        "vs",
    }
)

_STOPWORDS = frozenset(
    {
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "also",
        "among",
        "and",
        "any",
        "are",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "can",
        "could",
        "did",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "into",
        "its",
        "itself",
        "just",
        "may",
        "might",
        "more",
        "most",
        "much",
        "must",
        "nor",
        "not",
        "now",
        "off",
        "once",
        "only",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "show",
        "shown",
        "shows",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "under",
        "until",
        "used",
        "using",
        "very",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "within",
        "without",
        "would",
        "your",
        "yours",
    }
)

_CJK_LEAD_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_CJK_LIKE_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af"
    r"，。！？、；：“”‘’（）《》「」『』]"
)
_INLINE_HYPHEN_WRAP_RE = re.compile(
    r"([A-Za-z]{2,})[-\u2010\u2011\u2012\u2013]\s+([a-z]{2,})"
)
_CLOSER_SPACE_RE = re.compile(r"([.!?。！？])\s+([”’\"'\)\]\}）】》」』〉])")
_WORD_TOKEN_RE = re.compile(
    r"[A-Za-z0-9\u00c0-\u024f\u0400-\u04ff]+(?:[-'\u2019][A-Za-z0-9\u00c0-\u024f\u0400-\u04ff]+)*"
    r"|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]"
)
_MAJOR_PAUSE_CHARS = frozenset(".!?。！？;；:：")
_MINOR_PAUSE_CHARS = frozenset(",，、—–-)]）】》")


@dataclass(frozen=True)
class WordToken:
    """A single speakable word/character token with source offsets and timing weight."""

    start: int
    end: int
    text: str
    weight: float
    is_keyword: bool = False
    pause_strength: float = 0.0


@dataclass(frozen=True)
class KeywordMatch:
    """Character span of a matched keyword inside a text segment."""

    start: int
    end: int
    keyword: str


@dataclass(frozen=True)
class TimedHighlightUnit:
    """A unit scheduled for highlighting within a spoken audio chunk."""

    unit: object
    start_ms: int
    end_ms: int
    duration_ms: int
    is_keyword: bool = False


def normalize_speech_text(text: str) -> str:
    """Normalize PDF/DOM text for clean TTS synthesis without losing natural cadence."""
    cleaned = str(text or "").replace("\u2029", "\n").replace("\u00ad", "")
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = _INLINE_HYPHEN_WRAP_RE.sub(r"\1\2", cleaned)
    cleaned = _CLOSER_SPACE_RE.sub(r"\1\2", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _is_cjk_char(ch: str) -> bool:
    return bool(ch and _CJK_LEAD_RE.match(ch))


def _is_narration_lead(ch: str) -> bool:
    return _is_cjk_char(ch) or ch in {"（", "【", "“", "‘"}


def _is_non_breaking_period(text: str, dot_index: int) -> bool:
    """Return True if ``text[dot_index] == '.'`` should NOT end a sentence."""
    if dot_index < 0 or dot_index >= len(text) or text[dot_index] != ".":
        return False

    prev_ch = text[dot_index - 1] if dot_index > 0 else ""
    next_ch = text[dot_index + 1] if dot_index + 1 < len(text) else ""

    # Decimal numbers / version numbers / section numbers: 3.14, 0.05, v1.2.3
    if prev_ch.isdigit() and next_ch.isdigit():
        return True

    # Leading decimal like .05
    if next_ch.isdigit() and (not prev_ch or not prev_ch.isalnum()):
        return True

    # Domain names, file extensions, inline dotted identifiers: example.org/a.pdf
    if prev_ch.isalnum() and next_ch.isalnum():
        return True

    # Dotted acronyms such as e.g., i.e., U.S.A., Ph.D.
    if dot_index >= 2 and text[dot_index - 2] == "." and prev_ch.isalpha():
        return True
    if (
        dot_index + 2 < len(text)
        and next_ch.isalpha()
        and text[dot_index + 2] == "."
        and prev_ch.isalpha()
    ):
        return True

    # Extract preceding alphabetic token.
    tok_start = dot_index
    while tok_start > 0 and text[tok_start - 1].isalpha():
        tok_start -= 1
    token = text[tok_start:dot_index]
    if not token:
        return False

    # Single-letter middle/first initial (e.g., "J. Smith", "A. B. Chen").
    if len(token) == 1 and token.isupper():
        if tok_start == 0 or not text[tok_start - 1].isalnum():
            # Followed by whitespace + capitalized word or another initial.
            probe = dot_index + 1
            while probe < len(text) and text[probe].isspace():
                probe += 1
            if probe < len(text) and text[probe].isupper():
                return True

    lowered = token.lower()
    if lowered in _NON_BREAKING_ABBREVS:
        # "et al." before comma or lowercase continuation should not split;
        # even before a year/citation ("et al., 2024" or "et al. [12]") don't split.
        probe = dot_index + 1
        while probe < len(text) and text[probe].isspace():
            probe += 1
        if probe >= len(text):
            return False
        after = text[probe]
        # Only allow "al." to end a sentence if followed by a clear new sentence
        # starting after whitespace with a capital word (and not comma/bracket).
        if lowered == "al":
            if after in {",", ";", ":", ")", "]", "("} or after.islower():
                return True
            if after == "[" or after.isdigit():
                return True
            return False
        # Abbreviations like Fig. 1, Eq. (2), Dr. Smith, e.g., i.e., vs.
        if after in {",", ";", ":", "(", "["} or after.isdigit() or after.islower():
            return True
        if lowered in {
            "dr",
            "mr",
            "mrs",
            "ms",
            "prof",
            "st",
            "fig",
            "figs",
            "eq",
            "eqs",
            "sec",
            "secs",
            "ch",
            "chap",
            "ref",
            "refs",
            "no",
            "nos",
            "vol",
            "vols",
            "tab",
            "art",
            "supp",
        }:
            return True

    return False


def split_text_into_sentence_ranges(text: str) -> list[tuple[int, int]]:
    """Split ``text`` into ``(start, end)`` sentence ranges aware of academic syntax & CJK."""
    s = str(text or "")
    if not s:
        return []

    ranges: list[tuple[int, int]] = []
    start = 0
    n = len(s)
    i = 0
    while i < n:
        ch = s[i]
        if ch not in _SENTENCE_END_CHARS:
            i += 1
            continue

        if ch == "." and _is_non_breaking_period(s, i):
            i += 1
            continue

        end = i + 1
        # Consume consecutive end punctuation (e.g. "?!", "...", "。。").
        while end < n and s[end] in _SENTENCE_END_CHARS:
            end += 1
        punct_end = end

        # Ellipsis ("..." followed by lowercase word) usually continues the sentence.
        if punct_end - i >= 2 and all(c == "." for c in s[i:punct_end]):
            probe_ell = punct_end
            while probe_ell < n and s[probe_ell].isspace():
                probe_ell += 1
            if probe_ell < n and s[probe_ell].islower():
                i = punct_end
                continue

        # Attach trailing quotes/brackets even if PDF extraction inserted spaces.
        probe = end
        for _ in range(8):
            ws = probe
            while ws < n and s[ws].isspace():
                ws += 1
            if ws < n and s[ws] in _CLOSER_CHARS:
                probe = ws + 1
                while probe < n and s[probe] in _CLOSER_CHARS:
                    probe += 1
                end = probe
                continue
            break

        # Keep CJK quote-ended speech glued to immediately following narration
        # (e.g., "。”说完...").
        has_quote_closer = any(
            (not c.isspace()) and (c in _QUOTE_CLOSER_CHARS) for c in s[punct_end:end]
        )
        if has_quote_closer:
            k = end
            while k < n and s[k].isspace() and (k - end) < 48:
                k += 1
            if k < n and _is_narration_lead(s[k]):
                i = end
                continue

        ranges.append((start, end))
        start = end
        i = end

    if start < n:
        ranges.append((start, n))

    trimmed: list[tuple[int, int]] = []
    for a, b in ranges:
        while a < b and s[a].isspace():
            a += 1
        while b > a and s[b - 1].isspace():
            b -= 1
        if b > a:
            trimmed.append((a, b))
    return trimmed


def split_text_into_sentences(text: str) -> list[str]:
    """Return normalized sentences extracted from ``text``."""
    raw = str(text or "")
    ranges = split_text_into_sentence_ranges(raw)
    sentences: list[str] = []
    for start, end in ranges:
        seg = normalize_speech_text(raw[start:end])
        if seg:
            sentences.append(seg)
    return sentences


def chunk_speech_text(text: str, *, max_chars: int = 420) -> list[str]:
    """Split text into TTS-friendly chunks without cutting words in half."""
    limit = max(40, int(max_chars))
    cleaned = normalize_speech_text(text)
    if not cleaned:
        return []
    sentences = split_text_into_sentences(cleaned) or [cleaned]

    chunks: list[str] = []
    for sentence in sentences:
        if len(sentence) <= limit:
            chunks.append(sentence)
            continue
        chunks.extend(_split_long_sentence(sentence, limit))
    return [c for c in chunks if c]


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    """Split an over-long sentence on clause or word boundaries."""
    parts: list[str] = []
    remaining = sentence.strip()
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        # Prefer clause punctuation first, then whitespace.
        cut = -1
        for sep in ("; ", "；", ": ", "：", ", ", "，", "、", " "):
            pos = window.rfind(sep)
            if pos >= max_chars // 3:
                cut = pos + len(sep)
                break
        if cut <= 0:
            cut = max_chars
        head = remaining[:cut].strip()
        if head:
            parts.append(head)
        remaining = remaining[cut:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def group_web_spans_into_sentences(
    texts: Sequence[str], indices: Sequence[int]
) -> tuple[list[list[int]], list[str], dict[int, str]]:
    """Group aligned PDF.js span texts and indices into clean spoken sentences.

    Returns ``(span_groups, sentences, span_text_map)``.
    """
    paired: list[tuple[int, str]] = []
    for raw_text, raw_idx in zip(texts, indices):
        cleaned = str(raw_text or "").replace("\u00ad", "").replace("\u00a0", " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            continue
        try:
            idx = int(raw_idx)
        except Exception:
            continue
        paired.append((idx, cleaned))

    if not paired:
        return [], [], {}

    span_text_map: dict[int, str] = {}
    combined = ""
    parts: list[tuple[int, int, int]] = []  # (span_idx, start, end)
    for idx, text in paired:
        span_text_map[idx] = text
        sep = ""
        if combined:
            prev_ch = combined[-1]
            next_ch = text[0]
            if not prev_ch.isspace() and not (
                _CJK_LIKE_RE.match(prev_ch) and _CJK_LIKE_RE.match(next_ch)
            ):
                sep = " "
        start = len(combined) + len(sep)
        combined += sep + text
        end = len(combined)
        parts.append((idx, start, end))

    ranges = split_text_into_sentence_ranges(combined)
    if not ranges:
        all_indices = [idx for idx, _ in paired]
        sentence = normalize_speech_text(combined)
        return ([all_indices], [sentence], span_text_map) if sentence else ([], [], {})

    groups: list[list[int]] = []
    sentences: list[str] = []
    for start, end in ranges:
        sentence = normalize_speech_text(combined[start:end])
        if not sentence:
            continue
        group = [
            idx for idx, p_start, p_end in parts if p_end > start and p_start < end
        ]
        if not group:
            group = [paired[0][0]]
        # Preserve order while deduplicating.
        deduped = list(dict.fromkeys(group))
        groups.append(deduped)
        sentences.append(sentence)
    return groups, sentences, span_text_map


def parse_keywords(raw: str | Sequence[str] | None) -> list[str]:
    """Parse user-provided keyword specification into deduplicated clean terms."""
    if raw is None:
        return []
    if isinstance(raw, str):
        items = re.split(r"[,;\n，；]+", raw)
    else:
        items = []
        for entry in raw:
            items.extend(re.split(r"[,;\n，；]+", str(entry or "")))

    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        term = re.sub(r"\s+", " ", str(item or "").strip())
        if not term:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(term)
    return result


def find_keyword_spans(text: str, keywords: Sequence[str]) -> list[KeywordMatch]:
    """Find non-overlapping keyword matches in ``text``.

    Uses word-boundary checks for ASCII/Latin keywords so short terms like
    ``"rat"`` do not falsely match ``"strategy"``, while supporting substring
    matching for CJK terms where scripts have no spaces.
    """
    source = str(text or "")
    if not source:
        return []
    terms = parse_keywords(keywords)
    if not terms:
        return []

    # Longest terms first so multi-word phrases win over sub-words.
    ordered_terms = sorted(terms, key=lambda t: (-len(t), t.casefold()))
    raw_matches: list[KeywordMatch] = []
    for term in ordered_terms:
        has_cjk = bool(_CJK_LEAD_RE.search(term))
        escaped = re.escape(term)
        if has_cjk:
            pattern = re.compile(escaped, re.IGNORECASE)
        else:
            # Allow flexible internal whitespace for multi-word keywords.
            escaped_ws = re.sub(r"(?:\\ )+", r"\\s+", escaped)
            prefix = r"(?<![A-Za-z0-9_])" if term[0].isalnum() else ""
            suffix = r"(?![A-Za-z0-9_])" if term[-1].isalnum() else ""
            pattern = re.compile(f"{prefix}{escaped_ws}{suffix}", re.IGNORECASE)
        for match in pattern.finditer(source):
            if match.start() < match.end():
                raw_matches.append(KeywordMatch(match.start(), match.end(), term))

    if not raw_matches:
        return []

    # Resolve overlaps greedily (earliest start, then longest span).
    raw_matches.sort(key=lambda m: (m.start, -(m.end - m.start)))
    resolved: list[KeywordMatch] = []
    cursor = -1
    for item in raw_matches:
        if item.start < cursor:
            continue
        resolved.append(item)
        cursor = item.end
    return resolved


def extract_auto_keywords(text: str, *, max_keywords: int = 5) -> list[str]:
    """Deterministically extract salient keywords/key terms from a text block.

    Prefers acronyms (e.g. ``SAM2``, ``EEG``, ``TCN``), hyphenated technical
    compounds, and content words outside the common English stopword list.
    """
    source = str(text or "").strip()
    if not source or max_keywords <= 0:
        return []

    scores: dict[str, tuple[float, str, int]] = {}
    for pos, match in enumerate(_WORD_TOKEN_RE.finditer(source)):
        token = match.group(0).strip("-'\u2019")
        if not token:
            continue
        if _is_cjk_char(token[0]):
            continue
        lowered = token.lower()
        if lowered in _STOPWORDS:
            continue
        if token.isdigit():
            continue
        is_acronym = len(token) >= 2 and sum(1 for c in token if c.isupper()) >= 2
        if not is_acronym and len(token) < 5:
            continue

        bonus = 0.0
        if is_acronym:
            bonus += 3.0
        if "-" in token:
            bonus += 1.5
        if len(token) >= 8:
            bonus += 1.2
        elif len(token) >= 6:
            bonus += 0.6
        if token[0].isupper() and pos > 0:
            bonus += 0.8

        prev_score, prev_display, first_pos = scores.get(lowered, (0.0, token, pos))
        scores[lowered] = (prev_score + 1.0 + bonus, prev_display, first_pos)

    ranked = sorted(
        scores.values(),
        key=lambda item: (-item[0], item[2]),
    )
    return [display for _, display, _ in ranked[: int(max_keywords)]]


def _trailing_pause_strength(trailing_text: str) -> float:
    """Return prosodic pause strength immediately following a token."""
    for ch in trailing_text:
        if ch in _MAJOR_PAUSE_CHARS:
            return 2.0
        if ch in _MINOR_PAUSE_CHARS:
            return 1.0
    return 0.0


def _token_speech_weight(text: str, trailing_text: str = "") -> float:
    """Estimate relative spoken duration weight for a word or CJK character."""
    if not text:
        return 1.0
    # CJK characters correspond to full syllables; Latin words scale roughly
    # with characters + mild floor for short function words.
    cjk_count = sum(1 for ch in text if _is_cjk_char(ch))
    if cjk_count > 0:
        weight = float(cjk_count * 2.6 + (len(text) - cjk_count) * 0.9)
    else:
        weight = max(1.8, float(len(text)) * 0.85 + 0.8)

    # Add natural cadence pause weights for punctuation immediately following.
    pause = _trailing_pause_strength(trailing_text)
    if pause >= 2.0:
        weight += 3.2
    elif pause >= 1.0:
        weight += 1.6
    return max(1.0, weight)


def extract_word_tokens(
    text: str,
    *,
    base_offset: int = 0,
    keywords: Sequence[str] = (),
) -> list[WordToken]:
    """Extract word/CJK tokens with exact offsets, speech weights, and keyword flags."""
    source = str(text or "")
    if not source:
        return []
    kw_spans = find_keyword_spans(source, keywords) if keywords else []
    matches = list(_WORD_TOKEN_RE.finditer(source))
    tokens: list[WordToken] = []
    for idx, m in enumerate(matches):
        start, end = m.start(), m.end()
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(source)
        trailing = source[end:next_start]
        is_kw = any(kw.start < end and kw.end > start for kw in kw_spans)
        tokens.append(
            WordToken(
                start=base_offset + start,
                end=base_offset + end,
                text=m.group(0),
                weight=_token_speech_weight(m.group(0), trailing),
                is_keyword=is_kw,
                pause_strength=_trailing_pause_strength(trailing),
            )
        )
    return tokens


def analyze_audio_speech_segments(
    samples: object,
    sample_rate: int,
    *,
    frame_ms: int = 10,
    min_silence_ms: int = 90,
    min_speech_ms: int = 35,
    pad_ms: int = 15,
) -> list[tuple[int, int]]:
    """Detect active voiced speech intervals ``[(start_ms, end_ms), ...]`` in PCM audio.

    Trims leading/trailing silence and identifies real internal pauses so word
    highlighting tracks actual vocal delivery instead of drifting across silent
    margins or prosodic pauses. Works with NumPy arrays (lazily imported) or
    plain Python sequences.
    """
    sr = int(sample_rate or 0)
    if samples is None or sr <= 0:
        return []

    frame_len = max(1, int(round(sr * max(4, int(frame_ms)) / 1000.0)))
    rms_values: list[float] = []
    total_samples = 0

    try:
        import numpy as np  # type: ignore[import-not-found]

        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim > 1:
            arr = np.mean(arr, axis=1, dtype=np.float32)
        total_samples = int(arr.shape[0])
        if total_samples <= 0:
            return []
        # Normalize int-scaled waveforms (e.g. int16 [-32768, 32767]) to [-1, 1].
        peak_abs = float(np.max(np.abs(arr)))
        if peak_abs <= 0.0:
            return []
        if peak_abs > 1.5:
            arr = arr / peak_abs
        n_full = total_samples // frame_len
        if n_full > 0:
            frames = arr[: n_full * frame_len].reshape(n_full, frame_len)
            rms_arr = np.sqrt(np.mean(frames * frames, axis=1, dtype=np.float32))
            rms_values = [float(v) for v in rms_arr]
        rem = total_samples - n_full * frame_len
        if rem > 0:
            tail = arr[n_full * frame_len :]
            rms_values.append(float(np.sqrt(np.mean(tail * tail, dtype=np.float32))))
    except Exception:
        try:
            seq = list(samples)  # type: ignore[arg-type]
        except Exception:
            return []
        total_samples = len(seq)
        if total_samples <= 0:
            return []
        mono: list[float] = []
        peak_abs = 0.0
        for item in seq:
            if isinstance(item, (list, tuple)) and item:
                val = sum(float(x) for x in item) / len(item)
            else:
                val = float(item)
            mono.append(val)
            a = abs(val)
            if a > peak_abs:
                peak_abs = a
        if peak_abs <= 0.0:
            return []
        scale = peak_abs if peak_abs > 1.5 else 1.0
        for start in range(0, total_samples, frame_len):
            chunk = mono[start : start + frame_len]
            if not chunk:
                continue
            mean_sq = sum((x / scale) ** 2 for x in chunk) / len(chunk)
            rms_values.append(mean_sq**0.5)

    if not rms_values:
        return []

    total_ms = max(1, int(round((total_samples / float(sr)) * 1000.0)))
    peak_rms = max(rms_values)
    if peak_rms <= 1e-5:
        return []

    sorted_rms = sorted(rms_values)
    noise_floor = sorted_rms[max(0, int(len(sorted_rms) * 0.15))]
    threshold = max(peak_rms * 0.055, noise_floor * 2.5, 1e-4)
    # Guard against near-constant signals where noise_floor approaches peak_rms.
    if threshold >= peak_rms * 0.85:
        threshold = peak_rms * 0.25

    active = [val >= threshold for val in rms_values]
    raw_segments: list[tuple[int, int]] = []
    seg_start: int | None = None
    frame_step_ms = (frame_len / float(sr)) * 1000.0
    for idx, is_voiced in enumerate(active):
        if is_voiced and seg_start is None:
            seg_start = idx
        elif not is_voiced and seg_start is not None:
            s_ms = int(round(seg_start * frame_step_ms))
            e_ms = min(total_ms, int(round(idx * frame_step_ms)))
            if e_ms > s_ms:
                raw_segments.append((s_ms, e_ms))
            seg_start = None
    if seg_start is not None:
        s_ms = int(round(seg_start * frame_step_ms))
        if total_ms > s_ms:
            raw_segments.append((s_ms, total_ms))

    if not raw_segments:
        return []

    # Merge short internal dips (plosives / stop consonants) shorter than min_silence_ms.
    min_gap = max(20, int(min_silence_ms))
    merged: list[list[int]] = [[raw_segments[0][0], raw_segments[0][1]]]
    for s_ms, e_ms in raw_segments[1:]:
        if s_ms - merged[-1][1] < min_gap:
            merged[-1][1] = max(merged[-1][1], e_ms)
        else:
            merged.append([s_ms, e_ms])

    # Drop tiny noise clicks and apply subtle onset/offset padding without overlapping.
    min_seg = max(10, int(min_speech_ms))
    filtered = [seg for seg in merged if (seg[1] - seg[0]) >= min_seg]
    if not filtered:
        # Keep the single longest voiced span if all segments were brief.
        longest = max(merged, key=lambda seg: seg[1] - seg[0])
        filtered = [longest]

    pad = max(0, int(pad_ms))
    padded: list[tuple[int, int]] = []
    for i, (s_ms, e_ms) in enumerate(filtered):
        lo = max(0, s_ms - pad)
        hi = min(total_ms, e_ms + pad)
        if padded and lo < padded[-1][1]:
            mid = (padded[-1][1] + lo) // 2
            padded[-1] = (padded[-1][0], max(padded[-1][0] + 1, mid))
            lo = max(padded[-1][1], mid)
        if hi > lo:
            padded.append((lo, hi))
    return padded


def _normalize_speech_segments(
    speech_segments: Sequence[tuple[int, int]] | Sequence[Sequence[int]] | None,
    total_ms: int,
) -> list[tuple[int, int]]:
    """Sanitize and sort non-overlapping ``(start_ms, end_ms)`` speech intervals."""
    if not speech_segments or total_ms <= 0:
        return []
    cleaned: list[tuple[int, int]] = []
    for seg in speech_segments:
        if not isinstance(seg, (list, tuple)) or len(seg) < 2:
            continue
        try:
            s = max(0, min(total_ms, int(round(float(seg[0])))))
            e = max(0, min(total_ms, int(round(float(seg[1])))))
        except Exception:
            continue
        if e > s:
            cleaned.append((s, e))
    if not cleaned:
        return []
    cleaned.sort(key=lambda pair: (pair[0], pair[1]))
    merged: list[list[int]] = [[cleaned[0][0], cleaned[0][1]]]
    for s, e in cleaned[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(seg[0], seg[1]) for seg in merged]


def _select_clause_split_indices(
    weights: Sequence[float],
    pause_strengths: Sequence[float],
    segments: Sequence[tuple[int, int]],
) -> list[int] | None:
    """Choose ``len(segments) - 1`` token break indices matching audio pause positions."""
    k_breaks = len(segments) - 1
    n = len(weights)
    if k_breaks <= 0 or n <= k_breaks:
        return None
    if len(pause_strengths) != n:
        return None

    # Candidate break after token i (0 <= i < n - 1) where punctuation indicates a pause.
    candidates = [i for i in range(n - 1) if float(pause_strengths[i] or 0.0) > 0.0]
    if len(candidates) < k_breaks:
        return None
    if len(candidates) == k_breaks:
        return candidates

    # Target cumulative fraction of voiced audio at each pause boundary.
    total_voiced = float(sum(max(1, e - s) for s, e in segments))
    target_fracs: list[float] = []
    acc_v = 0.0
    for seg in segments[:-1]:
        acc_v += max(1, seg[1] - seg[0])
        target_fracs.append(acc_v / total_voiced)

    total_w = float(sum(max(0.1, float(w)) for w in weights))
    cum_w: list[float] = []
    running = 0.0
    for w in weights:
        running += max(0.1, float(w))
        cum_w.append(running / total_w)

    # Greedily pick strongest & best-positioned candidate per audio pause, preserving order.
    chosen: list[int] = []
    min_idx = 0
    for b_idx, target in enumerate(target_fracs):
        remaining_needed = k_breaks - b_idx - 1
        valid = [
            c
            for c in candidates
            if c >= min_idx
            and (len([x for x in candidates if x > c]) >= remaining_needed)
        ]
        if not valid:
            return None
        best = max(
            valid,
            key=lambda c: (
                float(pause_strengths[c]),
                -abs(cum_w[c] - target),
            ),
        )
        chosen.append(best)
        min_idx = best + 1
    return chosen


def _select_proportional_word_splits(
    weights: Sequence[float],
    segments: Sequence[tuple[int, int]],
) -> list[int] | None:
    """Partition words across multiple speech segments at word boundaries closest to audio energy."""
    k_breaks = len(segments) - 1
    n = len(weights)
    if k_breaks <= 0 or n <= k_breaks:
        return None

    total_voiced = float(sum(max(1, e - s) for s, e in segments))
    target_fracs: list[float] = []
    acc_v = 0.0
    for seg in segments[:-1]:
        acc_v += max(1, seg[1] - seg[0])
        target_fracs.append(acc_v / total_voiced)

    total_w = float(sum(max(0.1, float(w)) for w in weights))
    cum_w: list[float] = []
    running = 0.0
    for w in weights:
        running += max(0.1, float(w))
        cum_w.append(running / total_w)

    chosen: list[int] = []
    min_idx = 0
    for b_idx, target in enumerate(target_fracs):
        max_idx = n - (k_breaks - b_idx) - 1
        if min_idx > max_idx:
            return None
        best = min(
            range(min_idx, max_idx + 1),
            key=lambda c: abs(cum_w[c] - target),
        )
        chosen.append(best)
        min_idx = best + 1
    return chosen


def _voiced_offset_to_wall_ms(
    voiced_ms: float, segments: Sequence[tuple[int, int]], *, is_end: bool = False
) -> int:
    """Map a cumulative voiced offset onto wall-clock ms across speech segments."""
    if not segments:
        return max(0, int(round(voiced_ms)))
    rem = max(0.0, float(voiced_ms))
    for idx, (s_ms, e_ms) in enumerate(segments):
        span = float(max(1, e_ms - s_ms))
        in_seg = (rem <= span) if is_end else (rem < span)
        if in_seg or idx == len(segments) - 1:
            return int(round(min(float(e_ms), float(s_ms) + rem)))
        rem -= span
    return int(segments[-1][1])


def allocate_weighted_durations_ms(
    total_ms: int, weights: Sequence[float]
) -> list[int]:
    """Distribute ``total_ms`` across positive weights so durations sum to ``total_ms``."""
    total = int(total_ms)
    if total <= 0 or not weights:
        return []
    clean_weights = [max(0.1, float(w)) for w in weights]
    weight_sum = sum(clean_weights)
    if weight_sum <= 0:
        base = max(1, total // len(clean_weights))
        return [base for _ in clean_weights]

    raw = [total * (w / weight_sum) for w in clean_weights]
    durations = [max(1, int(x)) for x in raw]
    remainder = total - sum(durations)
    if remainder == 0:
        return durations

    fractional = [x - int(x) for x in raw]
    order = sorted(
        range(len(clean_weights)),
        key=lambda i: fractional[i],
        reverse=remainder > 0,
    )
    remaining = abs(remainder)
    idx = 0
    max_iters = remaining + len(order) * 2
    iters = 0
    while remaining > 0 and order and iters < max_iters:
        pos = order[idx % len(order)]
        if remainder > 0:
            durations[pos] += 1
            remaining -= 1
        elif durations[pos] > 1:
            durations[pos] -= 1
            remaining -= 1
        idx += 1
        iters += 1
    return durations


def downsample_weighted_units(
    units: Sequence[object],
    weights: Sequence[float],
    target_len: int,
    keyword_flags: Sequence[bool] | None = None,
) -> tuple[list[object], list[float], list[bool]]:
    """Downsample units while preserving first/last alignment and keyword flags."""
    flags = (
        list(keyword_flags)
        if keyword_flags is not None and len(keyword_flags) == len(units)
        else [False] * len(units)
    )
    if target_len <= 0 or target_len >= len(units):
        return list(units), [float(w) for w in weights], flags
    if target_len == 1:
        return [units[0]], [float(sum(weights))], [any(flags)]

    out_units: list[object] = []
    out_weights: list[float] = []
    out_flags: list[bool] = []
    n = len(units)
    for bucket in range(target_len):
        b_start = int(bucket * n / target_len)
        b_end = max(b_start + 1, int((bucket + 1) * n / target_len))
        b_end = min(n, b_end)
        # Prefer highlighting a keyword inside the bucket if present.
        pick = b_start
        for candidate in range(b_start, b_end):
            if flags[candidate]:
                pick = candidate
                break
        out_units.append(units[pick])
        out_weights.append(float(sum(weights[b_start:b_end])))
        out_flags.append(any(flags[b_start:b_end]))
    return out_units, out_weights, out_flags


def build_word_timing_schedule(
    units: Sequence[object],
    weights: Sequence[float],
    total_ms: int,
    *,
    keyword_flags: Sequence[bool] | None = None,
    min_interval_ms: int = 55,
    speech_segments: Sequence[tuple[int, int]] | Sequence[Sequence[int]] | None = None,
    pause_strengths: Sequence[float] | None = None,
) -> list[TimedHighlightUnit]:
    """Build monotonic word highlight schedule aligned to audio speech segments.

    When ``speech_segments`` (measured voiced intervals from PCM audio) are
    provided, leading and trailing silence are trimmed and multi-segment audio
    pauses align either directly to prosodic clause breaks (``pause_strengths``)
    or via voiced-time warping across active segments.
    """
    total = int(total_ms)
    if total <= 0 or not units or len(units) != len(weights):
        return []
    clean_weights = [max(0.1, float(w)) for w in weights]
    flags = (
        [bool(f) for f in keyword_flags]
        if keyword_flags is not None and len(keyword_flags) == len(units)
        else [False] * len(units)
    )
    pauses = (
        [float(p or 0.0) for p in pause_strengths]
        if pause_strengths is not None and len(pause_strengths) == len(units)
        else [0.0] * len(units)
    )

    segments = _normalize_speech_segments(speech_segments, total)
    active_voiced_ms = sum(max(1, e - s) for s, e in segments) if segments else total
    max_units = max(1, active_voiced_ms // max(1, int(min_interval_ms)))
    if len(units) > max_units:
        sampled_units, sampled_weights, sampled_flags = downsample_weighted_units(
            units, clean_weights, max_units, keyword_flags=flags
        )
        sampled_pauses = [0.0] * len(sampled_units)
    else:
        sampled_units = list(units)
        sampled_weights = clean_weights
        sampled_flags = flags
        sampled_pauses = pauses

    # Strategy 1 & 2: Multi-segment alignment (punctuation clause breaks first,
    # then word-boundary partitioning so no single word straddles a silent pause).
    if len(segments) > 1:
        split_indices = _select_clause_split_indices(
            sampled_weights, sampled_pauses, segments
        )
        if split_indices is None:
            split_indices = _select_proportional_word_splits(sampled_weights, segments)
        if split_indices is not None:
            schedule: list[TimedHighlightUnit] = []
            clause_start = 0
            boundaries = [*split_indices, len(sampled_units) - 1]
            for seg_idx, clause_end_inclusive in enumerate(boundaries):
                seg_start, seg_end = segments[seg_idx]
                seg_dur = max(1, seg_end - seg_start)
                c_units = sampled_units[clause_start : clause_end_inclusive + 1]
                c_weights = sampled_weights[clause_start : clause_end_inclusive + 1]
                c_flags = sampled_flags[clause_start : clause_end_inclusive + 1]
                c_durs = allocate_weighted_durations_ms(seg_dur, c_weights)
                cursor = seg_start
                for unit, dur, is_kw in zip(c_units, c_durs, c_flags):
                    schedule.append(
                        TimedHighlightUnit(
                            unit=unit,
                            start_ms=cursor,
                            end_ms=cursor + dur,
                            duration_ms=dur,
                            is_keyword=is_kw,
                        )
                    )
                    cursor += dur
                clause_start = clause_end_inclusive + 1
            return schedule

    # Strategy 3: Voiced-time warping across active speech segments (single segment or fewer words than segments).
    if segments:
        voiced_durations = allocate_weighted_durations_ms(
            active_voiced_ms, sampled_weights
        )
        if len(voiced_durations) != len(sampled_units):
            return []
        schedule = []
        v_cursor = 0
        for idx, (unit, v_dur, is_kw) in enumerate(
            zip(sampled_units, voiced_durations, sampled_flags)
        ):
            s_ms = _voiced_offset_to_wall_ms(v_cursor, segments, is_end=False)
            v_cursor += v_dur
            e_ms = (
                segments[-1][1]
                if idx == len(sampled_units) - 1
                else _voiced_offset_to_wall_ms(v_cursor, segments, is_end=True)
            )
            e_ms = max(s_ms + 1, e_ms)
            schedule.append(
                TimedHighlightUnit(
                    unit=unit,
                    start_ms=s_ms,
                    end_ms=e_ms,
                    duration_ms=e_ms - s_ms,
                    is_keyword=is_kw,
                )
            )
        return schedule

    # Strategy 3: Weighted distribution across full chunk duration.
    durations = allocate_weighted_durations_ms(total, sampled_weights)
    if len(durations) != len(sampled_units):
        return []

    schedule = []
    cursor = 0
    for unit, dur, is_kw in zip(sampled_units, durations, sampled_flags):
        schedule.append(
            TimedHighlightUnit(
                unit=unit,
                start_ms=cursor,
                end_ms=cursor + dur,
                duration_ms=dur,
                is_keyword=is_kw,
            )
        )
        cursor += dur
    return schedule


def active_schedule_index_at_ms(
    schedule: Sequence[TimedHighlightUnit], elapsed_ms: float
) -> int:
    """Return the index in ``schedule`` active at ``elapsed_ms`` (or ``-1`` if finished).

    During an internal pause gap ``schedule[i].end_ms <= elapsed_ms < schedule[i+1].start_ms``,
    holds cleanly on ``i`` instead of jumping prematurely to ``i + 1``.
    """
    if not schedule:
        return -1
    if elapsed_ms < schedule[0].start_ms:
        return 0
    if elapsed_ms >= schedule[-1].end_ms:
        return -1
    lo, hi = 0, len(schedule) - 1
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        item = schedule[mid]
        if elapsed_ms < item.start_ms:
            hi = mid - 1
        elif elapsed_ms >= item.end_ms:
            best = mid
            lo = mid + 1
        else:
            return mid
    return best


__all__ = [
    "KeywordMatch",
    "TimedHighlightUnit",
    "WordToken",
    "active_schedule_index_at_ms",
    "allocate_weighted_durations_ms",
    "analyze_audio_speech_segments",
    "build_word_timing_schedule",
    "chunk_speech_text",
    "downsample_weighted_units",
    "extract_auto_keywords",
    "extract_word_tokens",
    "find_keyword_spans",
    "group_web_spans_into_sentences",
    "normalize_speech_text",
    "parse_keywords",
    "split_text_into_sentence_ranges",
    "split_text_into_sentences",
]
