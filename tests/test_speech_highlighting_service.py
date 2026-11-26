from __future__ import annotations

import math
import sys

from annolid.services.speech_highlighting import (
    active_schedule_index_at_ms,
    allocate_weighted_durations_ms,
    analyze_audio_speech_segments,
    build_word_timing_schedule,
    chunk_speech_text,
    extract_auto_keywords,
    extract_word_tokens,
    find_keyword_spans,
    group_web_spans_into_sentences,
    normalize_speech_text,
    parse_keywords,
    split_text_into_sentence_ranges,
    split_text_into_sentences,
)


def test_sentence_splitting_preserves_scientific_abbreviations_and_decimals() -> None:
    text = (
        "As shown by Smith et al., 2024 in Fig. 2 and Eq. (3), "
        "accuracy reached 94.5% (p < 0.05, e.g., v1.2.3 on example.org/a.pdf). "
        "Dr. A. B. Chen confirmed the result! Did it generalize?"
    )
    sentences = split_text_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].startswith("As shown by Smith et al., 2024 in Fig. 2")
    assert sentences[0].endswith("example.org/a.pdf).")
    assert sentences[1] == "Dr. A. B. Chen confirmed the result!"
    assert sentences[2] == "Did it generalize?"


def test_sentence_splitting_handles_cjk_and_quote_continuations() -> None:
    text = "“准备好了吗？”说完她打开了门。实验正式开始！"
    sentences = split_text_into_sentences(text)
    assert sentences == ["“准备好了吗？”说完她打开了门。", "实验正式开始！"]


def test_chunk_speech_text_never_cuts_words_in_half() -> None:
    long_sentence = " ".join(f"word{i}" for i in range(60))
    chunks = chunk_speech_text(long_sentence, max_chars=80)
    assert len(chunks) > 2
    for chunk in chunks:
        assert len(chunk) <= 80
        # Every token in each chunk must be a complete "wordN" token.
        for tok in chunk.split():
            assert tok.startswith("word") and tok[4:].isdigit()


def test_normalize_speech_text_dehyphenates_and_fixes_closer_spacing() -> None:
    assert (
        normalize_speech_text("synergis- tic tracking . ” Next")
        == "synergistic tracking .” Next"
    )
    assert (
        normalize_speech_text("synergis- tic tracking. ”") == "synergistic tracking.”"
    )
    assert normalize_speech_text("non-linear tracking") == "non-linear tracking"


def test_group_web_spans_into_sentences_keeps_alignment_and_abbreviations() -> None:
    texts = ["Smith et al.", "\u00ad  ", "(2024) found p < 0.05.", "Next sentence."]
    indices = [10, 11, 12, 13]
    groups, sentences, span_map = group_web_spans_into_sentences(texts, indices)
    assert len(groups) == 2
    assert groups[0] == [10, 12]
    assert groups[1] == [13]
    assert sentences[0] == "Smith et al. (2024) found p < 0.05."
    assert sentences[1] == "Next sentence."
    assert 11 not in span_map
    assert span_map[10] == "Smith et al."


def test_keyword_matching_enforces_word_boundaries_phrases_and_cjk() -> None:
    assert parse_keywords("dopamine, social interaction; dopamine\n多巴胺") == [
        "dopamine",
        "social interaction",
        "多巴胺",
    ]

    text = "Our strategy measured rat social   interaction and dopamine in open field."
    matches = find_keyword_spans(
        text, ["rat", "social interaction", "open field", "field"]
    )
    matched_texts = [text[m.start : m.end] for m in matches]
    # "rat" must NOT match inside "strategy", and "open field" beats overlapping "field".
    assert matched_texts == ["rat", "social   interaction", "open field"]

    cjk_text = "实验记录了小鼠多巴胺释放的变化。"
    cjk_matches = find_keyword_spans(cjk_text, ["多巴胺"])
    assert len(cjk_matches) == 1
    assert cjk_text[cjk_matches[0].start : cjk_matches[0].end] == "多巴胺"


def test_extract_auto_keywords_prioritizes_acronyms_and_technical_terms() -> None:
    text = (
        "We evaluated SAM2 and EEG recordings for pose-estimation in rodents "
        "using a temporal convolutional architecture."
    )
    terms = extract_auto_keywords(text, max_keywords=4)
    lowered = [t.lower() for t in terms]
    assert "sam2" in lowered
    assert "eeg" in lowered
    assert "pose-estimation" in lowered
    assert "using" not in lowered


def test_word_tokens_and_weighted_timing_schedule_exact_sum_and_lookup() -> None:
    text = "First, dopamine rises; then freezing drops."
    tokens = extract_word_tokens(
        text, base_offset=10, keywords=["dopamine", "freezing"]
    )
    assert [t.text for t in tokens] == [
        "First",
        "dopamine",
        "rises",
        "then",
        "freezing",
        "drops",
    ]
    assert [t.is_keyword for t in tokens] == [False, True, False, False, True, False]
    # Punctuated "First," and "rises;" carry extra cadence pause weight.
    assert tokens[0].weight > tokens[3].weight

    schedule = build_word_timing_schedule(
        [(t.start, t.end) for t in tokens],
        [t.weight for t in tokens],
        total_ms=2000,
        keyword_flags=[t.is_keyword for t in tokens],
    )
    assert len(schedule) == len(tokens)
    assert sum(item.duration_ms for item in schedule) == 2000
    assert schedule[0].start_ms == 0
    assert schedule[-1].end_ms == 2000
    assert active_schedule_index_at_ms(schedule, 0) == 0
    assert active_schedule_index_at_ms(schedule, 1999) == len(schedule) - 1
    assert active_schedule_index_at_ms(schedule, 2000) == -1

    # Tight budget downsampling preserves keyword flags in buckets.
    down = build_word_timing_schedule(
        [(t.start, t.end) for t in tokens],
        [t.weight for t in tokens],
        total_ms=150,
        keyword_flags=[t.is_keyword for t in tokens],
        min_interval_ms=60,
    )
    assert len(down) == 2
    assert sum(item.duration_ms for item in down) == 150
    assert any(item.is_keyword for item in down)

    assert sum(allocate_weighted_durations_ms(1000, [1.0, 2.5, 3.2])) == 1000
    assert split_text_into_sentence_ranges("") == []


def _synth_voiced_waveform(
    sample_rate: int,
    total_ms: int,
    active_windows_ms: list[tuple[int, int]],
    *,
    amplitude: float = 0.5,
) -> list[float]:
    n_samples = int(round(sample_rate * total_ms / 1000.0))
    samples = [0.0] * n_samples
    for s_ms, e_ms in active_windows_ms:
        s_idx = max(0, int(round(sample_rate * s_ms / 1000.0)))
        e_idx = min(n_samples, int(round(sample_rate * e_ms / 1000.0)))
        for i in range(s_idx, e_idx):
            samples[i] = amplitude * math.sin(2.0 * math.pi * 220.0 * i / sample_rate)
    return samples


def test_analyze_audio_speech_segments_trims_silence_and_detects_internal_pause(
    monkeypatch,
) -> None:
    sr = 8000
    # 200ms leading silence, 300ms burst 1, 300ms pause, 500ms burst 2, 200ms trailing silence
    pcm = _synth_voiced_waveform(sr, 1500, [(200, 500), (800, 1300)])

    # 1) NumPy float32 path
    import numpy as np

    segments_np = analyze_audio_speech_segments(np.asarray(pcm, dtype=np.float32), sr)
    assert len(segments_np) == 2
    assert 170 <= segments_np[0][0] <= 215
    assert 485 <= segments_np[0][1] <= 530
    assert 770 <= segments_np[1][0] <= 815
    assert 1285 <= segments_np[1][1] <= 1330

    # 2) Int16 stereo path
    stereo_int16 = np.column_stack(
        [np.asarray(pcm) * 20000.0, np.asarray(pcm) * 20000.0]
    ).astype(np.int16)
    segments_stereo = analyze_audio_speech_segments(stereo_int16, sr)
    assert len(segments_stereo) == 2

    # 3) Pure-Python fallback when numpy is unavailable
    monkeypatch.setitem(sys.modules, "numpy", None)
    segments_py = analyze_audio_speech_segments(pcm, sr)
    assert len(segments_py) == 2
    assert 170 <= segments_py[0][0] <= 215
    assert 770 <= segments_py[1][0] <= 815

    # All-silent buffer returns empty segments (falls back cleanly to duration-only schedule).
    assert analyze_audio_speech_segments([0.0] * 4000, sr) == []


def test_audio_aligned_word_schedule_respects_leading_silence_and_clause_pauses() -> (
    None
):
    text = "First, dopamine rises quickly."
    tokens = extract_word_tokens(text, keywords=["dopamine"])
    assert tokens[0].pause_strength == 1.0  # comma after "First,"
    assert tokens[1].pause_strength == 0.0

    # Measured audio: 200ms leading silence, clause 1 [200..500], 300ms pause, clause 2 [800..1300], 200ms trailing silence.
    schedule = build_word_timing_schedule(
        [t.text for t in tokens],
        [t.weight for t in tokens],
        total_ms=1500,
        keyword_flags=[t.is_keyword for t in tokens],
        speech_segments=[(200, 500), (800, 1300)],
        pause_strengths=[t.pause_strength for t in tokens],
    )
    assert [item.unit for item in schedule] == ["First", "dopamine", "rises", "quickly"]
    # Clause 1 ("First") aligns strictly to [200, 500].
    assert schedule[0].start_ms == 200
    assert schedule[0].end_ms == 500
    # Clause 2 ("dopamine rises quickly") aligns strictly to [800, 1300].
    assert schedule[1].start_ms == 800
    assert schedule[1].is_keyword is True
    assert schedule[-1].end_ms == 1300

    # During the 300ms internal pause (t=650ms), active index holds on "First" (0) instead of racing to "dopamine" (1).
    assert active_schedule_index_at_ms(schedule, 650) == 0
    assert active_schedule_index_at_ms(schedule, 800) == 1
    # Finishes cleanly at end of voiced speech (1300ms), not after trailing silence (1500ms).
    assert active_schedule_index_at_ms(schedule, 1300) == -1


def test_audio_aligned_word_schedule_voiced_time_warping_without_punctuation() -> None:
    # No commas/punctuation, yet speaker paused between [150..550] and [850..1250].
    tokens = extract_word_tokens("one two three four")
    schedule = build_word_timing_schedule(
        [t.text for t in tokens],
        [t.weight for t in tokens],
        total_ms=1400,
        speech_segments=[(150, 550), (850, 1250)],
        pause_strengths=[t.pause_strength for t in tokens],
    )
    assert len(schedule) == 4
    assert schedule[0].start_ms == 150
    assert schedule[-1].end_ms == 1250
    # Words 2 and 3 land in the second voiced burst after the 300ms pause.
    assert schedule[2].start_ms >= 850
    assert active_schedule_index_at_ms(schedule, 700) == 1
