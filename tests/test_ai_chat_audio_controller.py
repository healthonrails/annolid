from annolid.gui.widgets.ai_chat_audio_controller import ChatAudioController


def test_iter_tts_chunks_preserves_sentence_boundaries() -> None:
    text = "Sentence one. Sentence two! Sentence three?"
    chunks = list(ChatAudioController._iter_tts_chunks(text, max_chunk_chars=200))
    assert chunks == ["Sentence one.", "Sentence two!", "Sentence three?"]


def test_iter_tts_chunks_splits_long_text() -> None:
    text = " ".join(["word"] * 500)
    chunks = list(
        ChatAudioController._iter_tts_chunks(
            text, max_chunk_chars=120, fallback_chunk_chars=60
        )
    )
    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert " ".join(chunks).replace("  ", " ").strip() == text


def test_iter_tts_chunks_empty_input() -> None:
    assert list(ChatAudioController._iter_tts_chunks("")) == []
