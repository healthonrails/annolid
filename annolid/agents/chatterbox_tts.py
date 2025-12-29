from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import os
import sys

from annolid.utils.logger import logger

MODEL_ID = "ResembleAI/chatterbox-turbo-ONNX"
SAMPLE_RATE = 24000

START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
SILENCE_TOKEN = 4299

NUM_KV_HEADS = 16
HEAD_DIM = 64

SUPPORTED_DTYPES = ("fp32", "fp16", "q8", "q4", "q4f16")


class RepetitionPenaltyLogitsProcessor:
    def __init__(self, penalty: float):
        if not isinstance(penalty, (float, int)) or not (float(penalty) > 0):
            raise ValueError(
                f"`penalty` must be a strictly positive float, but is {penalty}"
            )
        self.penalty = float(penalty)

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        score = np.take_along_axis(scores, input_ids, axis=1)
        score = np.where(score < 0, score * self.penalty, score / self.penalty)
        scores_processed = scores.copy()
        np.put_along_axis(scores_processed, input_ids, score, axis=1)
        return scores_processed


def _select_providers() -> Optional[list[str]]:
    try:
        import onnxruntime
    except Exception:
        return None

    available = list(onnxruntime.get_available_providers())

    # Allow explicit override, e.g.:
    #   ANNOLID_CHATTERBOX_ORT_PROVIDERS=CPUExecutionProvider
    override = os.getenv("ANNOLID_CHATTERBOX_ORT_PROVIDERS", "").strip()
    if override:
        requested = [p.strip() for p in override.split(",") if p.strip()]
        filtered = [p for p in requested if p in available]
        if filtered:
            return filtered
        logger.warning(
            f"ANNOLID_CHATTERBOX_ORT_PROVIDERS requested {requested} but none are available ({available}). Falling back to auto."
        )

    # CoreML EP currently has issues loading ONNX models with external data on some
    # onnxruntime builds (it can lose the model path during optimisation), so keep
    # Chatterbox on CPU by default on macOS. A user can override via env var above.
    if sys.platform == "darwin" and "CPUExecutionProvider" in available:
        return ["CPUExecutionProvider"]

    providers = [p for p in available if p != "TensorrtExecutionProvider"]
    return providers or None


def _resolve_filename(name: str, dtype: str) -> str:
    dtype = (dtype or "fp32").strip().lower()
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"Unsupported dtype '{dtype}'. Expected one of {SUPPORTED_DTYPES}."
        )
    suffix = "" if dtype == "fp32" else "_quantized" if dtype == "q8" else f"_{dtype}"
    return f"{name}{suffix}.onnx"


def _download_model(name: str, dtype: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise ImportError(
            "huggingface_hub is required to download Chatterbox ONNX assets."
        ) from exc

    filename = _resolve_filename(name, dtype)
    graph = hf_hub_download(MODEL_ID, subfolder="onnx", filename=filename)
    hf_hub_download(MODEL_ID, subfolder="onnx", filename=f"{filename}_data")
    return graph


@lru_cache(maxsize=8)
def _get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID)


@lru_cache(maxsize=4)
def _get_sessions(dtype: str):
    try:
        import onnxruntime
    except Exception as exc:
        raise ImportError(
            "onnxruntime is required for the Chatterbox ONNX TTS engine."
        ) from exc

    providers = _select_providers()
    conditional_decoder_path = _download_model(
        "conditional_decoder", dtype=dtype)
    speech_encoder_path = _download_model("speech_encoder", dtype=dtype)
    embed_tokens_path = _download_model("embed_tokens", dtype=dtype)
    language_model_path = _download_model("language_model", dtype=dtype)

    def make_session(path: str):
        return onnxruntime.InferenceSession(path, providers=providers)

    try:
        speech_encoder_session = make_session(speech_encoder_path)
        embed_tokens_session = make_session(embed_tokens_path)
        language_model_session = make_session(language_model_path)
        cond_decoder_session = make_session(conditional_decoder_path)
    except Exception as exc:
        message = str(exc)
        if (
            providers
            and "CoreMLExecutionProvider" in providers
            and "model_path must not be empty" in message
            and "CPUExecutionProvider" in onnxruntime.get_available_providers()
        ):
            logger.warning(
                "Chatterbox ONNX failed to initialise with CoreML; retrying with CPUExecutionProvider only."
            )
            providers = ["CPUExecutionProvider"]
            speech_encoder_session = onnxruntime.InferenceSession(
                speech_encoder_path, providers=providers
            )
            embed_tokens_session = onnxruntime.InferenceSession(
                embed_tokens_path, providers=providers
            )
            language_model_session = onnxruntime.InferenceSession(
                language_model_path, providers=providers
            )
            cond_decoder_session = onnxruntime.InferenceSession(
                conditional_decoder_path, providers=providers
            )
        else:
            raise
    return (
        speech_encoder_session,
        embed_tokens_session,
        language_model_session,
        cond_decoder_session,
    )


def _speaker_cache_key(voice_wav_path: str) -> Tuple[str, int]:
    resolved = Path(voice_wav_path).expanduser().resolve()
    stat = resolved.stat()
    return str(resolved), int(stat.st_mtime_ns)


@lru_cache(maxsize=4)
def _encode_speaker(dtype: str, resolved_path: str, mtime_ns: int):
    del mtime_ns  # cache key only
    speech_encoder_session, _, _, _ = _get_sessions(dtype)
    try:
        import librosa
    except Exception as exc:
        raise ImportError(
            "librosa is required to load the voice prompt audio."
        ) from exc

    audio_values, _ = librosa.load(resolved_path, sr=SAMPLE_RATE, mono=True)
    audio_values = audio_values[np.newaxis, :].astype(np.float32)
    encoder_input_name = speech_encoder_session.get_inputs()[0].name
    encoder_input = {encoder_input_name: audio_values}
    cond_emb, prompt_token, speaker_embeddings, speaker_features = (
        speech_encoder_session.run(None, encoder_input)
    )
    return cond_emb, prompt_token, speaker_embeddings, speaker_features


def _init_past_key_values(
    language_model_session, batch_size: int
) -> Tuple[Dict[str, np.ndarray], list[str]]:
    past_inputs = [
        inp for inp in language_model_session.get_inputs() if "past_key_values" in inp.name
    ]
    past_key_names = [inp.name for inp in past_inputs]
    past_key_values: Dict[str, np.ndarray] = {}
    for inp in past_inputs:
        dtype = np.float16 if inp.type == "tensor(float16)" else np.float32
        past_key_values[inp.name] = np.zeros(
            [batch_size, NUM_KV_HEADS, 0, HEAD_DIM], dtype=dtype
        )
    return past_key_values, past_key_names


def _update_past_key_values(
    language_model_session,
    past_key_names: list[str],
    present_values: list[np.ndarray],
) -> Dict[str, np.ndarray]:
    outputs = language_model_session.get_outputs()
    present_outputs = [
        out.name for out in outputs if "present_key_values" in out.name]
    if present_outputs and len(present_outputs) == len(past_key_names) == len(present_values):
        return {key: value for key, value in zip(past_key_names, present_values)}
    if len(past_key_names) == len(present_values):
        return {key: value for key, value in zip(past_key_names, present_values)}
    raise RuntimeError(
        f"Could not align present_key_values (got {len(present_values)}) with past_key_values (got {len(past_key_names)})."
    )


def text_to_speech(
    text: str,
    *,
    voice_wav_path: str,
    output_path: Optional[str] = None,
    dtype: str = "fp32",
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.2,
    apply_watermark: bool = False,
) -> Optional[Tuple[np.ndarray, int]]:
    """
    Voice-cloning TTS via ResembleAI/chatterbox-turbo-ONNX.

    Returns (samples, sample_rate) or None on failure.
    """
    text = (text or "").strip()
    voice_wav_path = (voice_wav_path or "").strip()
    if not text:
        return None
    if not voice_wav_path:
        logger.warning("Chatterbox TTS requires a voice prompt WAV path.")
        return None
    try:
        tokenizer = _get_tokenizer()
        (
            _speech_encoder_session,
            embed_tokens_session,
            language_model_session,
            cond_decoder_session,
        ) = _get_sessions(dtype.strip().lower())

        resolved_path, mtime_ns = _speaker_cache_key(voice_wav_path)
        cond_emb, prompt_token, speaker_embeddings, speaker_features = _encode_speaker(
            dtype.strip().lower(), resolved_path, mtime_ns
        )

        input_ids = tokenizer(text, return_tensors="np")[
            "input_ids"].astype(np.int64)
        embed_input_name = embed_tokens_session.get_inputs()[0].name
        prompt_embeds = embed_tokens_session.run(
            None, {embed_input_name: input_ids})[0]
        inputs_embeds = np.concatenate((cond_emb, prompt_embeds), axis=1)

        batch_size, seq_len, _ = inputs_embeds.shape
        past_key_values, past_key_names = _init_past_key_values(
            language_model_session, batch_size
        )
        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
        position_ids = (
            np.arange(seq_len, dtype=np.int64).reshape(
                1, -1).repeat(batch_size, axis=0)
        )

        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(
            penalty=repetition_penalty
        )
        generate_tokens = np.array([[START_SPEECH_TOKEN]], dtype=np.int64)

        current_embeds = inputs_embeds
        for _step in range(int(max_new_tokens)):
            outputs = language_model_session.run(
                None,
                dict(
                    inputs_embeds=current_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **past_key_values,
                ),
            )
            logits = outputs[0]
            present_values = outputs[1:]

            logits = logits[:, -1, :]
            next_token_logits = repetition_penalty_processor(
                generate_tokens, logits)
            next_token = np.argmax(next_token_logits, axis=-1, keepdims=True).astype(
                np.int64
            )

            generate_tokens = np.concatenate(
                (generate_tokens, next_token), axis=-1)
            if (next_token.flatten() == STOP_SPEECH_TOKEN).all():
                break

            past_key_values = _update_past_key_values(
                language_model_session, past_key_names, present_values
            )
            attention_mask = np.concatenate(
                [attention_mask, np.ones((batch_size, 1), dtype=np.int64)], axis=1
            )
            position_ids = position_ids[:, -1:] + 1
            current_embeds = embed_tokens_session.run(None, {embed_input_name: next_token})[
                0
            ]

        speech_tokens = generate_tokens[:, 1:]
        if speech_tokens.shape[1] and (speech_tokens[:, -1] == STOP_SPEECH_TOKEN).all():
            speech_tokens = speech_tokens[:, :-1]

        silence_tokens = np.full(
            (speech_tokens.shape[0], 3), SILENCE_TOKEN, dtype=np.int64)
        prompt_token = np.asarray(prompt_token).astype(np.int64)
        speech_tokens = np.concatenate(
            [prompt_token, speech_tokens, silence_tokens], axis=1
        )

        wav = cond_decoder_session.run(
            None,
            dict(
                speech_tokens=speech_tokens,
                speaker_embeddings=speaker_embeddings,
                speaker_features=speaker_features,
            ),
        )[0].squeeze(axis=0)

        if apply_watermark:
            try:
                import perth

                watermarker = perth.PerthImplicitWatermarker()
                wav = watermarker.apply_watermark(wav, sample_rate=SAMPLE_RATE)
            except Exception as exc:
                logger.warning(f"Failed to apply watermark: {exc}")

        if output_path:
            import soundfile as sf

            sf.write(str(output_path), wav, SAMPLE_RATE)

        return np.asarray(wav), SAMPLE_RATE
    except Exception as exc:
        logger.warning(f"Chatterbox TTS failed: {exc}")
        return None


def prewarm(*, voice_wav_path: str = "", dtype: str = "fp32") -> bool:
    """
    Best-effort warmup to reduce time-to-first-audio.

    - Downloads model assets (if missing)
    - Creates ORT sessions
    - Loads tokenizer
    - Optionally encodes the speaker prompt (if `voice_wav_path` is provided)
    """
    try:
        dtype = (dtype or "fp32").strip().lower()
        _get_tokenizer()
        _get_sessions(dtype)
        voice_wav_path = (voice_wav_path or "").strip()
        if voice_wav_path:
            resolved_path, mtime_ns = _speaker_cache_key(voice_wav_path)
            _encode_speaker(dtype, resolved_path, mtime_ns)
        return True
    except Exception as exc:
        logger.warning(f"Chatterbox prewarm failed: {exc}")
        return False
