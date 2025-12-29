from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import os

from annolid.utils.logger import logger

MODEL_ID = "ResembleAI/chatterbox-turbo-ONNX"
SAMPLE_RATE = 24000

START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
SILENCE_TOKEN = 4299

NUM_KV_HEADS = 16
HEAD_DIM = 64

SUPPORTED_DTYPES = ("fp32", "fp16", "q8", "q4", "q4f16")
_FLAG_VALUES = {"1", "true", "yes", "on"}
_GPU_PROVIDER_ORDER = (
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
    "OpenVINOExecutionProvider",
)


class RepetitionPenaltyLogitsProcessor:
    def __init__(self, penalty: float):
        if not isinstance(penalty, (float, int)) or not (float(penalty) > 0):
            raise ValueError(
                f"`penalty` must be a strictly positive float, but is {penalty}"
            )
        self.penalty = float(penalty)

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        # scores: (batch, vocab), input_ids: (batch, seq)
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
    disable_coreml = (
        os.getenv("ANNOLID_CHATTERBOX_DISABLE_COREML",
                  "").strip().lower() in _FLAG_VALUES
    )
    if disable_coreml and "CoreMLExecutionProvider" in available:
        available = [p for p in available if p != "CoreMLExecutionProvider"]

    override = os.getenv("ANNOLID_CHATTERBOX_ORT_PROVIDERS", "").strip()
    if override:
        requested = [p.strip() for p in override.split(",") if p.strip()]
        filtered = [p for p in requested if p in available]
        if filtered:
            return filtered
        logger.warning(
            f"ANNOLID_CHATTERBOX_ORT_PROVIDERS requested {requested} but none are available ({available}). Falling back to auto."
        )

    # Preferred order: GPU -> MPS -> CPU.
    providers: list[str] = []
    for provider in _GPU_PROVIDER_ORDER:
        if provider in available:
            providers.append(provider)
            break
    # macOS accelerator of choice when CUDA/ROCm/DirectML/OpenVINO aren't present.
    if not providers and "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")
    if not providers and "MPSExecutionProvider" in available:
        providers.append("MPSExecutionProvider")
    if "CPUExecutionProvider" in available and "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")

    if not providers:
        providers = [
            p for p in available if p != "TensorrtExecutionProvider"
        ]
    return providers or None


def _is_coreml_runtime_failure(exc: Exception) -> bool:
    msg = str(exc)
    if "CoreMLExecutionProvider" not in msg and "coreml_execution_provider.cc" not in msg:
        return False
    needles = (
        "dynamic shape",
        "zero elements",
        "not supported by the CoreML EP",
        "RegisterModelInputOutput Unable to get shape",
        "model_path must not be empty",
    )
    return any(n in msg for n in needles)


def _available_ort_providers() -> list[str]:
    try:
        import onnxruntime
    except Exception:
        return []
    try:
        return list(onnxruntime.get_available_providers())
    except Exception:
        return []


def _fallback_provider_overrides_for_runtime() -> list[Tuple[str, ...]]:
    """Fallback order after a CoreML runtime failure: MPS → CPU."""
    avail = _available_ort_providers()
    fallbacks: list[Tuple[str, ...]] = []
    if "MPSExecutionProvider" in avail:
        if "CPUExecutionProvider" in avail:
            fallbacks.append(("MPSExecutionProvider", "CPUExecutionProvider"))
        else:
            fallbacks.append(("MPSExecutionProvider",))
    if "CPUExecutionProvider" in avail:
        fallbacks.append(("CPUExecutionProvider",))
    return fallbacks


def _resolve_filename(name: str, dtype: str) -> str:
    dtype = (dtype or "fp32").strip().lower()
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"Unsupported dtype '{dtype}'. Expected one of {SUPPORTED_DTYPES}."
        )
    suffix = "" if dtype == "fp32" else "_quantized" if dtype == "q8" else f"_{dtype}"
    return f"{name}{suffix}.onnx"


def _inline_external_initializers(onnx_path: str) -> str:
    """
    If the model uses external data (<model>.onnx_data), rewrite it to a single-file ONNX
    with inline initializers. This avoids CoreML EP crashes where partitioned subgraphs
    lose model_path.
    """
    p = Path(onnx_path)
    inline_path = p.with_name(p.stem + "_inline.onnx")

    # Fast path: already created and up-to-date
    try:
        if inline_path.exists() and inline_path.stat().st_mtime_ns >= p.stat().st_mtime_ns:
            return str(inline_path)
    except Exception:
        pass

    # Detect common external-data sidecar names
    sidecars = [
        Path(str(p) + "_data"),           # e.g. "model.onnx_data"
        p.with_suffix(p.suffix + "_data")  # same as above, just explicit
    ]
    has_sidecar = any(s.exists() for s in sidecars)

    if not has_sidecar:
        # No external data; safe to use original
        return onnx_path

    try:
        import onnx
    except Exception as exc:
        logger.warning(
            "Model appears to use external data (*.onnx_data) but `onnx` is not installed; "
            "CoreML EP may crash. Install with: pip install onnx"
        )
        return onnx_path

    # Load + external data, then save inline
    model = onnx.load_model(str(p), load_external_data=True)
    onnx.save_model(model, str(inline_path), save_as_external_data=False)
    return str(inline_path)


def _download_model(name: str, dtype: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise ImportError(
            "huggingface_hub is required to download Chatterbox ONNX assets."
        ) from exc

    filename = _resolve_filename(name, dtype)
    graph = hf_hub_download(MODEL_ID, subfolder="onnx", filename=filename)

    # Try to fetch external data sidecar (if present)
    try:
        hf_hub_download(MODEL_ID, subfolder="onnx",
                        filename=f"{filename}_data")
    except Exception:
        pass

    # rewrite to single-file ONNX if external data exists
    graph = _inline_external_initializers(graph)
    return graph


@lru_cache(maxsize=8)
def _get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID)


def _make_session_options():
    # Conservative ORT options that help with very large graphs / many partitions.
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return so


@lru_cache(maxsize=8)
def _get_sessions(dtype: str, providers_override: Tuple[str, ...] = ()):
    import onnxruntime

    dtype = (dtype or "fp32").strip().lower()
    base_providers = list(
        providers_override) if providers_override else _select_providers()
    so = _make_session_options()

    conditional_decoder_path = _download_model(
        "conditional_decoder", dtype=dtype)
    speech_encoder_path = _download_model("speech_encoder", dtype=dtype)
    embed_tokens_path = _download_model("embed_tokens", dtype=dtype)
    language_model_path = _download_model("language_model", dtype=dtype)

    def make_session(path: str, providers_list: Optional[list[str]]):
        return onnxruntime.InferenceSession(path, sess_options=so, providers=providers_list)

    def cpu_only() -> Optional[list[str]]:
        avail = onnxruntime.get_available_providers()
        return ["CPUExecutionProvider"] if "CPUExecutionProvider" in avail else None

    def provider_attempts() -> list[list[str]]:
        """
        Attempts for encoder/embed/LM sessions only.
        - If override is supplied: try exactly override, then CPU.
        - Else: try base_providers, then MPS+CPU (if available), then CPU.
        """
        attempts: list[list[str]] = []
        if base_providers:
            attempts.append(base_providers)

        cpu = cpu_only()

        if providers_override:
            if cpu and cpu not in attempts:
                attempts.append(cpu)
            return attempts

        avail = onnxruntime.get_available_providers()
        if "MPSExecutionProvider" in avail:
            mps = ["MPSExecutionProvider"]
            if "CPUExecutionProvider" in avail:
                mps.append("CPUExecutionProvider")
            if mps not in attempts:
                attempts.append(mps)

        if cpu and cpu not in attempts:
            attempts.append(cpu)

        return attempts

    # Decoder providers: CoreML often fails on ISTFT output-shape inference.
    # Default: CPU-only ALWAYS. You can override this setting:
    #   ANNOLID_CHATTERBOX_DECODER_PROVIDERS="CoreMLExecutionProvider,CPUExecutionProvider"
    decoder_override = os.getenv(
        "ANNOLID_CHATTERBOX_DECODER_PROVIDERS", "").strip()
    if decoder_override:
        decoder_providers = [p.strip()
                             for p in decoder_override.split(",") if p.strip()]
        # Filter to available
        avail = set(onnxruntime.get_available_providers())
        decoder_providers = [
            p for p in decoder_providers if p in avail] or None
    else:
        decoder_providers = cpu_only()

    last_exc: Exception | None = None

    for providers_list in provider_attempts():
        try:
            # Encoder / LM side: try fast providers first
            speech_encoder_session = make_session(
                speech_encoder_path, providers_list)
            embed_tokens_session = make_session(
                embed_tokens_path, providers_list)
            language_model_session = make_session(
                language_model_path, providers_list)

            # Decoder: CPU-only by default (avoids /istft/... shape failure on CoreML)
            cond_decoder_session = make_session(
                conditional_decoder_path, decoder_providers)

            last_exc = None
            break

        except Exception as exc:
            last_exc = exc

            msg = str(exc)
            # If failure is specifically the ISTFT shape error, don't keep retrying the same provider set.
            # We'll just move on to the next provider attempt for encoder/LM.
            logger.warning(
                f"Chatterbox ONNX failed to initialize with providers {providers_list}; trying fallback ({exc})."
            )
            continue

    if last_exc is not None:
        raise last_exc

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


@lru_cache(maxsize=8)
def _encode_speaker(
    dtype: str,
    resolved_path: str,
    mtime_ns: int,
    providers_override: Tuple[str, ...] = (),
):
    del mtime_ns  # cache key only
    speech_encoder_session, _, _, _ = _get_sessions(dtype, providers_override)
    try:
        import librosa
    except Exception as exc:
        raise ImportError(
            "librosa is required to load the voice prompt audio.") from exc

    audio_values, _ = librosa.load(resolved_path, sr=SAMPLE_RATE, mono=True)
    audio_values = audio_values[np.newaxis, :].astype(np.float32)

    encoder_input_name = speech_encoder_session.get_inputs()[0].name
    encoder_input = {encoder_input_name: audio_values}
    cond_emb, prompt_token, speaker_embeddings, speaker_features = speech_encoder_session.run(
        None, encoder_input
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


def _coerce_past_key_values_for_session(
    past_key_values: Dict[str, np.ndarray],
    language_model_session,
) -> Dict[str, np.ndarray]:
    """Casts KV cache tensors to the dtypes expected by the given session inputs."""
    coerced = dict(past_key_values)
    for inp in language_model_session.get_inputs():
        if "past_key_values" not in inp.name:
            continue
        if inp.name not in coerced:
            continue
        expected = np.float16 if inp.type == "tensor(float16)" else np.float32
        value = coerced[inp.name]
        if value.dtype != expected:
            coerced[inp.name] = value.astype(expected)
    return coerced


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

    def run_with_sessions(
        sessions: tuple[object, object, object, object],
        *,
        dtype_key: str,
        providers_override: Tuple[str, ...],
    ) -> Tuple[np.ndarray, int]:
        tokenizer = _get_tokenizer()
        (
            _speech_encoder_session,
            embed_tokens_session,
            language_model_session,
            cond_decoder_session,
        ) = sessions

        resolved_path, mtime_ns = _speaker_cache_key(voice_wav_path)
        cond_emb, prompt_token, speaker_embeddings, speaker_features = _encode_speaker(
            dtype_key, resolved_path, mtime_ns, providers_override
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
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1).repeat(
            batch_size, axis=0
        )

        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(
            penalty=repetition_penalty
        )
        generate_tokens = np.array([[START_SPEECH_TOKEN]], dtype=np.int64)

        def step_lm(
            lm_session,
            embeds: np.ndarray,
            attention_mask: np.ndarray,
            position_ids: np.ndarray,
            past_key_values: Dict[str, np.ndarray],
        ) -> tuple[np.ndarray, list[np.ndarray]]:
            outputs = lm_session.run(
                None,
                dict(
                    inputs_embeds=embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **past_key_values,
                ),
            )
            return outputs[0], list(outputs[1:])

        # CoreML EP can fail when past_key_values have a 0-length sequence dimension.
        # Work around by running the first LM step on CPU to "prime" the cache,
        # then continue on the requested provider.
        try:
            session_providers = (
                language_model_session.get_providers()
                if hasattr(language_model_session, "get_providers")
                else []
            )
        except Exception:
            session_providers = []
        use_coreml = "CoreMLExecutionProvider" in list(session_providers)

        past_key_names = [
            inp.name
            for inp in language_model_session.get_inputs()
            if "past_key_values" in inp.name
        ]

        remaining_steps = int(max_new_tokens)
        current_embeds = inputs_embeds

        if use_coreml and not providers_override:
            try:
                cpu_sessions = _get_sessions(
                    dtype_key, ("CPUExecutionProvider",))
                _cpu_speech, cpu_embed, cpu_lm, _cpu_decoder = cpu_sessions
                cpu_past_key_values, cpu_past_key_names = _init_past_key_values(
                    cpu_lm, batch_size
                )
                logits, present_values = step_lm(
                    cpu_lm,
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    cpu_past_key_values,
                )
                logits = logits[:, -1, :]
                next_token_logits = repetition_penalty_processor(
                    generate_tokens, logits
                )
                next_token = np.argmax(
                    next_token_logits, axis=-1, keepdims=True
                ).astype(np.int64)
                generate_tokens = np.concatenate(
                    (generate_tokens, next_token), axis=-1
                )
                if not (next_token.flatten() == STOP_SPEECH_TOKEN).all():
                    past_key_values = _update_past_key_values(
                        cpu_lm, cpu_past_key_names, present_values
                    )
                    past_key_values = _coerce_past_key_values_for_session(
                        past_key_values, language_model_session
                    )
                    attention_mask = np.concatenate(
                        [attention_mask, np.ones(
                            (batch_size, 1), dtype=np.int64)],
                        axis=1,
                    )
                    position_ids = position_ids[:, -1:] + 1
                    current_embeds = embed_tokens_session.run(
                        None, {embed_input_name: next_token}
                    )[0]
                    remaining_steps = max(0, remaining_steps - 1)
                else:
                    remaining_steps = 0
            except Exception:
                # If priming fails, fall back to normal execution (and let the outer retry handle).
                past_key_values, _ = _init_past_key_values(
                    language_model_session, batch_size
                )
        else:
            past_key_values, past_key_names = _init_past_key_values(
                language_model_session, batch_size
            )

        for _step in range(int(remaining_steps)):
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
            [prompt_token, speech_tokens, silence_tokens], axis=1)

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

    dtype_key = (dtype or "fp32").strip().lower()
    try:
        sessions = _get_sessions(dtype_key, ())
        return run_with_sessions(
            sessions, dtype_key=dtype_key, providers_override=()
        )
    except Exception as exc:
        # CoreML EP can fail at runtime for dynamic/empty shapes; retry with MPS then CPU.
        if _is_coreml_runtime_failure(exc):
            for override in _fallback_provider_overrides_for_runtime():
                try:
                    sessions = _get_sessions(dtype_key, override)
                    return run_with_sessions(
                        sessions,
                        dtype_key=dtype_key,
                        providers_override=override,
                    )
                except Exception:
                    continue

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
        _get_sessions(dtype, ())
        voice_wav_path = (voice_wav_path or "").strip()
        if voice_wav_path:
            resolved_path, mtime_ns = _speaker_cache_key(voice_wav_path)
            _encode_speaker(dtype, resolved_path, mtime_ns, ())
        return True
    except Exception as exc:
        logger.warning(f"Chatterbox prewarm failed: {exc}")
        return False
