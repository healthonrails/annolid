"""Pure ONNX Runtime multimodal embedding adapter for Google EmbeddingGemma 2.

Supports unified text, image, multi-frame video clip, and interleaved
multimodal embeddings via ``onnx-community/embeddinggemma-2-ONNX`` (or a local
exported ONNX directory) without requiring PyTorch or gated checkpoints.

Key properties verified against ``onnx-community/embeddinggemma-2-ONNX``:
- Modular ONNX graphs under ``onnx/``:
  - ``model*.onnx`` (+ ``.onnx_data``): text/multimodal backbone emitting
    L2-normalized 768-d ``sentence_embedding`` (and ``last_hidden_state``).
  - ``vision_encoder*.onnx`` (+ ``.onnx_data``): 170M Gemma 4 vision encoder
    emitting projected 512-d ``image_features`` soft tokens.
- Precision variants: ``q4`` (default, ~270 MB text+vision), ``quantized``
  (INT8, ~485 MB), ``q4f16`` (~242 MB), ``fp32`` (~1.67 GB), ``fp16`` (~837 MB).
- Image preprocessing matches ``Gemma4ImageProcessor`` / ``processor_config.json``:
  aspect-ratio-preserving BICUBIC resize to multiples of
  ``patch_size * pooling_kernel_size`` (48 px), rescaling uint8 ``[0, 255]`` to
  float32 ``[0.0, 1.0]`` (``vision_encoder*.onnx`` internally applies
  ``(x - 0.5) * 2.0``), patchification to ``[N_patches, 768]``, and 2D
  ``pixel_position_ids`` ``(x=col, y=row)`` with ``(-1, -1)`` padding.
- Native Matryoshka Representation Learning (MRL) truncation to
  ``{128, 256, 512, 768}`` with L2 renormalization after slicing.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from annolid.core.models.base import (
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    RuntimeModel,
)

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDINGGEMMA2_ONNX_REPO = "onnx-community/embeddinggemma-2-ONNX"
DEFAULT_EMBEDDINGGEMMA2_QUANTIZATION = "q4"
VALID_MATRYOSHKA_DIMS: Tuple[int, ...] = (128, 256, 512, 768)
SUPPORTED_MAX_SOFT_TOKENS: Tuple[int, ...] = (70, 140, 280, 560, 1120)

# Official task prefixes from onnx-community/embeddinggemma-2-ONNX
# (config_sentence_transformers.json). Standalone image/video embeddings must
# NOT receive a text prefix.
EMBEDDINGGEMMA2_PROMPTS: Dict[str, str] = {
    "SearchQuery": "task: search result | query: ",
    "query": "task: search result | query: ",
    "Retrieval": "task: search result | query: ",
    "Retrieval-query": "task: search result | query: ",
    "Reranking": "task: search result | query: ",
    "BitextMining": "task: search result | query: ",
    "Document": "title: none | text: ",
    "document": "title: none | text: ",
    "Retrieval-document": "title: none | text: ",
    "Classification": "task: classification | query: ",
    "MultilabelClassification": "task: classification | query: ",
    "Clustering": "task: clustering | query: ",
    "CodeRetrieval": "task: code retrieval | query: ",
    "InstructionRetrieval": "task: code retrieval | query: ",
    "FactChecking": "task: fact checking | query: ",
    "QuestionAnswering": "task: question answering | query: ",
    "STS": "task: sentence similarity | query: ",
    "SentenceSimilarity": "task: sentence similarity | query: ",
    "PairClassification": "task: sentence similarity | query: ",
    "Summarization": "task: sentence similarity | query: ",
}

_QUANTIZATION_ALIASES: Dict[str, str] = {
    "q4": "q4",
    "int4": "q4",
    "4bit": "q4",
    "q4f16": "q4f16",
    "quantized": "quantized",
    "int8": "quantized",
    "q8": "quantized",
    "uint8": "quantized",
    "fp32": "fp32",
    "float32": "fp32",
    "full": "fp32",
    "default": "fp32",
    "fp16": "fp16",
    "float16": "fp16",
}

_QUANTIZATION_SUFFIX: Dict[str, str] = {
    "fp32": "",
    "fp16": "_fp16",
    "quantized": "_quantized",
    "q4": "_q4",
    "q4f16": "_q4f16",
}


def normalize_quantization_preset(value: Optional[str]) -> str:
    """Normalize user/config quantization names to canonical repo presets."""
    raw = str(value or DEFAULT_EMBEDDINGGEMMA2_QUANTIZATION).strip().lower()
    if raw not in _QUANTIZATION_ALIASES:
        supported = ", ".join(sorted(_QUANTIZATION_SUFFIX))
        raise ValueError(
            f"Unsupported EmbeddingGemma 2 ONNX quantization {value!r}. "
            f"Choose one of: {supported}."
        )
    return _QUANTIZATION_ALIASES[raw]


def format_embeddinggemma2_text(
    text: str,
    *,
    prompt_name: Optional[str] = None,
    prompt_prefix: Optional[str] = None,
) -> str:
    """Apply an EmbeddingGemma 2 task prefix unless already present."""
    clean = str(text or "").strip()
    if not clean:
        return ""
    if prompt_prefix is not None:
        prefix = str(prompt_prefix)
    elif prompt_name:
        key = str(prompt_name).strip()
        prefix = EMBEDDINGGEMMA2_PROMPTS.get(key)
        if prefix is None:
            lowered = key.lower()
            for cand_k, cand_v in EMBEDDINGGEMMA2_PROMPTS.items():
                if cand_k.lower() == lowered:
                    prefix = cand_v
                    break
        if prefix is None:
            raise ValueError(
                f"Unknown EmbeddingGemma 2 prompt_name {prompt_name!r}. "
                f"Known prompts: {', '.join(sorted(EMBEDDINGGEMMA2_PROMPTS))}."
            )
    else:
        prefix = ""

    if not prefix:
        return clean
    if clean.startswith(prefix):
        return clean
    return f"{prefix}{clean}"


def l2_normalize_np(
    arr: np.ndarray, *, axis: int = -1, eps: float = 1e-12
) -> np.ndarray:
    """Return float32 L2-normalized vectors without dividing by zero."""
    out = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(out, axis=axis, keepdims=True)
    safe_norms = np.maximum(norms, np.float32(eps))
    return (out / safe_norms).astype(np.float32, copy=False)


def apply_matryoshka_truncation_np(
    embeddings: np.ndarray,
    matryoshka_dim: Optional[int],
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Slice Matryoshka prefix ``[..., :matryoshka_dim]`` and L2-renormalize."""
    arr = np.asarray(embeddings, dtype=np.float32)
    if matryoshka_dim is not None:
        target = int(matryoshka_dim)
        if target <= 0:
            raise ValueError(f"matryoshka_dim must be positive, got {matryoshka_dim!r}")
        if target < int(arr.shape[-1]):
            arr = arr[..., :target].copy()
    if normalize:
        arr = l2_normalize_np(arr, axis=-1)
    return arr


def get_aspect_ratio_preserving_size(
    height: int,
    width: int,
    *,
    patch_size: int = 16,
    max_patches: int = 2520,
    pooling_kernel_size: int = 3,
) -> Tuple[int, int]:
    """Compute Gemma 4 aspect-ratio-preserving ``(target_h, target_w)``.

    Mirrors ``Gemma4ImageProcessor.get_aspect_ratio_preserving_size``:
    - Both dimensions are positive multiples of ``patch_size * pooling_kernel_size``.
    - Total patches ``(target_h // patch_size) * (target_w // patch_size)`` never
      exceed ``max_patches``.
    """
    h = max(1, int(height))
    w = max(1, int(width))
    ps = max(1, int(patch_size))
    pk = max(1, int(pooling_kernel_size))
    side_mult = ps * pk
    max_p = max(pk * pk, int(max_patches))

    total_px = float(h * w)
    target_px = float(max_p * ps * ps)
    factor = math.sqrt(target_px / total_px)
    ideal_h = factor * float(h)
    ideal_w = factor * float(w)

    target_h = int(math.floor(ideal_h / side_mult)) * side_mult
    target_w = int(math.floor(ideal_w / side_mult)) * side_mult

    if target_h == 0 and target_w == 0:
        target_h = side_mult
        target_w = side_mult
    elif target_h == 0:
        target_h = side_mult
        max_w_blocks = max(1, max_p // ((target_h // ps) * pk))
        target_w = min(
            max(side_mult, int(math.floor(ideal_w / side_mult)) * side_mult),
            max_w_blocks * side_mult,
        )
    elif target_w == 0:
        target_w = side_mult
        max_h_blocks = max(1, max_p // ((target_w // ps) * pk))
        target_h = min(
            max(side_mult, int(math.floor(ideal_h / side_mult)) * side_mult),
            max_h_blocks * side_mult,
        )

    # Guarantee patch budget constraint under rounding edge cases.
    while (target_h // ps) * (target_w // ps) > max_p:
        if target_h >= target_w and target_h > side_mult:
            target_h -= side_mult
        elif target_w > side_mult:
            target_w -= side_mult
        else:
            break

    return int(target_h), int(target_w)


def _to_rgb_pil(image: Union[Image.Image, np.ndarray, str, Path]) -> Image.Image:
    if isinstance(image, (str, Path)):
        with Image.open(Path(image)) as opened:
            return opened.convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"Expected an HxWx3 RGB image array, got shape {tuple(arr.shape)}."
        )
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.0:
            arr = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def preprocess_image_gemma4(
    image: Union[Image.Image, np.ndarray, str, Path],
    *,
    patch_size: int = 16,
    pooling_kernel_size: int = 3,
    max_soft_tokens: int = 280,
    rescale_factor: float = 1.0 / 255.0,
    do_normalize: bool = False,
    image_mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    image_std: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Preprocess one RGB image into Gemma 4 ONNX vision encoder tensors.

    Returns
    -------
    patches : np.ndarray
        ``[num_patches, patch_size * patch_size * 3]`` float32 in ``[0, 1]``
        (unless ``do_normalize=True``).
    position_ids : np.ndarray
        ``[num_patches, 2]`` int64 ``(x=col, y=row)`` patch grid coordinates.
    num_soft_tokens : int
        Exact number of pooled vision soft tokens produced by ``vision_encoder``.
    """
    pil = _to_rgb_pil(image)
    width, height = pil.size
    max_patches = max(1, int(max_soft_tokens)) * (int(pooling_kernel_size) ** 2)
    target_h, target_w = get_aspect_ratio_preserving_size(
        height=height,
        width=width,
        patch_size=patch_size,
        max_patches=max_patches,
        pooling_kernel_size=pooling_kernel_size,
    )
    if (width, height) != (target_w, target_h):
        resample = getattr(
            getattr(Image, "Resampling", Image),
            "BICUBIC",
            Image.BICUBIC,
        )
        pil = pil.resize((target_w, target_h), resample=resample)

    arr = np.asarray(pil, dtype=np.float32) * np.float32(rescale_factor)
    if do_normalize:
        mean = np.asarray(image_mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(image_std, dtype=np.float32).reshape(1, 1, 3)
        arr = (arr - mean) / std

    ps = int(patch_size)
    pk = int(pooling_kernel_size)
    patch_h = target_h // ps
    patch_w = target_w // ps

    # Equivalent to (C, H, W) -> (C, ph, p, pw, p) -> permute(1, 3, 2, 4, 0) -> (ph*pw, p*p*C)
    patches = (
        arr.reshape(patch_h, ps, patch_w, ps, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(patch_h * patch_w, ps * ps * 3)
        .astype(np.float32, copy=False)
    )

    xs, ys = np.meshgrid(
        np.arange(patch_w, dtype=np.int64),
        np.arange(patch_h, dtype=np.int64),
        indexing="xy",
    )
    position_ids = np.stack([xs, ys], axis=-1).reshape(patch_h * patch_w, 2)
    num_soft_tokens = (patch_h // pk) * (patch_w // pk)
    return patches, position_ids, int(num_soft_tokens)


def preprocess_images_batch_gemma4(
    images: Sequence[Union[Image.Image, np.ndarray, str, Path]],
    *,
    patch_size: int = 16,
    pooling_kernel_size: int = 3,
    max_soft_tokens: int = 280,
    rescale_factor: float = 1.0 / 255.0,
    do_normalize: bool = False,
    image_mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    image_std: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Batch-preprocess RGB images with ``(-1, -1)`` position padding if needed."""
    if not images:
        patch_dim = int(patch_size) * int(patch_size) * 3
        return (
            np.empty((0, 0, patch_dim), dtype=np.float32),
            np.empty((0, 0, 2), dtype=np.int64),
            [],
        )

    patch_list: List[np.ndarray] = []
    pos_list: List[np.ndarray] = []
    soft_counts: List[int] = []
    for img in images:
        patches, pos_ids, soft_n = preprocess_image_gemma4(
            img,
            patch_size=patch_size,
            pooling_kernel_size=pooling_kernel_size,
            max_soft_tokens=max_soft_tokens,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
        )
        patch_list.append(patches)
        pos_list.append(pos_ids)
        soft_counts.append(soft_n)

    max_len = max(p.shape[0] for p in patch_list)
    patch_dim = patch_list[0].shape[1]
    batch_size = len(patch_list)

    if all(p.shape[0] == max_len for p in patch_list):
        pixel_values = np.stack(patch_list, axis=0).astype(np.float32, copy=False)
        position_ids = np.stack(pos_list, axis=0).astype(np.int64, copy=False)
        return pixel_values, position_ids, soft_counts

    pixel_values = np.zeros((batch_size, max_len, patch_dim), dtype=np.float32)
    position_ids = np.full((batch_size, max_len, 2), -1, dtype=np.int64)
    for i, (patches, pos_ids) in enumerate(zip(patch_list, pos_list)):
        n = patches.shape[0]
        pixel_values[i, :n] = patches
        position_ids[i, :n] = pos_ids
    return pixel_values, position_ids, soft_counts


def create_gemma4_mel_filterbank(
    *,
    n_freqs: int = 257,
    f_min: float = 0.0,
    f_max: float = 8000.0,
    n_mels: int = 128,
    sample_rate: int = 16000,
    fft_length: int = 512,
) -> np.ndarray:
    """Build the HTK triangular Mel filterbank ``[n_freqs, n_mels]`` for Gemma 4 audio."""
    all_freqs = np.arange(n_freqs, dtype=np.float32) * (
        float(sample_rate) / float(fft_length)
    )
    m_min = 2595.0 * math.log10(1.0 + (float(f_min) / 700.0))
    m_max = 2595.0 * math.log10(1.0 + (float(f_max) / 700.0))
    m_pts = np.linspace(m_min, m_max, int(n_mels) + 2, dtype=np.float64)
    f_pts = 700.0 * (np.power(10.0, m_pts / 2595.0) - 1.0)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = np.expand_dims(f_pts, 0) - np.expand_dims(all_freqs.astype(np.float64), 1)
    down_slopes = (-1.0 * slopes[:, :-2]) / np.maximum(f_diff[:-1], 1e-12)
    up_slopes = slopes[:, 2:] / np.maximum(f_diff[1:], 1e-12)
    fb = np.maximum(0.0, np.minimum(down_slopes, up_slopes))
    return fb.astype(np.float32)


def _load_audio_waveform(
    audio: Union[np.ndarray, Sequence[float], Tuple[Any, int], str, Path],
    *,
    default_sample_rate: int = 16000,
) -> Tuple[np.ndarray, int]:
    """Coerce waveform array, ``(waveform, sr)`` tuple, or ``.wav`` path to mono float32."""
    sample_rate = int(default_sample_rate)
    if isinstance(audio, (str, Path)):
        path = Path(audio).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {path}")
        with wave.open(str(path), "rb") as wf:
            sample_rate = int(wf.getframerate())
            n_channels = int(wf.getnchannels())
            sampwidth = int(wf.getsampwidth())
            raw_frames = wf.readframes(wf.getnframes())
        if sampwidth == 1:
            arr = (
                np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32) - 128.0
            ) / 128.0
        elif sampwidth == 2:
            arr = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            arr = (
                np.frombuffer(raw_frames, dtype=np.int32).astype(np.float32)
                / 2147483648.0
            )
        else:
            raise ValueError(
                f"Unsupported WAV sample width ({sampwidth} bytes) in {path}."
            )
        if n_channels > 1:
            arr = arr.reshape(-1, n_channels).mean(axis=1)
        return arr.astype(np.float32, copy=False), sample_rate

    if (
        isinstance(audio, tuple)
        and len(audio) == 2
        and isinstance(audio[1], (int, float))
    ):
        raw_wav, sr = audio
        sample_rate = max(1, int(sr))
        arr = np.asarray(raw_wav, dtype=np.float32)
    else:
        arr = np.asarray(audio, dtype=np.float32)

    if arr.ndim == 2:
        # Downmix multi-channel audio ([T, C] or [C, T]) to mono.
        if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=-1)
    elif arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(np.float32, copy=False), sample_rate


def preprocess_audio_gemma4(
    audio: Union[np.ndarray, Sequence[float], Tuple[Any, int], str, Path],
    *,
    sample_rate: int = 16000,
    target_sample_rate: int = 16000,
    frame_length: int = 320,
    hop_length: int = 160,
    fft_length: int = 512,
    n_mels: int = 128,
    min_frequency: float = 0.0,
    max_frequency: float = 8000.0,
    mel_floor: float = 0.001,
    preemphasis: float = 0.0,
    preemphasis_htk_flavor: bool = True,
    input_scale_factor: float = 1.0,
    per_bin_mean: Optional[np.ndarray] = None,
    per_bin_stddev: Optional[np.ndarray] = None,
    max_samples: int = 480_000,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Convert one audio clip into Gemma 4 log-Mel features, bool mask, and token count."""
    wav, src_sr = _load_audio_waveform(audio, default_sample_rate=sample_rate)
    tgt_sr = max(1, int(target_sample_rate))
    if src_sr != tgt_sr and wav.size > 1:
        new_len = max(1, int(round(float(wav.size) * float(tgt_sr) / float(src_sr))))
        old_x = np.linspace(0.0, 1.0, wav.size, dtype=np.float64)
        new_x = np.linspace(0.0, 1.0, new_len, dtype=np.float64)
        wav = np.interp(new_x, old_x, wav.astype(np.float64)).astype(np.float32)

    if max_samples > 0 and wav.size > int(max_samples):
        wav = wav[: int(max_samples)]
    if float(input_scale_factor) != 1.0:
        wav = wav * np.float32(input_scale_factor)

    fl = max(1, int(frame_length))
    hl = max(1, int(hop_length))
    n_fft = max(fl, int(fft_length))
    pad_left = fl // 2
    wav_padded = np.pad(wav, (pad_left, 0), mode="constant")
    need_len = fl + (1 if float(preemphasis) > 0.0 else 0)
    if wav_padded.size < need_len:
        wav_padded = np.pad(
            wav_padded, (0, need_len - wav_padded.size), mode="constant"
        )

    n_frames = max(1, (wav_padded.size - need_len) // hl + 1)
    idx = (
        np.arange(need_len, dtype=np.int64)[None, :]
        + (np.arange(n_frames, dtype=np.int64) * hl)[:, None]
    )
    frames = wav_padded[idx]

    pe = float(preemphasis)
    if pe > 0.0:
        if preemphasis_htk_flavor:
            first = frames[:, :1] * (1.0 - pe)
            rest = frames[:, 1:-1] - pe * frames[:, :-2]
            frames = np.concatenate([first, rest], axis=-1)
        else:
            frames = frames[:, 1:] - pe * frames[:, :-1]

    hann_arange = np.arange(fl, dtype=np.float32)
    window = (0.5 * (1.0 - np.cos(2.0 * np.pi * hann_arange / float(fl)))).astype(
        np.float32
    )
    stft = np.fft.rfft(frames * window[None, :], n=n_fft, axis=-1)
    mag = np.abs(stft).astype(np.float32)
    fb = create_gemma4_mel_filterbank(
        n_freqs=(n_fft // 2) + 1,
        f_min=min_frequency,
        f_max=max_frequency,
        n_mels=n_mels,
        sample_rate=tgt_sr,
        fft_length=n_fft,
    )
    mel = mag @ fb
    log_mel = np.log(np.maximum(mel, np.float32(mel_floor))).astype(np.float32)
    if per_bin_mean is not None:
        log_mel = log_mel - np.asarray(per_bin_mean, dtype=np.float32).reshape(1, -1)
    if per_bin_stddev is not None:
        log_mel = log_mel / np.asarray(per_bin_stddev, dtype=np.float32).reshape(1, -1)

    mask = np.ones((log_mel.shape[0],), dtype=bool)
    # Gemma 4 audio subsampling conv stacks reduce time frames by 4x (ceil(T / 4)).
    num_audio_tokens = (int(log_mel.shape[0]) + 3) // 4
    return log_mel, mask, int(num_audio_tokens)


def preprocess_audios_batch_gemma4(
    audios: Sequence[Union[np.ndarray, Sequence[float], Tuple[Any, int], str, Path]],
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Batch-preprocess variable-length audio clips with right-zero / False mask padding."""
    if not audios:
        n_mels = int(kwargs.get("n_mels", 128))
        return (
            np.empty((0, 0, n_mels), dtype=np.float32),
            np.empty((0, 0), dtype=bool),
            [],
        )

    feat_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []
    token_counts: List[int] = []
    for item in audios:
        feats, mask, tok_n = preprocess_audio_gemma4(item, **kwargs)
        feat_list.append(feats)
        mask_list.append(mask)
        token_counts.append(tok_n)

    max_t = max(f.shape[0] for f in feat_list)
    n_mels = feat_list[0].shape[1]
    batch_size = len(feat_list)
    padded_feats = np.zeros((batch_size, max_t, n_mels), dtype=np.float32)
    padded_masks = np.zeros((batch_size, max_t), dtype=bool)
    for i, (feats, mask) in enumerate(zip(feat_list, mask_list)):
        t = feats.shape[0]
        padded_feats[i, :t] = feats
        padded_masks[i, :t] = mask
    return padded_feats, padded_masks, token_counts


@dataclass(frozen=True)
class EmbeddingGemma2OnnxConfig:
    """Configuration for ``EmbeddingGemma2OnnxAdapter``."""

    model_id: str = DEFAULT_EMBEDDINGGEMMA2_ONNX_REPO
    quantization: str = DEFAULT_EMBEDDINGGEMMA2_QUANTIZATION
    matryoshka_dim: Optional[int] = None
    max_length: int = 8192
    image_max_soft_tokens: int = 280
    video_max_soft_tokens: int = 140
    default_text_prompt_name: Optional[str] = "SearchQuery"
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    providers: Optional[Tuple[str, ...]] = None
    intra_op_num_threads: Optional[int] = None
    inter_op_num_threads: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            str(self.model_id or DEFAULT_EMBEDDINGGEMMA2_ONNX_REPO).strip(),
        )
        object.__setattr__(
            self,
            "quantization",
            normalize_quantization_preset(self.quantization),
        )
        if self.matryoshka_dim is not None:
            dim = int(self.matryoshka_dim)
            if dim <= 0:
                raise ValueError(f"matryoshka_dim must be positive, got {dim!r}")
            object.__setattr__(self, "matryoshka_dim", dim)
        object.__setattr__(self, "max_length", max(16, int(self.max_length)))
        object.__setattr__(
            self, "image_max_soft_tokens", max(1, int(self.image_max_soft_tokens))
        )
        object.__setattr__(
            self, "video_max_soft_tokens", max(1, int(self.video_max_soft_tokens))
        )
        if self.providers is not None:
            object.__setattr__(
                self,
                "providers",
                tuple(str(p).strip() for p in self.providers if str(p).strip()),
            )


def _require_onnxruntime():
    try:
        import onnxruntime as ort  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "EmbeddingGemma 2 ONNX requires 'onnxruntime'. "
            "Install it with: pip install onnxruntime"
        ) from exc
    except ImportError as exc:
        raise ImportError(
            "Failed to import 'onnxruntime' for EmbeddingGemma 2 ONNX. "
            f"Original error: {exc}"
        ) from exc
    return ort


def _require_tokenizers():
    try:
        from tokenizers import Tokenizer  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "EmbeddingGemma 2 ONNX requires 'tokenizers'. "
            "Install it with: pip install tokenizers"
        ) from exc
    except ImportError as exc:
        raise ImportError(
            "Failed to import 'tokenizers' for EmbeddingGemma 2 ONNX. "
            f"Original error: {exc}"
        ) from exc
    return Tokenizer


def _select_ort_providers(
    available_providers: Sequence[str],
    requested: Optional[Sequence[str]] = None,
) -> List[str]:
    """Select reliable ONNX Runtime providers for dynamic multimodal graphs.

    Prefers CUDA when available and falls back to CPU. CoreML is excluded by
    default because dynamic transformer sequence shapes commonly fail on CoreML;
    callers can opt into it explicitly via ``config.providers``.
    """
    available = list(available_providers or [])
    available_set = set(available)

    if requested:
        filtered = [p for p in requested if not available_set or p in available_set]
        if "CPUExecutionProvider" not in filtered:
            filtered.append("CPUExecutionProvider")
        return filtered

    providers: List[str] = []
    if "CUDAExecutionProvider" in available_set:
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available_set or not providers:
        providers.append("CPUExecutionProvider")
    return providers


def _onnx_filename(component: str, quantization: str) -> str:
    suffix = _QUANTIZATION_SUFFIX[normalize_quantization_preset(quantization)]
    return f"{component}{suffix}.onnx"


class EmbeddingGemma2OnnxAdapter(RuntimeModel):
    """Multimodal ONNX Runtime embedding adapter for Google EmbeddingGemma 2."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_EMBEDDINGGEMMA2_ONNX_REPO,
        quantization: str = DEFAULT_EMBEDDINGGEMMA2_QUANTIZATION,
        matryoshka_dim: Optional[int] = None,
        config: Optional[EmbeddingGemma2OnnxConfig] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        providers: Optional[Sequence[str]] = None,
        load_vision: bool = True,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = EmbeddingGemma2OnnxConfig(
                model_id=model_id,
                quantization=quantization,
                matryoshka_dim=matryoshka_dim,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                providers=tuple(providers) if providers else None,
            )
        self._load_vision_on_init = bool(load_vision)

        self._tokenizer: Any = None
        self._text_session: Any = None
        self._vision_session: Any = None
        self._audio_session: Any = None
        self._text_input_names: Tuple[str, ...] = ()
        self._text_output_names: Tuple[str, ...] = ()
        self._vision_input_names: Tuple[str, ...] = ()
        self._vision_output_names: Tuple[str, ...] = ()
        self._audio_input_names: Tuple[str, ...] = ()
        self._audio_output_names: Tuple[str, ...] = ()
        self._active_providers: List[str] = []

        # Architecture metadata populated from config.json / processor_config.json.
        self._hidden_size: int = 512
        self._embedding_dim: int = 768
        self._bos_token_id: int = 2
        self._eos_token_id: int = 1
        self._pad_token_id: int = 0
        self._boi_token_id: int = 255999
        self._image_token_id: int = 258880
        self._eoi_token_id: int = 258882
        self._video_token_id: int = 258884
        self._boa_token_id: int = 256000
        self._audio_token_id: int = 258881
        self._eoa_token_id: int = 258883

        self._patch_size: int = 16
        self._pooling_kernel_size: int = 3
        self._rescale_factor: float = 1.0 / 255.0
        self._do_normalize: bool = False
        self._image_mean: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._image_std: Tuple[float, float, float] = (1.0, 1.0, 1.0)
        self._audio_sample_rate: int = 16000
        self._audio_frame_length: int = 320
        self._audio_hop_length: int = 160
        self._audio_fft_length: int = 512
        self._audio_n_mels: int = 128
        self._audio_min_freq: float = 0.0
        self._audio_max_freq: float = 8000.0
        self._audio_mel_floor: float = 0.001
        self._audio_preemphasis: float = 0.0
        self._audio_preemphasis_htk_flavor: bool = True
        self._audio_input_scale_factor: float = 1.0
        self._audio_per_bin_mean: Optional[np.ndarray] = None
        self._audio_per_bin_stddev: Optional[np.ndarray] = None
        self._tokenizer_adds_bos: bool = True
        self._tokenizer_adds_eos: bool = True

        self._resolved_cache_dir: Optional[str] = (
            self._config.cache_dir or os.getenv("ANNOLID_ONNX_CACHE_DIR") or None
        )

    @property
    def config(self) -> EmbeddingGemma2OnnxConfig:
        return self._config

    @property
    def model_id(self) -> str:
        return (
            f"embeddinggemma2-onnx:{self._config.model_id}:{self._config.quantization}"
        )

    @property
    def embedding_dim(self) -> int:
        if self._config.matryoshka_dim is not None:
            return min(int(self._embedding_dim), int(self._config.matryoshka_dim))
        return int(self._embedding_dim)

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            tasks=("embed",),
            input_modalities=("text", "image", "video", "audio"),
            output_modalities=("embedding",),
        )

    def load(self, *, include_vision: Optional[bool] = None) -> None:
        """Load tokenizer, config metadata, text/multimodal backbone, and optionally vision."""
        want_vision = (
            self._load_vision_on_init
            if include_vision is None
            else bool(include_vision)
        )
        if self._tokenizer is None:
            self._load_metadata_and_tokenizer()
        if self._text_session is None:
            model_path = self._resolve_component_onnx("model")
            self._text_session = self._create_session(model_path)
            self._text_input_names = tuple(
                inp.name for inp in self._text_session.get_inputs()
            )
            self._text_output_names = tuple(
                out.name for out in self._text_session.get_outputs()
            )
        if want_vision and self._vision_session is None:
            self._ensure_vision_loaded()

    def _ensure_vision_loaded(self) -> None:
        if self._tokenizer is None or self._text_session is None:
            self.load(include_vision=False)
        if self._vision_session is None:
            vision_path = self._resolve_component_onnx("vision_encoder")
            self._vision_session = self._create_session(vision_path)
            self._vision_input_names = tuple(
                inp.name for inp in self._vision_session.get_inputs()
            )
            self._vision_output_names = tuple(
                out.name for out in self._vision_session.get_outputs()
            )

    def _ensure_audio_loaded(self) -> None:
        """Lazily resolve and load ``audio_encoder*.onnx`` only when audio is used."""
        if self._tokenizer is None or self._text_session is None:
            self.load(include_vision=False)
        if self._audio_session is None:
            audio_path = self._resolve_component_onnx("audio_encoder")
            self._audio_session = self._create_session(audio_path)
            self._audio_input_names = tuple(
                inp.name for inp in self._audio_session.get_inputs()
            )
            self._audio_output_names = tuple(
                out.name for out in self._audio_session.get_outputs()
            )

    def close(self) -> None:
        self._text_session = None
        self._vision_session = None
        self._audio_session = None
        self._tokenizer = None
        self._text_input_names = ()
        self._text_output_names = ()
        self._vision_input_names = ()
        self._vision_output_names = ()
        self._audio_input_names = ()
        self._audio_output_names = ()
        gc.collect()

    def _local_model_dir(self) -> Optional[Path]:
        candidate = Path(self._config.model_id).expanduser()
        if candidate.is_dir():
            return candidate
        return None

    def _download_or_find_file(
        self, filename: str, *, required: bool = True
    ) -> Optional[Path]:
        local_dir = self._local_model_dir()
        if local_dir is not None:
            direct = local_dir / filename
            if direct.is_file():
                return direct
            # Support flat local exports without an ``onnx/`` subdirectory.
            if filename.startswith("onnx/"):
                flat = local_dir / Path(filename).name
                if flat.is_file():
                    return flat
            if required:
                raise FileNotFoundError(
                    f"Required EmbeddingGemma 2 ONNX asset '{filename}' not found under {local_dir}."
                )
            return None

        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "huggingface-hub is required to download EmbeddingGemma 2 ONNX weights. "
                "Install it with: pip install 'huggingface-hub>=0.22.0'"
            ) from exc

        try:
            downloaded = hf_hub_download(
                repo_id=self._config.model_id,
                filename=filename,
                cache_dir=self._resolved_cache_dir,
                local_files_only=bool(self._config.local_files_only),
            )
            return Path(downloaded)
        except Exception as exc:
            if not required:
                return None
            raise RuntimeError(
                f"Failed to resolve EmbeddingGemma 2 ONNX asset '{filename}' from "
                f"'{self._config.model_id}': {exc}"
            ) from exc

    def _resolve_component_onnx(self, component: str) -> Path:
        onnx_name = _onnx_filename(component, self._config.quantization)
        rel_onnx = f"onnx/{onnx_name}"
        local_dir = self._local_model_dir()

        if local_dir is not None:
            candidates = [
                local_dir / rel_onnx,
                local_dir / onnx_name,
                local_dir / "onnx" / f"{component}.onnx",
                local_dir / f"{component}.onnx",
            ]
            for cand in candidates:
                if cand.is_file():
                    self._validate_onnx_artifact(cand)
                    sibling_data = Path(f"{cand}_data")
                    return self._stage_onnx_for_runtime(
                        cand, sibling_data if sibling_data.exists() else None
                    )
            raise FileNotFoundError(
                f"Could not locate ONNX file for '{component}' "
                f"(quantization={self._config.quantization!r}) in {local_dir}."
            )

        # Download both graph (.onnx) and sibling external weights (.onnx_data)
        # into the same HF snapshot directory so ONNX Runtime resolves external data.
        onnx_path = self._download_or_find_file(rel_onnx, required=True)
        assert onnx_path is not None
        data_rel = f"{rel_onnx}_data"
        data_path = self._download_or_find_file(data_rel, required=False)
        self._validate_onnx_artifact(onnx_path)
        return self._stage_onnx_for_runtime(onnx_path, data_path)

    @staticmethod
    def _stage_onnx_for_runtime(onnx_path: Path, data_path: Optional[Path]) -> Path:
        """Materialize same-directory hardlinks when external ``.onnx_data`` is a symlink.

        ONNX Runtime >=1.24 validates that resolved external data paths stay
        inside the model's directory. Hugging Face Hub snapshots store files as
        symlinks into ``../../blobs/<sha256>``, which trips that check unless
        staged as non-symlink hardlinks (zero extra disk usage on the same FS).
        """
        if data_path is None or not data_path.exists():
            return onnx_path

        model_dir_real = onnx_path.parent.resolve()
        data_real = data_path.resolve()
        try:
            data_real.relative_to(model_dir_real)
            return onnx_path
        except ValueError:
            pass

        onnx_real = onnx_path.resolve()
        # Place staged hardlinks in a sibling `.ort_staged/<rev>` directory on the
        # same filesystem (outside HF `snapshots/` so cache integrity stays untouched).
        parts = onnx_path.parts
        stage_dir: Path
        if "snapshots" in parts:
            snap_idx = len(parts) - 1 - parts[::-1].index("snapshots")
            repo_root = Path(*parts[:snap_idx])
            rev = parts[snap_idx + 1] if snap_idx + 1 < len(parts) else "default"
            stage_dir = repo_root / ".ort_staged" / rev
        else:
            stage_dir = onnx_path.parent / ".ort_staged"

        stage_dir.mkdir(parents=True, exist_ok=True)
        staged_onnx = stage_dir / onnx_path.name
        staged_data = stage_dir / data_path.name

        for src_real, dst in ((onnx_real, staged_onnx), (data_real, staged_data)):
            if dst.exists():
                try:
                    if os.path.samefile(src_real, dst):
                        continue
                except OSError:
                    pass
                dst.unlink()
            try:
                os.link(src_real, dst)
            except OSError:
                shutil.copy2(src_real, dst)

        return staged_onnx

    @staticmethod
    def _validate_onnx_artifact(path: Path) -> None:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(
                f"EmbeddingGemma 2 ONNX file is missing or empty: {path}"
            )

    def _load_metadata_and_tokenizer(self) -> None:
        Tokenizer = _require_tokenizers()

        cfg_path = self._download_or_find_file("config.json", required=False)
        if cfg_path is not None and cfg_path.is_file():
            raw_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            self._boi_token_id = int(raw_cfg.get("boi_token_id", self._boi_token_id))
            self._image_token_id = int(
                raw_cfg.get("image_token_id", self._image_token_id)
            )
            self._eoi_token_id = int(raw_cfg.get("eoi_token_id", self._eoi_token_id))
            self._video_token_id = int(
                raw_cfg.get("video_token_id", self._video_token_id)
            )
            self._boa_token_id = int(raw_cfg.get("boa_token_id", self._boa_token_id))
            self._audio_token_id = int(
                raw_cfg.get("audio_token_id", self._audio_token_id)
            )
            self._eoa_token_id = int(
                raw_cfg.get(
                    "eoa_token_id",
                    raw_cfg.get("eoa_token_index", self._eoa_token_id),
                )
            )
            text_cfg = raw_cfg.get("text_config") or {}
            if isinstance(text_cfg, dict):
                self._hidden_size = int(text_cfg.get("hidden_size", self._hidden_size))
                self._embedding_dim = int(
                    text_cfg.get("embedding_dim", self._embedding_dim)
                )
                self._bos_token_id = int(
                    text_cfg.get("bos_token_id", self._bos_token_id)
                )
                self._eos_token_id = int(
                    text_cfg.get("eos_token_id", self._eos_token_id)
                )
                self._pad_token_id = int(
                    text_cfg.get("pad_token_id", self._pad_token_id)
                )
            vision_cfg = raw_cfg.get("vision_config") or {}
            if isinstance(vision_cfg, dict):
                self._patch_size = int(vision_cfg.get("patch_size", self._patch_size))
                self._pooling_kernel_size = int(
                    vision_cfg.get("pooling_kernel_size", self._pooling_kernel_size)
                )

        proc_path = self._download_or_find_file("processor_config.json", required=False)
        if proc_path is not None and proc_path.is_file():
            raw_proc = json.loads(proc_path.read_text(encoding="utf-8"))
            img_proc = raw_proc.get("image_processor") or {}
            if isinstance(img_proc, dict):
                self._patch_size = int(img_proc.get("patch_size", self._patch_size))
                self._pooling_kernel_size = int(
                    img_proc.get("pooling_kernel_size", self._pooling_kernel_size)
                )
                self._rescale_factor = float(
                    img_proc.get("rescale_factor", self._rescale_factor)
                )
                self._do_normalize = bool(
                    img_proc.get("do_normalize", self._do_normalize)
                )
                mean = img_proc.get("image_mean")
                std = img_proc.get("image_std")
                if isinstance(mean, list) and len(mean) == 3:
                    self._image_mean = (float(mean[0]), float(mean[1]), float(mean[2]))
                if isinstance(std, list) and len(std) == 3:
                    self._image_std = (float(std[0]), float(std[1]), float(std[2]))
            feat_ext = raw_proc.get("feature_extractor") or {}
            if isinstance(feat_ext, dict):
                self._audio_sample_rate = int(
                    feat_ext.get("sampling_rate", self._audio_sample_rate)
                )
                self._audio_frame_length = int(
                    feat_ext.get("frame_length", self._audio_frame_length)
                )
                self._audio_hop_length = int(
                    feat_ext.get("hop_length", self._audio_hop_length)
                )
                self._audio_fft_length = int(
                    feat_ext.get("fft_length", self._audio_fft_length)
                )
                self._audio_n_mels = int(
                    feat_ext.get("feature_size", self._audio_n_mels)
                )
                self._audio_min_freq = float(
                    feat_ext.get("min_frequency", self._audio_min_freq)
                )
                self._audio_max_freq = float(
                    feat_ext.get("max_frequency", self._audio_max_freq)
                )
                self._audio_mel_floor = float(
                    feat_ext.get("mel_floor", self._audio_mel_floor)
                )
                self._audio_preemphasis = float(
                    feat_ext.get("preemphasis", self._audio_preemphasis)
                )
                self._audio_preemphasis_htk_flavor = bool(
                    feat_ext.get(
                        "preemphasis_htk_flavor",
                        self._audio_preemphasis_htk_flavor,
                    )
                )
                self._audio_input_scale_factor = float(
                    feat_ext.get("input_scale_factor", self._audio_input_scale_factor)
                )
                if feat_ext.get("per_bin_mean") is not None:
                    self._audio_per_bin_mean = np.asarray(
                        feat_ext["per_bin_mean"], dtype=np.float32
                    )
                if feat_ext.get("per_bin_stddev") is not None:
                    self._audio_per_bin_stddev = np.asarray(
                        feat_ext["per_bin_stddev"], dtype=np.float32
                    )

        tok_path = self._download_or_find_file("tokenizer.json", required=True)
        assert tok_path is not None
        tokenizer = Tokenizer.from_file(str(tok_path))
        tokenizer.no_padding()
        tokenizer.enable_truncation(max_length=self._config.max_length)
        self._tokenizer = tokenizer

        # Probe whether tokenizer post-processor automatically inserts BOS/EOS.
        probe = tokenizer.encode("hello")
        probe_ids = list(probe.ids)
        self._tokenizer_adds_bos = bool(
            probe_ids and int(probe_ids[0]) == self._bos_token_id
        )
        self._tokenizer_adds_eos = bool(
            probe_ids and int(probe_ids[-1]) == self._eos_token_id
        )

    def _create_session(self, model_path: Path):
        ort = _require_onnxruntime()
        available = list(ort.get_available_providers() or [])
        providers = _select_ort_providers(available, requested=self._config.providers)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        intra_threads = self._config.intra_op_num_threads
        if intra_threads is None:
            env_threads = str(os.getenv("ANNOLID_ONNX_THREADS") or "").strip()
            if env_threads.isdigit() and int(env_threads) > 0:
                intra_threads = int(env_threads)
        if intra_threads is not None:
            opts.intra_op_num_threads = max(1, int(intra_threads))
        if self._config.inter_op_num_threads is not None:
            opts.inter_op_num_threads = max(1, int(self._config.inter_op_num_threads))

        try:
            session = ort.InferenceSession(
                str(model_path), providers=providers, sess_options=opts
            )
            self._active_providers = list(session.get_providers())
            return session
        except Exception as exc:
            if providers != ["CPUExecutionProvider"]:
                logger.warning(
                    "ONNX Runtime failed with providers=%s for %s (%s); falling back to CPU.",
                    providers,
                    model_path.name,
                    exc,
                )
                session = ort.InferenceSession(
                    str(model_path),
                    providers=["CPUExecutionProvider"],
                    sess_options=opts,
                )
                self._active_providers = list(session.get_providers())
                return session
            raise RuntimeError(
                f"Failed to load EmbeddingGemma 2 ONNX session '{model_path}': {exc}"
            ) from exc

    def _encode_raw_text_ids(self, text: str) -> List[int]:
        if self._tokenizer is None:
            self.load(include_vision=False)
        enc = self._tokenizer.encode(str(text), add_special_tokens=False)
        return [int(t) for t in enc.ids]

    def _wrap_sequence_ids(self, body_ids: Sequence[int]) -> List[int]:
        max_len = int(self._config.max_length)
        ids: List[int] = [int(self._bos_token_id)]
        available_body = max(0, max_len - 2)
        ids.extend(int(t) for t in body_ids[:available_body])
        ids.append(int(self._eos_token_id))
        return ids

    def _pad_id_batch(
        self, sequences: Sequence[Sequence[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = len(sequences)
        max_len = max((len(seq) for seq in sequences), default=1)
        input_ids = np.full(
            (batch_size, max_len), int(self._pad_token_id), dtype=np.int64
        )
        attention_mask = np.zeros((batch_size, max_len), dtype=np.int64)
        for i, seq in enumerate(sequences):
            n = len(seq)
            input_ids[i, :n] = np.asarray(seq, dtype=np.int64)
            attention_mask[i, :n] = 1
        return input_ids, attention_mask

    def _run_text_backbone(
        self,
        *,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        image_features: Optional[np.ndarray] = None,
        video_features: Optional[np.ndarray] = None,
        audio_features: Optional[np.ndarray] = None,
        matryoshka_dim: Optional[int] = None,
    ) -> np.ndarray:
        if self._text_session is None:
            self.load(include_vision=False)
        empty = np.empty((0, int(self._hidden_size)), dtype=np.float32)

        candidate_feeds: Dict[str, np.ndarray] = {
            "input_ids": np.asarray(input_ids, dtype=np.int64),
            "attention_mask": np.asarray(attention_mask, dtype=np.int64),
            "image_features": (
                np.asarray(image_features, dtype=np.float32)
                if image_features is not None
                else empty
            ),
            "video_features": (
                np.asarray(video_features, dtype=np.float32)
                if video_features is not None
                else empty
            ),
            "audio_features": (
                np.asarray(audio_features, dtype=np.float32)
                if audio_features is not None
                else empty
            ),
        }
        feeds = {
            name: candidate_feeds[name]
            for name in self._text_input_names
            if name in candidate_feeds
        }

        if "sentence_embedding" in self._text_output_names:
            embeddings = self._text_session.run(["sentence_embedding"], feeds)[0]
            embeddings = np.asarray(embeddings, dtype=np.float32)
        else:
            out_name = (
                "last_hidden_state"
                if "last_hidden_state" in self._text_output_names
                else self._text_output_names[0]
            )
            hidden = np.asarray(
                self._text_session.run([out_name], feeds)[0], dtype=np.float32
            )
            if hidden.ndim == 3:
                mask = attention_mask[:, :, None].astype(np.float32)
                denom = np.maximum(mask.sum(axis=1), np.float32(1e-9))
                embeddings = (hidden * mask).sum(axis=1) / denom
            else:
                embeddings = hidden

        eff_dim = (
            matryoshka_dim
            if matryoshka_dim is not None
            else self._config.matryoshka_dim
        )
        return apply_matryoshka_truncation_np(embeddings, eff_dim, normalize=True)

    def _encode_vision_features(
        self,
        images: Sequence[Union[Image.Image, np.ndarray, str, Path]],
        *,
        max_soft_tokens: int,
    ) -> Tuple[np.ndarray, List[int]]:
        self._ensure_vision_loaded()
        pixel_values, position_ids, soft_counts = preprocess_images_batch_gemma4(
            images,
            patch_size=self._patch_size,
            pooling_kernel_size=self._pooling_kernel_size,
            max_soft_tokens=max_soft_tokens,
            rescale_factor=self._rescale_factor,
            do_normalize=self._do_normalize,
            image_mean=self._image_mean,
            image_std=self._image_std,
        )
        feeds = {}
        if "pixel_values" in self._vision_input_names:
            feeds["pixel_values"] = pixel_values
        elif self._vision_input_names:
            feeds[self._vision_input_names[0]] = pixel_values
        if "pixel_position_ids" in self._vision_input_names:
            feeds["pixel_position_ids"] = position_ids
        elif len(self._vision_input_names) > 1:
            feeds[self._vision_input_names[1]] = position_ids

        out_name = (
            "image_features"
            if "image_features" in self._vision_output_names
            else self._vision_output_names[0]
        )
        features = np.asarray(
            self._vision_session.run([out_name], feeds)[0],
            dtype=np.float32,
        )
        if features.ndim == 3:
            features = features.reshape(-1, features.shape[-1])
        expected_total = int(sum(soft_counts))
        actual_total = int(features.shape[0])
        if actual_total != expected_total:
            if len(soft_counts) == 1:
                soft_counts = [actual_total]
            elif actual_total % len(soft_counts) == 0:
                per_img = actual_total // len(soft_counts)
                soft_counts = [per_img] * len(soft_counts)
            else:
                raise RuntimeError(
                    f"Vision encoder produced {actual_total} soft tokens, "
                    f"expected {expected_total} for counts {soft_counts}."
                )
        return features, soft_counts

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        prompt_name: Optional[str] = "SearchQuery",
        prompt_prefix: Optional[str] = None,
        matryoshka_dim: Optional[int] = None,
    ) -> np.ndarray:
        """Embed one or more texts into ``[N, D]`` L2-normalized float32 vectors."""
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        self.load(include_vision=False)

        sequences: List[List[int]] = []
        for raw_text in texts:
            formatted = format_embeddinggemma2_text(
                raw_text,
                prompt_name=prompt_name,
                prompt_prefix=prompt_prefix,
            )
            if not formatted:
                raise ValueError("Cannot embed an empty text string.")
            enc = self._tokenizer.encode(formatted)
            ids = [int(t) for t in enc.ids]
            if not self._tokenizer_adds_bos and (
                not ids or ids[0] != self._bos_token_id
            ):
                ids = [int(self._bos_token_id), *ids]
            if not self._tokenizer_adds_eos and (
                not ids or ids[-1] != self._eos_token_id
            ):
                ids = [*ids, int(self._eos_token_id)]
            sequences.append(ids[: self._config.max_length])

        input_ids, attention_mask = self._pad_id_batch(sequences)
        return self._run_text_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            matryoshka_dim=matryoshka_dim,
        )

    def embed_images(
        self,
        images: Sequence[Union[Image.Image, np.ndarray, str, Path]],
        *,
        texts: Optional[Sequence[Optional[str]]] = None,
        max_soft_tokens: Optional[int] = None,
        matryoshka_dim: Optional[int] = None,
    ) -> np.ndarray:
        """Embed a batch of images (optionally interleaved with text) into ``[N, D]``."""
        if not images:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        if texts is not None and len(texts) != len(images):
            raise ValueError(
                f"texts length ({len(texts)}) must match images length ({len(images)})."
            )

        soft_budget = int(
            max_soft_tokens
            if max_soft_tokens is not None
            else self._config.image_max_soft_tokens
        )
        image_features, soft_counts = self._encode_vision_features(
            images, max_soft_tokens=soft_budget
        )

        sequences: List[List[int]] = []
        for idx, soft_n in enumerate(soft_counts):
            body: List[int] = [int(self._boi_token_id)]
            body.extend([int(self._image_token_id)] * int(soft_n))
            body.append(int(self._eoi_token_id))
            if texts is not None and texts[idx]:
                body.extend(self._encode_raw_text_ids(str(texts[idx])))
            sequences.append(self._wrap_sequence_ids(body))

        input_ids, attention_mask = self._pad_id_batch(sequences)
        return self._run_text_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_features=image_features,
            matryoshka_dim=matryoshka_dim,
        )

    def embed_video_clip(
        self,
        frames: Sequence[Union[Image.Image, np.ndarray, str, Path]],
        *,
        text: Optional[str] = None,
        max_soft_tokens_per_frame: Optional[int] = None,
        matryoshka_dim: Optional[int] = None,
    ) -> np.ndarray:
        """Embed a chronological multi-frame clip into a single ``[D]`` vector.

        Uses EmbeddingGemma 2's native ``video_token_id`` (``258884``) and
        ``video_features`` fusion slot so short temporal behaviors (e.g., rearing,
        attacking, grooming bouts) are represented jointly across frames.
        """
        if not frames:
            raise ValueError("embed_video_clip requires at least one frame.")
        if len(frames) == 1 and not text:
            return self.embed_images(
                frames,
                max_soft_tokens=max_soft_tokens_per_frame
                or self._config.image_max_soft_tokens,
                matryoshka_dim=matryoshka_dim,
            )[0]

        soft_budget = int(
            max_soft_tokens_per_frame
            if max_soft_tokens_per_frame is not None
            else self._config.video_max_soft_tokens
        )
        video_features, soft_counts = self._encode_vision_features(
            frames, max_soft_tokens=soft_budget
        )

        # Respect context budget across frames if many frames were supplied.
        max_body = max(16, int(self._config.max_length) - 2)
        text_ids = self._encode_raw_text_ids(str(text)) if text else []
        body: List[int] = []
        used_tokens = 0
        kept_slices: List[np.ndarray] = []
        cursor = 0
        for soft_n in soft_counts:
            needed = int(soft_n) + 2
            if used_tokens + needed + len(text_ids) > max_body and kept_slices:
                break
            body.append(int(self._boi_token_id))
            body.extend([int(self._video_token_id)] * int(soft_n))
            body.append(int(self._eoi_token_id))
            kept_slices.append(video_features[cursor : cursor + int(soft_n)])
            cursor += int(soft_n)
            used_tokens += needed
        if text_ids:
            body.extend(text_ids)

        merged_video_features = (
            np.concatenate(kept_slices, axis=0)
            if kept_slices
            else np.empty((0, int(self._hidden_size)), dtype=np.float32)
        )
        input_ids, attention_mask = self._pad_id_batch([self._wrap_sequence_ids(body)])
        return self._run_text_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            video_features=merged_video_features,
            matryoshka_dim=matryoshka_dim,
        )[0]

    def _encode_audio_features(
        self,
        audios: Sequence[
            Union[np.ndarray, Sequence[float], Tuple[Any, int], str, Path]
        ],
        *,
        sample_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        self._ensure_audio_loaded()
        sr = int(sample_rate) if sample_rate is not None else self._audio_sample_rate
        input_features, input_mask, token_counts = preprocess_audios_batch_gemma4(
            audios,
            sample_rate=sr,
            target_sample_rate=self._audio_sample_rate,
            frame_length=self._audio_frame_length,
            hop_length=self._audio_hop_length,
            fft_length=self._audio_fft_length,
            n_mels=self._audio_n_mels,
            min_frequency=self._audio_min_freq,
            max_frequency=self._audio_max_freq,
            mel_floor=self._audio_mel_floor,
            preemphasis=self._audio_preemphasis,
            preemphasis_htk_flavor=self._audio_preemphasis_htk_flavor,
            input_scale_factor=self._audio_input_scale_factor,
            per_bin_mean=self._audio_per_bin_mean,
            per_bin_stddev=self._audio_per_bin_stddev,
        )
        feeds: Dict[str, np.ndarray] = {}
        if "input_features" in self._audio_input_names:
            feeds["input_features"] = input_features
        elif self._audio_input_names:
            feeds[self._audio_input_names[0]] = input_features
        if "input_features_mask" in self._audio_input_names:
            feeds["input_features_mask"] = input_mask
        elif len(self._audio_input_names) > 1:
            feeds[self._audio_input_names[1]] = input_mask

        out_name = (
            "audio_features"
            if "audio_features" in self._audio_output_names
            else self._audio_output_names[0]
        )
        features = np.asarray(
            self._audio_session.run([out_name], feeds)[0],
            dtype=np.float32,
        )
        if features.ndim == 3:
            features = features.reshape(-1, features.shape[-1])
        expected_total = int(sum(token_counts))
        actual_total = int(features.shape[0])
        if actual_total != expected_total:
            if len(token_counts) == 1:
                token_counts = [actual_total]
            elif actual_total % len(token_counts) == 0:
                token_counts = [actual_total // len(token_counts)] * len(token_counts)
            else:
                raise RuntimeError(
                    f"Audio encoder produced {actual_total} tokens, "
                    f"expected {expected_total} for counts {token_counts}."
                )
        return features, token_counts

    def embed_audios(
        self,
        audios: Sequence[
            Union[np.ndarray, Sequence[float], Tuple[Any, int], str, Path]
        ],
        *,
        sample_rate: Optional[int] = None,
        texts: Optional[Sequence[Optional[str]]] = None,
        matryoshka_dim: Optional[int] = None,
    ) -> np.ndarray:
        """Lazily load ``audio_encoder`` and embed one or more audio clips into ``[N, D]``."""
        if not audios:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        if texts is not None and len(texts) != len(audios):
            raise ValueError(
                f"texts length ({len(texts)}) must match audios length ({len(audios)})."
            )

        audio_features, token_counts = self._encode_audio_features(
            audios, sample_rate=sample_rate
        )
        sequences: List[List[int]] = []
        for idx, tok_n in enumerate(token_counts):
            body: List[int] = [int(self._boa_token_id)]
            body.extend([int(self._audio_token_id)] * int(tok_n))
            body.append(int(self._eoa_token_id))
            if texts is not None and texts[idx]:
                body.extend(self._encode_raw_text_ids(str(texts[idx])))
            sequences.append(self._wrap_sequence_ids(body))

        input_ids, attention_mask = self._pad_id_batch(sequences)
        return self._run_text_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            matryoshka_dim=matryoshka_dim,
        )

    def embed_audio(
        self,
        audio: Union[np.ndarray, Sequence[float], Tuple[Any, int], str, Path],
        *,
        sample_rate: Optional[int] = None,
        text: Optional[str] = None,
        matryoshka_dim: Optional[int] = None,
    ) -> np.ndarray:
        """Embed a single audio clip (waveform, ``(wav, sr)``, or ``.wav`` path) into ``[D]``."""
        return self.embed_audios(
            [audio],
            sample_rate=sample_rate,
            texts=[text] if text else None,
            matryoshka_dim=matryoshka_dim,
        )[0]

    def predict(self, request: ModelRequest) -> ModelResponse:
        """Execute ``RuntimeModel`` embedding request for text, image, video, or audio."""
        params = dict(request.params or {})
        text = request.text
        if text is None and request.messages:
            text = str(request.messages[-1].get("content", ""))
        clean_text = str(text or "").strip()

        image = request.image if request.image is not None else request.image_path
        video = params.get("video") or params.get("video_frames")
        audio = params.get("audio") or params.get("audio_path")
        matryoshka_dim = params.get("matryoshka_dim")
        prompt_name = params.get(
            "prompt_name",
            self._config.default_text_prompt_name,
        )
        prompt_prefix = params.get("prompt_prefix")

        if audio is not None:
            vec = self.embed_audio(
                audio,
                sample_rate=params.get("sample_rate"),
                text=clean_text or None,
                matryoshka_dim=matryoshka_dim,
            )
            modality = "audio+text" if clean_text else "audio"
        elif video is not None:
            frames = self._resolve_video_frames(
                video, max_frames=int(params.get("max_frames", 8))
            )
            vec = self.embed_video_clip(
                frames,
                text=clean_text or None,
                max_soft_tokens_per_frame=params.get("max_soft_tokens"),
                matryoshka_dim=matryoshka_dim,
            )
            modality = "video+text" if clean_text else "video"
        elif image is not None:
            vec = self.embed_images(
                [image],
                texts=[clean_text] if clean_text else None,
                max_soft_tokens=params.get("max_soft_tokens"),
                matryoshka_dim=matryoshka_dim,
            )[0]
            modality = "image+text" if clean_text else "image"
        elif clean_text:
            vec = self.embed_texts(
                [clean_text],
                prompt_name=prompt_name,
                prompt_prefix=prompt_prefix,
                matryoshka_dim=matryoshka_dim,
            )[0]
            modality = "text"
        else:
            raise ValueError("Embedding request requires text, image, video, or audio.")

        return ModelResponse(
            task=request.task,
            output={"embedding": [float(v) for v in vec.tolist()]},
            meta={
                "model_id": self._config.model_id,
                "quantization": self._config.quantization,
                "modality": modality,
                "dim": int(vec.shape[0]),
                "providers": list(self._active_providers),
            },
        )

    @staticmethod
    def _resolve_video_frames(
        video: Any, *, max_frames: int = 8
    ) -> List[Union[Image.Image, np.ndarray, str, Path]]:
        if isinstance(video, (str, Path)):
            from annolid.core.media.video import CV2Video

            cap = CV2Video(Path(video))
            try:
                total = int(cap.total_frames())
                if total <= 0:
                    raise ValueError(f"Video contains no readable frames: {video}")
                count = max(1, min(int(max_frames), total))
                if count == 1:
                    indices = [0]
                else:
                    indices = sorted(
                        {
                            int(round(i * (total - 1) / float(count - 1)))
                            for i in range(count)
                        }
                    )
                frames: List[np.ndarray] = []
                for idx in indices:
                    frame = cap.load_frame(idx)
                    if frame is not None:
                        frames.append(frame)
                if not frames:
                    raise ValueError(f"Failed to decode frames from video: {video}")
                return frames
            finally:
                cap.release()
        if isinstance(video, Iterable):
            frames_list = list(video)
            if not frames_list:
                raise ValueError("Video frame sequence is empty.")
            return frames_list
        raise TypeError(f"Unsupported video payload type: {type(video).__name__}")


__all__ = [
    "DEFAULT_EMBEDDINGGEMMA2_ONNX_REPO",
    "DEFAULT_EMBEDDINGGEMMA2_QUANTIZATION",
    "EMBEDDINGGEMMA2_PROMPTS",
    "SUPPORTED_MAX_SOFT_TOKENS",
    "VALID_MATRYOSHKA_DIMS",
    "EmbeddingGemma2OnnxAdapter",
    "EmbeddingGemma2OnnxConfig",
    "apply_matryoshka_truncation_np",
    "create_gemma4_mel_filterbank",
    "format_embeddinggemma2_text",
    "get_aspect_ratio_preserving_size",
    "l2_normalize_np",
    "normalize_quantization_preset",
    "preprocess_audio_gemma4",
    "preprocess_audios_batch_gemma4",
    "preprocess_image_gemma4",
    "preprocess_images_batch_gemma4",
]
