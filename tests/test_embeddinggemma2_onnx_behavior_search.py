"""End-to-end deterministic tests for ONNX EmbeddingGemma 2 frame & behavior search."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest

from annolid.core.agent.tools.base import FrameBatch, FrameData, ToolContext
from annolid.core.agent.tools.embedding import EmbeddingTool
from annolid.core.agent.tools.vector_index import NumpyEmbeddingIndex
from annolid.core.models.adapters.embeddinggemma2_onnx import (
    EmbeddingGemma2OnnxAdapter,
    EmbeddingGemma2OnnxConfig,
    apply_matryoshka_truncation_np,
    format_embeddinggemma2_text,
    get_aspect_ratio_preserving_size,
    normalize_quantization_preset,
    preprocess_audio_gemma4,
    preprocess_audios_batch_gemma4,
    preprocess_image_gemma4,
    preprocess_images_batch_gemma4,
)
from annolid.core.models.base import ModelRequest
from annolid.core.types import FrameRef
from annolid.services.embedding_search import (
    EmbeddingGemma2OnnxBackend,
    EmbeddingSearchMatch,
    FrameEmbeddingBackend,
    ScoreSummary,
    backend_cache_key,
    compute_adaptive_threshold,
    default_video_index_cache_path,
    group_matches_into_bouts,
    normalize_backend_name,
    overlay_shapes_on_frame_rgb,
    prune_embedding_cache_dir,
    recommended_similarity_threshold,
    run_embedding_search,
    score_video_behaviors,
    search_video_behavior,
    summarize_similarity_scores,
)


def _create_mini_embeddinggemma2_onnx_dir(model_dir: Path) -> Path:
    """Create a real, tiny ONNX EmbeddingGemma 2 directory runnable in onnxruntime.

    Encodes visual color channels (R/G/B) and corresponding behavior tokens
    ('grooming'->R, 'rearing'->G, 'sniffing'->B) into the same 16-dim unit
    vector space (supporting Matryoshka prefixes 4/8/16), exercising the exact
    same input/output tensor names and multimodal token splicing contract as
    ``onnx-community/embeddinggemma-2-ONNX``.
    """
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    onnx_dir = model_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    # Small vocabulary with exact special-token roles declared in config.json.
    vocab: Dict[str, int] = {
        "<pad>": 0,
        "<eos>": 1,
        "<bos>": 2,
        "<unk>": 3,
        "task:": 4,
        "search": 5,
        "result": 6,
        "|": 7,
        "query:": 8,
        "title:": 9,
        "none": 10,
        "text:": 11,
        "mouse": 12,
        "grooming": 13,
        "rearing": 14,
        "sniffing": 15,
        "squeak": 16,
        "<|image>": 20,
        "<|image|>": 21,
        "<image|>": 22,
        "<|video|>": 23,
        "<|audio|>": 24,
        "<|audio>": 25,
        "<audio|>": 26,
    }
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", 2), ("<eos>", 1)],
    )
    tok.save(str(model_dir / "tokenizer.json"))

    hidden_size = 8
    embed_dim = 256

    config = {
        "model_type": "embedding_gemma2",
        "boi_token_id": 20,
        "image_token_id": 21,
        "eoi_token_id": 22,
        "video_token_id": 23,
        "boa_token_id": 25,
        "audio_token_id": 24,
        "eoa_token_index": 26,
        "text_config": {
            "bos_token_id": 2,
            "eos_token_id": 1,
            "pad_token_id": 0,
            "hidden_size": hidden_size,
            "embedding_dim": embed_dim,
            "vocab_size": 32,
        },
        "vision_config": {
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "hidden_size": 768,
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    proc_cfg = {
        "image_processor": {
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "max_soft_tokens": 280,
            "rescale_factor": 1.0 / 255.0,
            "do_normalize": False,
            "image_mean": [0.0, 0.0, 0.0],
            "image_std": [1.0, 1.0, 1.0],
        }
    }
    (model_dir / "processor_config.json").write_text(
        json.dumps(proc_cfg), encoding="utf-8"
    )

    # ---- 1. Build vision_encoder_q4.onnx ----
    # pixel_values: [B, P, 768] -> mean over 256 pixels per RGB channel ->
    # pool 9 patches (3x3) -> valid tokens only (pos >= 0) -> [N_soft, 8].
    proj_w = np.zeros((768, hidden_size), dtype=np.float32)
    for c in range(3):
        proj_w[c::3, c] = 1.0 / 256.0
    proj_w[:, 3] = 0.01 / 768.0

    v_inits = [
        numpy_helper.from_array(proj_w, name="W_vis"),
        numpy_helper.from_array(
            np.array([-1, 9, hidden_size], dtype=np.int64), name="pool_shape"
        ),
        numpy_helper.from_array(
            np.array([-1, 9, 2], dtype=np.int64), name="pos_pool_shape"
        ),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="axis_1"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="zero_i64"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="idx0"),
    ]
    v_nodes = [
        helper.make_node("MatMul", ["pixel_values", "W_vis"], ["patch_h"]),
        helper.make_node("Reshape", ["patch_h", "pool_shape"], ["grouped_h"]),
        helper.make_node(
            "ReduceMean", ["grouped_h", "axis_1"], ["soft_all"], keepdims=0
        ),
        # Check first patch of each 9-patch pool group has pos_x >= 0
        helper.make_node(
            "Reshape", ["pixel_position_ids", "pos_pool_shape"], ["grouped_pos"]
        ),
        helper.make_node("Gather", ["grouped_pos", "idx0"], ["first_pos"], axis=1),
        helper.make_node("Gather", ["first_pos", "idx0"], ["first_x"], axis=2),
        helper.make_node("Squeeze", ["first_x"], ["first_x_1d"]),
        helper.make_node("GreaterOrEqual", ["first_x_1d", "zero_i64"], ["valid_mask"]),
        helper.make_node(
            "Compress", ["soft_all", "valid_mask"], ["image_features"], axis=0
        ),
    ]
    v_graph = helper.make_graph(
        v_nodes,
        "mini_vision_encoder",
        inputs=[
            helper.make_tensor_value_info(
                "pixel_values", TensorProto.FLOAT, ["B", "P", 768]
            ),
            helper.make_tensor_value_info(
                "pixel_position_ids", TensorProto.INT64, ["B", "P", 2]
            ),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "image_features", TensorProto.FLOAT, ["N_soft", hidden_size]
            ),
        ],
        initializer=v_inits,
    )
    v_model = helper.make_model(v_graph, opset_imports=[helper.make_opsetid("", 18)])
    v_model.ir_version = 9
    onnx.save(v_model, str(onnx_dir / "vision_encoder_q4.onnx"))

    # ---- 2. Build model_q4.onnx ----
    # Embedding table maps behavior keywords to orthogonal RGB axes in hidden space:
    # 'grooming' (13) -> axis 0 (Red), 'rearing' (14) -> axis 1 (Green), 'sniffing' (15) -> axis 2 (Blue).
    embed_table = np.zeros((32, hidden_size), dtype=np.float32)
    embed_table[13, 0] = 10.0  # grooming -> Red
    embed_table[14, 1] = 10.0  # rearing -> Green
    embed_table[15, 2] = 10.0  # sniffing -> Blue
    embed_table[16, 4] = 10.0  # squeak -> audio high-freq axis 4
    embed_table[12, 3] = 0.05  # mouse -> shared small bias

    # Projection 8 -> 256 preserving leading channels for MRL prefix truncation!
    out_proj = np.zeros((hidden_size, embed_dim), dtype=np.float32)
    for i in range(hidden_size):
        out_proj[i, i] = 1.0
        out_proj[i, i + 128] = 0.25

    t_inits = [
        numpy_helper.from_array(embed_table, name="embed_table"),
        numpy_helper.from_array(out_proj, name="out_proj"),
        numpy_helper.from_array(
            np.array([-1, hidden_size], dtype=np.int64), name="flat_h_shape"
        ),
        numpy_helper.from_array(np.array([-1], dtype=np.int64), name="flat_1d"),
        numpy_helper.from_array(np.array(0, dtype=np.int64), name="c0"),
        numpy_helper.from_array(np.array(1, dtype=np.int64), name="c1"),
        numpy_helper.from_array(np.array(21, dtype=np.int64), name="img_tok_id"),
        numpy_helper.from_array(np.array(23, dtype=np.int64), name="vid_tok_id"),
        numpy_helper.from_array(np.array(24, dtype=np.int64), name="aud_tok_id"),
        numpy_helper.from_array(
            np.array([hidden_size], dtype=np.int64), name="h_dim_1d"
        ),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="axis_seq"),
        numpy_helper.from_array(np.array([-1], dtype=np.int64), name="axis_last"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="zero_f"),
        numpy_helper.from_array(np.array(1e-9, dtype=np.float32), name="eps_count"),
        numpy_helper.from_array(np.array(1e-12, dtype=np.float32), name="eps_norm"),
    ]
    # Exact multimodal merge pattern matching onnx-community/embeddinggemma-2-ONNX!
    t_nodes = [
        helper.make_node("Gather", ["embed_table", "input_ids"], ["text_emb"], axis=0),
        helper.make_node("Reshape", ["text_emb", "flat_h_shape"], ["flat_text"]),
        helper.make_node("Reshape", ["input_ids", "flat_1d"], ["flat_ids"]),
        helper.make_node("Size", ["input_ids"], ["n_ids"]),
        helper.make_node("Range", ["c0", "n_ids", "c1"], ["base_range"]),
        # Image feature splice (id == 21)
        helper.make_node("Equal", ["flat_ids", "img_tok_id"], ["is_img"]),
        helper.make_node("Where", ["is_img", "c1", "c0"], ["is_img_i"]),
        helper.make_node("CumSum", ["is_img_i", "c0"], ["img_cs"], exclusive=1),
        helper.make_node("Add", ["img_cs", "n_ids"], ["img_off"]),
        helper.make_node(
            "Where", ["is_img", "img_off", "base_range"], ["idx_after_img"]
        ),
        helper.make_node("Shape", ["image_features"], ["img_shape"]),
        helper.make_node("Gather", ["img_shape", "c0"], ["n_img"], axis=0),
        helper.make_node("Add", ["n_ids", "n_img"], ["vid_base"]),
        # Video feature splice (id == 23)
        helper.make_node("Equal", ["flat_ids", "vid_tok_id"], ["is_vid"]),
        helper.make_node("Where", ["is_vid", "c1", "c0"], ["is_vid_i"]),
        helper.make_node("CumSum", ["is_vid_i", "c0"], ["vid_cs"], exclusive=1),
        helper.make_node("Add", ["vid_cs", "vid_base"], ["vid_off"]),
        helper.make_node(
            "Where", ["is_vid", "vid_off", "idx_after_img"], ["idx_after_vid"]
        ),
        helper.make_node("Shape", ["video_features"], ["vid_shape"]),
        helper.make_node("Gather", ["vid_shape", "c0"], ["n_vid"], axis=0),
        helper.make_node("Add", ["vid_base", "n_vid"], ["aud_base"]),
        # Audio feature splice (id == 24)
        helper.make_node("Equal", ["flat_ids", "aud_tok_id"], ["is_aud"]),
        helper.make_node("Where", ["is_aud", "c1", "c0"], ["is_aud_i"]),
        helper.make_node("CumSum", ["is_aud_i", "c0"], ["aud_cs"], exclusive=1),
        helper.make_node("Add", ["aud_cs", "aud_base"], ["aud_off"]),
        helper.make_node(
            "Where", ["is_aud", "aud_off", "idx_after_vid"], ["final_idx"]
        ),
        # Gather from concat([flat_text, image_features, video_features, audio_features])
        helper.make_node(
            "Concat",
            ["flat_text", "image_features", "video_features", "audio_features"],
            ["all_feats"],
            axis=0,
        ),
        helper.make_node("Gather", ["all_feats", "final_idx"], ["merged_flat"], axis=0),
        helper.make_node("Shape", ["input_ids"], ["ids_shape"]),
        helper.make_node(
            "Concat", ["ids_shape", "h_dim_1d"], ["target_3d_shape"], axis=0
        ),
        helper.make_node("Reshape", ["merged_flat", "target_3d_shape"], ["merged_3d"]),
        helper.make_node("MatMul", ["merged_3d", "out_proj"], ["last_hidden_state"]),
        # Masked mean pooling + L2 norm -> sentence_embedding
        helper.make_node(
            "Cast", ["attention_mask"], ["mask_bool"], to=TensorProto.BOOL
        ),
        helper.make_node("Unsqueeze", ["mask_bool", "axis_last"], ["mask_3d"]),
        helper.make_node(
            "Where", ["mask_3d", "last_hidden_state", "zero_f"], ["masked_h"]
        ),
        helper.make_node("ReduceSum", ["masked_h", "axis_seq"], ["sum_h"], keepdims=0),
        helper.make_node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT),
        helper.make_node("ReduceSum", ["mask_f", "axis_seq"], ["sum_m"], keepdims=1),
        helper.make_node("Max", ["sum_m", "eps_count"], ["denom_m"]),
        helper.make_node("Div", ["sum_h", "denom_m"], ["mean_h"]),
        helper.make_node("ReduceL2", ["mean_h", "axis_last"], ["norm_h"], keepdims=1),
        helper.make_node("Max", ["norm_h", "eps_norm"], ["safe_norm"]),
        helper.make_node("Div", ["mean_h", "safe_norm"], ["sentence_embedding"]),
    ]
    t_graph = helper.make_graph(
        t_nodes,
        "mini_embeddinggemma2_text",
        inputs=[
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["B", "S"]),
            helper.make_tensor_value_info(
                "attention_mask", TensorProto.INT64, ["B", "S"]
            ),
            helper.make_tensor_value_info(
                "image_features", TensorProto.FLOAT, ["N_img", hidden_size]
            ),
            helper.make_tensor_value_info(
                "video_features", TensorProto.FLOAT, ["N_vid", hidden_size]
            ),
            helper.make_tensor_value_info(
                "audio_features", TensorProto.FLOAT, ["N_aud", hidden_size]
            ),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "last_hidden_state", TensorProto.FLOAT, ["B", "S", embed_dim]
            ),
            helper.make_tensor_value_info(
                "sentence_embedding", TensorProto.FLOAT, ["B", embed_dim]
            ),
        ],
        initializer=t_inits,
    )
    t_model = helper.make_model(t_graph, opset_imports=[helper.make_opsetid("", 18)])
    t_model.ir_version = 9
    onnx.save(t_model, str(onnx_dir / "model_q4.onnx"))
    return model_dir


def _write_color_segment_video(
    video_path: Path,
    segments: List[Tuple[Tuple[int, int, int], int]],
    *,
    fps: float = 10.0,
    size_wh: Tuple[int, int] = (96, 96),
) -> Path:
    """Write a synthetic video where each segment has a solid RGB color."""
    w, h = size_wh
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(fps),
        (w, h),
    )
    assert writer.isOpened(), f"Failed to open VideoWriter for {video_path}"
    try:
        for rgb, count in segments:
            bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
            frame = np.full((h, w, 3), bgr, dtype=np.uint8)
            for _ in range(count):
                writer.write(frame)
    finally:
        writer.release()
    return video_path


def test_gemma4_image_preprocessing_aspect_ratio_patchify_and_batch_padding() -> None:
    th, tw = get_aspect_ratio_preserving_size(
        480, 640, patch_size=16, max_patches=2520, pooling_kernel_size=3
    )
    assert th % 48 == 0 and tw % 48 == 0
    assert (th // 16) * (tw // 16) <= 2520

    # Extreme aspect ratio stays positive and within budget.
    eh, ew = get_aspect_ratio_preserving_size(
        20, 4000, patch_size=16, max_patches=2520, pooling_kernel_size=3
    )
    assert eh >= 48 and ew >= 48
    assert (eh // 16) * (ew // 16) <= 2520

    img = np.zeros((96, 144, 3), dtype=np.uint8)
    img[:, :, 0] = 255
    patches, pos_ids, soft_n = preprocess_image_gemma4(img, max_soft_tokens=280)
    assert patches.dtype == np.float32
    assert float(patches.min()) >= 0.0 and float(patches.max()) <= 1.0
    assert patches.shape[1] == 768
    assert pos_ids.shape == (patches.shape[0], 2)
    assert pos_ids.dtype == np.int64
    assert soft_n == patches.shape[0] // 9

    # Mixed aspect ratios pad shorter patch sequence with (-1, -1).
    img_square = np.full((96, 96, 3), 128, dtype=np.uint8)
    img_wide = np.full((96, 192, 3), 64, dtype=np.uint8)
    pv, pp, counts = preprocess_images_batch_gemma4(
        [img_square, img_wide], max_soft_tokens=70
    )
    assert pv.shape[0] == 2
    assert pp.shape == (2, pv.shape[1], 2)
    assert len(counts) == 2
    if counts[0] != counts[1]:
        short_i = 0 if counts[0] < counts[1] else 1
        assert int(pp[short_i, -1, 0]) == -1
        assert int(pp[short_i, -1, 1]) == -1


def test_prompt_formatting_quantization_and_matryoshka_renormalization() -> None:
    assert normalize_quantization_preset("int8") == "quantized"
    assert normalize_quantization_preset("q4") == "q4"
    with pytest.raises(
        ValueError, match="Unsupported EmbeddingGemma 2 ONNX quantization"
    ):
        normalize_quantization_preset("bogus")

    formatted = format_embeddinggemma2_text("mouse grooming", prompt_name="SearchQuery")
    assert formatted == "task: search result | query: mouse grooming"
    # Idempotent if prefix already supplied.
    assert (
        format_embeddinggemma2_text(formatted, prompt_name="SearchQuery") == formatted
    )

    vec = np.array([[3.0, 4.0, 12.0, 0.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    trunc = apply_matryoshka_truncation_np(vec, matryoshka_dim=4, normalize=True)
    assert trunc.shape == (1, 4)
    assert float(np.linalg.norm(trunc[0])) == pytest.approx(1.0, abs=1e-5)


def test_embeddinggemma2_onnx_adapter_text_image_video_clip_and_embedding_tool(
    tmp_path: Path,
) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")

    model_dir = _create_mini_embeddinggemma2_onnx_dir(tmp_path / "eg2_onnx")
    adapter = EmbeddingGemma2OnnxAdapter(
        config=EmbeddingGemma2OnnxConfig(
            model_id=str(model_dir),
            quantization="q4",
            matryoshka_dim=8,
            image_max_soft_tokens=70,
            video_max_soft_tokens=70,
        )
    )

    with adapter:
        text_embs = adapter.embed_texts(["grooming", "rearing", "sniffing"])
        assert text_embs.shape == (3, 8)
        assert np.allclose(np.linalg.norm(text_embs, axis=1), 1.0, atol=1e-5)

        red = np.full((96, 96, 3), (255, 0, 0), dtype=np.uint8)
        green = np.full((96, 96, 3), (0, 255, 0), dtype=np.uint8)
        blue = np.full((96, 96, 3), (0, 0, 255), dtype=np.uint8)
        img_embs = adapter.embed_images([red, green, blue])
        assert img_embs.shape == (3, 8)

        # Cross-modal alignment: grooming<->red, rearing<->green, sniffing<->blue
        sims = text_embs @ img_embs.T
        assert np.argmax(sims, axis=1).tolist() == [0, 1, 2]
        assert float(sims[0, 0]) > 0.9
        assert float(sims[0, 1]) < 0.2

        # Temporal multi-frame clip embedding via video_token_id / video_features
        clip_vec = adapter.embed_video_clip([green, green, green])
        assert clip_vec.shape == (8,)
        assert float(clip_vec @ text_embs[1]) > float(clip_vec @ text_embs[0])

        # RuntimeModel.predict + EmbeddingTool + NumpyEmbeddingIndex integration
        resp = adapter.predict(ModelRequest(task="embed", text="rearing"))
        assert len(resp.output["embedding"]) == 8
        assert resp.meta["modality"] == "text"

        tool = EmbeddingTool(model=adapter)
        batch = FrameBatch(
            frames=[
                FrameData(ref=FrameRef(frame_index=0), image_rgb=red),
                FrameData(ref=FrameRef(frame_index=1), image_rgb=green),
                FrameData(ref=FrameRef(frame_index=2), image_rgb=blue),
            ]
        )
        ctx = ToolContext(
            video_path=tmp_path / "dummy.mp4",
            results_dir=tmp_path,
            run_id="eg2-test",
        )
        emb_res = tool.run(ctx, batch)
        index = NumpyEmbeddingIndex(emb_res.embeddings, emb_res.frames)
        hits = index.search(resp.output["embedding"], top_k=1)
        assert hits[0].frame.frame_index == 1


def test_video_behavior_search_bouts_caching_clip_and_scoring(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")

    model_dir = _create_mini_embeddinggemma2_onnx_dir(tmp_path / "eg2_onnx")
    video_path = _write_color_segment_video(
        tmp_path / "mouse_behaviors.avi",
        [
            ((255, 0, 0), 6),  # frames 0..5: grooming (red)
            ((0, 255, 0), 6),  # frames 6..11: rearing (green)
            ((0, 0, 255), 6),  # frames 12..17: sniffing (blue)
        ],
        fps=10.0,
    )

    cache_npz = tmp_path / "mouse_behaviors_eg2.npz"
    params = {
        "model_id": str(model_dir),
        "quantization": "q4",
        "matryoshka_dim": 8,
        "image_max_soft_tokens": 70,
        "video_max_soft_tokens": 70,
    }

    # 1. Text-to-frame behavior search builds & caches .npz index + groups bouts
    res_groom = search_video_behavior(
        video_path,
        query_text="grooming",
        backend="embeddinggemma2_onnx",
        backend_params=params,
        stride=2,
        threshold=0.6,
        top_k=10,
        index_cache_path=cache_npz,
    )
    assert res_groom.query_mode == "text"
    assert cache_npz.is_file()
    groom_frames = sorted(m.frame_index for m in res_groom.matches)
    assert groom_frames == [0, 2, 4]
    assert len(res_groom.bouts) == 1
    assert res_groom.bouts[0].start_frame == 0
    assert res_groom.bouts[0].end_frame == 4

    # 2. Follow-up query reuses cached .npz index without invoking embed_frames_batch!
    backend = EmbeddingGemma2OnnxBackend(**params)
    try:
        called = {"frames": 0}
        orig_batch = backend.embed_frames_batch

        def _counting_batch(frames):
            called["frames"] += len(frames)
            return orig_batch(frames)

        backend.embed_frames_batch = _counting_batch  # type: ignore[method-assign]
        res_rear = search_video_behavior(
            video_path,
            query_text="rearing",
            backend=backend,
            stride=2,
            threshold=0.6,
            top_k=10,
            index_cache_path=cache_npz,
        )
        assert called["frames"] == 0
        assert sorted(m.frame_index for m in res_rear.matches) == [6, 8, 10]

        # 3. Frame-to-frame similarity search excludes query frame itself
        res_frame = search_video_behavior(
            video_path,
            query_frame_index=12,
            backend=backend,
            stride=2,
            threshold=0.6,
            top_k=10,
            index=res_rear.index,
        )
        assert res_frame.query_mode == "frame"
        assert sorted(m.frame_index for m in res_frame.matches) == [14, 16]

        # 4. Multi-frame temporal clip search
        res_clip = search_video_behavior(
            video_path,
            query_text="sniffing",
            backend=backend,
            stride=3,
            clip_frames=3,
            threshold=0.6,
            top_k=5,
        )
        assert res_clip.matches
        assert all(m.frame_index >= 12 for m in res_clip.matches)
        assert all(
            m.start_frame is not None and m.end_frame is not None
            for m in res_clip.matches
        )

        # 5. Zero-shot multi-behavior scoring
        scored = score_video_behaviors(
            video_path,
            ["grooming", "rearing", "sniffing"],
            backend=backend,
            stride=6,
        )
        top_labels = [row["behaviors"][0]["label"] for row in scored["frames"]]
        assert top_labels == ["grooming", "rearing", "sniffing"]
    finally:
        backend.close()


def test_folder_indexer_frame_search_and_behavior_timeline_agent(
    tmp_path: Path,
) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")

    from annolid.agents.behavior_timeline_agent import TimelineConfig, build_timeline
    from annolid.agents.frame_embedder import OnnxFrameIndexer
    from annolid.agents.frame_search import search_frames

    model_dir = _create_mini_embeddinggemma2_onnx_dir(tmp_path / "eg2_onnx")
    params = {
        "model_id": str(model_dir),
        "quantization": "q4",
        "matryoshka_dim": 8,
        "image_max_soft_tokens": 70,
        "video_max_soft_tokens": 70,
    }

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    specs = [
        ("vid_000000000.png", (255, 0, 0), {"grooming": True}, "grooming bout"),
        ("vid_000000001.png", (0, 255, 0), {"rearing": True}, "rearing bout"),
        ("vid_000000002.png", (0, 0, 255), {"sniffing": True}, "sniffing bout"),
    ]
    for name, rgb, flags, caption in specs:
        img_path = frames_dir / name
        cv2.imwrite(
            str(img_path),
            cv2.cvtColor(np.full((96, 96, 3), rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR),
        )
        ann_path = img_path.with_suffix(".json")
        ann_path.write_text(
            json.dumps({"flags": flags, "caption": caption, "shapes": []}),
            encoding="utf-8",
        )

    idx_path = tmp_path / "folder_index.npz"
    indexer = OnnxFrameIndexer(index_path=idx_path, backend_params=params)
    try:
        idx = indexer.index_folder(frames_dir)
        assert len(idx) == 3
        assert idx_path.is_file()

        # Text query over indexed folder frames
        text_hits = asyncio.run(indexer.search_by_text("rearing", limit=1))
        assert len(text_hits) == 1
        assert text_hits[0]["flags"] == ["rearing"]
    finally:
        indexer.close()

    # Standalone search_frames supports BOTH text and image queries on .npz index
    hits_text = search_frames(
        "sniffing", limit=1, index_path=idx_path, backend_params=params
    )
    assert hits_text[0]["flags"] == ["sniffing"]
    hits_img = search_frames(
        str(frames_dir / "vid_000000000.png"),
        limit=1,
        index_path=idx_path,
        backend_params=params,
    )
    assert hits_img[0]["flags"] == ["grooming"]

    # Zero-shot timeline honors cfg.hz=2 (fixing prior 1Hz-hardcoded bug!)
    video_path = _write_color_segment_video(
        tmp_path / "timeline_vid.avi",
        [((255, 0, 0), 10), ((0, 255, 0), 10)],
        fps=10.0,
    )
    backend = EmbeddingGemma2OnnxBackend(**params)
    try:
        timeline = build_timeline(
            str(video_path),
            TimelineConfig(
                hz=2,
                behaviors=("grooming", "rearing"),
                smooth_window=1,
                min_confidence=0.3,
            ),
            backend=backend,
        )
        # 20 frames at 10 fps with hz=2 -> step=5 -> 4 samples (frames 0, 5, 10, 15)
        assert [row["frame_index"] for row in timeline] == [0, 5, 10, 15]
        assert timeline[0]["behaviors"][0]["label"] == "grooming"
        assert timeline[2]["behaviors"][0]["label"] == "rearing"
    finally:
        backend.close()


def test_frame_similarity_service_cancellation_and_cache_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    from annolid.gui.frame_similarity_service import (
        FrameSimilaritySearchRequest,
        FrameSimilarityService,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    video_path = _write_color_segment_video(
        tmp_path / "qt_search.avi",
        [((255, 0, 0), 8), ((0, 255, 0), 8)],
        fps=10.0,
    )

    class _SlowFakeBackend(FrameEmbeddingBackend):
        supports_text = True
        supports_video_clip = True

        def __init__(self, delay_s: float = 0.0) -> None:
            self.delay_s = delay_s
            self.frame_calls = 0
            self.entered_frame_embed = threading.Event()

        @property
        def key(self) -> Tuple[str, ...]:
            return ("embeddinggemma2_onnx", "fake", "q4", "", "280", "140", "")

        def embed_frame_rgb(self, frame_rgb: np.ndarray) -> np.ndarray:
            self.frame_calls += 1
            self.entered_frame_embed.set()
            if self.delay_s > 0:
                time.sleep(self.delay_s)
            r = float(frame_rgb[:, :, 0].mean()) / 255.0
            g = float(frame_rgb[:, :, 1].mean()) / 255.0
            vec = np.array([r, g, 0.01], dtype=np.float32)
            return vec / float(np.linalg.norm(vec))

        def embed_text(
            self, text: str, *, prompt_name: Optional[str] = "SearchQuery"
        ) -> np.ndarray:
            _ = prompt_name
            if "rear" in text.lower():
                return np.array([0.0, 1.0, 0.0], dtype=np.float32)
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    fake_backend = _SlowFakeBackend(delay_s=0.0)
    monkeypatch.setattr(
        "annolid.gui.frame_similarity_service._build_backend",
        lambda _req: fake_backend,
    )
    monkeypatch.setattr(
        "annolid.gui.frame_similarity_service._backend_key_for_request",
        lambda _req: fake_backend.key,
    )

    svc = FrameSimilarityService()
    try:
        # 1. Text search builds index & finds red frames
        done_results: List[list] = []
        done_bouts: List[list] = []
        done_summaries: List[dict] = []
        errors: List[str] = []
        svc.finished.connect(done_results.append)
        svc.boutsFound.connect(done_bouts.append)
        svc.scoreSummary.connect(done_summaries.append)
        svc.error.connect(errors.append)

        req1 = FrameSimilaritySearchRequest(
            video_path=video_path,
            query_text="grooming",
            query_mode="text",
            stride=2,
            threshold=0.5,
            overlay_shapes=False,
            batch_size=1,
        )
        assert svc.request(req1) is True
        deadline = time.monotonic() + 5.0
        while not done_results and not errors and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert not errors
        assert len(done_results) == 1
        assert sorted(r["frame_index"] for r in done_results[0]) == [0, 2, 4, 6]
        assert done_bouts and len(done_bouts[0]) == 1
        assert done_summaries and done_summaries[0]["count"] == 8
        assert done_summaries[0]["threshold_mode"] == "fixed"
        first_call_count = fake_backend.frame_calls
        assert first_call_count == 8

        # 2. Second text query on same video hits worker index cache (0 new frame embeds!)
        req2 = FrameSimilaritySearchRequest(
            video_path=video_path,
            query_text="rearing",
            query_mode="text",
            stride=2,
            threshold=0.5,
            overlay_shapes=False,
            batch_size=1,
        )
        assert svc.request(req2) is True
        deadline = time.monotonic() + 5.0
        while len(done_results) < 2 and not errors and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert not errors
        assert sorted(r["frame_index"] for r in done_results[1]) == [8, 10, 12, 14]
        assert fake_backend.frame_calls == first_call_count

        # 3. Verify immediate Stop cancellation during an active slow scan!
        svc.clear_cache()
        fake_backend.delay_s = 0.08
        fake_backend.frame_calls = 0
        fake_backend.entered_frame_embed.clear()

        req3 = FrameSimilaritySearchRequest(
            video_path=video_path,
            query_text="grooming",
            query_mode="text",
            stride=1,
            threshold=0.5,
            overlay_shapes=False,
            batch_size=1,
        )
        assert svc.request(req3) is True
        assert fake_backend.entered_frame_embed.wait(timeout=2.0)
        svc.stop()
        deadline = time.monotonic() + 5.0
        while len(done_results) < 3 and not errors and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert not errors
        # Stopped well before embedding all 16 frames!
        assert fake_backend.frame_calls < 16
    finally:
        svc.close()


def test_embedding_search_widget_onnx_default_text_search_and_compact_width(
    tmp_path: Path,
) -> None:
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    from annolid.gui.frame_similarity_service import FrameSimilaritySearchRequest
    from annolid.gui.widgets.embedding_search_widget import EmbeddingSearchWidget

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app

    widget = EmbeddingSearchWidget()
    try:
        # Compact dock invariant must be preserved even with Auto threshold control
        assert widget.minimumSizeHint().width() <= 315
        assert widget._backend_combo.currentData() == "embeddinggemma2_onnx"
        assert widget._auto_threshold_chk.isChecked() is True
        assert widget._threshold_spin.isEnabled() is False
        assert widget._threshold_spin.value() == pytest.approx(0.66)
        assert (
            widget._results_list.selectionMode()
            == QtWidgets.QAbstractItemView.ExtendedSelection
        )

        # Wire a stub service to capture submitted requests without launching threads
        captured_requests: List[FrameSimilaritySearchRequest] = []

        class _StubService:
            def request(self, req: FrameSimilaritySearchRequest) -> bool:
                captured_requests.append(req)
                return True

            def stop(self) -> None:
                return None

            def close(self) -> None:
                return None

        widget._service = _StubService()
        widget._req_cls = FrameSimilaritySearchRequest
        video = tmp_path / "demo.mp4"
        video.write_bytes(b"")
        widget.set_video_path(video)

        # Text behavior search submits ONNX EmbeddingGemma 2 request with Auto threshold & 140t default
        widget._query_input.setText("mouse grooming")
        widget._run_text_search()
        assert len(captured_requests) == 1
        req = captured_requests[0]
        assert req.backend == "embeddinggemma2_onnx"
        assert req.query_mode == "text"
        assert req.query_text == "mouse grooming"
        assert req.query_frame_index is None
        assert req.threshold is None
        assert req.backend_params["quantization"] == "q4"
        assert req.backend_params["matryoshka_dim"] == 768
        assert req.backend_params["image_max_soft_tokens"] == 140
        assert widget.last_query_text() == "mouse grooming"

        # Finish search with auto score_summary and test multi-frame label emission
        statuses: List[str] = []
        widget.statusMessage.connect(statuses.append)
        widget._on_score_summary(
            {
                "count": 20,
                "min_similarity": 0.55,
                "median_similarity": 0.60,
                "mean_similarity": 0.62,
                "std_similarity": 0.08,
                "p90_similarity": 0.80,
                "max_similarity": 0.89,
                "best_frame_index": 15,
                "effective_threshold": 0.68,
                "threshold_mode": "auto",
            }
        )
        widget._on_bouts_found([{"start_frame": 10, "end_frame": 20, "peak_frame": 15}])
        widget._on_search_finished(
            [
                {"frame_index": 10, "similarity": 0.81},
                {"frame_index": 15, "similarity": 0.89},
            ]
        )
        assert any("auto ≥ 0.68" in s for s in statuses)
        assert widget._results_list.count() == 2
        for i in range(widget._results_list.count()):
            widget._results_list.item(i).setSelected(True)
        labeled: List[list] = []
        widget.labelFramesRequested.connect(labeled.append)
        widget._label_selected_frames()
        assert labeled == [[10, 15]]

        # Zero-match search surfaces actionable similarity range & best candidate frame
        statuses.clear()
        widget._on_score_summary(
            {
                "count": 12,
                "min_similarity": 0.52,
                "median_similarity": 0.56,
                "mean_similarity": 0.56,
                "std_similarity": 0.02,
                "p90_similarity": 0.59,
                "max_similarity": 0.61,
                "best_frame_index": 42,
                "effective_threshold": 0.66,
                "threshold_mode": "auto",
            }
        )
        widget._on_bouts_found([])
        widget._on_search_finished([])
        assert any(
            "No matches ≥ 0.66" in s and "Frame 000000042" in s and "0.610" in s
            for s in statuses
        )

        # Unchecking Auto + selecting fast 70t/256d preset passes explicit threshold & soft tokens
        widget._onnx_preset_combo.setCurrentIndex(1)
        widget._auto_threshold_chk.setChecked(False)
        assert widget._threshold_spin.isEnabled() is True
        widget._threshold_spin.setValue(0.71)
        widget._run_text_search()
        assert len(captured_requests) == 2
        req_fast = captured_requests[1]
        assert req_fast.threshold == pytest.approx(0.71)
        assert req_fast.backend_params["matryoshka_dim"] == 256
        assert req_fast.backend_params["image_max_soft_tokens"] == 70

        # Switching to DINOv3 rejects text search with actionable message
        statuses.clear()
        widget._backend_combo.setCurrentIndex(widget._backend_combo.findData("dinov3"))
        widget._run_text_search()
        assert len(captured_requests) == 2
        assert any("visual-only" in s.lower() for s in statuses)
    finally:
        widget.close()


def test_annolid_run_behavior_search_cli(tmp_path: Path, capsys) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    from annolid.engine.cli import main as annolid_run

    # Help lists behavior-search under Behavior group
    assert annolid_run(["--help"]) == 0
    help_text = capsys.readouterr().out
    assert "behavior-search" in help_text

    model_dir = _create_mini_embeddinggemma2_onnx_dir(tmp_path / "eg2_onnx")
    video_path = _write_color_segment_video(
        tmp_path / "cli_vid.avi",
        [((255, 0, 0), 5), ((0, 255, 0), 5)],
        fps=10.0,
    )
    out_json = tmp_path / "search_out.json"
    cache_npz = tmp_path / "cli_index.npz"

    rc = annolid_run(
        [
            "behavior-search",
            str(video_path),
            "--query",
            "rearing",
            "--model-id",
            str(model_dir),
            "--quantization",
            "q4",
            "--matryoshka-dim",
            "128",
            "--stride",
            "2",
            "--threshold",
            "0.5",
            "--index-cache",
            str(cache_npz),
            "--out",
            str(out_json),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["query_mode"] == "text"
    assert payload["backend"] == "embeddinggemma2_onnx"
    assert sorted(m["frame_index"] for m in payload["matches"]) == [6, 8]
    assert out_json.is_file()
    assert cache_npz.is_file()

    # Adaptive thresholding (omitting --threshold) + --soft-tokens 70
    rc_auto = annolid_run(
        [
            "behavior-search",
            str(video_path),
            "--query",
            "grooming",
            "--model-id",
            str(model_dir),
            "--quantization",
            "q4",
            "--soft-tokens",
            "70",
            "--matryoshka-dim",
            "128",
            "--stride",
            "2",
            "--json",
        ]
    )
    assert rc_auto == 0
    auto_payload = json.loads(capsys.readouterr().out)
    assert auto_payload["score_summary"]["threshold_mode"] == "auto"
    assert sorted(m["frame_index"] for m in auto_payload["matches"]) == [0, 2, 4]

    # Non-JSON zero-match output surfaces observed similarity diagnostics
    rc_zero = annolid_run(
        [
            "behavior-search",
            str(video_path),
            "--query",
            "sniffing",
            "--model-id",
            str(model_dir),
            "--quantization",
            "q4",
            "--soft-tokens",
            "70",
            "--matryoshka-dim",
            "128",
            "--stride",
            "2",
        ]
    )
    assert rc_zero == 0
    zero_out = capsys.readouterr().out
    assert "Found 0 match(es)" in zero_out
    assert "Observed similarity range:" in zero_out

    # Zero-shot multi-behavior scoring CLI mode
    rc2 = annolid_run(
        [
            "behavior-search",
            str(video_path),
            "--behaviors",
            "grooming,rearing",
            "--model-id",
            str(model_dir),
            "--quantization",
            "q4",
            "--matryoshka-dim",
            "128",
            "--stride",
            "5",
            "--json",
        ]
    )
    assert rc2 == 0
    score_payload = json.loads(capsys.readouterr().out)
    assert [f["behaviors"][0]["label"] for f in score_payload["frames"]] == [
        "grooming",
        "rearing",
    ]


def test_hf_cache_symlink_external_data_staging_for_ort(tmp_path: Path) -> None:
    """Regression: ONNX Runtime >=1.24 rejects symlinked HF Hub .onnx_data files.

    Reproduces the exact Hugging Face Hub layout where snapshot files are
    symlinks into ``../../blobs/<sha>`` and verifies ``EmbeddingGemma2OnnxAdapter``
    stages zero-copy hardlinked pairs into ``.ort_staged/<rev>/``.
    """
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")

    src_dir = _create_mini_embeddinggemma2_onnx_dir(tmp_path / "src_eg2")
    cache_root = tmp_path / "hf_repo_cache"
    blobs_dir = cache_root / "blobs"
    snap_rev = "rev123456"
    snap_dir = cache_root / "snapshots" / snap_rev
    snap_onnx = snap_dir / "onnx"
    blobs_dir.mkdir(parents=True)
    snap_onnx.mkdir(parents=True)

    # Copy tokenizer & config files into snapshot
    for fname in ("tokenizer.json", "config.json", "processor_config.json"):
        shutil.copy2(src_dir / fname, snap_dir / fname)

    # Re-save each ONNX model with external tensor data, move into blobs/, and symlink from snapshots/<rev>/onnx/
    blob_paths: Dict[str, Path] = {}
    for stem in ("model_q4", "vision_encoder_q4"):
        tmp_ext = tmp_path / f"ext_{stem}"
        tmp_ext.mkdir()
        model_proto = onnx.load(str(src_dir / "onnx" / f"{stem}.onnx"))
        onnx.save_model(
            model_proto,
            str(tmp_ext / f"{stem}.onnx"),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{stem}.onnx_data",
            size_threshold=128,
        )
        assert (tmp_ext / f"{stem}.onnx_data").stat().st_size > 0
        for suffix in (".onnx", ".onnx_data"):
            blob = blobs_dir / f"sha_{stem}{suffix}"
            shutil.copy2(tmp_ext / f"{stem}{suffix}", blob)
            blob_paths[f"{stem}{suffix}"] = blob
            link_path = snap_onnx / f"{stem}{suffix}"
            rel_target = os.path.relpath(blob, link_path.parent)
            os.symlink(rel_target, link_path)

    adapter = EmbeddingGemma2OnnxAdapter(
        config=EmbeddingGemma2OnnxConfig(
            model_id=str(snap_dir),
            quantization="q4",
            matryoshka_dim=8,
            image_max_soft_tokens=70,
            video_max_soft_tokens=70,
        )
    )
    with adapter:
        t_vec = adapter.embed_texts(["grooming"])[0]
        red = np.full((96, 96, 3), (255, 0, 0), dtype=np.uint8)
        i_vec = adapter.embed_images([red])[0]
        assert float(t_vec @ i_vec) > 0.9

    staged_dir = cache_root / ".ort_staged" / snap_rev
    for fname, blob in blob_paths.items():
        staged = staged_dir / fname
        assert staged.is_file()
        assert not staged.is_symlink()
        # Hardlinked on POSIX filesystems (zero extra disk duplication)
        assert os.stat(staged).st_ino == os.stat(blob).st_ino


def test_adaptive_threshold_calibration_and_score_summary(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")

    # 1. Calibrated per-backend/mode defaults
    assert recommended_similarity_threshold(
        "embeddinggemma2_onnx", "text"
    ) == pytest.approx(0.66)
    assert recommended_similarity_threshold(
        "embeddinggemma2_onnx", "frame"
    ) == pytest.approx(0.82)
    assert recommended_similarity_threshold(
        "embeddinggemma2_onnx", "hybrid"
    ) == pytest.approx(0.72)
    assert recommended_similarity_threshold("dinov3", "frame") == pytest.approx(0.35)

    # 2. Sparse positive cluster on realistic EmbeddingGemma 2 text scale
    sparse_scores = np.array(
        [0.56, 0.57, 0.58, 0.58, 0.59, 0.59, 0.60, 0.61, 0.71, 0.74], dtype=np.float32
    )
    thr_sparse = compute_adaptive_threshold(
        sparse_scores, backend="embeddinggemma2_onnx", query_mode="text"
    )
    assert 0.63 <= thr_sparse <= 0.70
    assert (sparse_scores >= thr_sparse).tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]

    # 3. Bimodal 50/50 split does not overshoot positive mode
    bimodal_scores = np.array([0.55, 0.56, 0.57, 0.70, 0.72, 0.74], dtype=np.float32)
    thr_bi = compute_adaptive_threshold(
        bimodal_scores, backend="embeddinggemma2_onnx", query_mode="text"
    )
    assert (bimodal_scores >= thr_bi).tolist() == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]

    # 4. Negative query below floor returns 0 false positives
    neg_scores = np.array([0.54, 0.56, 0.58, 0.60, 0.62], dtype=np.float32)
    thr_neg = compute_adaptive_threshold(
        neg_scores, backend="embeddinggemma2_onnx", query_mode="text"
    )
    assert not np.any(neg_scores >= thr_neg)

    # 5. Nearly flat slightly elevated negative query falls back to calibrated default (0.66)
    flat_neg = np.array([0.63, 0.64, 0.645, 0.65, 0.652], dtype=np.float32)
    thr_flat = compute_adaptive_threshold(
        flat_neg, backend="embeddinggemma2_onnx", query_mode="text"
    )
    assert thr_flat == pytest.approx(0.66)
    assert not np.any(flat_neg >= thr_flat)

    # 6. Same-modality frame-to-frame high baseline isolates similar frames
    frame_scores = np.array(
        [0.78, 0.79, 0.80, 0.80, 0.81, 0.92, 0.95], dtype=np.float32
    )
    thr_frame = compute_adaptive_threshold(
        frame_scores, backend="embeddinggemma2_onnx", query_mode="frame"
    )
    assert (frame_scores >= thr_frame).tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]

    # 7. summarize_similarity_scores excludes query frame cleanly
    summary = summarize_similarity_scores(
        np.array([0, 5, 10], dtype=np.int64),
        np.array([1.0, 0.62, 0.88], dtype=np.float32),
        effective_threshold=0.82,
        threshold_mode="auto",
        exclude_frame_index=0,
    )
    assert isinstance(summary, ScoreSummary)
    assert summary.count == 2
    assert summary.best_frame_index == 10
    assert summary.max_similarity == pytest.approx(0.88)

    # 8. End-to-end adaptive search_video_behavior: positive finds exact segment, negative returns 0
    model_dir = _create_mini_embeddinggemma2_onnx_dir(tmp_path / "eg2_adaptive")
    video_path = _write_color_segment_video(
        tmp_path / "adaptive_vid.avi",
        [((255, 0, 0), 6), ((0, 255, 0), 6)],
        fps=10.0,
    )
    params = {
        "model_id": str(model_dir),
        "quantization": "q4",
        "matryoshka_dim": 8,
        "image_max_soft_tokens": 70,
        "video_max_soft_tokens": 70,
    }
    pos_res = search_video_behavior(
        video_path,
        query_text="grooming",
        backend="embeddinggemma2_onnx",
        backend_params=params,
        stride=2,
        threshold=None,
    )
    assert pos_res.score_summary is not None
    assert pos_res.score_summary.threshold_mode == "auto"
    assert sorted(m.frame_index for m in pos_res.matches) == [0, 2, 4]

    neg_res = search_video_behavior(
        video_path,
        query_text="sniffing",
        backend="embeddinggemma2_onnx",
        backend_params=params,
        stride=2,
        threshold=None,
        index=pos_res.index,
    )
    assert neg_res.matches == []
    assert neg_res.score_summary is not None
    assert (
        neg_res.score_summary.max_similarity < neg_res.score_summary.effective_threshold
    )


@pytest.mark.skipif(
    os.getenv("ANNOLID_RUN_LIVE_ONNX_TESTS") != "1",
    reason="Opt-in live pretrained ONNX weights test (set ANNOLID_RUN_LIVE_ONNX_TESTS=1).",
)
def test_live_embeddinggemma2_onnx_pretrained_weights(tmp_path: Path) -> None:
    """Live end-to-end verification against real onnx-community/embeddinggemma-2-ONNX q4."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    pytest.importorskip("huggingface_hub")

    repo_root = Path(__file__).resolve().parents[1]
    img_specs = [
        (
            repo_root / "docs/imgs/mouse_keypoints.png",
            "a laboratory mouse with colored anatomical keypoint markers",
        ),
        (
            repo_root / "docs/imgs/AP_across_labeled_frames.png",
            "a scientific line chart plotting average precision scores",
        ),
        (
            repo_root / "docs/imgs/annolid_ui.png",
            "a desktop graphical user interface application window",
        ),
        (
            repo_root / "docs/imgs/annolid_qr.png",
            "a black and white square QR code pattern",
        ),
    ]
    for path, _ in img_specs:
        assert path.is_file(), f"Missing repo sample image: {path}"

    adapter = EmbeddingGemma2OnnxAdapter(
        config=EmbeddingGemma2OnnxConfig(
            quantization="q4",
            image_max_soft_tokens=140,
            video_max_soft_tokens=70,
        )
    )
    with adapter:
        images = [
            cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p, _ in img_specs
        ]
        img_embs = adapter.embed_images(images)
        assert img_embs.shape == (len(images), 768)
        assert np.all(np.isfinite(img_embs))
        assert np.allclose(np.linalg.norm(img_embs, axis=1), 1.0, atol=1e-4)

        queries = [q for _, q in img_specs]
        txt_embs = adapter.embed_texts(queries)
        sims = txt_embs @ img_embs.T
        # Every real query must retrieve its matching real image as rank-1!
        assert np.argmax(sims, axis=1).tolist() == list(range(len(images)))

        # Unrelated negative query must produce 0 matches under adaptive thresholding
        neg_vec = adapter.embed_texts(["a red sports car racing on a highway"])[0]
        neg_scores = img_embs @ neg_vec
        neg_thr = compute_adaptive_threshold(
            neg_scores, backend="embeddinggemma2_onnx", query_mode="text"
        )
        assert not np.any(neg_scores >= neg_thr)

        # Matryoshka 256d truncation preserves unit norm and rank-1 retrieval
        img_256 = apply_matryoshka_truncation_np(img_embs, 256, normalize=True)
        txt_256 = apply_matryoshka_truncation_np(txt_embs, 256, normalize=True)
        assert np.argmax(txt_256 @ img_256.T, axis=1).tolist() == list(
            range(len(images))
        )

        # Multi-frame video clip embedding via video_features path
        clip_vec = adapter.embed_video_clip([images[0], images[0]])
        assert clip_vec.shape == (768,)
        assert np.all(np.isfinite(clip_vec))
        assert float(np.linalg.norm(clip_vec)) == pytest.approx(1.0, abs=1e-4)
        assert float(clip_vec @ txt_embs[0]) > float(clip_vec @ txt_embs[1])

        # Verify audio_encoder was NOT loaded until explicitly requested
        assert adapter._audio_session is None
        sr = 16000
        t_arr = np.arange(sr * 2, dtype=np.float32) / sr
        beep = (
            0.5
            * np.sin(2.0 * np.pi * 3500.0 * t_arr)
            * (np.sin(2.0 * np.pi * 6.0 * t_arr) > 0)
        ).astype(np.float32)
        noise = (0.3 * np.random.default_rng(42).normal(0, 1, size=t_arr.shape)).astype(
            np.float32
        )
        aud_embs = adapter.embed_audios([beep, noise], sample_rate=sr)
        assert adapter._audio_session is not None
        assert aud_embs.shape == (2, 768)
        assert np.all(np.isfinite(aud_embs))
        assert np.allclose(np.linalg.norm(aud_embs, axis=1), 1.0, atol=1e-4)
        aud_txt = adapter.embed_texts(
            [
                "high pitched electronic beeping alarm tone",
                "white noise television static hiss",
            ]
        )
        assert np.argmax(aud_txt @ aud_embs.T, axis=1).tolist() == [0, 1]

        # Also test adaptive search_video_behavior on a short real-image video
        vid_path = tmp_path / "live_real_frames.avi"
        w, h = 256, 256
        writer = cv2.VideoWriter(
            str(vid_path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (w, h)
        )
        try:
            for img in images:
                resized = cv2.resize(img, (w, h))
                bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                for _ in range(2):
                    writer.write(bgr)
        finally:
            writer.release()

        backend = EmbeddingGemma2OnnxBackend(
            quantization="q4", image_max_soft_tokens=140, video_max_soft_tokens=70
        )
        try:
            res = search_video_behavior(
                vid_path,
                query_text="a laboratory mouse with colored anatomical keypoint markers",
                backend=backend,
                stride=2,
                threshold=None,
            )
            assert res.score_summary is not None
            assert res.score_summary.best_frame_index == 0
            assert any(m.frame_index == 0 for m in res.matches)
        finally:
            backend.close()


def test_deterministic_overlay_and_bout_grouping() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    shapes = [
        {
            "label": "mouse_1",
            "shape_type": "rectangle",
            "points": [[10, 10], [40, 40]],
        }
    ]
    out1 = overlay_shapes_on_frame_rgb(frame, shapes)
    out2 = overlay_shapes_on_frame_rgb(frame, shapes)
    assert np.array_equal(out1, out2)
    assert int(out1.sum()) > 0

    matches = [
        EmbeddingSearchMatch(frame_index=10, similarity=0.6),
        EmbeddingSearchMatch(frame_index=15, similarity=0.9),
        EmbeddingSearchMatch(frame_index=20, similarity=0.7),
        EmbeddingSearchMatch(frame_index=100, similarity=0.8),
    ]
    bouts = group_matches_into_bouts(matches, max_gap_frames=5, fps=10.0, label="groom")
    assert len(bouts) == 2
    assert bouts[0].start_frame == 10 and bouts[0].end_frame == 20
    assert bouts[0].peak_frame == 15
    assert bouts[0].start_time_sec == pytest.approx(1.0)
    assert bouts[1].start_frame == 100 and bouts[1].end_frame == 100

    # Verify backend alias normalization and cache keys
    assert normalize_backend_name("embeddinggemma-2") == "embeddinggemma2_onnx"
    assert backend_cache_key("onnx", {"quantization": "int8"})[2] == "quantized"
    # Verify run_embedding_search supports query_frame_index=None (text queries don't skip frame 0!)
    hits = run_embedding_search(
        query_vector=np.array([1.0, 0.0], dtype=np.float32),
        query_frame_index=None,
        total_frames=2,
        stride=1,
        max_frames=None,
        threshold=0.5,
        top_k=5,
        embed_frame=lambda _i: np.array([1.0, 0.0], dtype=np.float32),
    )
    assert [m.frame_index for m in hits] == [0, 1]


def _add_mini_audio_encoder_onnx(model_dir: Path, hidden_size: int = 8) -> Path:
    """Add a miniature ``onnx/audio_encoder_q4.onnx`` matching Gemma 4's exact contract.

    Inputs: ``input_features`` ``[B, T, 128]`` float32, ``input_features_mask`` ``[B, T]`` bool.
    Output: ``audio_features`` ``[N_aud, hidden_size]`` float32 via 4x time subsampling.
    Maps high-frequency mel energy (bins 64..127) onto axis 4 ('squeak').
    """
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    proj = np.zeros((128, hidden_size), dtype=np.float32)
    proj[64:, 4] = 1.0 / 64.0
    proj[:64, 5] = 1.0 / 64.0

    inits = [
        numpy_helper.from_array(proj, name="W_aud"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="s_start"),
        numpy_helper.from_array(np.array([999999], dtype=np.int64), name="s_end"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="s_axis"),
        numpy_helper.from_array(np.array([4], dtype=np.int64), name="s_step"),
        numpy_helper.from_array(
            np.array([-1, hidden_size], dtype=np.int64), name="flat_h"
        ),
        numpy_helper.from_array(np.array([-1], dtype=np.int64), name="flat_m"),
    ]
    nodes = [
        # 4x time subsampling: indices 0, 4, 8, ... -> exact ceil(T / 4) count!
        helper.make_node(
            "Slice",
            ["input_features", "s_start", "s_end", "s_axis", "s_step"],
            ["sub_feats"],
        ),
        helper.make_node(
            "Slice",
            ["input_features_mask", "s_start", "s_end", "s_axis", "s_step"],
            ["sub_mask"],
        ),
        helper.make_node("MatMul", ["sub_feats", "W_aud"], ["proj_h"]),
        helper.make_node("Reshape", ["proj_h", "flat_h"], ["flat_proj"]),
        helper.make_node("Reshape", ["sub_mask", "flat_m"], ["flat_mask"]),
        helper.make_node(
            "Compress", ["flat_proj", "flat_mask"], ["audio_features"], axis=0
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "mini_audio_encoder",
        inputs=[
            helper.make_tensor_value_info(
                "input_features", TensorProto.FLOAT, ["B", "T", 128]
            ),
            helper.make_tensor_value_info(
                "input_features_mask", TensorProto.BOOL, ["B", "T"]
            ),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "audio_features", TensorProto.FLOAT, ["N_aud", hidden_size]
            )
        ],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    out_path = model_dir / "onnx" / "audio_encoder_q4.onnx"
    onnx.save(model, str(out_path))
    return out_path


def test_lazy_onnx_audio_embedding_and_wav_preprocessing(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")

    model_dir = _create_mini_embeddinggemma2_onnx_dir(tmp_path / "eg2_audio")

    # 1. Text/vision load succeeds WITHOUT audio_encoder_q4.onnx present (lazy invariant!)
    adapter = EmbeddingGemma2OnnxAdapter(
        config=EmbeddingGemma2OnnxConfig(
            model_id=str(model_dir),
            quantization="q4",
            matryoshka_dim=8,
            image_max_soft_tokens=70,
        )
    )
    with adapter:
        assert adapter._audio_session is None
        t_squeak = adapter.embed_texts(["squeak"])[0]
        assert adapter._audio_session is None

        # 2. Add audio_encoder_q4.onnx and verify lazy loading on first audio embed
        _add_mini_audio_encoder_onnx(model_dir, hidden_size=8)

        sr = 16000
        t_arr = np.arange(sr // 2, dtype=np.float32) / sr
        high_chirp = (0.8 * np.sin(2.0 * np.pi * 4500.0 * t_arr)).astype(np.float32)
        low_hum = (0.8 * np.sin(2.0 * np.pi * 150.0 * t_arr)).astype(np.float32)

        # Write a real PCM-16 WAV file to verify stdlib wave loading + resampling path
        wav_path = tmp_path / "chirp.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(
                np.clip(high_chirp * 32767.0, -32768, 32767).astype(np.int16).tobytes()
            )

        mel, mask, tok_n = preprocess_audio_gemma4(wav_path)
        assert mel.ndim == 2 and mel.shape[1] == 128
        assert mask.dtype == np.bool_ and mask.shape == (mel.shape[0],)
        assert tok_n == (mel.shape[0] + 3) // 4

        # Batched variable-length preprocessing pads right with False mask
        short_wav = low_hum[: sr // 4]
        b_feats, b_mask, b_counts = preprocess_audios_batch_gemma4(
            [wav_path, (short_wav, 8000)]
        )
        assert b_feats.shape[0] == 2 and b_mask.shape == b_feats.shape[:2]
        assert len(b_counts) == 2

        a_vecs = adapter.embed_audios([wav_path, low_hum], sample_rate=sr)
        assert adapter._audio_session is not None
        assert a_vecs.shape == (2, 8)
        assert np.allclose(np.linalg.norm(a_vecs, axis=1), 1.0, atol=1e-5)
        # High-frequency chirp aligns more strongly with 'squeak' than low hum
        assert float(t_squeak @ a_vecs[0]) > float(t_squeak @ a_vecs[1])

        # RuntimeModel.predict(params={'audio': ...}) routes cleanly
        resp = adapter.predict(ModelRequest(task="embed", params={"audio": wav_path}))
        assert resp.meta["modality"] == "audio"
        assert len(resp.output["embedding"]) == 8


def test_persistent_bounded_disk_cache_cli_auto_cache_and_gui_eta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    from annolid.engine.cli import main as annolid_run

    cache_dir = tmp_path / "emb_cache"
    monkeypatch.setenv("ANNOLID_EMBEDDING_CACHE_DIR", str(cache_dir))

    model_dir = _create_mini_embeddinggemma2_onnx_dir(tmp_path / "eg2_cache")
    video_path = _write_color_segment_video(
        tmp_path / "cache_vid.avi",
        [((255, 0, 0), 6), ((0, 255, 0), 6)],
        fps=10.0,
    )
    params = {
        "model_id": str(model_dir),
        "quantization": "q4",
        "matryoshka_dim": 8,
        "image_max_soft_tokens": 70,
        "video_max_soft_tokens": 70,
    }

    # 1. First auto_disk_cache search computes and persists .npz under ANNOLID_EMBEDDING_CACHE_DIR
    res1 = search_video_behavior(
        video_path,
        query_text="grooming",
        backend="embeddinggemma2_onnx",
        backend_params=params,
        stride=2,
        threshold=None,
        auto_disk_cache=True,
    )
    assert res1.cache_hit is False
    cached_files = list(cache_dir.glob("*.npz"))
    assert len(cached_files) == 1

    # 2. Second call with a fresh backend hits disk cache with 0 frame embeddings!
    backend = EmbeddingGemma2OnnxBackend(**params)
    try:
        calls = {"n": 0}
        orig_embed = backend.embed_frames_batch

        def _count(frames):
            calls["n"] += len(frames)
            return orig_embed(frames)

        backend.embed_frames_batch = _count  # type: ignore[method-assign]
        res2 = search_video_behavior(
            video_path,
            query_text="rearing",
            backend=backend,
            stride=2,
            threshold=None,
            auto_disk_cache=True,
        )
        assert res2.cache_hit is True
        assert calls["n"] == 0
        assert sorted(m.frame_index for m in res2.matches) == [6, 8, 10]

        # overlay_shapes=True intentionally bypasses auto disk cache to avoid stale annotations
        res_overlay = search_video_behavior(
            video_path,
            query_text="grooming",
            backend=backend,
            stride=2,
            threshold=None,
            overlay_shapes=True,
            auto_disk_cache=True,
        )
        assert res_overlay.cache_hit is False
        assert calls["n"] > 0
    finally:
        backend.close()

    # 3. Bounded LRU pruning removes oldest .npz files
    f_old = cache_dir / "old_a.npz"
    f_mid = cache_dir / "mid_b.npz"
    f_new = cache_dir / "new_c.npz"
    for idx_f, p in enumerate((f_old, f_mid, f_new)):
        p.write_bytes(b"x" * 100)
        os.utime(p, (1000 + idx_f * 10, 1000 + idx_f * 10))
    removed = prune_embedding_cache_dir(cache_dir, max_files=2, max_bytes=10_000_000)
    assert removed >= 2
    assert not f_old.exists()
    assert f_new.exists()

    # 4. CLI --auto-cache hits disk cache on second query
    cache_dir2 = tmp_path / "cli_auto_cache"
    monkeypatch.setenv("ANNOLID_EMBEDDING_CACHE_DIR", str(cache_dir2))
    base_args = [
        "behavior-search",
        str(video_path),
        "--model-id",
        str(model_dir),
        "--quantization",
        "q4",
        "--soft-tokens",
        "70",
        "--matryoshka-dim",
        "128",
        "--stride",
        "2",
        "--auto-cache",
        "--json",
    ]
    assert annolid_run([*base_args, "--query", "grooming"]) == 0
    p1 = json.loads(capsys.readouterr().out)
    assert p1["cache_hit"] is False
    assert annolid_run([*base_args, "--query", "rearing"]) == 0
    p2 = json.loads(capsys.readouterr().out)
    assert p2["cache_hit"] is True
    assert sorted(m["frame_index"] for m in p2["matches"]) == [6, 8, 10]

    # 5. Deterministic cache path helper + GUI ETA progress formatting
    cp1 = default_video_index_cache_path(
        video_path=video_path,
        backend_key=("embeddinggemma2_onnx", "q4"),
        stride=5,
        max_frames=None,
        clip_frames=1,
        cache_dir=cache_dir2,
    )
    cp2 = default_video_index_cache_path(
        video_path=video_path,
        backend_key=("embeddinggemma2_onnx", "q4"),
        stride=5,
        max_frames=None,
        clip_frames=1,
        cache_dir=cache_dir2,
    )
    assert cp1 == cp2

    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    from annolid.gui.widgets.embedding_search_widget import EmbeddingSearchWidget

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    widget = EmbeddingSearchWidget()
    try:
        msgs: List[str] = []
        widget.statusMessage.connect(msgs.append)
        widget._search_started_monotonic = time.monotonic() - 10.0
        widget._on_search_progress(5, 25)
        assert msgs and "5/25" in msgs[-1] and "left)" in msgs[-1]
    finally:
        widget.close()
