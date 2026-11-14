"""Regression tests for large-N Cutie multi-instance scaling, chunking parity, and recovery safety."""

from __future__ import annotations

import math
from pathlib import Path
import types

import cv2
import numpy as np
from omegaconf import OmegaConf
import PIL.Image
import PIL.ImageDraw
import torch

from annolid.annotation.polygon_constraints import (
    make_instance_masks_exclusive,
    resolve_polygon_shape_conflicts,
)
from annolid.gui.shape import MaskShape, Shape
from annolid.segmentation.cutie_vos.engine import CutieEngine
from annolid.segmentation.cutie_vos.inference.inference_core import InferenceCore
from annolid.segmentation.cutie_vos.inference.memory_manager import MemoryManager
from annolid.segmentation.cutie_vos.inference.object_manager import ObjectManager
from annolid.segmentation.cutie_vos.interactive_utils import (
    color_id_mask,
    get_visualization,
    get_visualization_torch,
    overlay_davis,
    overlay_davis_torch,
)
from annolid.segmentation.cutie_vos.model.big_modules import MaskEncoder
from annolid.segmentation.cutie_vos.model.cutie import CUTIE
from annolid.segmentation.cutie_vos.model.transformer.object_summarizer import (
    ObjectSummarizer,
)
from annolid.segmentation.cutie_vos.model.transformer.object_transformer import (
    QueryTransformer,
)
from annolid.segmentation.cutie_vos.model.transformer.positional_encoding import (
    PositionalEncoding,
)
from annolid.segmentation.cutie_vos.model.utils.memory_utils import do_softmax
import annolid.segmentation.cutie_vos.predict as cutie_predict
from annolid.segmentation.cutie_vos.predict import (
    CutieCoreVideoProcessor,
    SeedFrame,
    SeedSegment,
)
from annolid.segmentation.cutie_vos.utils.tensor_utils import aggregate
from annolid.tracking.annotation_adapter import AnnotationAdapter
from annolid.tracking.configuration import CutieDinoTrackerConfig
from annolid.tracking.cutie_mask_manager import CutieMaskManager
from annolid.tracking.domain import InstanceRegistry, InstanceState
from annolid.tracking.identity_continuity import find_cross_instance_mask_overlaps
from annolid.utils.annotation_compat import shape_to_mask, shapes_to_label
from annolid.utils.mask_geometry import (
    compute_mask_centroid_and_area,
    compute_mask_mean_flow_magnitude,
    compute_mask_mean_scalar,
    extract_index_mask_bboxes,
    remap_index_mask_numpy,
    tight_mask_bbox,
)


def _reference_shape_to_mask(
    img_shape: tuple[int, int],
    points: list[list[float]],
    shape_type: str = "polygon",
    line_width: int = 10,
    point_size: int = 5,
) -> np.ndarray:
    mask = PIL.Image.fromarray(np.zeros(img_shape[:2], dtype=np.uint8))
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        (cx, cy), (px, py) = xy
        d = math.hypot(cx - px, cy - py)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "line":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        draw.polygon(xy=xy, outline=1, fill=1)
    return np.asarray(mask, dtype=bool)


def _tiny_transformer_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "value_dim": 32,
            "pixel_dim": 24,
            "pixel_pe_scale": 16,
            "pixel_pe_temperature": 64,
            "object_summarizer": {
                "embed_dim": 32,
                "num_summaries": 4,
                "add_pe": True,
            },
            "object_transformer": {
                "embed_dim": 32,
                "ff_dim": 64,
                "num_heads": 4,
                "num_blocks": 2,
                "num_queries": 4,
                "read_from_pixel": {
                    "input_norm": False,
                    "input_add_pe": True,
                    "add_pe_to_qkv": [True, True, False],
                },
                "read_from_past": {
                    "add_pe_to_qkv": [True, True, False],
                },
                "read_from_ memory": {},
                "read_from_query": {
                    "input_norm": False,
                    "add_pe_to_qkv": [True, True, False],
                    "output_norm": True,
                },
                "query_self_attention": {
                    "add_pe_to_qkv": [True, True, False],
                },
                "pixel_self_attention": {
                    "add_pe_to_qkv": [True, True, False],
                },
            },
        }
    )


def test_mask_geometry_primitives_match_dense_reference() -> None:
    rng = np.random.default_rng(7)
    index_mask = np.zeros((160, 220), dtype=np.int32)
    flow = rng.normal(size=(160, 220, 2)).astype(np.float32)
    scalar = rng.uniform(0.0, 255.0, size=(160, 220)).astype(np.float32)

    object_ids = list(range(1, 45)) + [70_001, 120_005]
    for idx, obj_id in enumerate(object_ids):
        y0 = (idx * 17) % 140
        x0 = (idx * 29) % 195
        index_mask[y0 : y0 + 9, x0 : x0 + 11] = obj_id

    bboxes = extract_index_mask_bboxes(index_mask)
    assert set(bboxes) == set(object_ids)

    for obj_id in [1, 22, 44, 70_001, 120_005]:
        binary = index_mask == obj_id
        ys, xs = np.where(binary)
        expected_bbox = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        )
        assert bboxes[obj_id] == expected_bbox
        assert tight_mask_bbox(binary) == expected_bbox

        moments = cv2.moments(binary.astype(np.uint8))
        expected_cx = float(moments["m10"] / moments["m00"])
        expected_cy = float(moments["m01"] / moments["m00"])
        expected_area = int(np.count_nonzero(binary))
        stats = compute_mask_centroid_and_area(binary, bbox=expected_bbox)
        assert stats is not None
        assert math.isclose(stats[0], expected_cx, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(stats[1], expected_cy, rel_tol=1e-9, abs_tol=1e-9)
        assert stats[2] == expected_area

        expected_scalar = float(scalar[binary].mean())
        got_scalar = compute_mask_mean_scalar(binary, scalar, bbox=expected_bbox)
        assert got_scalar is not None
        assert math.isclose(got_scalar, expected_scalar, rel_tol=1e-5, abs_tol=1e-5)

        dense_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        expected_flow = float(dense_mag[binary].mean())
        got_flow = compute_mask_mean_flow_magnitude(binary, flow, bbox=expected_bbox)
        assert got_flow is not None
        assert math.isclose(got_flow, expected_flow, rel_tol=1e-5, abs_tol=1e-5)

    # Test both LUT path (bounded IDs) and sparse-ID fallback path (>65536).
    bounded_mask = np.where(index_mask < 65536, index_mask, 0)
    bounded_map = {obj_id: obj_id * 10 for obj_id in range(1, 45)}
    expected_bounded = np.zeros_like(bounded_mask)
    for src, dst in bounded_map.items():
        expected_bounded[bounded_mask == src] = dst
    assert np.array_equal(
        remap_index_mask_numpy(bounded_mask, bounded_map), expected_bounded
    )

    sparse_map = {obj_id: idx + 500 for idx, obj_id in enumerate(object_ids)}
    expected_sparse = np.zeros_like(index_mask)
    for src, dst in sparse_map.items():
        expected_sparse[index_mask == src] = dst
    assert np.array_equal(
        remap_index_mask_numpy(index_mask, sparse_map), expected_sparse
    )


def test_extract_index_mask_bboxes_falls_back_without_scipy(monkeypatch) -> None:
    import sys

    bounded = np.zeros((48, 64), dtype=np.int32)
    bounded[0, 0] = -5
    bounded[7, 13] = 1
    bounded[12:20, 30:44] = 9
    expected_bounded = extract_index_mask_bboxes(bounded)

    sparse = bounded.copy()
    sparse[25:33, 5:18] = 75_000
    sparse[40, 60] = 150_123
    expected_sparse = extract_index_mask_bboxes(sparse)

    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.ndimage", None)

    assert extract_index_mask_bboxes(bounded) == expected_bounded
    assert extract_index_mask_bboxes(sparse) == expected_sparse
    assert extract_index_mask_bboxes(np.zeros((8, 8), dtype=np.int32)) == {}


def test_object_manager_and_inference_core_scale_above_255_objects() -> None:
    mgr = ObjectManager()
    global_ids = [idx * 2 + 3 for idx in range(265)]  # 265 instances, IDs up to 531
    corresponding_tmp_ids, inserted_ids = mgr.add_new_objects(global_ids)
    assert len(corresponding_tmp_ids) == 265
    assert inserted_ids == global_ids

    mask = torch.zeros((64, 64), dtype=torch.int32)
    for idx, obj_id in enumerate(global_ids):
        r = idx // 16
        c = idx % 16
        mask[r * 3 : r * 3 + 2, c * 4 : c * 4 + 3] = obj_id

    one_hot = mgr.make_one_hot(mask)
    assert one_hot.shape == (265, 64, 64)
    for tmp_id, obj_id in zip(corresponding_tmp_ids, global_ids):
        expected_channel = (mask == obj_id).float()
        assert torch.equal(one_hot[tmp_id - 1], expected_channel)

    tmp_mask = torch.argmax(
        torch.cat([torch.full((1, 64, 64), 0.1), one_hot], dim=0),
        dim=0,
    )
    restored = mgr.tmp_to_obj_cls(tmp_mask)
    assert torch.equal(restored.to(torch.int32), mask)

    core_auto = InferenceCore.__new__(InferenceCore)
    core_auto.chunk_size = -1
    assert core_auto._effective_chunk_size(12) == -1
    assert core_auto._effective_chunk_size(16) == -1
    assert core_auto._effective_chunk_size(17) == InferenceCore.DEFAULT_AUTO_CHUNK_SIZE
    assert core_auto._effective_chunk_size(265) == InferenceCore.DEFAULT_AUTO_CHUNK_SIZE

    core_explicit = InferenceCore.__new__(InferenceCore)
    core_explicit.chunk_size = 8
    assert core_explicit._effective_chunk_size(265) == 8


def test_interactive_visualizations_wrap_ids_above_255_without_crashing() -> None:
    image = np.full((32, 32, 3), 120, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.int32)
    mask[2:6, 2:6] = 1
    mask[8:12, 8:12] = 255
    mask[14:18, 14:18] = 256  # wraps to palette slot 1 (never 0/background!)
    mask[20:24, 20:24] = 300
    mask[26:30, 26:30] = 512

    vis = overlay_davis(image, mask)
    assert vis.shape == image.shape
    assert np.array_equal(vis[0, 0], image[0, 0])  # background untouched
    assert np.array_equal(
        vis[3, 3], vis[15, 15]
    )  # ID 256 wraps to ID 1, not background
    assert not np.array_equal(vis[15, 15], image[15, 15])

    rgb = color_id_mask(mask)
    assert rgb.shape == (32, 32, 3)
    assert np.array_equal(rgb[0, 0], [0, 0, 0])
    assert np.array_equal(rgb[3, 3], rgb[15, 15])
    assert np.any(rgb[21, 21] > 0)

    vis_davis = get_visualization("davis", image, mask, None, [1, 300])
    assert vis_davis.shape == image.shape

    image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    prob_t = torch.zeros((513, 32, 32), dtype=torch.float32)
    prob_t.scatter_(0, torch.from_numpy(mask).long().unsqueeze(0), 1.0)
    vis_t = overlay_davis_torch(image_t.clone(), prob_t)
    assert vis_t.shape == image.shape
    vis_mode_t = get_visualization_torch(
        "davis", image_t.clone(), prob_t, None, [1, 300]
    )
    assert vis_mode_t.shape == image.shape


def test_aggregate_preserves_small_n_parity_and_prevents_large_n_background_collapse() -> (
    None
):
    torch.manual_seed(11)
    prob_small = torch.sigmoid(torch.randn(16, 12, 12) - 2.0)
    out_small = aggregate(prob_small, dim=0)
    bg_exact = torch.prod(1.0 - prob_small, dim=0, keepdim=True)
    ref_prob = torch.cat([bg_exact, prob_small], dim=0).clamp(1e-7, 1.0 - 1e-7)
    ref_small = torch.log(ref_prob / (1.0 - ref_prob))
    assert torch.equal(out_small, ref_small)

    # With 200 instances and uniform 2% background noise on empty pixels,
    # unbounded (0.98)^200 = 0.0176 < 0.02 would cause empty background to flip
    # to foreground; bounded top-k keeps background winning while preserving
    # confident foreground predictions.
    prob_large = torch.full((200, 16, 16), 0.02, dtype=torch.float32)
    prob_large[77, 4:8, 4:8] = 0.92
    prob_large[185, 10:14, 10:14] = 0.89
    out_large = aggregate(prob_large, dim=0)
    winners = torch.argmax(out_large, dim=0)
    assert int(winners[0, 0].item()) == 0
    assert torch.all(winners[4:8, 4:8] == 78)
    assert torch.all(winners[10:14, 10:14] == 186)

    # Verify > 16 genuinely overlapping foreground objects at one pixel are NOT
    # capped at 16: all 24 overlapping objects (p=0.25) contribute to suppressing
    # background, while 176 background-noise channels (p=0.02) remain gated out.
    prob_dense_overlap = torch.full((200, 4, 4), 0.02, dtype=torch.float32)
    prob_dense_overlap[:24, 1, 2] = 0.25
    out_dense = aggregate(prob_dense_overlap, dim=0)
    expected_bg_24 = (1.0 - 0.25) ** 24
    expected_bg_16 = (1.0 - 0.25) ** 16
    expected_logit_24 = math.log(expected_bg_24 / (1.0 - expected_bg_24))
    expected_logit_16 = math.log(expected_bg_16 / (1.0 - expected_bg_16))
    got_bg_logit = float(out_dense[0, 1, 2].item())
    assert abs(got_bg_logit - expected_logit_24) < 1e-4
    assert abs(got_bg_logit - expected_logit_16) > 1.5
    # Pure noise pixel (0, 0) still stays background
    assert int(torch.argmax(out_dense[:, 0, 0]).item()) == 0


def test_mask_encoder_and_cutie_segment_compact_chunking_parity(monkeypatch) -> None:
    torch.manual_seed(23)
    # 1. MaskEncoder chunked forward matches unchunked and never passes all N objects
    # to distributor at once when chunk_size < N.
    encoder_cfg = OmegaConf.create(
        {
            "pixel_dim": 16,
            "value_dim": 16,
            "sensory_dim": 8,
            "mask_encoder": {"type": "resnet18", "final_dim": 256},
        }
    )
    # Avoid downloading ImageNet weights in unit tests by patching resnet18 pretrained=False
    from annolid.segmentation.cutie_vos.model.utils import resnet as cutie_resnet

    orig_resnet18 = cutie_resnet.resnet18
    orig_resnet50 = cutie_resnet.resnet50
    monkeypatch.setattr(
        cutie_resnet,
        "resnet18",
        lambda **_kw: orig_resnet18(
            pretrained=False, **{k: v for k, v in _kw.items() if k != "pretrained"}
        ),
    )
    monkeypatch.setattr(
        cutie_resnet,
        "resnet50",
        lambda **_kw: orig_resnet50(
            pretrained=False, **{k: v for k, v in _kw.items() if k != "pretrained"}
        ),
    )
    mask_encoder = MaskEncoder(encoder_cfg, single_object=False).eval()

    bs, num_objects, h, w = 1, 11, 32, 32
    image = torch.randn(bs, 3, h, w)
    pix_feat = torch.randn(bs, 16, h // 16, w // 16)
    sensory = torch.randn(bs, num_objects, 8, h // 16, w // 16)
    masks = torch.sigmoid(torch.randn(bs, num_objects, h, w))
    others = (masks.sum(dim=1, keepdim=True) - masks).clamp(0.0, 1.0)

    seen_distributor_n: list[int] = []
    orig_dist_forward = mask_encoder.distributor.forward

    def _spy_distributor(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        seen_distributor_n.append(int(g.shape[1]))
        return orig_dist_forward(x, g)

    monkeypatch.setattr(mask_encoder.distributor, "forward", _spy_distributor)
    with torch.no_grad():
        val_full, sens_full = mask_encoder(
            image, pix_feat, sensory, masks, others, deep_update=True, chunk_size=-1
        )
        seen_distributor_n.clear()
        val_chunked, sens_chunked = mask_encoder(
            image, pix_feat, sensory, masks, None, deep_update=True, chunk_size=3
        )
    assert seen_distributor_n == [3, 3, 3, 2]
    assert torch.allclose(val_full, val_chunked, atol=1e-5)
    assert torch.allclose(sens_full, sens_chunked, atol=1e-5)

    # 2. CUTIE._upsample_and_softmax_logits (return_logits=False) and segment_compact
    # match full (B, N+1, H, W) segment output while keeping last_mask at stride 16.
    engine = CutieEngine.__new__(CutieEngine)
    full_cfg = engine._load_hydra_config({"chunk_size": 4})
    cutie_net = CUTIE(full_cfg).eval()

    ms_feat = [
        torch.randn(1, full_cfg.model.pixel_encoder.ms_dims[0], 2, 3),
        torch.randn(1, full_cfg.model.pixel_encoder.ms_dims[1], 4, 6),
        torch.randn(1, full_cfg.model.pixel_encoder.ms_dims[2], 8, 12),
    ]
    mem_readout = torch.randn(1, 19, full_cfg.model.embed_dim, 2, 3)
    sens_in = torch.randn(1, 19, full_cfg.model.sensory_dim, 2, 3)

    with torch.no_grad():
        sens_ref, logits_ref, prob_ref = cutie_net.segment(
            ms_feat,
            mem_readout,
            sens_in,
            chunk_size=-1,
            update_sensory=True,
            return_logits=True,
        )
        sens_single_buf, logits_none, prob_single_buf = cutie_net.segment(
            ms_feat,
            mem_readout,
            sens_in,
            chunk_size=4,
            update_sensory=True,
            return_logits=False,
        )
        sens_compact, idx_compact, last_s16_compact = cutie_net.segment_compact(
            ms_feat, mem_readout, sens_in, chunk_size=4, update_sensory=True
        )

    assert logits_ref is not None and logits_none is None
    assert torch.allclose(sens_ref, sens_single_buf, atol=1e-5)
    assert torch.allclose(prob_ref, prob_single_buf, atol=1e-5)
    assert torch.allclose(sens_ref, sens_compact, atol=1e-5)
    expected_idx = torch.argmax(prob_ref, dim=1).to(torch.int32)
    assert torch.equal(idx_compact, expected_idx)
    expected_last_s16 = torch.nn.functional.interpolate(
        prob_ref[:, 1:], size=(2, 3), mode="area"
    )
    assert last_s16_compact.shape == (1, 19, 2, 3)
    assert torch.allclose(last_s16_compact, expected_last_s16, atol=1e-5)

    # 3. End-to-end InferenceCore.step keeps last_mask at stride 16 and matches
    # between compact non-memory steps and full probability steps.
    core_compact = InferenceCore(
        cutie_net, cfg=engine._load_hydra_config({"chunk_size": 5, "mem_every": 5})
    )
    core_full = InferenceCore(
        cutie_net, cfg=engine._load_hydra_config({"chunk_size": -1, "mem_every": 5})
    )
    seed_img = torch.rand(3, 32, 48)
    next_img = torch.rand(3, 32, 48)
    seed_mask = torch.zeros((32, 48), dtype=torch.int32)
    for obj_id in range(1, 20):
        r = ((obj_id - 1) // 5) * 7
        c = ((obj_id - 1) % 5) * 9
        seed_mask[r : r + 5, c : c + 6] = obj_id

    with torch.no_grad():
        out0_c = core_compact.step(
            seed_img,
            seed_mask,
            objects=list(range(1, 20)),
            idx_mask=True,
            return_index_mask=True,
        )
        out0_f = torch.argmax(
            core_full.step(
                seed_img,
                seed_mask,
                objects=list(range(1, 20)),
                idx_mask=True,
                return_index_mask=False,
            ),
            dim=0,
        ).to(torch.int32)
        assert torch.equal(out0_c, out0_f)
        assert core_compact.last_mask.shape == (1, 19, 2, 3)
        assert core_full.last_mask.shape == (1, 19, 2, 3)

        out1_c = core_compact.step(next_img, return_index_mask=True)
        out1_f = torch.argmax(
            core_full.step(next_img, return_index_mask=False), dim=0
        ).to(torch.int32)
        assert torch.equal(out1_c, out1_f)
        assert core_compact.last_mask.shape == (1, 19, 2, 3)


def test_memory_manager_compact_fp16_and_adaptive_kv_budget_for_fifo_and_long_term() -> (
    None
):
    torch.manual_seed(31)
    # 1. Verify do_softmax never crashes when memory tokens < top_k
    tiny_sim = torch.randn(1, 6, 12)
    aff = do_softmax(tiny_sim.clone(), top_k=30, inplace=False)
    assert aff.shape == (1, 6, 12)
    assert torch.allclose(aff.sum(dim=1), torch.ones(1, 12), atol=1e-5)

    # 2. Verify auto float16 promotion (<=16 stays float32; >16 promotes all values to float16)
    # and strict KV memory budget enforcement in FIFO mode.
    fifo_cfg = OmegaConf.create(
        {
            "model": {"sensory_dim": 8},
            "top_k": 10,
            "chunk_size": 8,
            "kv_storage_dtype": "auto",
            "max_kv_memory_mb": 0.25,  # 256 KB cap across 40 objects
            "save_aux": False,
            "use_long_term": False,
            "max_mem_frames": 8,
            "long_term": {"count_usage": False},
        }
    )
    mgr_fifo = ObjectManager()
    initial_ids = list(range(1, 11))
    mgr_fifo.add_new_objects(initial_ids)
    mem_fifo = MemoryManager(fifo_cfg, mgr_fifo)

    bs, ck, cv, h, w = 1, 8, 16, 4, 4  # HW = 16
    key = torch.randn(bs, ck, h, w, dtype=torch.float32)
    shrinkage = torch.ones(bs, 1, h, w, dtype=torch.float32)
    mem_fifo.add_memory(
        key,
        shrinkage,
        torch.randn(bs, 10, cv, h, w, dtype=torch.float32),
        torch.randn(bs, 10, 2, 9, dtype=torch.float32),
        initial_ids,
        as_permanent="first",
    )
    assert mem_fifo.work_mem.v[1].dtype == torch.float32  # <= 16 objects stays fp32

    # Expand to 40 objects -> promotes existing + new values to float16 and enforces 256 KB cap
    extra_ids = list(range(11, 41))
    mgr_fifo.add_new_objects(extra_ids)
    all_40 = list(range(1, 41))
    for _ in range(10):
        mem_fifo.add_memory(
            torch.randn(bs, ck, h, w),
            shrinkage,
            torch.randn(bs, 40, cv, h, w, dtype=torch.float32),
            torch.randn(bs, 40, 2, 9, dtype=torch.float32),
            all_40,
            as_permanent="no",
        )
    assert all(v.dtype == torch.float16 for v in mem_fifo.work_mem.v.values())
    non_perm_bytes = mem_fifo.work_mem.value_bytes(non_permanent_only=True)
    assert non_perm_bytes <= int(0.25 * 1024 * 1024)

    # 3. Verify long-term memory consolidation under a tight KV budget for 60 objects
    lt_cfg = OmegaConf.create(
        {
            "model": {"sensory_dim": 8},
            "top_k": 10,
            "chunk_size": 16,
            "kv_storage_dtype": "auto",
            "max_kv_memory_mb": 0.5,  # 512 KB total working + long-term budget
            "save_aux": False,
            "use_long_term": True,
            "max_mem_frames": 5,
            "long_term": {
                "count_usage": True,
                "max_mem_frames": 10,
                "min_mem_frames": 5,
                "num_prototypes": 64,
                "max_num_tokens": 2000,
                "buffer_tokens": 256,
            },
        }
    )
    mgr_lt = ObjectManager()
    ids_60 = list(range(1, 61))
    mgr_lt.add_new_objects(ids_60)
    mem_lt = MemoryManager(lt_cfg, mgr_lt)
    selection = torch.sigmoid(torch.randn(bs, ck, h, w))

    class _StubReadNet:
        single_object = False

        def pixel_fusion(self, pix_feat, visual_readout, sensory, last_mask, **_kw):
            _ = pix_feat, sensory, last_mask
            return visual_readout

        def readout_query(self, pixel_readout, obj_mem, **_kw):
            _ = obj_mem
            return pixel_readout, None

    mem_lt.initialize_sensory_if_needed(key, ids_60)
    stub_net = _StubReadNet()
    for step_idx in range(14):
        mem_lt.add_memory(
            torch.randn(bs, ck, h, w),
            shrinkage,
            torch.randn(bs, 60, cv, h, w, dtype=torch.float32),
            torch.randn(bs, 60, 2, 9, dtype=torch.float32),
            ids_60,
            selection=selection,
            as_permanent="first" if step_idx == 0 else "no",
        )
        # Exercise read() so usage counters update across working & long-term stores
        read_out = mem_lt.read(
            torch.randn(bs, cv, h, w),
            torch.randn(bs, ck, h, w),
            selection,
            torch.rand(bs, 60, h, w),
            stub_net,
            chunk_size=16,
        )
        assert len(read_out) == 60
        assert next(iter(read_out.values())).dtype == torch.float32

    assert mem_lt.long_mem.engaged(0)
    assert all(v.dtype == torch.float16 for v in mem_lt.work_mem.v.values())
    assert all(v.dtype == torch.float16 for v in mem_lt.long_mem.v.values())
    total_non_perm = mem_lt.work_mem.value_bytes(
        non_permanent_only=True
    ) + mem_lt.long_mem.value_bytes(non_permanent_only=True)
    assert (
        total_non_perm <= int(0.5 * 1024 * 1024) * 2
    )  # bounded well below unthrottled 12+ MB

    # 4. Verify adaptive prior seed limit in commit_masks_into_permanent_memory (>64 -> 1)
    proc = CutieCoreVideoProcessor.__new__(CutieCoreVideoProcessor)
    proc.cfg = types.SimpleNamespace(amp=False)
    proc.device = "cpu"
    proc.prior_reference_seed_limit = None
    proc._seed_frames = [
        SeedFrame(i * 10, Path(f"{i}.png"), Path(f"{i}.json")) for i in range(1, 6)
    ]
    proc._committed_seed_frames = {10, 20, 30, 40, 50}
    captured_prior_limits: list[int | None] = []
    orig_ref = CutieCoreVideoProcessor._reference_seed_frames_for_segment

    def _spy_ref(frame_num, seeds, segment_end=None, prior_limit=None):
        captured_prior_limits.append(prior_limit)
        return orig_ref(
            frame_num, seeds, segment_end=segment_end, prior_limit=prior_limit
        )

    proc._reference_seed_frames_for_segment = _spy_ref
    proc.commit_masks_into_permanent_memory(
        50, {}, active_labels=[f"a_{i}" for i in range(10)]
    )
    proc.commit_masks_into_permanent_memory(
        50, {}, active_labels=[f"a_{i}" for i in range(80)]
    )
    assert captured_prior_limits == [None, 1]


def test_object_summarizer_and_query_transformer_chunking_exact_parity() -> None:
    torch.manual_seed(42)
    model_cfg = _tiny_transformer_cfg()
    summarizer = ObjectSummarizer(model_cfg).eval()
    transformer = QueryTransformer(model_cfg).eval()

    bs, num_objects, h, w = 1, 11, 6, 8
    masks = torch.sigmoid(torch.randn(bs, num_objects, h, w))
    values = torch.randn(bs, num_objects, model_cfg.value_dim, h, w)

    with torch.no_grad():
        sum_full, weights_full = summarizer(
            masks, values, need_weights=True, chunk_size=-1
        )
        sum_chunked, weights_chunked = summarizer(
            masks, values, need_weights=True, chunk_size=3
        )
    assert torch.allclose(sum_full, sum_chunked, atol=1e-5)
    assert weights_full is not None and weights_chunked is not None
    assert torch.allclose(weights_full, weights_chunked, atol=1e-5)

    pixel = torch.randn(bs, num_objects, model_cfg.value_dim, h, w)
    obj_summaries = sum_full.unsqueeze(2)  # bs * num_objects * T=1 * Q * (embed_dim+1)
    selector = (
        torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )

    with torch.no_grad():
        pix_full, aux_full = transformer(
            pixel, obj_summaries, selector=selector, need_weights=True, chunk_size=-1
        )
        pix_chunked, aux_chunked = transformer(
            pixel, obj_summaries, selector=selector, need_weights=True, chunk_size=3
        )
    assert torch.allclose(pix_full, pix_chunked, atol=1e-5)
    assert len(aux_full["logits"]) == len(aux_chunked["logits"])
    for l_full, l_chunk in zip(aux_full["logits"], aux_chunked["logits"]):
        assert torch.allclose(l_full, l_chunk, atol=1e-5)
    assert torch.allclose(aux_full["q_weights"], aux_chunked["q_weights"], atol=1e-5)
    assert torch.allclose(aux_full["p_weights"], aux_chunked["p_weights"], atol=1e-5)


def test_positional_encoding_cache_handles_transposed_5d_and_dtype_changes() -> None:
    pe = PositionalEncoding(
        32, scale=16, temperature=64, channel_last=False, transpose_output=True
    )
    x_small_n = torch.zeros(1, 3, 24, 6, 8, dtype=torch.float32)
    x_large_n = torch.zeros(1, 90, 24, 6, 8, dtype=torch.float32)

    out_1 = pe(x_small_n)
    cached_ref = pe.cached_penc
    out_2 = pe(x_large_n)
    assert out_1.shape == (1, 1, 6, 8, 32)
    assert out_2.shape == (1, 1, 6, 8, 32)
    assert (
        pe.cached_penc is cached_ref
    )  # reused across any num_objects without reallocating!

    x_fp64 = torch.zeros(1, 5, 24, 6, 8, dtype=torch.float64)
    out_fp64 = pe(x_fp64)
    assert out_fp64.dtype == torch.float64
    assert pe.cached_penc is not cached_ref


def test_memory_manager_chunked_read_matches_unchunked_with_cross_object_last_others() -> (
    None
):
    class _FakeNetwork:
        def pixel_fusion(
            self,
            pix_feat: torch.Tensor,
            readout: torch.Tensor,
            sensory: torch.Tensor,
            last_mask: torch.Tensor,
            *,
            chunk_size: int = -1,
            last_others: torch.Tensor | None = None,
        ) -> torch.Tensor:
            _ = chunk_size
            num_objects = last_mask.shape[1]
            if last_others is None:
                total = last_mask.sum(dim=1, keepdim=True)
                last_others = (total - last_mask).clamp(0.0, 1.0)
            expanded_pix = pix_feat.unsqueeze(1).expand(-1, num_objects, -1, -1, -1)
            mask_term = (last_mask * 0.7 - last_others * 1.3).unsqueeze(2)
            return expanded_pix + readout + sensory + mask_term

        def readout_query(
            self,
            pixel: torch.Tensor,
            obj_memory: torch.Tensor,
            *,
            selector: torch.Tensor | None = None,
            need_weights: bool = False,
            chunk_size: int = -1,
        ):
            _ = selector, need_weights, chunk_size
            obj_bias = (
                obj_memory[:, :, 0, :, :-1].mean(dim=2).unsqueeze(-1).unsqueeze(-1)
            )
            return pixel + obj_bias, {"logits": pixel[:, :, 0]}

    torch.manual_seed(19)
    cfg = OmegaConf.create(
        {
            "model": {"sensory_dim": 8},
            "top_k": 4,
            "chunk_size": -1,
            "save_aux": False,
            "use_long_term": False,
            "max_mem_frames": 5,
            "long_term": {"count_usage": False},
        }
    )
    obj_mgr = ObjectManager()
    obj_mgr.add_new_objects([10, 20, 30, 40, 50])
    mem = MemoryManager(cfg, obj_mgr)

    bs, ck, cv, h, w = 1, 8, 8, 4, 5
    key = torch.randn(bs, ck, h, w)
    shrinkage = torch.ones(bs, 1, h, w)
    msk_val = torch.randn(bs, 5, cv, h, w)
    obj_val = torch.randn(bs, 5, 3, cv + 1)
    mem.initialize_sensory_if_needed(key, obj_mgr.all_obj_ids)
    mem.add_memory(
        key, shrinkage, msk_val, obj_val, obj_mgr.all_obj_ids, as_permanent="all"
    )

    query_key = torch.randn(bs, ck, h, w)
    selection = torch.ones(bs, ck, h, w)
    pix_feat = torch.randn(bs, cv, h, w)
    last_mask = torch.sigmoid(torch.randn(bs, 5, h, w))
    net = _FakeNetwork()

    out_full = mem.read(pix_feat, query_key, selection, last_mask, net, chunk_size=-1)
    out_chunked = mem.read(pix_feat, query_key, selection, last_mask, net, chunk_size=2)
    assert set(out_full) == set(out_chunked) == {10, 20, 30, 40, 50}
    for obj_id in out_full:
        assert torch.allclose(out_full[obj_id], out_chunked[obj_id], atol=1e-5)


def test_patch_rasterization_and_bbox_polygon_pipeline_equivalence() -> None:
    img_shape = (240, 320)
    cases = [
        ("polygon", [[-5.0, 10.0], [25.0, 8.0], [20.0, 35.0], [-2.0, 30.0]]),
        ("rectangle", [[40.0, 50.0], [95.0, 110.0]]),
        ("circle", [[150.0, 120.0], [168.0, 125.0]]),
        ("line", [[200.0, 20.0], [260.0, 90.0]]),
        ("point", [[290.0, 210.0]]),
    ]
    for shape_type, pts in cases:
        got = shape_to_mask(img_shape, pts, shape_type=shape_type)
        ref = _reference_shape_to_mask(img_shape, pts, shape_type=shape_type)
        assert np.array_equal(got, ref), f"Mismatch for shape_type={shape_type}"

    # Verify >255 instances through shapes_to_label
    shapes = []
    label_map = {"_background_": 0}
    for idx in range(1, 270):
        row = (idx - 1) // 20
        col = (idx - 1) % 20
        x0 = col * 15 + 2
        y0 = row * 15 + 2
        label = f"inst_{idx:03d}"
        label_map[label] = idx
        shapes.append(
            {
                "label": label,
                "points": [[x0, y0], [x0 + 9, y0], [x0 + 9, y0 + 9], [x0, y0 + 9]],
                "shape_type": "polygon",
                "group_id": idx,
            }
        )
    cls, ins = shapes_to_label(img_shape, shapes, label_map)
    assert int(cls.max()) == 269
    assert int(ins.max()) == 269
    assert len(np.unique(cls)) == 270

    # Verify MaskShape.toPolygons(..., bbox=bbox) matches full-frame toPolygons
    mask = np.zeros((240, 320), dtype=bool)
    mask[70:105, 110:155] = True
    ms_full = MaskShape(label="mouse_1")
    ms_full.mask = mask
    ms_bbox = MaskShape(label="mouse_1")
    ms_bbox.mask = mask
    bbox = tight_mask_bbox(mask)
    polys_full = ms_full.toPolygons(epsilon=1.5, fill_holes=False)
    polys_bbox = ms_bbox.toPolygons(epsilon=1.5, fill_holes=False, bbox=bbox)
    assert len(polys_full) == len(polys_bbox) == 1
    pts_full = [(p.x(), p.y()) for p in polys_full[0].points]
    pts_bbox = [(p.x(), p.y()) for p in polys_bbox[0].points]
    assert pts_full == pts_bbox

    # Verify make_instance_masks_exclusive and find_cross_instance_mask_overlaps
    m1 = np.zeros((80, 80), dtype=bool)
    m2 = np.zeros((80, 80), dtype=bool)
    m3 = np.zeros((80, 80), dtype=bool)
    m1[10:30, 10:30] = True
    m2[20:40, 20:40] = True
    m3[50:65, 50:65] = True
    overlaps = find_cross_instance_mask_overlaps({"a": m1, "b": m2, "c": m3})
    assert len(overlaps) == 1 and overlaps[0].pixel_count == 100
    excl = make_instance_masks_exclusive([m1, m2, m3], scores=[0.5, 0.9, 0.7])
    assert int(np.count_nonzero(excl[0] & excl[1])) == 0
    assert int(np.count_nonzero(excl[1])) == 400
    assert int(np.count_nonzero(excl[0])) == 300

    # Verify resolve_polygon_shape_conflicts across 40 polygons with local overlap
    poly_shapes: list[Shape] = []
    for idx in range(40):
        bx = (idx % 10) * 30 + 5
        by = (idx // 10) * 30 + 5
        s = Shape(label=f"obj_{idx}", shape_type="polygon")
        s.points = [[bx, by], [bx + 18, by], [bx + 18, by + 18], [bx, by + 18]]
        poly_shapes.append(s)
    # Introduce overlap between obj_0 and overlapper
    overlapper = Shape(label="obj_overlap", shape_type="polygon")
    overlapper.points = [[15, 10], [30, 10], [30, 25], [15, 25]]
    poly_shapes.append(overlapper)
    resolved = resolve_polygon_shape_conflicts(poly_shapes)
    assert len(resolved.shapes) == 41
    assert 40 in resolved.adjusted_shape_indices


def test_cutie_predict_identity_ambiguity_uses_cached_centroids_and_invalidates_on_mutation() -> (
    None
):
    processor = CutieCoreVideoProcessor.__new__(CutieCoreVideoProcessor)
    processor.identity_switch_protection_enabled = True
    processor._recent_instance_masks = {}
    processor._recent_instance_mask_frames = {}
    processor._recent_instance_centroids = {}
    processor._recent_instance_centroid_frames = {}
    processor._frame_mask_geometry_cache = {}
    processor._last_saved_instance_masks = {}
    processor._last_saved_instance_centroids = {}

    expected_labels = {f"fish_{i}" for i in range(25)}
    frame0_masks: dict[str, np.ndarray] = {}
    frame1_masks: dict[str, np.ndarray] = {}
    for i in range(25):
        cy = (i // 5) * 40 + 20
        cx = (i % 5) * 50 + 25
        m0 = np.zeros((240, 320), dtype=bool)
        m0[cy - 4 : cy + 5, cx - 4 : cx + 5] = True
        frame0_masks[f"fish_{i}"] = m0

        m1 = np.zeros((240, 320), dtype=bool)
        m1[cy - 4 : cy + 5, cx - 3 : cx + 6] = True
        frame1_masks[f"fish_{i}"] = m1

    processor._update_recent_instance_masks(0, frame0_masks)
    assert (
        processor._detect_identity_assignment_ambiguity(
            frame1_masks, expected_labels, 1
        )
        is None
    )

    # Swap fish_0 and fish_24 -> strong ambiguity must be detected!
    swapped_masks = dict(frame1_masks)
    swapped_masks["fish_0"], swapped_masks["fish_24"] = (
        frame0_masks["fish_24"].copy(),
        frame0_masks["fish_0"].copy(),
    )
    ambiguity = processor._detect_identity_assignment_ambiguity(
        swapped_masks, expected_labels, 1
    )
    assert ambiguity is not None
    assert ("fish_0", "fish_24") in ambiguity.changed_assignments

    # Mutate _recent_instance_masks directly without calling _update_recent_instance_masks:
    # object-identity guard must invalidate stale cached centroids!
    processor._recent_instance_masks["fish_0"] = swapped_masks["fish_0"].copy()
    processor._recent_instance_masks["fish_24"] = swapped_masks["fish_24"].copy()
    assert (
        processor._detect_identity_assignment_ambiguity(
            swapped_masks, expected_labels, 1
        )
        is None
    )


def test_multi_instance_partial_occlusion_throttles_and_rolls_back_failed_recovery(
    monkeypatch,
) -> None:
    created_cores = []

    class _DummyInferenceCore:
        def __init__(self, *_args, **_kwargs):
            self.step_calls = 0
            created_cores.append(self)

        def step(self, _frame_torch, mask_tensor=None, **_kwargs):
            self.step_calls += 1
            if mask_tensor is not None and len(created_cores) == 1:
                # Initial seed frame has all 4 instances
                return np.array([[1, 2], [3, 4]], dtype=np.int32)
            # Subsequent frames (including recovery attempts) have instances 3 & 4 occluded
            return np.array([[1, 2], [0, 0]], dtype=np.int32)

    class _DummyCap:
        def __init__(self, frames: int = 6):
            self._frames = frames
            self._pos = 0

        def get(self, _prop):
            return self._pos

        def set(self, _prop, value):
            self._pos = int(value)

        def isOpened(self):
            return self._pos < self._frames

        def read(self):
            if self._pos >= self._frames:
                return False, None
            self._pos += 1
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

    processor = CutieCoreVideoProcessor.__new__(CutieCoreVideoProcessor)
    processor.cutie = object()
    processor.cfg = types.SimpleNamespace(amp=False)
    processor.device = "cpu"
    processor.video_folder = Path("clip")
    processor.label_registry = {
        "_background_": 0,
        "m1": 1,
        "m2": 2,
        "m3": 3,
        "m4": 4,
    }
    processor._global_label_names = {}
    processor.compute_optical_flow = False
    processor.auto_missing_instance_recovery = True
    processor.auto_fill_missing_instances = False
    processor.automatic_pause_enabled = False
    processor.continue_on_missing_instances = True
    processor.debug = False
    processor._optical_flow_kwargs = {}
    processor.optical_flow_backend = "farneback"
    processor._recent_instance_masks = {}
    processor._recent_instance_mask_frames = {}
    processor._recovery_seed_frame_index = None
    processor._recovery_seed_frame = None
    processor._recovery_seed_masks = {}
    processor._last_saved_instance_masks = {}
    processor._flow_hsv = None
    processor._should_stop = lambda _worker=None: False
    processor.commit_masks_into_permanent_memory = lambda *_args, **_kwargs: dict(
        processor.label_registry
    )
    processor._register_active_objects = lambda _ids: None
    processor._update_tracking_frame_stat = lambda *_args, **_kwargs: None

    def _capture_save(_filename, mask_dict, _shape, shape_notes=None):
        _ = shape_notes
        processor._last_saved_instance_masks = {
            k: v.copy() for k, v in mask_dict.items()
        }

    processor._save_annotation_with_notes = _capture_save

    monkeypatch.setattr(cutie_predict, "InferenceCore", _DummyInferenceCore)
    monkeypatch.setattr(
        cutie_predict, "image_to_torch", lambda frame, device=None: frame
    )
    monkeypatch.setattr(cutie_predict, "torch_prob_to_numpy_mask", lambda pred: pred)

    segment = SeedSegment(
        seed=SeedFrame(0, Path("0.png"), Path("0.json")),
        start_frame=0,
        end_frame=5,
        mask=np.array([[1, 2], [3, 4]], dtype=np.int32),
        labels_map={"_background_": 0, "m1": 1, "m2": 2, "m3": 3, "m4": 4},
        active_labels=["m1", "m2", "m3", "m4"],
    )

    message, should_halt = processor._process_segment(
        cap=_DummyCap(frames=6),
        segment=segment,
        end_frame=5,
        fps=30.0,
    )

    assert should_halt is False
    assert message == "Stop at frame:\n#5"
    # Core #1 is the segment core; Core #2 was created once at frame 1 for recovery,
    # failed to recover m3/m4, rolled back to Core #1, and frames 2..5 were throttled!
    assert len(created_cores) == 2
    assert processor.processor is created_cores[0]


def test_cutie_engine_hydra_null_defaults_and_cutie_mask_manager_tmp_id_remap(
    tmp_path: Path,
) -> None:
    # 1. CutieEngine._load_hydra_config fills eval_config.yaml null fields safely
    engine = CutieEngine.__new__(CutieEngine)
    cfg = engine._load_hydra_config({"chunk_size": 12})
    assert cfg.mem_every == 5
    assert cfg.max_mem_frames == 5
    assert cfg.use_long_term is False
    assert cfg.chunk_size == 12

    # 2. CutieMaskManager.update_masks maps temporary IDs via output_prob_to_mask
    # and preserves deterministic fallback ordering.
    adapter = AnnotationAdapter(image_height=16, image_width=16)
    tracker_cfg = CutieDinoTrackerConfig(
        use_cutie_tracking=True,
        max_mask_fallback_frames=2,
        mask_dilation_iterations=0,
    )
    manager = CutieMaskManager(tmp_path / "video.mp4", adapter, tracker_cfg)

    class _StubCoreWithRemap:
        def step(self, *_args, **_kwargs):
            # Returns temporary ID 1 in top-left corner
            out = torch.zeros((16, 16), dtype=torch.int32)
            out[2:8, 2:8] = 1
            return out

        def output_prob_to_mask(self, output: torch.Tensor) -> torch.Tensor:
            # Maps temporary ID 1 -> persistent object ID 2 ("beta")
            return torch.where(output == 1, torch.tensor(2, dtype=torch.int32), 0)

    m_alpha = np.zeros((16, 16), dtype=bool)
    m_alpha[9:14, 9:14] = True
    m_beta = np.zeros((16, 16), dtype=bool)
    m_beta[2:8, 2:8] = True
    registry = InstanceRegistry(
        {
            "alpha": InstanceState(label="alpha", mask_bitmap=m_alpha),
            "beta": InstanceState(label="beta", mask_bitmap=m_beta),
        }
    )

    manager._core = _StubCoreWithRemap()
    manager._device = "cpu"
    manager._initialized = True
    manager._label_to_value = {"alpha": 1, "beta": 2}
    manager._value_to_label = {1: "alpha", 2: "beta"}
    manager._seed_last_results(registry)

    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    results = manager.update_masks(1, frame, registry)
    assert list(results.keys()) == ["beta", "alpha"]
    assert np.array_equal(results["beta"].mask_bitmap, m_beta)
    # "alpha" was missing from step() output and deterministically filled from fallback
    assert np.array_equal(results["alpha"].mask_bitmap, m_alpha)


def test_inference_core_streaming_seed_and_memory_frames_match_unchunked(
    monkeypatch,
) -> None:
    torch.manual_seed(53)
    from annolid.segmentation.cutie_vos.model.utils import resnet as cutie_resnet

    orig_resnet18 = cutie_resnet.resnet18
    orig_resnet50 = cutie_resnet.resnet50
    monkeypatch.setattr(
        cutie_resnet,
        "resnet18",
        lambda **_kw: orig_resnet18(
            pretrained=False, **{k: v for k, v in _kw.items() if k != "pretrained"}
        ),
    )
    monkeypatch.setattr(
        cutie_resnet,
        "resnet50",
        lambda **_kw: orig_resnet50(
            pretrained=False, **{k: v for k, v in _kw.items() if k != "pretrained"}
        ),
    )

    engine = CutieEngine.__new__(CutieEngine)
    net_cfg = engine._load_hydra_config({"chunk_size": 4, "mem_every": 2})
    cutie_net = CUTIE(net_cfg).eval()

    # Keep KV storage in float32 on both cores so multi-frame parity is exact.
    core_stream = InferenceCore(
        cutie_net,
        cfg=engine._load_hydra_config(
            {"chunk_size": 4, "mem_every": 2, "kv_storage_dtype": "float32"}
        ),
    )
    core_ref = InferenceCore(
        cutie_net,
        cfg=engine._load_hydra_config(
            {"chunk_size": -1, "mem_every": 2, "kv_storage_dtype": "float32"}
        ),
    )

    one_hot_called = False
    orig_make_one_hot = core_stream.object_manager.make_one_hot

    def _forbid_make_one_hot(*args, **kwargs):
        nonlocal one_hot_called
        one_hot_called = True
        return orig_make_one_hot(*args, **kwargs)

    monkeypatch.setattr(
        core_stream.object_manager, "make_one_hot", _forbid_make_one_hot
    )

    obj_ids = [idx * 10 for idx in range(1, 20)]  # 19 non-consecutive IDs
    h, w = 32, 48
    seed_mask_0 = torch.zeros((h, w), dtype=torch.int32)
    seed_mask_3 = torch.zeros((h, w), dtype=torch.int32)
    for pos, obj_id in enumerate(obj_ids):
        r = (pos // 5) * 7
        c = (pos % 5) * 9
        seed_mask_0[r : r + 5, c : c + 6] = obj_id
        seed_mask_3[r : r + 5, min(w, c + 1) : min(w, c + 7)] = obj_id

    frames = [torch.rand(3, h, w) for _ in range(5)]

    with torch.no_grad():
        # t=0: streaming indexed seed (permanent='first')
        out0_s = core_stream.step(
            frames[0],
            seed_mask_0,
            objects=obj_ids,
            idx_mask=True,
            return_index_mask=True,
        )
        out0_r = torch.argmax(
            core_ref.step(
                frames[0],
                seed_mask_0,
                objects=obj_ids,
                idx_mask=True,
                return_index_mask=False,
            ),
            dim=0,
        ).to(torch.int32)
        assert not one_hot_called
        assert torch.equal(out0_s, out0_r)

        # t=1: non-memory propagation frame
        out1_s = core_stream.step(frames[1], return_index_mask=True)
        out1_r = torch.argmax(
            core_ref.step(frames[1], return_index_mask=False), dim=0
        ).to(torch.int32)
        assert torch.equal(out1_s, out1_r)

        # t=2: memory propagation frame (`is_mem_frame=True` since mem_every=2)
        out2_s = core_stream.step(frames[2], return_index_mask=True)
        out2_r = torch.argmax(
            core_ref.step(frames[2], return_index_mask=False), dim=0
        ).to(torch.int32)
        assert core_stream.last_mem_ti == 2 and core_ref.last_mem_ti == 2
        assert torch.equal(out2_s, out2_r)

        # t=3: mid-run permanent seed correction (`force_permanent=True`)
        out3_s = core_stream.step(
            frames[3],
            seed_mask_3,
            objects=obj_ids,
            idx_mask=True,
            force_permanent=True,
            return_index_mask=True,
        )
        out3_r = torch.argmax(
            core_ref.step(
                frames[3],
                seed_mask_3,
                objects=obj_ids,
                idx_mask=True,
                force_permanent=True,
                return_index_mask=False,
            ),
            dim=0,
        ).to(torch.int32)
        assert not one_hot_called
        assert torch.equal(out3_s, out3_r)

        # t=4: post-correction propagation frame reading both permanent + working KV
        out4_s = core_stream.step(frames[4], return_index_mask=True)
        out4_r = torch.argmax(
            core_ref.step(frames[4], return_index_mask=False), dim=0
        ).to(torch.int32)
        assert torch.equal(out4_s, out4_r)

    assert torch.allclose(core_stream.last_mask, core_ref.last_mask, atol=1e-4)
    for obj_id in obj_ids:
        assert torch.allclose(
            core_stream.memory.sensory[obj_id],
            core_ref.memory.sensory[obj_id],
            atol=1e-4,
        )
        assert torch.allclose(
            core_stream.memory.obj_v[obj_id],
            core_ref.memory.obj_v[obj_id],
            atol=1e-4,
        )
        assert torch.allclose(
            core_stream.memory.work_mem.v[obj_id],
            core_ref.memory.work_mem.v[obj_id],
            atol=1e-4,
        )


def test_extreme_scale_600_objects_4k_spatial_pooling_and_permanent_budget() -> None:
    torch.manual_seed(61)
    # 600 instances on a 4K stride-16 grid (135 x 240 = 32,400 spatial tokens/frame).
    # Without sub-frame spatial token pooling and permanent-token trimming, even a
    # single float16 frame here (`600 * 8 * 32,400 * 2` = 296.6 MB) would exceed the
    # 2.0 MB budget by >148x.
    num_objects = 600
    obj_ids = list(range(1, num_objects + 1))
    bs, ck, cv, h16, w16 = 1, 4, 8, 135, 240
    budget_mb = 2.0
    budget_bytes = int(budget_mb * 1024 * 1024)

    class _Stub4KNet:
        single_object = False

        def pixel_fusion(self, _pix_feat, visual_readout, _sensory, _last_mask, **_kw):
            return visual_readout

        def readout_query(self, pixel_readout, _obj_mem, **_kw):
            return pixel_readout, None

    stub_net = _Stub4KNet()
    sample_msk_full = torch.randn(bs, 32, cv, h16, w16, dtype=torch.float32)
    sample_obj_full = torch.randn(bs, 32, 2, cv + 1, dtype=torch.float32)
    sample_pix = torch.randn(bs, cv, h16, w16)
    sample_qkey = torch.randn(bs, ck, h16, w16)
    sample_qsel = torch.ones(bs, ck, h16, w16)
    sample_last_mask = torch.full((bs, num_objects, h16, w16), 0.01)

    for use_long_term in (False, True):
        cfg = OmegaConf.create(
            {
                "model": {"sensory_dim": 4},
                "top_k": 15,
                "chunk_size": 32,
                "kv_storage_dtype": "auto",
                "max_kv_memory_mb": budget_mb,
                "save_aux": False,
                "use_long_term": use_long_term,
                "max_mem_frames": 5,
                "long_term": {
                    "count_usage": True,
                    "max_mem_frames": 8,
                    "min_mem_frames": 4,
                    "num_prototypes": 64,
                    "max_num_tokens": 4000,
                    "buffer_tokens": 256,
                },
            }
        )
        mgr = ObjectManager()
        mgr.add_new_objects(obj_ids)
        mem = MemoryManager(cfg, mgr)
        key0 = torch.randn(bs, ck, h16, w16)
        mem.initialize_sensory_if_needed(key0, obj_ids)

        # Commit 4 permanent seed frames + 6 temporary working frames via 32-object chunks
        schedule = ["first", "no", "all", "no", "all", "no", "no", "all", "no", "no"]
        for step_i, perm_mode in enumerate(schedule):
            key = torch.randn(bs, ck, h16, w16)
            shrinkage = torch.ones(bs, 1, h16, w16)
            selection = (
                torch.sigmoid(torch.randn(bs, ck, h16, w16)) if use_long_term else None
            )

            def _chunks(scale_factor: float = 1.0 + 0.05 * step_i):
                for start in range(0, num_objects, 32):
                    chunk_objs = obj_ids[start : start + 32]
                    c_len = len(chunk_objs)
                    yield (
                        chunk_objs,
                        sample_msk_full[:, :c_len] * scale_factor,
                        sample_obj_full[:, :c_len],
                    )

            mem.add_memory_chunks(
                key,
                shrinkage,
                _chunks(),
                obj_ids,
                selection=selection,
                as_permanent=perm_mode,
            )

            # Total KV bytes INCLUDING permanent tokens must strictly stay <= budget_bytes!
            total_kv_bytes = mem.work_mem.value_bytes(non_permanent_only=False)
            if use_long_term:
                total_kv_bytes += mem.long_mem.value_bytes(non_permanent_only=False)
            assert total_kv_bytes <= budget_bytes, (
                f"use_long_term={use_long_term}, perm_mode={perm_mode}: "
                f"{total_kv_bytes} > {budget_bytes}"
            )
            if use_long_term and mem.work_mem.non_perm_size(0) > 0:
                mem.work_mem.update_bucket_usage(
                    0, torch.rand(bs, mem.work_mem.size(0))
                )

        read_out = mem.read(
            sample_pix,
            sample_qkey,
            sample_qsel,
            sample_last_mask,
            stub_net,
            chunk_size=64,
        )
        assert len(read_out) == num_objects
        assert read_out[1].shape == (bs, cv, h16, w16)
        assert read_out[num_objects].shape == (bs, cv, h16, w16)
        assert mem.work_mem.size(0) < h16 * w16
        if use_long_term:
            assert mem.long_mem.engaged(0)
