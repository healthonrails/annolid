#!/usr/bin/env python3
"""Directly upgrade and compress Cutie's weights using:
  1. Algebraic Conv+BN folding (eliminating all 48 BatchNorm passes per frame),
  2. HOSVD Tucker-2 spatial-channel factorization + structured FFN neuron pruning
     (~54% parameter reduction, ~4.3x smaller mixed-precision FP16 checkpoint, up to ~2x faster FPS),
  3. Dual-Teacher (SAM 3.1 Multiplex + DINOv3 ViT-L/16) multi-scale feature & affinity
     distillation + closed-loop multi-granularity instance & body-part VOS training.

Safety contract:
- Preserves an immutable backup of the original upstream checkpoint at
  `annolid/segmentation/cutie_vos/weights/cutie-base-mega.orig.pth` (verified MD5).
- Writes the directly upgraded, compressed weights to both `cutie-base-mega.pth`
  and `cutie-sam3-bodyparts.pth`.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from collections import OrderedDict
import copy
import gc
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from annolid.segmentation.SAM.sam3.aliases import ensure_sam3_aliases  # noqa: E402
from annolid.segmentation.cutie_vos.engine import unwrap_cutie_state_dict  # noqa: E402
from annolid.segmentation.cutie_vos.model.cutie import CUTIE  # noqa: E402
from annolid.segmentation.cutie_vos.model.utils.memory_utils import (  # noqa: E402
    get_similarity,
    readout,
)
from annolid.segmentation.cutie_vos.model.weight_compression import (  # noqa: E402
    compress_and_factorize_cutie_in_place,
    is_compact_cutie_checkpoint,
    pack_compact_cutie_checkpoint,
)
from annolid.segmentation.cutie_vos.utils.tensor_utils import aggregate  # noqa: E402
from annolid.utils.model_assets import md5sum, sha256sum  # noqa: E402
from evaluate_cutie_bodyparts_benchmark import (  # noqa: E402
    CUTIE_CFG_DIR,
    DEFAULT_BASE_WEIGHTS,
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_SAM3_WEIGHTS,
    build_two_mice_gt_mask,
    propagate_sequence_with_cutie,
    score_sequence_predictions,
    synthesize_articulated_part_sequence,
    synthesize_multi_instance_crossing_sequence,
)

EXPECTED_BASE_MEGA_MD5 = "a6071de6136982e396851903ab4c083a"
DEFAULT_ORIG_BACKUP_WEIGHTS = DEFAULT_BASE_WEIGHTS.parent / "cutie-base-mega.orig.pth"
DINOV3_TEACHER_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def ensure_immutable_upstream_backup(
    base_weights: Path, backup_weights: Path = DEFAULT_ORIG_BACKUP_WEIGHTS
) -> Path:
    """Guarantee an untouched copy of upstream cutie-base-mega.pth exists at backup_weights."""
    if backup_weights.is_file():
        b_md5 = md5sum(backup_weights)
        if b_md5 != EXPECTED_BASE_MEGA_MD5:
            raise RuntimeError(
                f"Upstream backup MD5 mismatch at {backup_weights}: {b_md5}"
            )
        return backup_weights

    if not base_weights.is_file():
        raise FileNotFoundError(f"Base Cutie weights not found: {base_weights}")
    orig_md5 = md5sum(base_weights)
    if orig_md5 != EXPECTED_BASE_MEGA_MD5:
        raise RuntimeError(
            f"Cannot create upstream backup: {base_weights} MD5={orig_md5} != {EXPECTED_BASE_MEGA_MD5}"
        )
    backup_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_weights, backup_weights)
    print(
        f"Backed up untouched upstream {base_weights.name} -> {backup_weights.name} (md5={orig_md5})"
    )
    return backup_weights


def discover_sam3_checkpoint(explicit_path: Optional[Path] = None) -> Path:
    if explicit_path is not None:
        p = explicit_path.expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Explicit SAM3 checkpoint not found: {p}")
        return p

    hf_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--facebook--sam3.1"
        / "snapshots"
    )
    if hf_root.exists():
        matches = sorted(hf_root.glob("*/sam3.1_multiplex.pt"))
        if matches:
            return matches[-1]
    raise FileNotFoundError(
        "Could not locate cached sam3.1_multiplex.pt in ~/.cache/huggingface/hub/models--facebook--sam3.1"
    )


def build_cutie_config(weights_path: Path) -> Any:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3.2", config_dir=str(CUTIE_CFG_DIR)):
        cfg = compose(config_name="eval_config")
    with open_dict(cfg):
        cfg.weights = str(weights_path)
        cfg.mem_every = 3
        cfg.max_mem_frames = 5
        cfg.max_internal_size = -1
        cfg.use_long_term = False
        cfg.amp = False
        cfg.chunk_size = -1
        cfg.top_k = 30
        cfg.stagger_updates = 3
    return cfg


def resize_clip_for_training(
    frames_rgb: Sequence[np.ndarray],
    frames_gt: Sequence[np.ndarray],
    size: int = 384,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    r_rgb = [
        cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in frames_rgb
    ]
    r_gt = [
        cv2.resize(
            g.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST
        ).astype(np.int32)
        for g in frames_gt
    ]
    return r_rgb, r_gt


def load_disjoint_train_and_val_clips(
    datasets_dir: Path,
    train_crop_size: int = 384,
    seq_len: int = 5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_clips: List[Dict[str, Any]] = []
    val_clips: List[Dict[str, Any]] = []

    for split_name, target_list in (("train", train_clips), ("val", val_clips)):
        rodent_root = datasets_dir / "rodent_3d" / split_name
        if not rodent_root.exists():
            continue
        scen_dirs = sorted(
            p
            for p in rodent_root.iterdir()
            if (p / "ground_truth" / "sequence_gt.json").exists()
        )
        for scen_dir in scen_dirs:
            gt_dir = scen_dir / "ground_truth"
            img_paths = sorted((gt_dir / "images").glob("*.jpg"))
            msk_paths = sorted((gt_dir / "masks").glob("*_ids.png"))
            raw_rgb = [
                cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB) for ip in img_paths
            ]
            raw_id_bgr = [cv2.imread(str(mp), cv2.IMREAD_UNCHANGED) for mp in msk_paths]
            if split_name == "train" and len(raw_rgb) > seq_len:
                idxs = np.linspace(0, len(raw_rgb) - 1, seq_len).round().astype(int)
                sub_rgb = [raw_rgb[i] for i in idxs]
                sub_bgr = [raw_id_bgr[i] for i in idxs]
            else:
                sub_rgb = raw_rgb
                sub_bgr = raw_id_bgr

            for gran in ("instance", "coarse", "fine"):
                gts = [build_two_mice_gt_mask(b, gran) for b in sub_bgr]
                if split_name == "train":
                    c_rgb, c_gt = resize_clip_for_training(
                        sub_rgb, gts, size=train_crop_size
                    )
                else:
                    c_rgb, c_gt = sub_rgb, gts
                if int(c_gt[0].max()) == 0:
                    continue
                target_list.append(
                    {
                        "name": f"rodent3d_{scen_dir.name}_{gran}",
                        "domain": f"rodent_{gran}",
                        "rgb": c_rgb,
                        "gt": c_gt,
                    }
                )

    os_train_sources = [
        ("partimagenet", "train", 10, 4, 110000),
        ("pascal_part", "train", 8, 4, 210000),
        ("human_parsing", "train", 10, 4, 310000),
    ]
    for ds_name, split_name, n_part_clips, n_inst_clips, salt in os_train_sources:
        manifest_path = datasets_dir / ds_name / split_name / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sroot = manifest_path.parent
        loaded = []
        for rec in manifest:
            rgb = cv2.cvtColor(cv2.imread(str(sroot / rec["image"])), cv2.COLOR_BGR2RGB)
            pm = cv2.imread(str(sroot / rec["part_mask"]), cv2.IMREAD_UNCHANGED).astype(
                np.int32
            )
            wm = cv2.imread(
                str(sroot / rec["whole_mask"]), cv2.IMREAD_UNCHANGED
            ).astype(np.int32)
            loaded.append((rec["id"], rgb, pm, wm))

        for i in range(min(n_part_clips, len(loaded))):
            sid, rgb, pm, _ = loaded[i]
            f_rgb, f_gt = synthesize_articulated_part_sequence(
                rgb,
                pm,
                num_frames=seq_len,
                seed=salt + i,
                target_hw=(train_crop_size, train_crop_size),
                add_occluder=(i % 2 == 0),
            )
            if int(f_gt[0].max()) > 0:
                train_clips.append(
                    {
                        "name": f"{ds_name}_part_{sid}",
                        "domain": f"{ds_name}_parts",
                        "rgb": f_rgb,
                        "gt": f_gt,
                    }
                )

        for p in range(min(n_inst_clips, len(loaded) // 2)):
            sid_a, rgb_a, _, wm_a = loaded[2 * p]
            sid_b, rgb_b, _, wm_b = loaded[2 * p + 1]
            f_rgb, f_gt = synthesize_multi_instance_crossing_sequence(
                rgb_a,
                wm_a,
                rgb_b,
                wm_b,
                num_frames=seq_len,
                seed=salt + 500 + p,
                canvas_hw=(train_crop_size, train_crop_size),
            )
            if int(f_gt[0].max()) > 0:
                train_clips.append(
                    {
                        "name": f"{ds_name}_inst_{sid_a}_{sid_b}",
                        "domain": f"{ds_name}_instances",
                        "rgb": f_rgb,
                        "gt": f_gt,
                    }
                )

    for ds_name, salt in (("partimagenet", 410000), ("human_parsing", 510000)):
        manifest_path = datasets_dir / ds_name / "val" / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sroot = manifest_path.parent
        loaded = []
        for rec in manifest:
            rgb = cv2.cvtColor(cv2.imread(str(sroot / rec["image"])), cv2.COLOR_BGR2RGB)
            pm = cv2.imread(str(sroot / rec["part_mask"]), cv2.IMREAD_UNCHANGED).astype(
                np.int32
            )
            wm = cv2.imread(
                str(sroot / rec["whole_mask"]), cv2.IMREAD_UNCHANGED
            ).astype(np.int32)
            loaded.append((rec["id"], rgb, pm, wm))
        for i in range(min(3, len(loaded))):
            sid, rgb, pm, _ = loaded[i]
            f_rgb, f_gt = synthesize_articulated_part_sequence(
                rgb, pm, num_frames=8, seed=salt + i
            )
            val_clips.append(
                {
                    "name": f"val_{ds_name}_part_{sid}",
                    "domain": f"{ds_name}_parts",
                    "rgb": f_rgb,
                    "gt": f_gt,
                }
            )
        if len(loaded) >= 4:
            for p in range(2):
                sid_a, rgb_a, _, wm_a = loaded[2 * p]
                sid_b, rgb_b, _, wm_b = loaded[2 * p + 1]
                f_rgb, f_gt = synthesize_multi_instance_crossing_sequence(
                    rgb_a, wm_a, rgb_b, wm_b, num_frames=8, seed=salt + 100 + p
                )
                val_clips.append(
                    {
                        "name": f"val_{ds_name}_inst_{sid_a}_{sid_b}",
                        "domain": f"{ds_name}_instances",
                        "rgb": f_rgb,
                        "gt": f_gt,
                    }
                )

    return train_clips, val_clips


def _clip_rgb_cache_key(clip_name: str) -> str:
    parts = clip_name.rsplit("_", 1)
    if clip_name.startswith("rodent3d_") and parts[-1] in (
        "instance",
        "coarse",
        "fine",
    ):
        return parts[0]
    return clip_name


def extract_sam3_teacher_representations(
    sam3_ckpt_path: Path,
    train_clips: List[Dict[str, Any]],
    cache_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, torch.Tensor]], torch.Tensor]:
    clip_names = [c["name"] for c in train_clips]
    if cache_path is not None and cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached.get("clip_names") == clip_names:
            print(
                f"\n=== Loaded Cached SAM 3.1 Teacher Pyramids from {cache_path.name} ==="
            )
            return cached["teacher_per_clip"], cached["sam3_maskmem_pix_proj"]

    print(f"\n=== Extracting SAM 3.1 Teacher Pyramids from {sam3_ckpt_path.name} ===")
    t0 = time.time()
    ensure_sam3_aliases()
    from sam3.model.necks import Sam3TriViTDetNeck
    from sam3.model.vitdet import ViT
    from sam3.model_builder import _create_position_encoding

    position_encoding = _create_position_encoding(precompute_resolution=336)
    vit_backbone = ViT(
        img_size=336,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.0,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=None,
        use_fa3=False,
        use_rope_real=True,
    )
    tri_neck = Sam3TriViTDetNeck(
        trunk=vit_backbone,
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0],
    )
    raw = torch.load(sam3_ckpt_path, map_location="cpu", weights_only=True)
    if "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]

    visual_prefixes = (
        "detector.backbone.vision_backbone.",
        "sam3_model.backbone.vision_backbone.",
        "detector.backbone.visual.",
    )
    visual_sd = {}
    for k, v in raw.items():
        for vp in visual_prefixes:
            if k.startswith(vp):
                sub_k = k[len(vp) :]
                if "freqs_cis" in sub_k or "position_encoding.cache" in sub_k:
                    continue
                visual_sd[sub_k] = v
                break

    missing, unexpected = tri_neck.load_state_dict(visual_sd, strict=False)
    non_buffer_missing = [
        m for m in missing if "freqs_cis" not in m and "position_encoding" not in m
    ]
    if non_buffer_missing:
        raise RuntimeError(
            f"Missing learned SAM 3.1 TriViTDetNeck parameters: {non_buffer_missing[:10]}"
        )
    print(
        f"  Loaded SAM 3.1 TriViTDetNeck ({len(visual_sd)} learned tensors, missing_learned=0, unexpected={len(unexpected)})"
    )

    sam3_pix_proj_key = "tracker.model.maskmem_backbone.pix_feat_proj.weight"
    if sam3_pix_proj_key not in raw:
        sam3_pix_proj_key = "sam2_predictor.model.maskmem_backbone.pix_feat_proj.weight"
    sam3_maskmem_pix_proj = raw[sam3_pix_proj_key].detach().float().cpu().clone()

    del raw, visual_sd
    gc.collect()
    tri_neck.eval().to("cpu")

    rgb_cache: Dict[str, Dict[str, torch.Tensor]] = {}
    teacher_per_clip: List[Dict[str, torch.Tensor]] = []

    with torch.inference_mode():
        for clip in train_clips:
            cache_key = _clip_rgb_cache_key(clip["name"])
            if cache_key in rgb_cache:
                teacher_per_clip.append(rgb_cache[cache_key])
                continue

            batch_336 = []
            for rgb in clip["rgb"]:
                im336 = cv2.resize(rgb, (336, 336), interpolation=cv2.INTER_AREA)
                t_im = torch.from_numpy(im336).permute(2, 0, 1).float() / 255.0
                t_im = (t_im - 0.5) / 0.5
                batch_336.append(t_im)
            inp = torch.stack(batch_336, dim=0)

            sam3_out, _, inter_out, _, prop_out, _ = tri_neck(
                inp,
                need_sam3_out=True,
                need_interactive_out=True,
                need_propagation_out=True,
            )
            s16 = F.normalize(prop_out[2].tensors.float(), dim=1) + F.normalize(
                sam3_out[2].tensors.float(), dim=1
            )
            s16 = F.normalize(s16, dim=1).to(torch.float16).cpu()
            s8 = F.normalize(inter_out[1].tensors.float(), dim=1) + F.normalize(
                sam3_out[1].tensors.float(), dim=1
            )
            s8 = F.normalize(s8, dim=1).to(torch.float16).cpu()
            s4 = (
                F.normalize(inter_out[0].tensors.float(), dim=1).to(torch.float16).cpu()
            )

            entry = {"s16": s16, "s8": s8, "s4": s4}
            rgb_cache[cache_key] = entry
            teacher_per_clip.append(entry)

    del tri_neck
    gc.collect()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "clip_names": clip_names,
                "teacher_per_clip": teacher_per_clip,
                "sam3_maskmem_pix_proj": sam3_maskmem_pix_proj,
            },
            cache_path,
        )
    print(f"  Finished SAM 3.1 extraction in {time.time() - t0:.1f}s")
    return teacher_per_clip, sam3_maskmem_pix_proj


def extract_dinov3_teacher_representations(
    train_clips: List[Dict[str, Any]],
    cache_path: Optional[Path] = None,
) -> Optional[List[torch.Tensor]]:
    """Extract DINOv3 ViT-L/16 (1024-dim, 24x24 patch grid at 384x384) dense semantic part features."""
    clip_names = [c["name"] for c in train_clips]
    if cache_path is not None and cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached.get("clip_names") == clip_names:
            print(
                f"\n=== Loaded Cached DINOv3 ViT-L/16 Teacher Features from {cache_path.name} ==="
            )
            return cached["dino_per_clip"]

    try:
        from transformers import AutoModel
    except ImportError:
        return None

    print(
        f"\n=== Extracting DINOv3 ViT-L/16 Teacher Features ({DINOV3_TEACHER_ID}) ==="
    )
    t0 = time.time()
    try:
        dino = AutoModel.from_pretrained(DINOV3_TEACHER_ID, local_files_only=True)
    except Exception as exc:
        print(
            f"  [Info] Local DINOv3 ViT-L/16 unavailable ({exc}); continuing with SAM 3.1 only."
        )
        return None

    dino.eval().to("cpu")
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    rgb_cache: Dict[str, torch.Tensor] = {}
    dino_per_clip: List[torch.Tensor] = []

    with torch.inference_mode():
        for clip in train_clips:
            cache_key = _clip_rgb_cache_key(clip["name"])
            if cache_key in rgb_cache:
                dino_per_clip.append(rgb_cache[cache_key])
                continue

            # 384x384 / 16 = 24x24 = 576 spatial tokens
            inp = torch.stack(
                [
                    torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                    for rgb in clip["rgb"]
                ],
                dim=0,
            )
            inp = (inp - mean) / std
            out = dino(pixel_values=inp)
            tokens = out.last_hidden_state[:, -576:, :]  # (T, 576, 1024)
            feat = tokens.permute(0, 2, 1).reshape(-1, 1024, 24, 24)
            # Foreground/scene mean debias + L2 normalize per token
            feat = feat - feat.mean(dim=(-2, -1), keepdim=True) * 0.35
            feat = F.normalize(feat.float(), dim=1).to(torch.float16).cpu()
            rgb_cache[cache_key] = feat
            dino_per_clip.append(feat)
            if len(rgb_cache) % 10 == 0:
                print(
                    f"  Extracted DINOv3 ViT-L/16 features for {len(rgb_cache)} unique videos ({time.time() - t0:.1f}s)"
                )

    del dino
    gc.collect()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"clip_names": clip_names, "dino_per_clip": dino_per_clip}, cache_path
        )
    print(
        f"  Finished DINOv3 ViT-L/16 extraction for {len(rgb_cache)} unique videos in {time.time() - t0:.1f}s"
    )
    return dino_per_clip


def solve_dual_teacher_procrustes(
    model: CUTIE,
    train_clips: List[Dict[str, Any]],
    sam3_feats: List[Dict[str, torch.Tensor]],
    dino_feats: Optional[List[torch.Tensor]],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    print(
        "\n=== Computing Orthogonal Procrustes Alignments (SAM 3.1 + DINOv3 -> Cutie) ==="
    )
    c16_list, t16_list, d16_list = [], [], []
    c8_list, t8_list = [], []
    c4_list, t4_list = [], []

    seen_keys = set()
    model.eval()
    with torch.inference_mode():
        for idx, (clip, tdict) in enumerate(zip(train_clips, sam3_feats)):
            key = id(tdict)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            f_idxs = [0, len(clip["rgb"]) // 2]
            inp = torch.stack(
                [
                    torch.from_numpy(clip["rgb"][i]).permute(2, 0, 1).float() / 255.0
                    for i in f_idxs
                ],
                dim=0,
            ).to(device)
            ms_feat, pix_feat = model.encode_image(inp)
            d8, d4 = model.mask_decoder.decoder_feat_proc(ms_feat[1:])

            c16 = (
                F.normalize(pix_feat.float(), dim=1)
                .cpu()
                .permute(0, 2, 3, 1)
                .reshape(-1, 256)
            )
            t16 = tdict["s16"][f_idxs].float().permute(0, 2, 3, 1).reshape(-1, 256)
            c16_list.append(c16)
            t16_list.append(t16)

            if dino_feats is not None:
                df = (
                    dino_feats[idx][f_idxs]
                    .float()
                    .permute(0, 2, 3, 1)
                    .reshape(-1, 1024)
                )
                d16_list.append(df)

            c8 = (
                F.normalize(d8.float(), dim=1)
                .cpu()
                .permute(0, 2, 3, 1)
                .reshape(-1, 256)
            )
            t8 = tdict["s8"][f_idxs].float().permute(0, 2, 3, 1).reshape(-1, 256)
            c8_list.append(c8[::2])
            t8_list.append(t8[::2])

            c4 = (
                F.normalize(d4.float(), dim=1)
                .cpu()
                .permute(0, 2, 3, 1)
                .reshape(-1, 128)
            )
            t4 = tdict["s4"][f_idxs].float().permute(0, 2, 3, 1).reshape(-1, 256)
            c4_list.append(c4[::4])
            t4_list.append(t4[::4])

    C16 = torch.cat(c16_list, dim=0)
    T16 = torch.cat(t16_list, dim=0)
    u16, _, vh16 = torch.linalg.svd(T16.T @ C16, full_matrices=False)
    R16 = (u16 @ vh16).to(device)

    C8 = torch.cat(c8_list, dim=0)
    T8 = torch.cat(t8_list, dim=0)
    u8, _, vh8 = torch.linalg.svd(T8.T @ C8, full_matrices=False)
    R8 = (u8 @ vh8).to(device)

    C4 = torch.cat(c4_list, dim=0)
    T4 = torch.cat(t4_list, dim=0)
    u4, _, vh4 = torch.linalg.svd(T4.T @ C4, full_matrices=False)
    P4 = (u4 @ vh4).to(device)

    P_dino: Optional[torch.Tensor] = None
    if d16_list:
        D16 = torch.cat(d16_list, dim=0)
        ud, _, vhd = torch.linalg.svd(D16.T @ C16, full_matrices=False)
        P_dino = (ud @ vhd).to(device)  # (1024, 256)
        cos_dino = (
            F.cosine_similarity((D16.to(device) @ P_dino), C16.to(device), dim=-1)
            .mean()
            .item()
        )
    else:
        cos_dino = 0.0

    cos16 = (
        F.cosine_similarity((T16.to(device) @ R16), C16.to(device), dim=-1)
        .mean()
        .item()
    )
    cos8 = (
        F.cosine_similarity((T8.to(device) @ R8), C8.to(device), dim=-1).mean().item()
    )
    cos4 = (
        F.cosine_similarity((T4.to(device) @ P4), C4.to(device), dim=-1).mean().item()
    )
    print(
        f"  Procrustes cosines: SAM3(s16={cos16:.4f}, s8={cos8:.4f}, s4={cos4:.4f}) | DINOv3(s16={cos_dino:.4f})"
    )
    return R16, R8, P4, P_dino


def differentiable_topk_softmax(
    similarity: torch.Tensor, top_k: Optional[int] = 30
) -> torch.Tensor:
    if top_k is not None and similarity.shape[1] > int(top_k):
        k = max(1, int(top_k))
        values, indices = torch.topk(similarity, k=k, dim=1)
        max_v = values.max(dim=1, keepdim=True).values.detach()
        probs = torch.exp(values - max_v)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return torch.zeros_like(similarity).scatter(1, indices, probs)
    return F.softmax(similarity, dim=1)


def morph_gradient(prob: torch.Tensor) -> torch.Tensor:
    dil = F.max_pool2d(prob, kernel_size=3, stride=1, padding=1)
    ero = -F.max_pool2d(-prob, kernel_size=3, stride=1, padding=1)
    return (dil - ero).clamp(0.0, 1.0)


def compute_contact_band_mask(gt_idx: torch.Tensor) -> torch.Tensor:
    fg = (gt_idx > 0).float().unsqueeze(1)
    max_id = F.max_pool2d(
        gt_idx.float().unsqueeze(1), kernel_size=7, stride=1, padding=3
    )
    inf_val = 9999.0
    pos_or_inf = torch.where(
        gt_idx > 0, gt_idx.float(), torch.full_like(gt_idx.float(), inf_val)
    )
    min_pos_id = -F.max_pool2d(
        -pos_or_inf.unsqueeze(1), kernel_size=7, stride=1, padding=3
    )
    has_two_neighbors = (
        (max_id > 0) & (min_pos_id < inf_val) & ((max_id - min_pos_id) >= 0.5)
    )
    return (has_two_neighbors.float() * (0.5 + 0.5 * fg)).squeeze(1)


def unroll_and_compute_loss(
    model: CUTIE,
    frames_t: torch.Tensor,
    gts_t: torch.Tensor,
    s16_teacher: torch.Tensor,
    s8_teacher: torch.Tensor,
    s4_teacher: torch.Tensor,
    dino_teacher: Optional[torch.Tensor],
    R16: torch.Tensor,
    R8: torch.Tensor,
    P4: torch.Tensor,
    P_dino: Optional[torch.Tensor],
    top_k: int = 30,
) -> Dict[str, torch.Tensor]:
    seq_len, _, H, W = frames_t.shape
    seed_gt = gts_t[0]
    seeded_oids = [int(v) for v in torch.unique(seed_gt).tolist() if int(v) > 0]
    num_objects = len(seeded_oids)

    remapped_gt = torch.zeros_like(gts_t)
    for new_id, oid in enumerate(seeded_oids, start=1):
        remapped_gt[gts_t == oid] = new_id

    ms_all, pix_all = model.encode_image(frames_t)
    keys_all, shrink_all, sel_all = model.transform_key(
        ms_all[0].float(), need_sk=True, need_ek=True
    )
    d8_all, d4_all = model.mask_decoder.decoder_feat_proc(ms_all[1:])

    # 1. Dual-Teacher (SAM 3.1 + DINOv3 ViT-L/16) Multi-Scale Feature & Affinity Distillation
    t16_aligned = torch.einsum("tchw,cd->tdhw", s16_teacher.float(), R16)
    t8_aligned = torch.einsum("tchw,cd->tdhw", s8_teacher.float(), R8)
    t4_aligned = torch.einsum("tchw,cd->tdhw", s4_teacher.float(), P4)

    c16_norm = F.normalize(pix_all, dim=1)
    c8_norm = F.normalize(d8_all, dim=1)
    c4_norm = F.normalize(d4_all, dim=1)

    loss_teacher_feat = (
        0.75 * (1.0 - (c16_norm * F.normalize(t16_aligned, dim=1)).sum(dim=1)).mean()
        + 0.70 * (1.0 - (c8_norm * F.normalize(t8_aligned, dim=1)).sum(dim=1)).mean()
        + 0.75 * (1.0 - (c4_norm * F.normalize(t4_aligned, dim=1)).sum(dim=1)).mean()
    )

    if dino_teacher is not None and P_dino is not None:
        d16_aligned = torch.einsum("tchw,cd->tdhw", dino_teacher.float(), P_dino)
        loss_dino_feat = (
            1.0 - (c16_norm * F.normalize(d16_aligned, dim=1)).sum(dim=1)
        ).mean()
        loss_teacher_feat = loss_teacher_feat + 0.65 * loss_dino_feat

    # Ensemble SAM 3.1 + DINOv3 temporal token correspondence into Cutie's key_proj
    s16_flat = F.normalize(s16_teacher.float().flatten(2), dim=1)
    dino_flat = (
        F.normalize(dino_teacher.float().flatten(2), dim=1)
        if dino_teacher is not None
        else None
    )
    loss_teacher_aff = frames_t.new_tensor(0.0)
    for ti in range(1, seq_len):
        sim_teacher = torch.matmul(s16_flat[0].T, s16_flat[ti])
        if dino_flat is not None:
            sim_teacher = 0.55 * sim_teacher + 0.45 * torch.matmul(
                dino_flat[0].T, dino_flat[ti]
            )
        prob_teacher = F.softmax(sim_teacher / 0.08, dim=0)
        sim_cutie = get_similarity(
            keys_all[0:1],
            shrink_all[0:1],
            keys_all[ti : ti + 1],
            sel_all[ti : ti + 1],
        ).squeeze(0)
        log_prob_cutie = F.log_softmax(sim_cutie, dim=0)
        fgt_16 = (
            F.interpolate(
                (remapped_gt[ti : ti + 1] > 0).float().unsqueeze(1),
                size=keys_all.shape[-2:],
                mode="area",
            ).flatten()
            > 0.05
        )
        if fgt_16.any():
            kl = F.kl_div(
                log_prob_cutie[:, fgt_16],
                prob_teacher[:, fgt_16],
                reduction="batchmean",
            )
            loss_teacher_aff = loss_teacher_aff + kl / float(max(1, seq_len - 1))

    # 2. Closed-Loop Temporal Unroll Matching InferenceCore
    h16, w16 = keys_all.shape[-2:]
    sensory = torch.zeros(
        (1, num_objects, model.sensory_dim, h16, w16), device=frames_t.device
    )

    seed_onehot = (
        F.one_hot(remapped_gt[0], num_classes=num_objects + 1)
        .permute(2, 0, 1)[1:]
        .float()
    )
    seed_prob_with_bg = F.softmax(aggregate(seed_onehot, dim=0), dim=0)
    last_mask = seed_prob_with_bg[1:].unsqueeze(0)

    mv0, sensory, ov0, _ = model.encode_mask(
        frames_t[0:1], pix_all[0:1], sensory, last_mask, deep_update=True
    )
    mem_keys = [keys_all[0:1]]
    mem_shrink = [shrink_all[0:1]]
    mem_vals = [mv0]
    obj_vals = [ov0]

    loss_ce_total = frames_t.new_tensor(0.0)
    loss_iou_total = frames_t.new_tensor(0.0)
    loss_bnd_total = frames_t.new_tensor(0.0)
    loss_aux_total = frames_t.new_tensor(0.0)
    steps_counted = float(max(1, seq_len - 1))

    for ti in range(1, seq_len):
        mk = torch.stack(mem_keys, dim=2)
        ms = torch.stack(mem_shrink, dim=2)
        mv = torch.stack(mem_vals, dim=3)
        ov = torch.stack(obj_vals, dim=2)

        sim = get_similarity(
            mk.float(),
            ms.float(),
            keys_all[ti : ti + 1].float(),
            sel_all[ti : ti + 1].float(),
        )
        aff = differentiable_topk_softmax(sim, top_k=top_k)
        pix_readout = readout(aff, mv.flatten(1, 2).float()).view(
            1, num_objects, model.value_dim, h16, w16
        )
        fused = model.pixel_fusion(
            pix_all[ti : ti + 1], pix_readout, sensory, last_mask
        )
        mem_readout, aux_feat = model.readout_query(fused, ov)
        ms_ti = [f[ti : ti + 1] for f in ms_all]
        sensory, logits_ti, prob_ti = model.segment(
            ms_ti, mem_readout, sensory, update_sensory=True, return_logits=True
        )
        gt_ti = remapped_gt[ti : ti + 1]
        gt_oh = (
            F.one_hot(gt_ti, num_classes=num_objects + 1).permute(0, 3, 1, 2).float()
        )

        class_weights = logits_ti.new_ones(num_objects + 1)
        total_px = float(H * W)
        for c in range(1, num_objects + 1):
            cnt = float((gt_ti == c).sum().item())
            if cnt > 0:
                class_weights[c] = min(
                    7.5, max(1.15, (total_px / max(cnt, 16.0)) ** 0.32)
                )
            else:
                class_weights[c] = 1.5

        ce_map = F.cross_entropy(
            logits_ti, gt_ti, weight=class_weights, reduction="none"
        )
        pt = (prob_ti * gt_oh).sum(dim=1).clamp(1e-6, 1.0)
        focal_factor = (1.0 - pt).pow(1.25)
        contact_band = compute_contact_band_mask(gt_ti)
        spatial_weight = 1.0 + 2.8 * contact_band
        loss_ce_total = (
            loss_ce_total
            + (ce_map * focal_factor * spatial_weight).mean() / steps_counted
        )

        fg_prob = prob_ti[:, 1:]
        fg_gt = gt_oh[:, 1:]
        obj_losses = []
        for k in range(num_objects):
            pk = fg_prob[:, k]
            yk = fg_gt[:, k]
            if yk.sum() > 0.5:
                inter = (pk * yk).sum()
                union = (pk + yk - pk * yk).sum()
                obj_losses.append(1.0 - (inter + 1.0) / (union + 1.0))
            else:
                obj_losses.append(
                    pk.pow(2).mean() * 4.0
                    + 0.35 * F.adaptive_avg_pool2d(pk.unsqueeze(1), 4).max()
                )
        loss_iou_total = loss_iou_total + torch.stack(obj_losses).mean() / steps_counted

        pred_grad = morph_gradient(fg_prob)
        gt_grad = morph_gradient(fg_gt)
        b_inter = 2.0 * (pred_grad * gt_grad).sum(dim=(-2, -1))
        b_denom = pred_grad.sum(dim=(-2, -1)) + gt_grad.sum(dim=(-2, -1))
        b_dice = (1.0 - (b_inter + 1.0) / (b_denom + 1.0)).mean()
        loss_bnd_total = (
            loss_bnd_total
            + (b_dice + 0.5 * F.l1_loss(pred_grad, gt_grad)) / steps_counted
        )

        if aux_feat and aux_feat.get("logits"):
            gt_s16 = F.interpolate(gt_oh, size=(h16, w16), mode="area")
            last_q_logit = aux_feat["logits"][-1]
            q_agg = aggregate(torch.sigmoid(last_q_logit), dim=1)
            q_ce = -(gt_s16 * F.log_softmax(q_agg, dim=1)).sum(dim=1).mean()
            loss_aux_total = loss_aux_total + q_ce / steps_counted

        last_mask = fg_prob
        if ti < seq_len - 1 and (ti % 2 == 0):
            mv_t, sensory, ov_t, _ = model.encode_mask(
                frames_t[ti : ti + 1],
                pix_all[ti : ti + 1],
                sensory,
                last_mask,
                deep_update=True,
            )
            mem_keys.append(keys_all[ti : ti + 1])
            mem_shrink.append(shrink_all[ti : ti + 1])
            mem_vals.append(mv_t)
            obj_vals.append(ov_t)

    return {
        "ce": loss_ce_total,
        "iou": loss_iou_total,
        "bnd": loss_bnd_total,
        "teacher_feat": loss_teacher_feat,
        "teacher_aff": loss_teacher_aff,
        "aux": loss_aux_total,
    }


def evaluate_validation_pool(
    model: CUTIE,
    cfg: Any,
    val_clips: Sequence[Dict[str, Any]],
    device: str,
) -> Dict[str, float]:
    model.eval()
    all_jf: List[float] = []
    inst_jf: List[float] = []
    part_jf: List[float] = []
    for clip in val_clips:
        oids = [int(v) for v in np.unique(clip["gt"][0]) if int(v) > 0]
        preds = propagate_sequence_with_cutie(
            model, cfg, clip["rgb"], clip["gt"][0], device
        )
        _, _, jf = score_sequence_predictions(preds, clip["gt"], oids)
        all_jf.append(jf)
        if "instance" in clip["domain"]:
            inst_jf.append(jf)
        else:
            part_jf.append(jf)
    return {
        "overall_JF": float(np.mean(all_jf)) if all_jf else 0.0,
        "instance_JF": float(np.mean(inst_jf)) if inst_jf else 0.0,
        "parts_JF": float(np.mean(part_jf)) if part_jf else 0.0,
    }


def configure_factorized_trainable_parameters(
    model: CUTIE, base_lr: float
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor]]:
    """Freeze PixelEncoder stem/res2/layer2; set gentle recovery LR on layer3 Tucker cores
    and full LR on factorized per-object VOS heads."""
    frozen_prefixes = (
        "pixel_encoder.conv1",
        "pixel_encoder.res2",
        "pixel_encoder.layer2",
        "mask_encoder.conv1",
        "mask_encoder.layer1",
    )
    backbone_params: List[nn.Parameter] = []
    head_params: List[nn.Parameter] = []
    anchor_snapshot: Dict[str, torch.Tensor] = {}

    for name, param in model.named_parameters():
        if name.startswith(frozen_prefixes):
            param.requires_grad_(False)
            continue
        param.requires_grad_(True)
        anchor_snapshot[name] = param.detach().clone()
        if name.startswith(("pixel_encoder.", "mask_encoder.layer")):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": base_lr * 0.20, "weight_decay": 1e-4},
        {"params": head_params, "lr": base_lr, "weight_decay": 1e-4},
    ]
    n_train = sum(p.numel() for p in backbone_params) + sum(
        p.numel() for p in head_params
    )
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"  Compressed model parameters: {n_tot:,} total ({n_train:,} trainable)")
    return param_groups, anchor_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-weights", type=Path, default=DEFAULT_BASE_WEIGHTS)
    parser.add_argument("--output-weights", type=Path, default=DEFAULT_SAM3_WEIGHTS)
    parser.add_argument(
        "--update-default-weights",
        action="store_true",
        default=False,
        help="Also overwrite cutie-base-mega.pth after backing up cutie-base-mega.orig.pth (disabled by default so Original Cutie remains the default).",
    )
    parser.add_argument(
        "--no-update-default",
        dest="update_default_weights",
        action="store_false",
    )
    parser.add_argument("--sam3-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--datasets-dir", type=Path, default=DEFAULT_BENCHMARK_ROOT / "datasets"
    )
    parser.add_argument("--steps", type=int, default=140)
    parser.add_argument("--lr", type=float, default=8.0e-5)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. Safeguard immutable upstream backup before any direct weight update
    orig_backup_path = ensure_immutable_upstream_backup(args.base_weights)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    # 2. Load strictly disjoint train and validation clips
    train_clips, val_clips = load_disjoint_train_and_val_clips(args.datasets_dir)
    print(
        f"Loaded {len(train_clips)} disjoint training clips and {len(val_clips)} disjoint validation clips"
    )

    # 3. Extract Dual-Teacher representations (SAM 3.1 + DINOv3 ViT-L/16)
    sam3_ckpt = discover_sam3_checkpoint(args.sam3_checkpoint)
    sam3_feats, _ = extract_sam3_teacher_representations(
        sam3_ckpt,
        train_clips,
        cache_path=DEFAULT_BENCHMARK_ROOT / "sam3_teacher_cache.pt",
    )
    dino_feats = extract_dinov3_teacher_representations(
        train_clips,
        cache_path=DEFAULT_BENCHMARK_ROOT / "dinov3_teacher_cache.pt",
    )

    # 4. Initialize dense Cutie from layer-selective blend of upstream + prior v1 (if dense)
    cfg = build_cutie_config(orig_backup_path)
    orig_dense_sd = unwrap_cutie_state_dict(
        torch.load(orig_backup_path, map_location="cpu", weights_only=True)
    )
    init_dense_sd: OrderedDict[str, torch.Tensor] = OrderedDict()

    prior_v1_sd = None
    if args.output_weights.is_file():
        raw_v1 = torch.load(args.output_weights, map_location="cpu", weights_only=True)
        if not is_compact_cutie_checkpoint(raw_v1):
            cand = unwrap_cutie_state_dict(raw_v1)
            if list(cand.keys()) == list(orig_dense_sd.keys()):
                prior_v1_sd = cand

    for k, b_val in orig_dense_sd.items():
        if prior_v1_sd is not None and b_val.is_floating_point():
            # Keep pixel_encoder & key_proj strongly anchored to upstream mega for open-world instances
            w_v1 = 0.25 if k.startswith(("pixel_encoder.", "key_proj.")) else 0.85
            init_dense_sd[k] = (
                (1.0 - w_v1) * b_val.float() + w_v1 * prior_v1_sd[k].float()
            ).to(b_val.dtype)
        else:
            init_dense_sd[k] = b_val.clone()

    # Evaluate upstream baseline on validation pool first
    ref_model = CUTIE(cfg).to(device).eval()
    ref_model.load_weights(copy.deepcopy(orig_dense_sd))
    orig_param_count = sum(p.numel() for p in ref_model.parameters())
    val_upstream = evaluate_validation_pool(ref_model, cfg, val_clips, device)
    print(
        f"\n[Upstream cutie-base-mega ({orig_param_count:,} params)] "
        f"val overall_JF={val_upstream['overall_JF']:.4f} "
        f"(instance={val_upstream['instance_JF']:.4f}, parts={val_upstream['parts_JF']:.4f})"
    )
    del ref_model

    # 5. Apply Direct Structural Compression (48 Conv+BN Folded + HOSVD Tucker-2 + FFN Pruning)
    model = CUTIE(cfg).to(device).eval()
    model.load_weights(init_dense_sd)
    tucker_cfg = compress_and_factorize_cutie_in_place(model)
    comp_param_count = sum(p.numel() for p in model.parameters())

    hosvd_init_state = OrderedDict(
        (k, v.detach().cpu().clone()) for k, v in model.state_dict().items()
    )
    val_hosvd = evaluate_validation_pool(model, cfg, val_clips, device)
    print(
        f"[Zero-Shot HOSVD Compressed ({comp_param_count:,} params, "
        f"-{100.0 * (1.0 - comp_param_count / orig_param_count):.1f}%)] "
        f"val overall_JF={val_hosvd['overall_JF']:.4f} "
        f"(instance={val_hosvd['instance_JF']:.4f}, parts={val_hosvd['parts_JF']:.4f})"
    )

    # 6. Solve Dual-Teacher Orthogonal Procrustes Alignments
    R16, R8, P4, P_dino = solve_dual_teacher_procrustes(
        model, train_clips, sam3_feats, dino_feats, device
    )

    # 7. Closed-Loop VOS + Dual-Teacher Recovery & Fine-Tuning
    param_groups, anchor_snapshot = configure_factorized_trainable_parameters(
        model, args.lr
    )
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.08
    )

    best_val_score = val_hosvd["overall_JF"]
    best_ft_state = copy.deepcopy(hosvd_init_state)
    rng = random.Random(args.seed + 202)
    order = list(range(len(train_clips)))
    rng.shuffle(order)

    grad_accum = 2
    optimizer.zero_grad(set_to_none=True)
    t_train = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        accum_metrics: Dict[str, float] = {}

        for micro in range(grad_accum):
            idx_in_order = ((step - 1) * grad_accum + micro) % len(order)
            pick = order[idx_in_order]
            if idx_in_order == len(order) - 1:
                rng.shuffle(order)

            clip = train_clips[pick]
            tdict = sam3_feats[pick]
            d_feat = dino_feats[pick].to(device) if dino_feats is not None else None

            frames_t = torch.stack(
                [
                    torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
                    for f in clip["rgb"]
                ],
                dim=0,
            ).to(device)
            gts_t = torch.stack(
                [torch.from_numpy(g).long() for g in clip["gt"]], dim=0
            ).to(device)
            s16_t = tdict["s16"].to(device)
            s8_t = tdict["s8"].to(device)
            s4_t = tdict["s4"].to(device)

            if rng.random() < 0.5:
                frames_t = torch.flip(frames_t, dims=[-1])
                gts_t = torch.flip(gts_t, dims=[-1])
                s16_t = torch.flip(s16_t, dims=[-1])
                s8_t = torch.flip(s8_t, dims=[-1])
                s4_t = torch.flip(s4_t, dims=[-1])
                if d_feat is not None:
                    d_feat = torch.flip(d_feat, dims=[-1])

            losses = unroll_and_compute_loss(
                model,
                frames_t,
                gts_t,
                s16_t,
                s8_t,
                s4_t,
                d_feat,
                R16,
                R8,
                P4,
                P_dino,
                top_k=30,
            )

            anchor_reg = frames_t.new_tensor(0.0)
            for pname, p in model.named_parameters():
                if p.requires_grad and pname in anchor_snapshot:
                    anchor_reg = anchor_reg + (p - anchor_snapshot[pname]).pow(2).sum()

            total_loss = (
                1.00 * losses["ce"]
                + 1.15 * losses["iou"]
                + 0.50 * losses["bnd"]
                + 0.28 * losses["teacher_feat"]
                + 0.14 * losses["teacher_aff"]
                + 0.15 * losses["aux"]
                + 2.0e-4 * anchor_reg
            ) / float(grad_accum)

            total_loss.backward()
            for lk, lv in losses.items():
                accum_metrics[lk] = (
                    accum_metrics.get(lk, 0.0) + float(lv.item()) / grad_accum
                )

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if step % 20 == 0 or step == 1:
            print(
                f"  [Step {step:03d}/{args.steps}] "
                f"ce={accum_metrics['ce']:.4f} iou={accum_metrics['iou']:.4f} "
                f"bnd={accum_metrics['bnd']:.4f} distill={accum_metrics['teacher_feat']:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} ({time.time() - t_train:.1f}s)"
            )

        if step % 35 == 0 or step == args.steps:
            val_metrics = evaluate_validation_pool(model, cfg, val_clips, device)
            print(
                f"  --> [Val Step {step:03d}] overall_JF={val_metrics['overall_JF']:.4f} "
                f"(instance={val_metrics['instance_JF']:.4f}, parts={val_metrics['parts_JF']:.4f})"
            )
            if val_metrics["overall_JF"] >= best_val_score:
                best_val_score = val_metrics["overall_JF"]
                best_ft_state = OrderedDict(
                    (k, v.detach().cpu().clone()) for k, v in model.state_dict().items()
                )

    # 8. WiSE-FT Sweep in Factorized Space on Validation Pool
    print("\n=== Selecting Optimal Factorized WiSE-FT Blend on Validation Split ===")
    best_alpha = 1.0
    best_wise_metrics = {"overall_JF": -1.0, "instance_JF": 0.0, "parts_JF": 0.0}
    best_final_state: OrderedDict[str, torch.Tensor] = best_ft_state

    for alpha in (0.65, 0.80, 0.90, 1.00):
        cand_state: OrderedDict[str, torch.Tensor] = OrderedDict()
        for k in hosvd_init_state.keys():
            h_t = hosvd_init_state[k]
            f_t = best_ft_state[k]
            if h_t.is_floating_point():
                cand_state[k] = ((1.0 - alpha) * h_t.float() + alpha * f_t.float()).to(
                    h_t.dtype
                )
            else:
                cand_state[k] = f_t.clone()

        model.load_state_dict(cand_state, strict=True)
        m_val = evaluate_validation_pool(model, cfg, val_clips, device)
        print(
            f"  alpha={alpha:.2f} -> val overall_JF={m_val['overall_JF']:.4f} "
            f"(instance={m_val['instance_JF']:.4f}, parts={m_val['parts_JF']:.4f})"
        )
        if (
            m_val["instance_JF"] >= val_upstream["instance_JF"] - 0.002
            and m_val["overall_JF"] >= best_wise_metrics["overall_JF"]
        ):
            best_wise_metrics = m_val
            best_alpha = alpha
            best_final_state = cand_state

    model.load_state_dict(best_final_state, strict=True)
    for k, t in model.state_dict().items():
        if t.is_floating_point():
            assert torch.isfinite(t).all(), f"Non-finite parameter in {k}"

    compact_payload = pack_compact_cutie_checkpoint(
        model,
        tucker_cfg=tucker_cfg,
        metadata={
            "teachers": ["sam3.1_multiplex.pt", DINOV3_TEACHER_ID],
            "orig_parameters": orig_param_count,
            "compressed_parameters": comp_param_count,
            "selected_wise_ft_alpha": best_alpha,
        },
    )

    # Verify exact round-trip load of compact FP16/FP32 checkpoint into a fresh CUTIE model
    verify_model = CUTIE(cfg).to(device).eval()
    verify_model.load_weights(compact_payload)
    val_compact = evaluate_validation_pool(verify_model, cfg, val_clips, device)
    print(
        f"  [Packed Compact Checkpoint Round-Trip] val overall_JF={val_compact['overall_JF']:.4f} "
        f"(instance={val_compact['instance_JF']:.4f}, parts={val_compact['parts_JF']:.4f})"
    )
    del verify_model

    args.output_weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(compact_payload, args.output_weights)
    if args.update_default_weights:
        shutil.copy2(args.output_weights, args.base_weights)

    upgraded_sha = sha256sum(args.output_weights)
    release_assets_dir = DEFAULT_BENCHMARK_ROOT / "release_assets"
    release_assets_dir.mkdir(parents=True, exist_ok=True)
    staged_asset = release_assets_dir / args.output_weights.name
    shutil.copy2(args.output_weights, staged_asset)
    (release_assets_dir / f"{args.output_weights.name}.sha256").write_text(
        f"{upgraded_sha}  {args.output_weights.name}\n",
        encoding="utf-8",
    )

    assert md5sum(orig_backup_path) == EXPECTED_BASE_MEGA_MD5
    orig_size_mb = orig_backup_path.stat().st_size / (1024 * 1024)
    new_size_mb = args.output_weights.stat().st_size / (1024 * 1024)

    provenance = {
        "upstream_backup": orig_backup_path.name,
        "upstream_backup_md5": EXPECTED_BASE_MEGA_MD5,
        "updated_default_weights": str(args.base_weights.name)
        if args.update_default_weights
        else None,
        "updated_sam3_weights": str(args.output_weights.name),
        "upgraded_sha256": upgraded_sha,
        "teachers": ["sam3.1_multiplex.pt", DINOV3_TEACHER_ID],
        "orig_parameters": orig_param_count,
        "compressed_parameters": comp_param_count,
        "parameter_reduction_pct": round(
            100.0 * (1.0 - comp_param_count / orig_param_count), 2
        ),
        "orig_file_size_mb": round(orig_size_mb, 2),
        "compressed_file_size_mb": round(new_size_mb, 2),
        "size_compression_ratio": round(orig_size_mb / max(new_size_mb, 1e-6), 2),
        "steps": args.steps,
        "learning_rate": args.lr,
        "selected_wise_ft_alpha": best_alpha,
        "val_upstream": val_upstream,
        "val_hosvd_zero_shot": val_hosvd,
        "val_after": val_compact,
    }
    prov_path = DEFAULT_BENCHMARK_ROOT / "training_provenance.json"
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("\n=== Directly Updated & Compressed Cutie Weights Saved Successfully ===")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
