# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from iopath.common.file_io import g_pathmgr
from sam3.model.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerDecoderLayerv2,
    TransformerEncoderCrossAttention,
)
from sam3.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from sam3.model.geometry_encoders import SequenceGeometryEncoder
from sam3.model.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from sam3.model.memory import (
    CXBlock,
    SimpleFuser,
    SimpleMaskDownSampler,
    SimpleMaskEncoder,
)
from sam3.model.model_misc import (
    DotProductScoring,
    MLP,
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper,
)
from sam3.model.necks import Sam3DualViTDetNeck
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3.model.sam3_image import Sam3Image, Sam3ImageOnVideoMultiGPU
from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor
from sam3.model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity
from sam3.model.sam3_video_predictor import Sam3VideoPredictorMultiGPU
from sam3.model.text_encoder_ve import VETextEncoder
from sam3.model.tokenizer_ve import SimpleTokenizer
from sam3.model.vitdet import ViT
from sam3.model.vl_combiner import SAM3VLBackbone
from sam3.sam.transformer import RoPEAttention
from sam3.utils import select_device


@dataclass
class Sam3ImageConfig:
    """Configuration for building a SAM3 image model."""

    bpe_path: Optional[str] = None
    device: Optional[str] = None
    eval_mode: bool = True
    checkpoint_path: Optional[str] = None
    load_from_HF: bool = True
    enable_segmentation: bool = True
    enable_inst_interactivity: bool = False
    compile: bool = False


@dataclass
class Sam3VideoConfig:
    """Configuration for building a SAM3 video model."""

    checkpoint_path: Optional[str] = None
    load_from_HF: bool = True
    bpe_path: Optional[str] = None
    has_presence_token: bool = True
    geo_encoder_use_img_cross_attn: bool = True
    strict_state_dict_loading: bool = True
    apply_temporal_disambiguation: bool = True
    device: Optional[str] = None
    compile: bool = False
    score_threshold_detection: Optional[float] = None
    new_det_thresh: Optional[float] = None


# Setup TensorFloat-32 for Ampere GPUs if available
def _setup_tf32() -> None:
    """Enable TensorFloat-32 for Ampere GPUs if available."""
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


_setup_tf32()


def _create_position_encoding(precompute_resolution=None, device=None):
    """Create position encoding for visual backbone."""
    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
        device=device,
    )


def _create_vit_backbone(compile_mode=None):
    """Create ViT backbone for visual feature extraction."""
    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
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
        compile_mode=compile_mode,
    )


def _create_vit_neck(position_encoding, vit_backbone, enable_inst_interactivity=False):
    """Create ViT neck for feature pyramid."""
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=enable_inst_interactivity,
    )


def _create_vl_backbone(vit_neck, text_encoder):
    """Create visual-language backbone."""
    return SAM3VLBackbone(visual=vit_neck, text=text_encoder, scalp=1)


def _create_transformer_encoder() -> TransformerEncoderFusion:
    """Create transformer encoder with its layer."""
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
    )

    encoder = TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )
    return encoder


def _create_transformer_decoder(
    device=None, has_presence_token: bool = True
) -> TransformerDecoder:
    """Create transformer decoder with its layer."""
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
        ),
        n_heads=8,
        use_text_cross_attention=True,
    )

    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=1008,
        stride=14,
        use_act_checkpoint=True,
        presence_token=has_presence_token,
        device=device,
    )
    return decoder


def _create_dot_product_scoring():
    """Create dot product scoring module."""
    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)


def _create_segmentation_head(compile_mode=None):
    """Create segmentation head with pixel decoder."""
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=256,
        compile_mode=compile_mode,
    )

    cross_attend_prompt = MultiheadAttention(
        num_heads=8,
        dropout=0,
        embed_dim=256,
    )

    segmentation_head = UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=True,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )
    return segmentation_head


def _create_geometry_encoder(device=None):
    """Create geometry encoder with all its components."""
    # Create position encoding for geometry encoder
    geo_pos_enc = _create_position_encoding(device=device)
    # Create CX block for fuser
    cx_block = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )
    # Create geometry encoder layer
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
    )

    # Create geometry encoder
    input_geometry_encoder = SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
    )
    return input_geometry_encoder


def _create_sam3_model(
    backbone,
    transformer,
    input_geometry_encoder,
    segmentation_head,
    dot_prod_scoring,
    inst_interactive_predictor,
    eval_mode,
):
    """Create the SAM3 image model."""
    common_params = {
        "backbone": backbone,
        "transformer": transformer,
        "input_geometry_encoder": input_geometry_encoder,
        "segmentation_head": segmentation_head,
        "num_feature_levels": 1,
        "o2m_mask_predict": True,
        "dot_prod_scoring": dot_prod_scoring,
        "use_instance_query": False,
        "multimask_output": True,
        "inst_interactive_predictor": inst_interactive_predictor,
    }

    # Training matcher is omitted in the inference-only build.
    common_params["matcher"] = None
    model = Sam3Image(**common_params)

    return model


def _create_tracker_maskmem_backbone():
    """Create the SAM3 Tracker memory encoder."""
    # Position encoding for mask memory backbone
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=64,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008,
    )

    # Mask processing components
    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3, stride=2, padding=1, interpol_size=[1152, 1152]
    )

    cx_block_layer = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )

    fuser = SimpleFuser(layer=cx_block_layer, num_layers=2)

    maskmem_backbone = SimpleMaskEncoder(
        out_dim=64,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )

    return maskmem_backbone


def _create_tracker_transformer():
    """Create the SAM3 Tracker transformer components."""
    # Self attention
    self_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        use_fa3=False,
        use_rope_real=False,
    )

    # Cross attention
    cross_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        kv_in_dim=64,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        rope_k_repeat=True,
        use_fa3=False,
        use_rope_real=False,
    )

    # Encoder layer
    encoder_layer = TransformerDecoderLayerv2(
        cross_attention_first=False,
        activation="relu",
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=self_attention,
        d_model=256,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        cross_attention=cross_attention,
    )

    # Encoder
    encoder = TransformerEncoderCrossAttention(
        remove_cross_attention_layers=[],
        batch_first=True,
        d_model=256,
        frozen=False,
        pos_enc_at_input=True,
        layer=encoder_layer,
        num_layers=4,
        use_act_checkpoint=False,
    )

    # Transformer wrapper
    transformer = TransformerWrapper(
        encoder=encoder,
        decoder=None,
        d_model=256,
    )

    return transformer


def build_tracker(
    apply_temporal_disambiguation: bool, with_backbone: bool = False, compile_mode=None
) -> Sam3TrackerPredictor:
    """
    Build the SAM3 Tracker module for video tracking.

    Returns:
        Sam3TrackerPredictor: Wrapped SAM3 Tracker module
    """

    # Create model components
    maskmem_backbone = _create_tracker_maskmem_backbone()
    transformer = _create_tracker_transformer()
    backbone = None
    if with_backbone:
        vision_backbone = _create_vision_backbone(compile_mode=compile_mode)
        backbone = SAM3VLBackbone(scalp=1, visual=vision_backbone, text=None)
    # Create the Tracker module
    model = Sam3TrackerPredictor(
        image_size=1008,
        num_maskmem=7,
        backbone=backbone,
        backbone_stride=14,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        # SAM parameters
        multimask_output_in_sam=True,
        # Evaluation
        forward_backbone_per_frame_for_eval=True,
        trim_past_non_cond_mem_for_eval=False,
        # Multimask
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        # Additional settings
        always_start_from_first_ann_frame=False,
        # Mask overlap
        non_overlap_masks_for_mem_enc=False,
        non_overlap_masks_for_output=False,
        max_cond_frames_in_attn=4,
        offload_output_to_cpu_for_eval=False,
        # SAM decoder settings
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        clear_non_cond_mem_around_input=True,
        fill_hole_area=0,
        use_memory_selection=apply_temporal_disambiguation,
    )

    return model


def _create_text_encoder(bpe_path: str) -> VETextEncoder:
    """Create SAM3 text encoder."""
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    return VETextEncoder(
        tokenizer=tokenizer,
        d_model=256,
        width=1024,
        heads=16,
        layers=24,
    )


def _create_vision_backbone(
    compile_mode=None, enable_inst_interactivity=True, device=None
) -> Sam3DualViTDetNeck:
    """Create SAM3 visual backbone with ViT and neck."""
    # Position encoding
    position_encoding = _create_position_encoding(
        precompute_resolution=1008, device=device
    )
    # ViT backbone
    vit_backbone: ViT = _create_vit_backbone(compile_mode=compile_mode)
    vit_neck: Sam3DualViTDetNeck = _create_vit_neck(
        position_encoding,
        vit_backbone,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    # Visual neck
    return vit_neck


def _create_sam3_transformer(has_presence_token: bool = True, device=None) -> TransformerWrapper:
    """Create SAM3 transformer encoder and decoder."""
    encoder: TransformerEncoderFusion = _create_transformer_encoder()
    decoder: TransformerDecoder = _create_transformer_decoder(
        device=device, has_presence_token=has_presence_token
    )

    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)


def _load_checkpoint(model, checkpoint_path):
    """Load model checkpoint from file."""
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    sam3_image_ckpt = {
        k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k
    }
    if model.inst_interactive_predictor is not None:
        sam3_image_ckpt.update(
            {
                k.replace("tracker.", "inst_interactive_predictor.model."): v
                for k, v in ckpt.items()
                if "tracker" in k
            }
        )
    missing_keys, _ = model.load_state_dict(sam3_image_ckpt, strict=False)
    if len(missing_keys) > 0:
        print(
            f"loaded {checkpoint_path} and found "
            f"missing and/or unexpected keys:\n{missing_keys=}"
        )


def _setup_device_and_mode(model, device, eval_mode):
    """Setup model device and evaluation mode."""
    resolved_device = (
        device if isinstance(device, torch.device) else select_device(device)
    )
    model = model.to(resolved_device)
    if eval_mode:
        model.eval()
    return model


def _build_video_inference(
    detector,
    tracker,
    *,
    score_thresh: float,
    new_det_thresh: float,
    apply_temporal_disambiguation: bool,
    compile_model: bool,
):
    """
    Build a Sam3VideoInferenceWithInstanceInteractivity with shared defaults,
    only toggling hotstart/reconditioning heuristics as needed.
    """
    common_kwargs = dict(
        detector=detector,
        tracker=tracker,
        score_threshold_detection=score_thresh,
        assoc_iou_thresh=0.1,
        det_nms_thresh=0.1,
        new_det_thresh=new_det_thresh,
        suppress_unmatched_only_within_hotstart=True,
        min_trk_keep_alive=-1,
        max_trk_keep_alive=30,
        init_trk_keep_alive=30,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
        suppress_det_close_to_boundary=False,
        fill_hole_area=16,
        masklet_confirmation_enable=False,
        decrease_trk_keep_alive_for_empty_masklets=False,
        image_size=1008,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        compile_model=compile_model,
    )

    if apply_temporal_disambiguation:
        common_kwargs.update(
            hotstart_delay=15,
            hotstart_unmatch_thresh=8,
            hotstart_dup_thresh=8,
            recondition_every_nth_frame=16,
        )
    else:
        # A version without heuristics for ablation or debugging.
        common_kwargs.update(
            hotstart_delay=0,
            hotstart_unmatch_thresh=0,
            hotstart_dup_thresh=0,
            recondition_every_nth_frame=0,
        )

    return Sam3VideoInferenceWithInstanceInteractivity(**common_kwargs)


def _resolve_image_config(
    config: Optional[Sam3ImageConfig],
    *,
    bpe_path,
    device,
    eval_mode,
    checkpoint_path,
    load_from_HF,
    enable_segmentation,
    enable_inst_interactivity,
    compile,
) -> Sam3ImageConfig:
    """Combine legacy args with an optional config object."""
    if config is not None:
        return config
    return Sam3ImageConfig(
        bpe_path=bpe_path,
        device=device,
        eval_mode=eval_mode,
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_HF,
        enable_segmentation=enable_segmentation,
        enable_inst_interactivity=enable_inst_interactivity,
        compile=compile,
    )


def build_sam3_image_model(
    bpe_path=None,
    device=None,
    eval_mode=True,
    checkpoint_path=None,
    load_from_HF=True,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
    config: Optional[Sam3ImageConfig] = None,
):
    """
    Build SAM3 image model

    Args:
        bpe_path: Path to the BPE tokenizer vocabulary
        device: Device to load the model on ('cuda' or 'cpu')
        eval_mode: Whether to set the model to evaluation mode
        checkpoint_path: Optional path to model checkpoint
        enable_segmentation: Whether to enable segmentation head
        enable_inst_interactivity: Whether to enable instance interactivity (SAM 1 task)
        compile_mode: To enable compilation, set to "default"

    Returns:
        A SAM3 image model
    """
    cfg = _resolve_image_config(
        config,
        bpe_path=bpe_path,
        device=device,
        eval_mode=eval_mode,
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_HF,
        enable_segmentation=enable_segmentation,
        enable_inst_interactivity=enable_inst_interactivity,
        compile=compile,
    )

    device = select_device(cfg.device)
    if cfg.bpe_path is None:
        bpe_path = os.path.join(
            os.path.dirname(
                __file__), "..", "assets", "bpe_simple_vocab_16e6.txt.gz"
        )
    else:
        bpe_path = cfg.bpe_path
    # Create visual components
    compile_mode = "default" if cfg.compile else None
    vision_encoder = _create_vision_backbone(
        compile_mode=compile_mode,
        enable_inst_interactivity=cfg.enable_inst_interactivity,
        device=device,
    )

    # Create text components
    text_encoder = _create_text_encoder(bpe_path)

    # Create visual-language backbone
    backbone = _create_vl_backbone(vision_encoder, text_encoder)

    # Create transformer components
    transformer = _create_sam3_transformer()

    # Create dot product scoring
    dot_prod_scoring = _create_dot_product_scoring()

    # Create segmentation head if enabled
    segmentation_head = (
        _create_segmentation_head(compile_mode=compile_mode)
        if cfg.enable_segmentation
        else None
    )

    # Create geometry encoder
    input_geometry_encoder = _create_geometry_encoder(device=device)
    if cfg.enable_inst_interactivity:
        sam3_pvs_base = build_tracker(apply_temporal_disambiguation=False)
        inst_predictor = SAM3InteractiveImagePredictor(sam3_pvs_base)
    else:
        inst_predictor = None
    # Create the SAM3 model
    model = _create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring,
        inst_predictor,
        cfg.eval_mode,
    )
    if cfg.load_from_HF and cfg.checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf()
    else:
        checkpoint_path = cfg.checkpoint_path
    # Load checkpoint if provided
    if checkpoint_path is not None:
        _load_checkpoint(model, checkpoint_path)

    # Setup device and mode
    model = _setup_device_and_mode(model, device, cfg.eval_mode)

    return model


def download_ckpt_from_hf():
    """Download SAM3 checkpoint, honoring optional env overrides."""
    env_override = os.environ.get("SAM3_CKPT_PATH")
    if env_override:
        candidate = os.path.expanduser(env_override)
        if os.path.exists(candidate):
            return candidate
    SAM3_MODEL_ID = "facebook/sam3"
    SAM3_CKPT_NAME = "sam3.pt"
    SAM3_CFG_NAME = "config.json"
    try:
        _ = hf_hub_download(repo_id=SAM3_MODEL_ID, filename=SAM3_CFG_NAME)
        checkpoint_path = hf_hub_download(
            repo_id=SAM3_MODEL_ID, filename=SAM3_CKPT_NAME
        )
    except Exception as exc:
        raise RuntimeError(
            "Unable to download SAM3 checkpoint from HuggingFace. "
            "Set SAM3_CKPT_PATH to a local checkpoint to skip download."
        ) from exc
    return checkpoint_path


def _resolve_video_config(
    config: Optional[Sam3VideoConfig],
    *,
    checkpoint_path,
    load_from_HF,
    bpe_path,
    has_presence_token,
    geo_encoder_use_img_cross_attn,
    strict_state_dict_loading,
    apply_temporal_disambiguation,
    device,
    compile,
) -> Sam3VideoConfig:
    """Combine legacy args with an optional config object."""
    if config is not None:
        return config
    return Sam3VideoConfig(
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_HF,
        bpe_path=bpe_path,
        has_presence_token=has_presence_token,
        geo_encoder_use_img_cross_attn=geo_encoder_use_img_cross_attn,
        strict_state_dict_loading=strict_state_dict_loading,
        apply_temporal_disambiguation=apply_temporal_disambiguation,
        device=device,
        compile=compile,
    )


def build_sam3_video_model(
    checkpoint_path: Optional[str] = None,
    load_from_HF=True,
    bpe_path: Optional[str] = None,
    has_presence_token: bool = True,
    geo_encoder_use_img_cross_attn: bool = True,
    strict_state_dict_loading: bool = True,
    apply_temporal_disambiguation: bool = True,
    device=None,
    compile=False,
    config: Optional[Sam3VideoConfig] = None,
) -> Sam3VideoInferenceWithInstanceInteractivity:
    """
    Build SAM3 dense tracking model.

    Args:
        checkpoint_path: Optional path to checkpoint file
        bpe_path: Path to the BPE tokenizer file

    Returns:
        Sam3VideoInferenceWithInstanceInteractivity: The instantiated dense tracking model
    """
    cfg = _resolve_video_config(
        config,
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_HF,
        bpe_path=bpe_path,
        has_presence_token=has_presence_token,
        geo_encoder_use_img_cross_attn=geo_encoder_use_img_cross_attn,
        strict_state_dict_loading=strict_state_dict_loading,
        apply_temporal_disambiguation=apply_temporal_disambiguation,
        device=device,
        compile=compile,
    )

    device = select_device(cfg.device)
    if cfg.bpe_path is None:
        bpe_path = os.path.join(
            os.path.dirname(
                __file__), "..", "assets", "bpe_simple_vocab_16e6.txt.gz"
        )
    else:
        bpe_path = cfg.bpe_path

    # Build Tracker module
    tracker = build_tracker(
        apply_temporal_disambiguation=cfg.apply_temporal_disambiguation)

    # Build Detector components
    visual_neck = _create_vision_backbone(device=device)
    text_encoder = _create_text_encoder(bpe_path)
    backbone = SAM3VLBackbone(scalp=1, visual=visual_neck, text=text_encoder)
    transformer = _create_sam3_transformer(
        has_presence_token=cfg.has_presence_token, device=device
    )
    segmentation_head: UniversalSegmentationHead = _create_segmentation_head()
    input_geometry_encoder = _create_geometry_encoder()

    # Create main dot product scoring
    main_dot_prod_scoring = _create_dot_product_scoring()

    # Build Detector module
    detector = Sam3ImageOnVideoMultiGPU(
        num_feature_levels=1,
        backbone=backbone,
        transformer=transformer,
        segmentation_head=segmentation_head,
        semantic_segmentation_head=None,
        input_geometry_encoder=input_geometry_encoder,
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=main_dot_prod_scoring,
        supervise_joint_box_scores=has_presence_token,
    )

    # Build the main SAM3 video model
    # Allow Annolid to relax detection thresholds via environment variables
    # so dense tracking is more robust on challenging videos.
    score_thresh = (
        float(os.environ.get("SAM3_SCORE_THRESH", "0.05"))
        if cfg.score_threshold_detection is None
        else cfg.score_threshold_detection
    )
    new_det_thresh = (
        float(os.environ.get("SAM3_NEW_DET_THRESH", "0.06"))
        if cfg.new_det_thresh is None
        else cfg.new_det_thresh
    )
    model = _build_video_inference(
        detector=detector,
        tracker=tracker,
        score_thresh=score_thresh,
        new_det_thresh=new_det_thresh,
        apply_temporal_disambiguation=cfg.apply_temporal_disambiguation,
        compile_model=cfg.compile,
    )

    # Load checkpoint if provided
    if cfg.load_from_HF and cfg.checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf()
    else:
        checkpoint_path = cfg.checkpoint_path
    if checkpoint_path is not None:
        with g_pathmgr.open(checkpoint_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        missing_keys, unexpected_keys = model.load_state_dict(
            ckpt, strict=cfg.strict_state_dict_loading
        )
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

    model.to(device=device)
    return model


def build_sam3_video_predictor(*model_args, gpus_to_use=None, **model_kwargs):
    return Sam3VideoPredictorMultiGPU(
        *model_args, gpus_to_use=gpus_to_use, **model_kwargs
    )
