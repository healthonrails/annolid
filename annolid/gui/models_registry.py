from dataclasses import dataclass


@dataclass
class ModelConfig:
    display_name: str
    identifier: str
    weight_file: str


# List of model configurations
MODEL_REGISTRY = [
    ModelConfig("SAM_HQ", "sam_hq", "sam_hq.pt"),
    ModelConfig("EfficientVit_SAM", "efficientvit_sam", "efficientvit_sam.pt"),
    ModelConfig("EfficientTAM_s", "efficienttam_s", "efficienttam_s.pt"),
    ModelConfig("EfficientTAM_ti", "efficienttam_ti", "efficienttam_ti.pt"),
    ModelConfig("Cutie", "Cutie", "Cutie.pt"),
    ModelConfig("CoTracker", "CoTracker", "CoTracker.pt"),
    ModelConfig("sam2_hiera_s", "sam2_hiera_s", "sam2_hiera_s.pt"),
    ModelConfig("sam2_hiera_l", "sam2_hiera_l", "sam2_hiera_l.pt"),
    ModelConfig("SAM3", "sam3", "sam3"),
    ModelConfig("YOLO11n", "yolo11n", "yolo11n-seg.pt"),
    ModelConfig("YOLO11x", "yolo11x", "yolo11x-seg.pt"),
    ModelConfig("YOLO11n-pose", "yolo11n-pose", "yolo11n-pose.pt"),
    ModelConfig("YOLO11x-pose", "yolo11x-pose", "yolo11x-pose.pt"),
    ModelConfig("yoloe-11s-seg.pt", "yoloe-11s-seg.pt", "yoloe-11s-seg.pt"),
    ModelConfig("yoloe-11l-seg.pt", "yoloe-11l-seg.pt", "yoloe-11l-seg.pt"),
    ModelConfig("DINOv3 Keypoint Tracker",
                "dinov3_keypoint_tracker", "DINO_KEYPOINT_TRACKER"),
    ModelConfig(
        "Cutie + DINOv3 Keypoint Segmentation",
        "dino_kpseg_tracker",
        "runs/dino_kpseg/train/weights/best.pt",
    ),
    ModelConfig(
        "DINOv3 Keypoint Segmentation",
        "dino_kpseg",
        "runs/dino_kpseg/train/weights/best.pt",
    ),
]


# Registry for patch-similarity (DINO) backbones. These identifiers correspond
# to Hugging Face model IDs and may require gated-access for certain DINOv3
# checkpoints.
PATCH_SIMILARITY_MODELS = [
    ModelConfig("DINOv2 Base (open)", "facebook/dinov2-base", ""),
    ModelConfig("DINOv2 Large (open)", "facebook/dinov2-large", ""),
    ModelConfig("DINOv3 ViT-S/16 (gated)",
                "facebook/dinov3-vits16-pretrain-lvd1689m", ""),
    ModelConfig("DINOv3 ViT-S/16+ (gated)",
                "facebook/dinov3-vits16plus-pretrain-lvd1689m", ""),
    ModelConfig("DINOv3 ViT-L/16 (gated)",
                "facebook/dinov3-vitl16-pretrain-lvd1689m", ""),
    ModelConfig("DINOv3 ViT-H/16+ (gated)",
                "facebook/dinov3-vith16plus-pretrain-lvd1689m", ""),
    ModelConfig("DINOv3 ViT-7B/16 (gated)",
                "facebook/dinov3-vit7b16-pretrain-lvd1689m", ""),
]

PATCH_SIMILARITY_DEFAULT_MODEL = PATCH_SIMILARITY_MODELS[2].identifier
