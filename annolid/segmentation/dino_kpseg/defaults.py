"""Centralized default hyperparameters for DinoKPSEG training."""

from __future__ import annotations

# Backbone/features
MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"
SHORT_SIDE = 640
LAYERS = "-1,-2"

# Supervision/instance handling
RADIUS_PX = 5.5
MASK_TYPE = "gaussian"
INSTANCE_MODE = "auto"
BBOX_SCALE = 1.15

# Optimization/model capacity
HIDDEN_DIM = 192
LR = 3e-4
EPOCHS = 120
BATCH = 8
THRESHOLD = 0.40
HEAD_TYPE = "hybrid"
ATTN_HEADS = 4
ATTN_LAYERS = 2
COORD_WARMUP_EPOCHS = 5

# Loss weighting
BCE_TYPE = "bce"
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
DICE_LOSS_WEIGHT = 0.35
COORD_LOSS_WEIGHT = 0.50
COORD_LOSS_TYPE = "smooth_l1"

# Left-right regularization (helps reduce swap errors when flip_idx is available)
LR_PAIR_LOSS_WEIGHT = 0.06
LR_PAIR_MARGIN_PX = 4.0
LR_SIDE_LOSS_WEIGHT = 0.04
LR_SIDE_LOSS_MARGIN = 0.05

# Early stopping/model selection
EARLY_STOP_PATIENCE = 20
EARLY_STOP_MIN_DELTA = 0.001
EARLY_STOP_MIN_EPOCHS = 12
BEST_METRIC = "pck_weighted"
EARLY_STOP_METRIC = "auto"
PCK_WEIGHTED_WEIGHTS = "1,1,2,2"

# Augmentation defaults (used when augmentations are enabled)
AUGMENT_ENABLED = True
HFLIP = 0.25
DEGREES = 3.0
TRANSLATE = 0.01
SCALE = 0.03
BRIGHTNESS = 0.03
CONTRAST = 0.03
SATURATION = 0.02

# TensorBoard projector defaults (expensive; keep opt-in by default)
TB_PROJECTOR = False
TB_PROJECTOR_SPLIT = "val"
TB_PROJECTOR_MAX_IMAGES = 64
TB_PROJECTOR_MAX_PATCHES = 4000
TB_PROJECTOR_PER_IMAGE_PER_KEYPOINT = 3
TB_PROJECTOR_POS_THRESHOLD = 0.35
TB_PROJECTOR_CROP_PX = 96
TB_PROJECTOR_SPRITE_BORDER_PX = 3
TB_PROJECTOR_ADD_NEGATIVES = False
TB_PROJECTOR_NEG_THRESHOLD = 0.02
TB_PROJECTOR_NEGATIVES_PER_IMAGE = 6
