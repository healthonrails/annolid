# config.py
import torch


class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VIDEO_FOLDER = "behaivor_videos"
    CHECKPOINT_DIR = "checkpoints"
    TENSORBOARD_LOG_DIR = "runs"
    LOG_DIR = "logs"
    LOG_FILE_NAME = "training.log"
    LOG_LEVEL = "INFO"  # Can be "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    VALIDATION_SPLIT = 0.2
    BATCH_SIZE = 1
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    MODEL_NAME = 'slowfast'  # Default model
    PRETRAINED = True
    NUM_WORKERS = 4
    NUM_FRAMES = 30
    CLIP_LEN = 1  # in seconds
    FPS = 30
    FEATURE_BACKBONE = "clip"
    DINOV3_MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"
    TRANSFORMER_DIM = 768
