import argparse
import logging
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from annolid.behavior.data_loading.datasets import BehaviorDataset
from annolid.behavior.data_loading.transforms import IdentityTransform, ResizeCenterCropNormalize
from annolid.behavior.models.classifier import BehaviorClassifier
from annolid.behavior.models.feature_extractors import (
    CLIPFeatureExtractor,
    Dinov3BehaviorFeatureExtractor,
    ResNetFeatureExtractor,
)

# Configuration
VIDEO_FOLDER = "inference_videos"  # Folder with videos for inference
CHECKPOINT_PATH = "checkpoints/best_model.pth"  # Path to the saved model
BATCH_SIZE = 1  # Batch size for inference

logger = logging.getLogger(__name__)
BACKBONE_CHOICES = ("clip", "resnet18", "dinov3")
DEFAULT_DINOV3_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run behavior inference.")
    parser.add_argument("--video_folder", type=str, default=VIDEO_FOLDER,
                        help="Path to the video folder.")
    parser.add_argument("--checkpoint_path", type=str, default=CHECKPOINT_PATH,
                        help="Path to the trained model checkpoint.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size for inference.")
    parser.add_argument("--feature_backbone", type=str, choices=BACKBONE_CHOICES,
                        default="clip", help="Feature extraction backbone to use.")
    parser.add_argument("--dinov3_model_name", type=str, default=DEFAULT_DINOV3_MODEL,
                        help="DINOv3 checkpoint to use when --feature_backbone=dinov3.")
    parser.add_argument("--feature_dim", type=int, default=None,
                        help="Optional feature dimension override for compatible backbones.")
    parser.add_argument("--transformer_dim", type=int, default=768,
                        help="Transformer embedding dimension (d_model).")
    return parser.parse_args()


def build_feature_extractor(
    backbone: str,
    device: torch.device,
    *,
    dinov3_model: str,
    feature_dim: Optional[int],
) -> Tuple[torch.nn.Module, int]:
    if backbone == "clip":
        extractor = CLIPFeatureExtractor()
        if feature_dim is not None and feature_dim != extractor.feature_dim:
            raise ValueError(
                "CLIP backbone does not support overriding feature_dim.")
        return extractor.to(device), extractor.feature_dim
    if backbone == "resnet18":
        target_dim = feature_dim or 512
        extractor = ResNetFeatureExtractor(feature_dim=target_dim)
        return extractor.to(device), extractor.feature_dim
    if backbone == "dinov3":
        target_dim = feature_dim or 768
        extractor = Dinov3BehaviorFeatureExtractor(
            model_name=dinov3_model,
            feature_dim=target_dim,
            device=device.type,
            freeze=True,
        )
        return extractor.to(device), extractor.feature_dim
    raise ValueError(
        f"Unsupported backbone '{backbone}'. Valid options: {BACKBONE_CHOICES}")


def load_model(checkpoint_path, num_classes, device, backbone, *, dinov3_model, feature_dim, transformer_dim):
    """Loads the trained model from the checkpoint."""
    feature_extractor, backbone_dim = build_feature_extractor(
        backbone,
        device,
        dinov3_model=dinov3_model,
        feature_dim=feature_dim,
    )
    model = BehaviorClassifier(
        feature_extractor,
        num_classes=num_classes,
        d_model=transformer_dim,
        feature_dim=backbone_dim,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    logger.info(f"Model loaded from {checkpoint_path}")
    return model


def predict(model, data_loader, device):
    """Runs inference and prints predictions."""
    predictions = []
    with torch.no_grad():
        for i, (inputs, labels, video_names) in enumerate(data_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Get the predicted class index
            _, predicted = torch.max(outputs, 1)
            predictions.extend(zip(video_names, predicted.cpu().numpy()))

            # Print progress
            logger.info(f"Processed batch {i+1}/{len(data_loader)}")

    return predictions


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    transform = IdentityTransform(
    ) if args.feature_backbone == "dinov3" else ResizeCenterCropNormalize()

    try:
        # Dataset and DataLoader for inference
        dataset = BehaviorDataset(args.video_folder, transform=transform)
        num_of_classes = dataset.get_num_classes()

        data_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )
    except Exception as e:
        logger.error(f"Error creating dataset for inference: {e}")
        exit(1)

    # Load the trained model
    model = load_model(
        args.checkpoint_path,
        num_of_classes,
        device,
        args.feature_backbone,
        dinov3_model=args.dinov3_model_name,
        feature_dim=args.feature_dim,
        transformer_dim=args.transformer_dim,
    )

    # Run inference
    logger.info("Starting inference...")
    predictions = predict(model, data_loader, device)

    # Output predictions
    logger.info("Inference completed. Results:")
    for video_name, pred in predictions:
        logger.info(f"Video: {video_name}, Predicted Class: {pred}")


if __name__ == "__main__":
    main()
