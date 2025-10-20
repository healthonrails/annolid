import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
import argparse
from annolid.behavior.data_loading.datasets import BehaviorDataset
from annolid.behavior.data_loading.transforms import IdentityTransform, ResizeCenterCropNormalize
from annolid.behavior.models.classifier import BehaviorClassifier
from annolid.behavior.models.feature_extractors import (
    CLIPFeatureExtractor,
    Dinov3BehaviorFeatureExtractor,
    ResNetFeatureExtractor,
)
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard

# Configuration (Best practice: Move these to a separate configuration file or use command-line arguments)
BATCH_SIZE = 1
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
# Replace with your video folder
VIDEO_FOLDER = "behaivor_videos"  # Replace with your actual path
CHECKPOINT_DIR = "checkpoints"  # Directory to save checkpoints
VALIDATION_SPLIT = 0.2  # Proportion of the dataset to use for validation
TENSORBOARD_LOG_DIR = "runs"  # Directory for TensorBoard logs

logger = logging.getLogger(__name__)

BACKBONE_CHOICES = ("clip", "resnet18", "dinov3")
DEFAULT_DINOV3_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


def build_feature_extractor(
    backbone: str,
    device: torch.device,
    *,
    dinov3_model: str,
    feature_dim: Optional[int],
    unfreeze_dino: bool,
) -> Tuple[nn.Module, int]:
    """Factory for feature extractors."""
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
            freeze=not unfreeze_dino,
            device=device.type,
        )
        return extractor.to(device), extractor.feature_dim
    raise ValueError(
        f"Unsupported backbone '{backbone}'. Valid options: {BACKBONE_CHOICES}")


def train_model(model, train_loader, val_loader, num_epochs, device, optimizer, criterion, checkpoint_dir, writer):
    """Trains the model and evaluates it on a validation set, logging to TensorBoard."""

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    global_step = 0  # Track training steps for TensorBoard

    for epoch in range(num_epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            try:
                inputs, labels, _ = batch
                inputs, labels = inputs.to(device), labels.to(device)
            except Exception as e:
                logger.error(f"Error processing batch: {e}. Skipping batch.")
                continue

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            progress_info = f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
            print(progress_info)

            # Log training loss to TensorBoard
            writer.add_scalar('Loss/train', loss.item(), global_step)

            if (i + 1) % 10 == 0:  # Log every 10 steps
                logger.info(progress_info)

            global_step += 1

        # Validation after each epoch
        val_loss, val_accuracy = validate_model(
            model, val_loader, criterion, device)
        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Log validation loss and accuracy to TensorBoard
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/validation',
                          val_accuracy / 100.0, epoch)  # Scale to 0-1

        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"New best model saved at {checkpoint_path}")

    logger.info("Training finished.")


def validate_model(model, val_loader, criterion, device):
    """Evaluates the model on the validation set and calculates accuracy."""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss, accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Train animal behavior classifier.")
    parser.add_argument("--video_folder", type=str,
                        default=VIDEO_FOLDER, help="Path to the video folder.")
    parser.add_argument("--batch_size", type=int,
                        default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--num_epochs", type=int,
                        default=NUM_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float,
                        default=LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--checkpoint_dir", type=str,
                        default=CHECKPOINT_DIR, help="Checkpoint directory.")
    parser.add_argument("--tensorboard_log_dir", type=str,
                        default=TENSORBOARD_LOG_DIR, help="Directory for TensorBoard logs.")
    parser.add_argument("--feature_backbone", type=str, choices=BACKBONE_CHOICES,
                        default="dinov3", help="Feature extraction backbone.")
    parser.add_argument("--dinov3_model_name", type=str, default=DEFAULT_DINOV3_MODEL,
                        help="DINOv3 checkpoint to use when --feature_backbone=dinov3.")
    parser.add_argument("--unfreeze_dinov3", action="store_true",
                        help="Unfreeze DINOv3 weights for fine-tuning.")
    parser.add_argument("--feature_dim", type=int, default=None,
                        help="Optional feature dimension override for the backbone projection.")
    parser.add_argument("--transformer_dim", type=int, default=768,
                        help="Transformer embedding dimension (d_model).")

    args = parser.parse_args()

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("training.log", mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    transform = IdentityTransform(
    ) if args.feature_backbone == "dinov3" else ResizeCenterCropNormalize()

    try:
        dataset = BehaviorDataset(args.video_folder, transform=transform)
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        exit(1)

    num_of_classes = dataset.get_num_classes()

    # Split dataset into train and validation sets
    val_size = int(VALIDATION_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    feature_extractor, backbone_dim = build_feature_extractor(
        args.feature_backbone,
        device,
        dinov3_model=args.dinov3_model_name,
        feature_dim=args.feature_dim,
        unfreeze_dino=args.unfreeze_dinov3,
    )
    model = BehaviorClassifier(
        feature_extractor,
        num_classes=num_of_classes,
        d_model=args.transformer_dim,
        feature_dim=backbone_dim,
    ).to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(args.tensorboard_log_dir)
    # Add graph to TensorBoard
    try:
        # Get a sample input from the DataLoader to determine the correct shape
        sample_batch = next(iter(train_loader))
        # Get the first item from the batch and add a batch dimension
        dummy_input = sample_batch[0][0].unsqueeze(0).to(device)
        writer.add_graph(model, dummy_input)
    except Exception as e:
        logger.warning(f"Failed to add graph to TensorBoard: {e}")

    train_model(model, train_loader, val_loader, args.num_epochs,
                device, optimizer, criterion, args.checkpoint_dir, writer)

    # Load best model for final evaluation
    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path))
    final_val_loss, final_val_accuracy = validate_model(
        model, val_loader, criterion, device)
    logger.info(
        f"Final Validation Loss: {final_val_loss:.4f}, Final Validation Accuracy: {final_val_accuracy:.2f}%")

    logger.info("Training and validation completed.")
    writer.close()  # Close the TensorBoard writer


if __name__ == "__main__":
    main()
