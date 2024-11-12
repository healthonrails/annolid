import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
import argparse
from annolid.behavior.data_loading.datasets import BehaviorDataset
from annolid.behavior.data_loading.transforms import ResizeCenterCropNormalize
from annolid.behavior.models.classifier import BehaviorClassifier
from annolid.behavior.models.feature_extractors import ResNetFeatureExtractor,CLIPFeatureExtractor

# Configuration (Best practice: Move these to a separate configuration file or use command-line arguments)
BATCH_SIZE = 1
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
# Replace with your video folder
VIDEO_FOLDER = "behaivor_videos"  # Replace with your actual path
CHECKPOINT_DIR = "checkpoints"  # Directory to save checkpoints
VALIDATION_SPLIT = 0.2  # Proportion of the dataset to use for validation

logger = logging.getLogger(__name__)


def train_model(model, train_loader, val_loader, num_epochs, device, optimizer, criterion, checkpoint_dir):
    """Trains the model and evaluates it on a validation set."""

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

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

            if (i + 1) % 10 == 0:  # Log every 10 steps
                logger.info(progress_info)

        # Validation after each epoch
        val_loss, val_accuracy = validate_model(
            model, val_loader, criterion, device)
        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

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

    transform = ResizeCenterCropNormalize()

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

    feature_extractor = CLIPFeatureExtractor().to(device)
    model = BehaviorClassifier(
        feature_extractor, num_classes=num_of_classes).to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, val_loader, args.num_epochs,
                device, optimizer, criterion, args.checkpoint_dir)

    # Load best model for final evaluation
    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path))
    final_val_loss, final_val_accuracy = validate_model(
        model, val_loader, criterion, device)
    logger.info(
        f"Final Validation Loss: {final_val_loss:.4f}, Final Validation Accuracy: {final_val_accuracy:.2f}%")

    logger.info("Training and validation completed.")


if __name__ == "__main__":
    main()
