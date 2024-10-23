import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
from annolid.behavior.data_loading.datasets import BehaviorDataset
from annolid.behavior.data_loading.transforms import ResizeCenterCropNormalize
from annolid.behavior.models.classifier import BehaviorClassifier, ResNetFeatureExtractor

# Configuration (Best practice: Move these to a separate configuration file or use command-line arguments)
BATCH_SIZE = 1
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
VIDEO_FOLDER = "/path/to/video/folder"  # Replace with your video folder
CHECKPOINT_DIR = "checkpoints"  # Directory to save checkpoints


logger = logging.getLogger(__name__)


def train_model(model, dataloader, num_epochs, device, optimizer, criterion, checkpoint_dir):
    """
    Trains the behavior classification model.

    Args:
        model: The model to train.
        dataloader: The DataLoader for the training data.
        num_epochs: The number of training epochs.
        device: The device to use for training (e.g., "cuda" or "cpu").
        optimizer: The optimizer.
        criterion: The loss function.
        checkpoint_dir: The directory to save model checkpoints.
    """

    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):

            try:  # Handle potential errors in batch data
                inputs, labels, _ = batch  # _ for video_paths
                inputs, labels = inputs.to(device), labels.to(device)
            except Exception as e:
                logger.error(f"Error processing batch: {e}. Skipping batch.")
                continue  # Skip to the next batch

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:  # Print/log every 10 batches
                logger.info(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(
            checkpoint_dir, f"epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Checkpoint saved at {checkpoint_path}")


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

    # Set up logging to both console and a log file.
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(
        "training.log", mode='w')  # Overwrite file if exist
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)  # Use a file handler for logging to a file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    feature_extractor = ResNetFeatureExtractor().to(device)
    model = BehaviorClassifier(feature_extractor).to(device)
    print(model)
    transform = ResizeCenterCropNormalize()

    try:
        # Use command-line argument
        dataset = BehaviorDataset(args.video_folder, transform=transform)
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        exit(1)

    filtered_dataset = list(filter(lambda x: x is not None, dataset))
    dataloader = DataLoader(
        filtered_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # Use command-line argument for learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Use command-line args for checkpoint_dir, etc
    train_model(model, dataloader, args.num_epochs, device,
                optimizer, criterion, args.checkpoint_dir)

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
