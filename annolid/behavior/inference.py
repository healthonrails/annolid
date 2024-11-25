import logging
import torch
from torch.utils.data import DataLoader
from annolid.behavior.data_loading.datasets import BehaviorDataset
from annolid.behavior.data_loading.transforms import ResizeCenterCropNormalize
from annolid.behavior.models.classifier import BehaviorClassifier
from annolid.behavior.models.feature_extractors import CLIPFeatureExtractor

# Configuration
VIDEO_FOLDER = "inference_videos"  # Folder with videos for inference
CHECKPOINT_PATH = "checkpoints/best_model.pth"  # Path to the saved model
BATCH_SIZE = 1  # Batch size for inference

logger = logging.getLogger(__name__)


def load_model(checkpoint_path, num_classes, device):
    """Loads the trained model from the checkpoint."""
    feature_extractor = CLIPFeatureExtractor().to(device)
    model = BehaviorClassifier(feature_extractor, num_classes=num_classes).to(device)
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

            _, predicted = torch.max(outputs, 1)  # Get the predicted class index
            predictions.extend(zip(video_names, predicted.cpu().numpy()))

            # Print progress
            logger.info(f"Processed batch {i+1}/{len(data_loader)}")
    
    return predictions


def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    transform = ResizeCenterCropNormalize()

    try:
        # Dataset and DataLoader for inference
        dataset = BehaviorDataset(VIDEO_FOLDER, transform=transform)
        num_of_classes = dataset.get_num_classes()

        data_loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
        )
    except Exception as e:
        logger.error(f"Error creating dataset for inference: {e}")
        exit(1)

    # Load the trained model
    model = load_model(CHECKPOINT_PATH, num_of_classes, device)

    # Run inference
    logger.info("Starting inference...")
    predictions = predict(model, data_loader, device)

    # Output predictions
    logger.info("Inference completed. Results:")
    for video_name, pred in predictions:
        logger.info(f"Video: {video_name}, Predicted Class: {pred}")


if __name__ == "__main__":
    main()