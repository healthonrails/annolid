from torchvision import transforms
import logging

logger = logging.getLogger(__name__)

IMG_SIZE = 224  # Default image size


class IdentityTransform:
    """Returns inputs unchanged. Useful when raw frames must be preserved."""

    def __call__(self, image):
        return image

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class ResizeCenterCropNormalize(transforms.Compose):
    """
    Resizes, center crops, and normalizes an image.

    Args:
        size (int, optional): The target size of the image after resizing and cropping. Defaults to 224.
    """

    def __init__(self, size: int = IMG_SIZE):
        super().__init__(
            [
                # Resize with a buffer for better quality
                transforms.Resize(int(size * 1.14)),
                transforms.CenterCrop(size),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet normalization
            ]
        )
