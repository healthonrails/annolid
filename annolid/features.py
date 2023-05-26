import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2

try:
    import clip
except ImportError:
    # Handle the case where CLIP is not installed
    print("CLIP library is not installed. Please install it to use CLIP-related functionality.")
    # Perform fallback or error handling actions
    print("pip install git+https://github.com/openai/CLIP.git")


class CLIPModel:
    """
    Wrapper class for CLIP model and related operations.
    """

    def __init__(self, clip_version="ViT-B/32"):
        """
        Initializes the CLIP model.

        clip_version (str): The version of the CLIP model to load.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load(
            clip_version, device=self.device)

    def encode_image(self, image):
        """
        Encodes the given image using the CLIP model.

        image (PIL.Image): The input image to encode.

        Returns:
            torch.Tensor: The encoded image tensor.
        """
        image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.clip_model.encode_image(image)
        return image_features.cpu().detach().numpy()

    def encode_text(self, text):
        """
        Encodes the given text using the CLIP model.

        text (str): The input text to encode.

        Returns:
            torch.Tensor: The encoded text tensor.
        """
        text = clip.tokenize(text).to(self.device)
        text_features = self.clip_model.encode_text(text)
        return text_features.cpu().detach().numpy()


class Embedding():
    """
    Use the resnet18 pretrained weights on imagenet to extract features
    """

    def __init__(self, layer_output_size=512):
        self.net = models.resnet18(pretrained=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.layer = self.net._modules.get('avgpool')
        self.net.eval()
        self.net.to(self.device)
        self.size = (224, 224)
        self.layer_output_size = layer_output_size

        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def resize(self, img):
        img = cv2.resize(img.astype(np.float32)/255., self.size)
        return img

    def __call__(self, imgs):

        img_batch = torch.cat([
            self.norm(self.resize(im)).unsqueeze(0)
            for im in imgs
        ], dim=0).float()
        img_batch = img_batch.to(self.device)
        _embedding = torch.zeros(len(imgs), self.layer_output_size, 1, 1)

        def copy_layer(m, i, o):
            _embedding.copy_(o.data)

        hook = self.layer.register_forward_hook(copy_layer)
        self.net(img_batch)

        hook.remove()
        return _embedding.cpu().numpy()[:, :, 0, 0]
