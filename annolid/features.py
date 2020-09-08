import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
import logging
import glob


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
