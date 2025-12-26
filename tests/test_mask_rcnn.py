from PIL import Image
import torch
from torchvision.transforms import functional as F

from annolid.segmentation.maskrcnn.model import get_maskrcnn_model


def test_load_pretrained_maskrcnn_model():
    num_classes = 2
    model = get_maskrcnn_model(num_classes, pretrained=False)
    model.eval()
    params = [p for p in model.parameters()]
    assert len(params) > 3


def test_predict_maskrcnn_model_structure():
    model = get_maskrcnn_model(finetuning=False, pretrained=False)
    model.eval()
    img = Image.new("RGB", (64, 64), color=(0, 0, 0))
    img_t = F.to_tensor(img)
    with torch.no_grad():
        prediction = model([img_t])

    assert isinstance(prediction, list)
    assert len(prediction) == 1
    assert "masks" in prediction[0]
        

    
