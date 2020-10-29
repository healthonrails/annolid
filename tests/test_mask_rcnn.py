from pathlib import Path
from PIL import Image
from annolid.segmentation.maskrcnn.model import (
    get_maskrcnn_model,
    predict_coco
)



def test_load_pretrained_maskrcnn_model():
    num_classes = 2
    model = get_maskrcnn_model(num_classes)
    model.eval()
    params = [p for p in model.parameters()]
    assert len(params) > 3


def test_predict_pretained_maskrcnn_model():
    img_dir = Path(__file__).parent.parent / 'docs' / 'imgs'
    img_path = img_dir / "mutiple_animal_tracking.png"
    img = Image.open(str(img_path)).convert("RGB")
    prediction = predict_coco(img)
    print(prediction)
    assert 'masks' in prediction[0]
        

    
