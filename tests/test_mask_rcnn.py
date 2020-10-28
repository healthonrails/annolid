from annolid.segmentation.maskrcnn.model import get_maskrcnn_model

def test_load_pretrained_maskrcnn_model():
    num_classes = 2
    model = get_maskrcnn_model(num_classes)
    model.eval()
    params = [p for p in model.parameters()]
    assert len(params) > 3