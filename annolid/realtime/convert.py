from ultralytics import YOLO
from annolid.yolo import configure_ultralytics_cache

configure_ultralytics_cache()
# load a pretrained model (recommended for training)
model = YOLO("yolo11n-seg.pt")
model.export(format="openvino")  # export the model to ONNX format
