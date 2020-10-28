import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_maskrcnn_model(num_classes):
    # load mask rcnn model pretrained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pretrained head
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )

    # the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    hidden_layer = 256
    # replace the mask predictor
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model
