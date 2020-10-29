import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F


def get_maskrcnn_model(
        num_classes=None,
        num_hidden_layer=256,
        finetuning=True):
    """ Get the pretrained mask rcnn model for finetuning or unmodified

    Args:
        num_classes ([int], optional): number of class for custom dataset. Defaults to None.
        num_hidden_layer (int, optional): number of hidden units. Defaults to 256.
        finetuning (bool, optional): finetuning for the custom dataset. Defaults to True.

    Returns:
        [model]: [mask rcnn model]
    """
    # load mask rcnn model pretrained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    if finetuning:
        # number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pretrained head
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            num_classes
        )

        # the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

        hidden_layer = num_hidden_layer
        # replace the mask predictor
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )

    return model


def predict_coco(img, device=None):
    """predict with the default pretained mask rcnn model pretrained on COCO

    Args:
        img ([torch.Tensor]): channel, width, height
        device ([torch.device]): cuda or cpu

    Returns:
        [dict]: prediction with bbox, masks, and labels
    """
    if device is None:
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

    if not isinstance(img, torch.Tensor):
        img = F.to_tensor(img)
    coco_model = get_maskrcnn_model(finetuning=False)
    coco_model.to(device)
    coco_model.eval()

    with torch.no_grad():
        prediction = coco_model([img.to(device)])
        return prediction
