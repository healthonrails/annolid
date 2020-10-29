import decord
import torch
import cv2
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_maskrcnn_model(
        num_classes=None,
        num_hidden_layer=256,
        finetuning=True):
    """ Get the pretrained mask rcnn model for finetuning or unmodified

    Args:
        num_classes ([int], optional): [number of class for custom dataset]. Defaults to None.
        num_hidden_layer (int, optional): number of hidden units. Defaults to 256.
        finetuning (bool, optional): [finetuning for the custom dataset]. Defaults to True.

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


def predict_coco(img, device):
    """predict with the default pretained mask rcnn model pretrained on COCO

    Args:
        img ([torch.Tensor]): [channel, width, height]
        device ([torch.device]): [cuda or cpu]

    Returns:
        [dict]: [prediction with bbox, masks, and labels]
    """
    coco_model = get_maskrcnn_model(finetuning=False)
    coco_model.to(device)
    coco_model.eval()
    with torch.no_grad():
        prediction = coco_model([img.to(device)])
        return prediction


if __name__ == "__main__":
    video_url = 'my_video.mkv'
    vr = decord.VideoReader(video_url)
    decord.bridge.set_bridge('torch')
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    for frame in vr:
        frame = frame.permute(2, 0, 1)
        frame = frame / 255.0
        prediction = predict_coco(frame, device)
        print(prediction)
        predict_img = prediction[0]['masks'][0, 0].mul(255).cpu().numpy()
        cv2.imshow("predicted mask", predict_img)
        cv2.waitKey(0)

        break
    cv2.destroyAllWindows()
