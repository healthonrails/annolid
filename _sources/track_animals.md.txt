# Track animals and Auto labeling

Click `Track Animals` button on the toolbar, fill the info in the opened dialog as follows. 

![Track Animals and objects](../imgs/track_animals.png)

Use `Detectron2` as the default model type

Choose the video file path and provide the trained model file with .pth format

Select a class threshold between 0 and 1

Provide the data.yaml file path in the COCO dataset folder

The output result folder is optional.

Note. You need to [install `Detectron2`](https://healthonrails.github.io/annolid/install.html#install-detectron2-locally) on your local device. If your workstation does not have a GPU card, it will only extract the key frames from the provided video and will save predicted results as json format in the same png image folder. 
Here is an example of predicted polygon annotions. 

![](../imgs/predicted_polygons.png)
The GPU workstation will run inference for all the frames in the provided video and will save the predicted results into a CSV file.

# Output CSV format 
Here are the columns of the Annolid CSV output format: 
frame_number: int, 0 based numbers for frames e.g. 10 the 11th frame
x1: float, the top left x value of the instance bounding box
y1: float, the top left y value of the instance bounding box
x2: float, the bottom right x value of the instance bounding box
y2: float, the bottom right y value of the instance bounding box
instance_name: string, the unique name of the instances or the class name
class_score: float, the confidence score between 0 to 1 for the class or instance name
segmentation: run length encoding of the instance binary mask
cx: float, optional, the center x value of the instance bounding box
cy: float, optional, the center y value of the instance bounding box

