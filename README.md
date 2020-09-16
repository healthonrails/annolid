# annolid
An annotation and instance segmentation based multiple animal tracking and behavior analysis package.

![Multiple Animal Tracking](docs/imgs/mutiple_animal_tracking.png)

## Installation

* Clone the code repo and change into the directory
```bash
git clone https://github.com/healthonrails/annolid.git
cd annolid 

# install the package
pip install -e .
```

## Extract desired number of frames from a video based on optical flow

```bash
python annolid/main.py -v /path/to/my_video.mp4 --extract_frames=100
```
The above command will extract 100 frames from the provided video and save them to a default folder called extracted_frames in the current annolid repo folder. 

## Display optical flow while extracting frames with **--show_flow=True**
```bash
python annolid/main.py -v /path/to/my_video.mp4 --extract_frames=100 --show_flow=True
```

## Save all the frames as images
```bash
python annolid/main.py  -v /path/to/my_video.mp4 --extract_frames=-1
```
## Select frames uniformally 
```bash
python annolid/main.py  -v /path/to/my_video.mp4 --extract_frames=100 --algo=uniform
```

## Threshold based object segmenation
Added track bars for users to select HSV values to 
segment ROIs in the provided video. 
```bash
python annolid/main.py -v /path/to/my_video.mp4 --segmentation=threshold
```
![Threshold based segmentation](docs/imgs/threshold_based_segmentation.png)

## Convert WMV format to mp4 format using ffmpeg
```bash
ffmpeg -i /path/to/my_video.wmv -c:a aac /path/to/my_video.mp4
```

## Save the extracted frames to a user selected output directory 
If not selected, it will save the extracted frames to a folder named with the video name without extension. For example, if the input video path is /path/to/my_video.mp4, the extracted frames will be saved in the folder /path/to/my_video. 
The output directory is provided, the extracted frames will be saved /path/to/dest/my_video. 
```bash
cd annolid
python main.py -v /path/to/my_video.mp4 --extract_frames=20 --to /path/to/dest --algo=uniform
```

## How to track multiple objects in the video? 
Currently, it just works for person class with pretrained YOLOV3 or YOLOV5 weights.
```bash
python annolid/main.py -v /path/to/my_video.mp4 --track=YOLOV5
```

## How to convert coco annonation format to YOLOV5 format? 
The dataset is structured as follows. 
```
my_yolo_dataset
├── images
│   ├── train
│   └── val
└── labels
    ├── train
    └── val
data.yaml
```
```
python main.py --coco2yolo=/path/to/my_coco_dataset/annotations.json --to my_yolo_dataset --dataset_type=val
```
## How to train a custom YOLOV5 model? 
```bash
cd annolid/detector
git clone https://github.com/healthonrails/yolov5.git
cp -r my_yolo_dataset annolid/detector
cd yolov5
# change nc (number of classes) in the models/yolo5x.ym, default is 80

python train.py --img 640 --batch=8 --epochs 30 --data ../my_dataset_yolo/data.yaml --cfg ./models/yolov5x.yaml --weights yolov5x.pt --name yolov5x_my_model --cache
```

## How to track objects in a video with the trained model? 

```bash
cd annold
python main.py -v /path/to/my_video.mp4 --track=YOLOV5 --weights=detector/yolov5/runs/exp5_yolov5x_my_model/weights/best.pt
```





