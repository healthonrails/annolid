# annolid
An annotation and instance segmenation based mutiple animal tracking package.

![Multiple Animal Tracking](docs/imgs/mutiple_animal_tracking.png)

## Installation

* Clone the code repo and change into the directory
```bash
git clone https://github.com/healthonrails/annolid.git
cd annolid 

# install the package
pip install -e .
```

## Extract desired number of frames from video based on optical flow

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
python annolid/main.py  -v /Users/chenyang/Downloads/zebrafish_video.mp4 --extract_frames=100 --algo='uniform'
```