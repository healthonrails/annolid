# Video preparation
In our paper, we find that when compress and downsample the video size will significally increase the processing speed and saving computation resources and time. 
Here is one example:
```
ffmpeg -i mouse.avi -vcodec libx264 -vf scale=720x480 mouse.mp4
```
## Video Compression and Rescaling with FFmpeg

FFmpeg is a potent tool for video processing, offering capabilities for compression and rescaling. This guide presents the process of compressing and rescaling videos using FFmpeg.

## Installing FFmpeg

```shell
conda create --name annolid-env
conda activate annolid-env
conda install -c conda-forge ffmpeg

```
## Verify FFmpeg installation

```
ffmpeg -version
```
## Compressing a Video
Reduce file size while maintaining quality
```
ffmpeg -i input.mp4 -c:v libx264 -crf 18 output.mp4
```
Reduce file size and quality

```
ffmpeg -i input.mp4 -c:v libx264 -b:v 500k output.mp4
```
## Rescaling a Video
Resize to a specific resolution
```
ffmpeg -i input.mp4 -filter:v scale=w=1280:h=720 output.mp4
```
Resize to a specific aspect ratio
```
ffmpeg -i input.mp4 -filter:v scale=w=1280:h=-1 output.mp4
```

## Additional Tips
Utilize the -preset option to adjust encoding speed (e.g., -preset fast for faster encoding)
Specify the audio codec with -c:a (e.g., -c:a aac for AAC audio)
Set the audio bitrate using -b:a (e.g., -b:a 128k for 128 kbps audio)