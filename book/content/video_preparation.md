# Video preparation
In our paper, we find that compressing and downsampling videos can significantly increase processing speed and save compute resources and time.
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

## Compress Videos with Annolid GUI

1. **Click `File -> Downsample Videos`.**
2. **Select the folder** containing your original videos.
3. **Select an output folder** to save all the compressed videos.
4. **Set the scale factor**:
    - For example, selecting `0.5` will reduce the size of the video by half.
5. **Check the `Rescale Video` checkbox**.
6. **Click the `Run Scaling` button**.

This will compress and downsample all the files in the selected folder.

## Additional Tips
Utilize the -preset option to adjust encoding speed (e.g., -preset fast for faster encoding)
Specify the audio codec with -c:a (e.g., -c:a aac for AAC audio)
Set the audio bitrate using -b:a (e.g., -b:a 128k for 128 kbps audio)
