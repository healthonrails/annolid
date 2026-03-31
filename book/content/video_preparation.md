# Video preparation

In our paper, we find that compressing and downsampling videos can significantly increase processing speed and save compute resources and time.

For a quick one-line example:

```bash
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

The GUI version is a better fit when you need to process a folder of videos, apply the same defaults to most files, and keep a few file-specific exceptions.

1. **Click `File -> Downsample Videos`.**
2. **Select the folder** containing your original videos.
3. **Open the `Input / Output` tab** and set an output folder to save all the processed videos.
   - If you leave it blank, Annolid uses a sibling folder with `_downsampled` appended to the input folder name.
4. **Open the `Processing` tab** and set folder defaults:
   - For example, selecting `0.5` will reduce the size of the video by half.
   - Leave FPS blank unless you need a fixed frame rate.
   - Enable crop only if the same crop applies to every video.
5. If one or more videos need different settings, click **`Review Videos One by One (Folder Input)`** and use **`Save & Next`** for exceptions or **`Skip`** to keep the defaults.
6. **Open the `Run` tab**.
7. **Check the `Rescale Video` checkbox**.
8. **Click the `Run Processing` button**.

This workflow compresses and downsamples the selected folder while keeping per-video changes explicit and reproducible.

## Additional Tips

- Utilize the `-preset` option to adjust encoding speed, for example `-preset fast`.
- Specify the audio codec with `-c:a`, for example `-c:a aac`.
- Set the audio bitrate using `-b:a`, for example `-b:a 128k`.
- If you need a deeper walkthrough for folder workflows, see the dedicated downsampling guide in the documentation.
