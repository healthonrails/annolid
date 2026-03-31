# Video FFmpeg Processing with Annolid Bot

Annolid provides a built-in agent capability (the `video-ffmpeg` skill) that allows you to easily process, enhance, and convert videos using natural language commands. Under the hood, this uses FFmpeg and the `video_ffmpeg_process` tool, but you don't need to write complex command-line arguments.

If you prefer a direct GUI workflow for folder-based processing, use **`File` > `Downsample Videos`** in the Annolid desktop app. The dialog keeps shared defaults in one place and offers a sequential per-video review button for exceptions.

## What it can do

You can ask the Annolid bot to perform common video operations, such as:

* **Improve Quality:** Denoise and auto-contrast noisy lab videos.
* **Auto Contrast:** Enhance brightness and contrast automatically.
* **Downsample:** Reduce video resolution and frame rate to save space.
* **Denoise:** Apply spatial and temporal denoising.
* **Crop:** Crop the video to a specific region of interest.

## Example Workflows

Here are some examples of how to interact with the bot to process your videos.

### 1. Simple Downsampling

If you have a large high-resolution, high-framerate video and want to compress it for easier sharing or faster processing:

> **You:** "Please downsample the video at `/path/to/my_video.mp4` to reduce its file size."
>
> **Bot:** The bot will automatically analyze the source video using `video_info`, determine its current resolution and FPS, and then run `video_ffmpeg_process` with the `downsample` preset (which halves the resolution and frame rate). It will save the result as `/path/to/my_video_processed.mp4`.

### 2. Enhancing Noisy Videos

If you have a dark or noisy recording from an experimental setup:

> **You:** "Improve the quality of `/path/to/dark_recording.mp4`. Make it brighter and remove the noise."
>
> **Bot:** The bot will map this to the `improve_quality` preset and increase the contrast strength. It will apply hqdn3d denoising and auto-contrast enhancement.

### 3. Precision Cropping

If you only want to analyze a specific quadrant of the arena:

> **You:** "Crop `/path/to/behavior_video.mp4` to the top-left region. The area is x=0, y=0, width=800, height=600."
>
> **Bot:** The bot will execute the `crop` action, passing exactly those coordinates to FFmpeg to generate a cropped duplicate of your video.

### 4. Custom Parameters

You can guide the bot with more specific technical requirements:

> **You:** "Process my video but downsample it specifically to 15 fps and scale it to 25% of its original size."
>
> **Bot:** The bot understands the parameters `target_fps=15` and `scale_factor=0.25` and will execute a custom FFmpeg command matching your specifications exactly.

## Troubleshooting and Tips

* **File Paths:** Always provide absolute paths to your videos (e.g. `/Users/name/Desktop/data/video.mp4`).
* **Overwriting:** By default, the bot will *not* overwrite existing processed files to protect your data. If you want it to replace an existing file, explicitly say, "overwrite the existing output file."
* **Inspect First:** It is often helpful to ask the bot to "inspect" or "give me info" on a video first. This allows both you and the bot to confirm the source properties (resolution, original FPS) before deciding how to process it.
* **FFmpeg Required:** This skill requires FFmpeg to be installed on your system. If the bot reports that FFmpeg is missing, you may need to install it (e.g. `brew install ffmpeg` on macOS or `sudo apt install ffmpeg` on Linux).
* **GUI Batch Workflow:** For batches where most videos share the same settings and a few need exceptions, the GUI review workflow is usually easier than describing each exception to the bot.
