This guide will walk you through using the **Video Rescaling Tool** in Annolid. This powerful feature allows you to prepare your video files for analysis by resizing (downsampling), changing the frame rate, denoising, and cropping them in batches.

The tool ensures that all your processing steps are logged, making your workflow reproducible and easy to track.

## Getting Started: Opening the Tool

First, you need to open the Video Rescaling dialog from the Annolid main window.

1.  Launch the Annolid application.
2.  From the top menu bar, navigate to **`File`** > **`Downsample Videos`**.

This will open the **Video Rescaling** dialog window, where you will configure all the settings.

---

## Step-by-Step Guide

Follow these steps to configure and run the video processing task.

### Step 1: Select Your Input and Output Folders

This is the most important first step. You need to tell the tool where to find your original videos and where to save the new, processed ones.

-   **Input Folder:** Click the `Select Folder` button and choose the directory containing the videos you want to process.
-   **Output Folder:** Click the `Select Folder` button and choose the directory where you want the new videos to be saved.

> **💡 Tip:** It is highly recommended to use a different folder for your output to avoid overwriting your original files.

### Step 2: Configure Resizing and Frame Rate (FPS)

-   **Scale Factor:** This determines the new size of your videos.
    -   Use the **slider** for a quick estimate.
    -   For precision, type a decimal value into the **text box**. For example, `0.5` will make the videos half of their original width and height, while `0.25` will make them a quarter of the size.
-   **Frames Per Second (FPS):**
    -   To force all videos to a new, consistent frame rate (e.g., `30` FPS), enter the number and check the box for **`Use specified FPS for all videos`**.
    -   If you leave this box unchecked, each video will be processed while retaining its own original frame rate.

### Step 3: Crop Your Videos (The Interactive Method)

The interactive crop feature is the easiest way to select a specific region of interest to keep.

1.  **Launch the Cropper:** Click the **`Preview & Crop First Frame`** button. (You must have an input folder selected for this to work).
2.  **Draw Your Region:** A new window will pop up, showing the first frame of your first video. **Click and drag your mouse** over the image to draw a rectangle. You'll see a red dashed box representing your crop area.
3.  **Confirm Selection:** When you are happy with the region, click the `OK` button in the preview window. The `x`, `y`, `width`, and `height` coordinates will now be filled in for you automatically.
4.  **Activate the Crop:** **This is a crucial step!** You must check the **`Enable Crop Region`** checkbox. If you do not check this box, the cropping will be ignored, even if the coordinates are filled in.

### Step 4: Choose Final Options and Run

-   **Apply Denoise:** Check this box if your videos have visual noise that you'd like to reduce.
-   **Select Run Mode:**
    -   ✅ **`Rescale Video`**: Choose this for the main task. It will apply all your settings (scaling, cropping, etc.) and create new video files in the output folder.
    -   ✅ **`Collect Metadata Only`**: Use this if you only want to generate a report (`metadata.csv`) of the properties of your *original* videos without changing them.

Finally, click the **`Run Rescaling`** button. The button will change to "Processing..." and you will see a confirmation message when the task is complete.

---

## Understanding Your Results

After the process finishes, navigate to your chosen **Output Folder**. You will find:

1.  **Processed Video Files:** Your new, smaller, and/or cropped videos.
2.  **`metadata.csv`:** A single spreadsheet file that lists the technical details (like new dimensions, duration, FPS) of all the videos you just created. This is great for data management.
3.  **Markdown Reports (`.md` files):** For *every* video processed, there will be a matching `.md` report file. This file is your permanent record and contains:
    -   The exact parameters you used (scale factor, crop coordinates).
    -   The full `FFmpeg` command that was executed.
    -   The final metadata of the new video.

### Example Markdown Report (`my_video.md`)

```markdown
# Metadata and Processing Info for my_video.mp4

**Processing Parameters:**
- Scale Factor: 0.5
- FPS: 30
- Apply Denoise: True
- Crop Region: x=120, y=200, width=640, height=480

**FFmpeg Command:**

```
ffmpeg -i /path/to/input/my_video.mp4 -vf "crop=640:480:120:200,scale=iw0.5:ih0.5,hqdn3d" -r 30 -c:v libx264 -y /path/to/output/my_video.mp4
```

**Video Metadata:**
- **filename**: my_video.mp4
- **width**: 320
- **height**: 240
- **duration**: 60.5
- **fps**: 30.0
- **codec_name**: h264

