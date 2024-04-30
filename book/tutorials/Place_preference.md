# Place preference

# Converting and Featuring SLEAP Keypoint Predictions with Annolid
[[](https://i9.ytimg.com/vi_webp/tvLHj-Nwu-Y/mq2.webp?sqp=COCIxbEG-oaymwEmCMACELQB8quKqQMa8AEB-AH-CYAC0AWKAgwIABABGGUgZShlMA8=&rs=AOn4CLA8dAP3_nEJsDAFYf02EJ9ct147UQ)](https://youtu.be/tvLHj-Nwu-Y?si=G7gXjQM4Oh9Ob5ix)
<iframe width="560" height="315" src="https://www.youtube.com/embed/tvLHj-Nwu-Y?si=ddyWwA2Wedypnoim" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

In this tutorial, we'll walk through the process of converting SLEAP keypoint predictions into JSON files and featuring them with Annolid for tracking purposes. Let's dive in!

## Prerequisites

Before getting started, ensure you have the following:

- SLEAP prediction H5 format file
- AI Polygon for zone creation
- Annolid installed and configured

## Steps

### 1. Create Zone File

- Open your video file.
- Use AI Polygon to automatically create zone shapes.
- Name the zones and specify the keyword "zone" in the label description field.
  
### 2. Load SLEAP Prediction File

- Go to the file menu and select "Load SLEAP h5"
- Click the "run" button to initiate the conversion to Labelme JSON format.

### 3. Save Keypoints to JSON

- Once the conversion is complete, save the JSON files containing the keypoints on your desktop.

### 4. Load Files into Annolid

- Open Annolid.
- Load the the video file and Annolid will load JSON files automatically for further reviewing or editing.

### 5. Conversion Verification

- Check the conversions across frames to ensure the predictions are accurate.

### 6. Optional: Convert to CSV

- If needed, convert the JSON file format of the keypoints to CSV format.
- Click on the file menu and select "Save CSV."
- Locate the folder containing the JSON files and click the "run" button to generate and save the CSV files to disk.

### 7. Additional Features

- Annolid automatically converts the keypoints to smaller circles around their center with a radius of 10 pixels.
- Random 10 points are drawn on each circle.

## Conclusion

By following these steps, you can efficiently convert SLEAP keypoint predictions into JSON files and utilize Annolid for tracking and analysis purposes. Happy tracking!

