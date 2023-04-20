import pandas as pd
"""
The choice of tracking metrics to report in research papers depends on
 the specific application and goals of the research. 
 However, some commonly used tracking metrics include:

Precision and Recall: Precision measures the percentage of correctly
 identified tracks out of the total number of identified tracks,
   while recall measures the percentage of correctly identified 
   tracks out of the total number of ground truth tracks. 
These metrics are often used to evaluate the accuracy of a tracking algorithm.

Multiple Object Tracking Accuracy (MOTA): MOTA is a comprehensive metric that takes
into account various sources of tracking errors, including missed detections,
 false positives, and identity switches. 
 It is calculated as 1 - (Total number of False Positives + 
 Total number of Missed Targets + Total number of Identity Switches)
   / Total number of Ground Truth Objects.

Multiple Object Tracking Precision (MOTP): MOTP measures the average
 distance between the predicted track and the ground truth track across all frames. 
 It is calculated as the sum of all distances between matched predicted and ground 
 truth tracks, divided by the total number of matches.

ID F1 Score: This metric measures the harmonic mean of precision and recall 
for each tracked identity. It is calculated as 2 * (Precision * Recall) / (Precision + Recall),
 where Precision and Recall are calculated for each identity separately.

Fragmentation Index (Frag): Frag measures the number of times a single ground 
truth object is tracked by multiple predicted tracks. It is calculated as the 
total number of identity switches divided by the total number of ground truth objects.

Mostly Tracked Targets (MT): MT measures the percentage of ground truth objects
 that are tracked for at least 80% of their lifespan.

Mostly Lost Targets (ML): ML measures the percentage of ground truth objects
 that are tracked for less than 20% of their lifespan.

False Positive Rate (FPR): FPR measures the percentage of falsely detected 
objects out of the total number of detections.

False Negative Rate (FNR): FNR measures the percentage of ground truth
 objects that are not detected by the tracker.

Processing Speed: Processing speed measures the number of frames processed 
per unit time, such as frames per second (FPS) or milliseconds per frame. 
This metric is often reported to evaluate the computational efficiency of a tracking algorithm.

These are just some of the commonly used tracking metrics in research papers,
 and there may be other application-specific metrics used as well.

"""


def extract_digits(s: str) -> str:
    """
    Extracts all digits at the end of a string and returns them as a string.

    :param s: The input string.
    :return: The string of digits at the end of the input string.
    """
    # Start at the end of the string and look for the first non-digit character
    i = len(s) - 1
    while i >= 0 and s[i].isdigit():
        i -= 1

    # Extract the digits that occur after the last non-digit character
    digits = s[i+1:]

    return digits


def convert_annolid_tracking_csv_to_mot(input_filename: str,
                                        output_filename: str = None) -> None:
    """
    Reads a CSV file with columns 'frame_number', 'x1', 'y1', 'x2', 'y2', 'instance_name', 'class_score',
    'segmentation', and 'tracking_id', and converts it to a new MOT CSV file with columns '<frame number>',
    '<object id>', '<bb_left>', '<bb_top>', '<bb_width>', '<bb_height>', '<confidence>', 'class', 'visiblity', 'z'.

    https://arxiv.org/pdf/2003.09003.pdf MOT
    Position Name Description 
    1 Frame number Indicate at which frame the object is present
    2 Identity number Each pedestrian trajectory is identified by a unique ID (−1 for detections)
    3 Bounding box left Coordinate of the top-left corner of the pedestrian bounding box
    4 Bounding box top Coordinate of the top-left corner of the pedestrian bounding box
    5 Bounding box width Width in pixels of the pedestrian bounding box
    6 Bounding box height Height in pixels of the pedestrian bounding box
    7 Confidence score DET: Indicates how confident the detector is that this instance is a pedestrian.
    GT: It acts as a flag whether the entry is to be considered (1) or ignored (0).
    8 Class GT: Indicates the type of object annotated
    9 Visibility GT: Visibility ratio, a number between 0 and 1 that says how much of that object is visible. Can be due
    to occlusion and due to image border cropping.

    :param input_filename: The path to the input CSV file.
    :param output_filename: The path to the output CSV file.
    """
    if output_filename is None:
        output_filename = input_filename.replace('.csv', '_mot.csv')

    # Load the input CSV file into a pandas dataframe
    df = pd.read_csv(input_filename)

    # Initialize a list to store the output rows
    output_rows = []

    # Iterate over each row in the input dataframe
    for index, row in df.iterrows():
        # Extract values from the input dataframe row
        frame_number = row['frame_number']
        x1 = int(row['x1'])
        y1 = int(row['y1'])
        x2 = int(row['x2'])
        y2 = int(row['y2'])
        instance_name = row['instance_name']
        class_score = row['class_score']
        tracking_id = row['tracking_id']

        # Calculate bounding box dimensions
        bb_left = x1
        bb_top = y1
        bb_width = x2 - x1
        bb_height = y2 - y1
        x = -1
        y = -1
        z = -1

        # Use the tracking ID to assign an object ID
        object_id = int(extract_digits(instance_name))
        if not object_id:
            if tracking_id[0].isdigit():
                object_id = tracking_id
            else:
                tracking_id = -1

        # Append a new row to the output list
        output_rows.append([frame_number, object_id, bb_left, bb_top,
                           bb_width, bb_height, class_score, x, y, z])

    # Create a new dataframe from the output list
    output_df = pd.DataFrame(output_rows, columns=[
                             'frame number', 'object id', 'bb_left',
                             'bb_top', 'bb_width', 'bb_height',
                             'confidence', 'x', 'y', 'z'])

    # Write the output dataframe to a CSV file
    output_df.to_csv(output_filename, index=False)


def mot_metrics_enhanced_calculator(gt_source: str, t_source: str) -> None:
    """
    Calculates MOT metrics for the provided ground truth and tracking output files.

    :param gt_source: The path to the ground truth file.
    :param t_source: The path to the tracking output file.
    # a guide for calculating MOT metrics for your custom dataset.
    # https://github.com/cheind/py-motmetrics
    """
    # Import required packages
    import motmetrics as mm
    import numpy as np

    # Load ground truth and tracking output
    gt = np.loadtxt(gt_source, delimiter=',', skiprows=1)
    t = np.loadtxt(t_source, delimiter=',', skiprows=1)

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    # Iterate over frames
    for frame in range(int(gt[:, 0].max())):
        frame += 1  # Detection and frame numbers begin at 1
        # Select id, x, y, width, height for current frame
        gt_dets = gt[gt[:, 0] == frame, 1:6]  # Select all detections in gt
        t_dets = t[t[:, 0] == frame, 1:6]  # Select all detections in t
        # Compute intersection over union matrix
        C = mm.distances.iou_matrix(gt_dets[:, 1:], t_dets[:, 1:], max_iou=0.5)
        # Update accumulator
        acc.update(gt_dets[:, 0].astype('int').tolist(),
                   t_dets[:, 0].astype('int').tolist(), C)

    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr',
                                       'recall', 'precision', 'num_objects',
                                       'mostly_tracked', 'partially_tracked',
                                       'mostly_lost', 'num_false_positives',
                                       'num_misses', 'num_switches',
                                       'num_fragmentations', 'mota', 'motp'],
                         name='acc')

    # Render summary as a string
    strsummary = mm.io.render_summary(
        summary,
        namemap={'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll',
                 'precision': 'Prcn', 'num_objects': 'GT',
                 'mostly_tracked': 'MT', 'partially_tracked': 'PT',
                 'mostly_lost': 'ML', 'num_false_positives': 'FP',
                 'num_misses': 'FN', 'num_switches': 'IDsw',
                 'num_fragmentations': 'FM', 'mota': 'MOTA', 'motp': 'MOTP'}
    )

    # Print summary
    print(strsummary)


if __name__ == '__main__':
    gt_truth = 'gt_truth_mask_rcnn_tracking_results_with_segmenation_mot.csv'
    tracker_output = 'mask_rcnn_tracking_results_with_segmenation_mot.csv'
    # convert_annolid_tracking_csv_to_mot(input_file_name)
    mot_metrics_enhanced_calculator(gt_truth, tracker_output)
