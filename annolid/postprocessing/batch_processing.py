import argparse
import os
import sys
import glob
import pandas as pd
from annolid.postprocessing.glitter import tracks2nix
from annolid.inference.predict import Segmentor


def get_videos(videos_dir="/data/videos/",
               video_file_pattern='*/*.mp4'):
    videos = glob.glob(os.path.join(videos_dir, video_file_pattern))
    assert len(videos) > 0
    return videos


def get_tracking_csvs(csv_dir='/data/csvs/',
                      csv_file_pattern='*/*.csv'
                      ):
    tracking_res = {}
    csv_files = glob.glob(os.path.join(csv_dir, csv_file_pattern))
    assert len(csv_files) > 0
    for cs in csv_files:
        video_name_no_ext = os.path.basename(cs).split('dataset_')[
            1].split('_mask_r')[0]
        tracking_res[video_name_no_ext] = cs

    return tracking_res


def get_segmentor(
    model_path='model_final.path',
    dataset_dir='/data/coco/dataset',
    score_threshold=0.1
):
    """create an intance segmentation model

    Args:
        model_path (str, optional): file path to the trained model. Defaults to 'model_final.path'.
        dataset_dir (str, optional): folder contains the coco dataset. Defaults to '/data/coco/dataset'.
        score_threshold (float, optional): class threshold. Defaults to 0.1.

    Returns:
        model : Detectron2 model
    """
    segmentor = Segmentor(
        dataset_dir,
        model_path,
        score_threshold
    )

    return segmentor


def get_freezing_results(results_dir, video_name):
    filtered_results = []
    res_files = os.listdir(results_dir)
    for rf in res_files:
        if video_name in rf:
            if '_results.mp4' in rf:
                filtered_results.append(rf)
            if '_tracked.mp4' in rf:
                filtered_results.append(rf)
            if '_motion.csv' in rf:
                filtered_results.append(rf)
            if 'nix.csv' in rf:
                filtered_results.append(rf)
    return filtered_results


def fill_missing_predictions(
    tracking_csv_file,
    excluded_instances=None,
    moving_window=5
):
    """Fill missing predictions.
    Use the last present location as the hidden animails' predictions.

    Args:
        tracking_csv_file (_type_): Tracking result csv file path
    """
    df = pd.read_csv(tracking_csv_file)

    # disable false positive warning
    pd.options.mode.chained_assignment = None
    all_instance_names = set(df.instance_name.unique())
    count = 0
    if excluded_instances is None:
        excluded_instances = set([])
    # exclude instances not interested
    all_instance_names = all_instance_names - excluded_instances
    print("Fill the instane with name in the list: ", all_instance_names)
    missing_predictions = []
    max_frame_number = df.frame_number.max()
    for frame_number in df.frame_number:
        pred_instance = set(
            df[df.frame_number == frame_number].instance_name.unique()
        )
        missing_instance = all_instance_names - pred_instance
        for instance_name in missing_instance:
            frame_range_end = frame_number + moving_window
            if frame_range_end > max_frame_number:
                df_instance = df[(df.frame_number.between(max_frame_number-moving_window,
                                                          max_frame_number)) &
                                 (df.instance_name == instance_name)
                                 ]

            else:

                df_instance = df[
                    (df.frame_number.between(frame_number,
                                             frame_range_end))
                    & (df.instance_name == instance_name)
                ]
            if df_instance.shape[0] >= 1:
                fill_value = df_instance.iloc[0]
            else:
                #(f"No instances {instance_name} in this window")
                # move to the next frame
                continue
            fill_value.frame_number = frame_number
            missing_predictions.append(fill_value)
            count += 1
            if count % 1000 == 0:
                print(f'Filling {count} missing {instance_name}')
    df = df.append(missing_predictions, ignore_index=True)
    return df


def parse_arges(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder",
                        help="folder contains all the video files",
                        required=True
                        )
    parser.add_argument('--video_pattern',
                        help="pattern for matching video files",
                        default='*/*.mp4',
                        )
    parser.add_argument('--csv_folder',
                        help='folder with tracking csv files',
                        default='/data/'
                        )
    parser.add_argument('--csv_pattern',
                        help="pattern for matching tracking csv files",
                        default='*.csv')
    parser.add_argument('--model_path',
                        help='file path to the trained Detectron2 model',
                        default='/data/model_final.pth'
                        )
    parser.add_argument('--dataset_dir',
                        help='folder path to COCO format dataset that were used to trained the model',
                        default='/data/coco_dataset'
                        )
    parser.add_argument('--motion_threshold',
                        help='motion threshold between 0 to 1',
                        default=0.01)
    parser.add_argument('--score_threshold',
                        help="class score threshold",
                        default=0.1
                        )
    return parser.parse_args(args)


def main():
    args = parse_arges()
    segmentor = None
    videos = get_videos(args.video_folder, args.video_pattern)
    csvs = get_tracking_csvs(args.csv_folder, args.csv_pattern)
    if os.path.exists(args.model_path) and os.path.exists(args.dataset_dir):
        segmentor = get_segmentor(args.model_path,
                                  args.dataset_dir, args.score_threshold)
    for v in videos:
        if segmentor:
            segmentor.on_video(v)
        video_name = os.path.basename(v).split('.')[0]
        try:
            csv_file = csvs[video_name]
        except KeyError:
            print('No tracking csv for video: ', video_name)
            continue
        tracks2nix(v,
                   csv_file,
                   score_threshold=args.score_threshold,
                   motion_threshold=args.motion_threshold
                   )
        print('Finish tracking video: ', video_name)


if __name__ == '__main__':
    main()
