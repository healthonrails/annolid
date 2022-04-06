import argparse
import os
import sys
import glob
from annolid.postprocessing.glitter import tracks2nix


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
                        required=True
                        )
    parser.add_argument('--csv_pattern',
                        help="pattern for matching tracking csv files",
                        default='*.csv')
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
    videos = get_videos(args.video_folder, args.video_pattern)
    csvs = get_tracking_csvs(args.csv_folder, args.csv_pattern)
    for v in videos:
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
