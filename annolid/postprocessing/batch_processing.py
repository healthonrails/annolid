import os
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


def main(videos_dir,
         video_file_pattern='20*.mp4',
         csv_dir='.',
         csv_file_pattern='C57*.csv',
         motion_threshold=0.01,
         score_threshold=0.15
         ):
    videos = get_videos(videos_dir, video_file_pattern)
    csvs = get_tracking_csvs(csv_dir, csv_file_pattern)
    for v in videos:
        video_name = os.path.basename(v).split('.')[0]
        try:
            csv_file = csvs[video_name]
        except KeyError:
            print('No tracking csv for video: ', video_name)
            continue
        tracks2nix(v,
                   csv_file,
                   score_threshold=score_threshold,
                   motion_threshold=motion_threshold
                   )
        print('Finish tracking video: ', video_name)


if __name__ == '__main__':
    results_dir = '/data'
    main(results_dir, csv_dir=results_dir)