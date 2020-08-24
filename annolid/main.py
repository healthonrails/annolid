import argparse
from data.videos import extract_frames


def parse_args():
    arg_builder = argparse.ArgumentParser(
        description="Multiple Animal Tracking"
    )
    arg_builder.add_argument("-v", "--video", type=str,
                             help="path to a input video file"
                             )
    arg_builder.add_argument('--extract_frames', type=int, default=0,
                             help="Number of frames to be extracted"
                             )
    args = vars(arg_builder.parse_args())
    return args

def main():
    args = parse_args()
    print("Start extracting video frames...")
    if args['extract_frames'] > 0:
        extract_frames(args['video'], args['extract_frames'])


if __name__ == "__main__":
    main()