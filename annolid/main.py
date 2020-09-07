import argparse
from segmentation.threshold import InRange
from data.videos import extract_frames


def parse_args():
    arg_builder = argparse.ArgumentParser(
        description="Multiple Animal Tracking"
    )
    arg_builder.add_argument("-v", "--video", type=str,
                             help="path to a input video file"
                             )
    arg_builder.add_argument('--extract_frames', type=int, default=0,
                             help="Number of frames to be extracted. \
                                 if -1 then save all the frames"
                             )
    arg_builder.add_argument('--to', type=str, default=None,
                             help='destination directory for saving \
                                  extracted frames '
                             )
    arg_builder.add_argument('--show_flow', type=bool, default=False,
                             help="Display optical flow while extracting"
                             )
    arg_builder.add_argument('--algo', type=str, default='flow',
                             help="Select 100 frame uniformly or by flow \
                                 options:flow|uniform"
                             )
    arg_builder.add_argument('--segmentation', type=str, default=None,
                             help="Segmentation based on support methods \
                                 options: threshold|yolact"
                             )

    arg_builder.add_argument('--min_area', type=int, default=50,
                             help="min area of the object \
                                 default is 50 pixels"
                             )

    arg_builder.add_argument('--max_area', type=int, default=150,
                             help="min area of the object \
                                 default is 50 pixels"
                             )

    args = vars(arg_builder.parse_args())
    return args


def main():
    args = parse_args()

    if args['extract_frames'] != 0:
        print("Start extracting video frames...")
        extract_frames(args['video'],
                       args['extract_frames'],
                       show_flow=args['show_flow'],
                       algo=args['algo'],
                       out_dir=args['to']
                       )
    if args['segmentation'] == 'threshold':
        ir = InRange()
        ir.run(args['video'],
               args['min_area'],
               args['max_area'])


if __name__ == "__main__":
    main()
