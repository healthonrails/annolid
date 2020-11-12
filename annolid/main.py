import os
import argparse
from segmentation.threshold import InRange
from annotation import coco2yolo
from data.videos import extract_frames, track
from postprocessing.glitter import tracks2nix


def parse_args():
    arg_builder = argparse.ArgumentParser(
        description="Multiple Animal Tracking"
    )
    arg_builder.add_argument("-v", "--video", type=str,
                             default=None,
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
    arg_builder.add_argument('--track', type=str, default=None,
                             help="Track objects in the video \
                                 with detector YOLOV5|YOLOV3"
                             )
    arg_builder.add_argument('--weights', type=str, default=None,
                             help="path to the trained  weights  \
                                 e.g. ./detector/yolov5/weights/latest.pt"
                             )
    arg_builder.add_argument('--show_flow', type=bool, default=False,
                             help="Display optical flow while extracting"
                             )
    arg_builder.add_argument('--algo', type=str, default='flow',
                             help="Select 100 frame uniformly or by flow \
                                 options:flow|random"
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

    arg_builder.add_argument('--coco2yolo', type=str, default=None,
                             help="coco annotation file path e.g. ./annotaitons.json"
                             )

    arg_builder.add_argument('--dataset_type', type=str, default='train',
                             help="create a train or val dataset"
                             )

    arg_builder.add_argument('--labelme2coco', type=str, default=None,
                             help="input dir for labelme annotation e.g. ./extract_frames"
                             )
    arg_builder.add_argument('--keypoints2labelme', type=str, default=None,
                             help="input dir for keypoitns image dir e.g. ./mouse_m8s4"
                             )
    arg_builder.add_argument('--keypoints', type=str, default=None,
                             help="keypoints annotation file e.g. ./CollectedData_xxx.h5  "
                             )

    arg_builder.add_argument('--labels', type=str, default='labels.txt',
                             help="text file with all the class names "
                             )
    arg_builder.add_argument('--vis', type=bool, default=False,
                             help="Visualize the labeled images"
                             )
    arg_builder.add_argument('--tracks2glitter', type=str, default=None,
                             help="tracking results csv file."
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

    if args['track'] is not None:
        track(args["video"],
              args['track'],
              args['weights']
              )
    if args["coco2yolo"] is not None:
        names = coco2yolo.create_dataset(args['coco2yolo'],
                                         results_dir=args['to'],
                                         dataset_type=args['dataset_type']
                                         )
        print(names)
        print("Done.")

    if args['labelme2coco'] is not None:
        from annotation import labelme2coco
        label_gen = labelme2coco.convert(
            args['labelme2coco'],
            output_annotated_dir=args['to'],
            labels_file=args['labels'],
            vis=args['vis']
        )
        for image_id, content in label_gen:
            print(
                f'Converting {content} as {image_id % 100:.2f} % of the labeled images.')

    if args['tracks2glitter'] is not None:
        assert(os.path.isfile(
            args['tracks2glitter'])), \
            "Please provide the correct tracking results csv file"

        if args['to'] is not None:
            dest_file = os.path.join(args['to'],
                                     os.path.basename(args['tracks2glitter']).replace('.csv', '_nix.csv'))
        else:
            dest_file = args['tracks2glitter'].replace('.csv', '_nix.csv')

        tracks2nix(args['video'],
                   args['tracks2glitter'],
                   dest_file
                   )

    if (args['keypoints2labelme'] is not None) and \
            (args['keypoints'] is not None):
        from annolid.annotation import keypoints
        keypoints.to_labelme(
            args['keypoints2labelme'],
            args['keypoints']
        )


if __name__ == "__main__":
    main()
