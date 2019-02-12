#!/usr/bin/python3
from extractor import extract_frames
import os
import sys
import argparse
import json
from fileutils import download_file, extract_file_from_tar, create_zip
from detector import detector as det
sys.path.append('.')
# from compare import compare_models


def get_input_data(dir_name, class_lalbels_file, allowed_file_types={'mp4': True, 'avi': True}):
    with open(class_lalbels_file, "r") as labels_file:
        labels_str = labels_file.read()
        class_labels_to_filter = labels_str.replace('\n', ',')
        class_labels_to_filter = list(
            filter(None, class_labels_to_filter.split(',')))
        input_video_files = list(
            filter(lambda f: allowed_file_types.get(f), os.listdir(dir_name)))
        return input_video_files, class_labels_to_filter
    print('error opening '+class_lalbels_file)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Tool to extract frames of a video containing specified object(s)')
    parser.add_argument('--videos_dir', dest='videos_dir',
                        help='Directory/Folder Containing input video file(s)', default='videos', type=str)
    parser.add_argument('--class_labels', dest='class_labels_file',
                        help='Object class labels to filter frames', default='class_labels.txt', type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='Destination Directory/Folder for output', default='output', type=str)
    return parser.parse_args()


# {"annotation": {"data_filename": "1.jpg", "data_type": "image", "data_annotation": {"bounding_polygon": [], "bounding_box": [{"classification_label": "label1", "point_2D": ["98,192", "185,278"]}, {
#     "classification_label": "label2", "point_2D": ["282,153", "364,293"]}, {"classification_label": "label2", "point_2D": ["421,152", "508,250"]}, {"classification_label": "label1", "point_2D": ["144,70", "213,147"]}]}}}


def convert_to_annotations(fname, output):  # call once per file
    annotations = {'annotation': {'data_filename': fname, 'data_type': 'image', 'data_annotation': {'bounding_polygon': [], 'bounding_box': ''}}
    regions = None
    for item in output.items():
        regions = ''
    annotations['annotation']['data_annotation']['bounding_box'] = regions


def main():
    input_file = download_file(
        'https://github.com/manuhg/masknet/raw/master/input_video.mp4')
    class_labels_to_filter_by = ['person']
    annotations_file = 'annotations.json'
    prepared = True
    detector = det('ssd')
    detector_model_class = detector.get_model_class()
    detector_model = detector_model_class(prepared)
    detector_model.prepare(prepared)
    output = extract_frames(
        detector_model, input_file, class_labels_to_filter_by, interval=2)
    with open(annotations_file, 'w+') as af:
        json.dump(output, af)
    output.update({annotations_file: annotations_file})
    create_zip('detections.zip', output.keys())
    # if len(sys.argv) >= 3:
    #   pass
    # TODO ADD functionality to use zip files as i/p and o/p instead of dir later on
    # args = parse_args()
    # videos_dir=args.videos_dir
    # class_labels_file=args.class_labels_file
    # output_dir=args.output_dir
    #####################
    # dir_name = sys.argv[1]
    # class_labels_file = sys.argv[2]
    # if len(sys.argv)>3:
    #   dest_dir = sys.argv[3]

    # input_video_files,class_labels_to_filter = get_input_data(dir_name,class_labels_file)
    # for input_video_file in input_video_files:
    #   prediction_stats = extract_frames(input_video_file,class_labels,dest_dir,show_popup=True)
    #   with open(input_video_file+'-prediction_stats.txt',"w+") as sf:
    #     sf.write(json.dump(prediction_stats))
    # compare_models() TODO
   # else:
    #    print('Format:\npython main.py <directory/zip file containing input videos> <text file containing class labels to filter> <destination (optional) > ')


# if __name__ == "main":
main()
