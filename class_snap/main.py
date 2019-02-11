#!/usr/bin/python3
from extractor import extract_frames
import os
import sys
import argparse
import json
from fileutils import download_file, extract_file_from_tar
from models.ssd import ssd
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


required_envs = {'ssd': {'name': 'ssd', 'dir': 'models/ssd',
                         'env_dir': 'env_dir', 'env_script': 'prepare_env.sh'}}


def main():
    dest_dir = '.'
    model_name = 'ssd'
    model = required_envs[model_name]
    model_dir = model['dir']+'/'
    env_dir = model_dir+model['env_dir']
    # if os.path.isdir(env_dir):
    #    print('\nRequired environment for '+model+' found. Skipping setup.')
    # else:
    #print('setting up '+env_dir)
    #exec_cmd('rm -rf '+model['env_dir'])
    # make a local dir, do all stuff then move it to models/<model_name>/
    #exec_cmd('mkdir '+model['env_dir'])
    # copy the models/<model name>/<script> to the local dir
    #exec_cmd('cp '+model_dir+model['env_script']+' '+model['env_dir']+'/')
    #exec_cmd('cd '+model_dir+' && bash '+model['env_script'])
    input_file = download_file(
        'https://github.com/manuhg/masknet/raw/master/input_video.mp4')
    class_labels_to_filter_by = ['person']
    prepared = True
    ssd_obj = ssd(prepared)
    ssd_obj.prepare(prepared)
    labels_matched = extract_frames(
        ssd_obj, input_file, class_labels_to_filter_by, interval=2)
    create_zip('detections.zip', labels_matched.keys())
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
