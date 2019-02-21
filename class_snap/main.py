#!/usr/bin/python2
from __future__ import print_function
from extractor import extractor
import os
import sys
import argparse
# from compare import compare_models
allowed_file_types = {'mp4': True, 'avi': True}
def get_input_data(input_video, class_labels_file,input_videos_dir=None, allowed_file_types=None):
    allowed_file_types = allowed_file_types if allowed_file_types else {'mp4': True, 'avi': True}
    input_video_files = []
    class_labels_to_filter = []
    try:
        if len(str(input_video)):
            input_video_files.append(input_video)
        
        with open(class_labels_file, "r") as labels_file:
            labels_str = labels_file.read()
            class_labels_to_filter = labels_str.replace('\n', ',')
            class_labels_to_filter = list(filter(None, class_labels_to_filter.split(',')))
            if input_videos_dir:
                input_video_files = input_video_files + list(filter(lambda f: allowed_file_types.get(f), os.listdir(input_videos_dir)))
    except Exception as e:
        print('error opening '+class_labels_file,'\n',e)
    return input_video_files, class_labels_to_filter

def build_args_parser():
    parser = argparse.ArgumentParser(description='Tool to extract frames of a video containing specified object(s)')
    parser.add_argument('-i','--input_video', dest='input_video',help='input video file(s)', default='input_video.mp4', type=str,required=True)
    parser.add_argument('-t','--interval', dest='interval',help='recuring interval (in seconds) at which to take a frame and process', default=1, type=int,required=True)                        
    parser.add_argument('-c','--class_labels', dest='class_labels_file',help='Object class labels to filter frames', default='class_labels.txt', type=str,required=True)
    parser.add_argument('-m','--model', dest='model_name',help='model name', default='ssd', type=str)
    #parser.add_argument('-id','--input_videos_dir', dest='input_videos_dir',help='Directory Containing input video file(s)', default='videos', type=str)
    #parser.add_argument('-od','--output_dir', dest='output_dir',help='Destination Directory for output', default='output', type=str)
    parser.add_argument('-o','--output_file', dest='zip_name',help='Output zip name', default='detections.zip', type=str)
    return parser

def main():
    #input_file = download_file('https://github.com/manuhg/masknet/raw/master/input_video.mp4',nc=True)
    #interval = 1
    parser = build_args_parser()
    if len(sys.argv) >= 3:
        args = parser.parse_args()
        input_video = args.input_video
        class_labels_file = args.class_labels_file
        interval = args.interval
        model_name = args.model_name
        zip_name = args.zip_name
        #input_videos_dir,output_dir,  = args.input_videos_dir, args.output_dir,
        
        input_video_files, class_labels_to_filter = get_input_data(input_video, class_labels_file)
        #,input_videos_dir=None, allowed_file_types=allowed_file_types)
        
        if (not input_video_files) or (not class_labels_to_filter) or (not interval):#(not output_dir) or
            print('\nError: Arguments empty/invalid')
            print('input_video_files:',input_video_files)
            print('class_labels_to_filter:',class_labels_to_filter)
            #print('output_dir:',output_dir)
            print('interval:',interval)
            print('Please check the above')
            exit()
        extractor_ = extractor(model_name=model_name,load=True)
        for input_video_file in input_video_files:
            print(input_video_file,class_labels_to_filter,interval,zip_name)
            extractor_.process(input_video_file,class_labels_to_filter,interval,zip_name=zip_name)
    else:
        print('\nExample: python main.py -i input_video.mp4 -t 20 -c class_labels.txt\n')
        parser.print_help()

if __name__ == "__main__":
    main()