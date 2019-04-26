#!/usr/bin/python2
from __future__ import print_function
from extractor import extractor
from compare import compare, get_ground_truth_ann, compare_models
from generate_report import generate_report
import os
import sys
import argparse
import time

# from compare import compare_models
allowed_file_types = {'mp4': True, 'avi': True}

def file_type_allowed(fname):
    try:
        return fname.split('/')[-1].split('.')[-1] in allowed_file_types
    except Exception as e:
        print(e)

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
                input_video_files = input_video_files + list(map(lambda f : input_videos_dir+'/'+f,  list(filter(file_type_allowed, os.listdir(input_videos_dir))) ))
    except Exception as e:
        print('error opening '+class_labels_file,'\n',e)
    input_video_files = list(filter(None,input_video_files))
    return input_video_files, class_labels_to_filter

def build_args_parser():
    parser = argparse.ArgumentParser(description='Tool to extract frames of a video containing specified object(s)')
    parser.add_argument('-i','--input_video', dest='input_video',help='input video file(s) or youtube url of a video', default='', type=str)
    parser.add_argument('-t','--interval', dest='interval',help='recuring interval (in seconds) at which to take a frame and process', default=1, type=int,required=True)                        
    parser.add_argument('-c','--class_labels', dest='class_labels_file',help='Object class labels to filter frames', default='class_labels.txt', type=str,required=True)
    parser.add_argument('-m','--model', dest='model_name',help='model name', default='yolo', type=str)
    parser.add_argument('-v','--model_variant', dest='model_variant',help='model variant', type=str)
    parser.add_argument('-a','--annotations_dir', dest='annotations_dir',help='Directory containing ground truth annotations',default='./', type=str)
    parser.add_argument('-id','--input_videos_dir', dest='input_videos_dir',help='Directory Containing input video file(s)', default='', type=str)
    parser.add_argument('-od','--output_dir', dest='output_dir',help='Destination Directory for output', default='./', type=str)
    parser.add_argument('-o','--output_file', dest='zip_name',help='Output zip name', default='detections.zip', type=str)
    parser.add_argument('-vz','--visualize', dest='visualize',help='visualize annotations in output images', default='', type=str)
    parser.add_argument('-cm','--compare_models', dest='compare_models',help='compare models by running given input', default='', type=str)

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
        model_variant = args.model_variant
        zip_name = args.zip_name
        annotations_dir = args.annotations_dir
        output_dir = args.output_dir
        input_videos_dir = args.input_videos_dir
        annotations_dir = annotations_dir+'/' if annotations_dir[-1]!='/' else annotations_dir
        input_video_files, class_labels_to_filter = get_input_data(input_video, class_labels_file,input_videos_dir)
        visualize = args.visualize
        compare_models_ = args.compare_models

        if len(input_video_files)<1:
            print('THERE ARE NO VIDEO FILES AT ../yotube_download or SPECIFY ALTERNATE DIRECTORY --id some_dir')
            exit()

        if (not input_video_files) or (not class_labels_to_filter) or (not interval):#(not output_dir) or
            print('\nError: Arguments empty/invalid')
            print('input_video_files:',input_video_files)
            print('class_labels_to_filter:',class_labels_to_filter)
            #print('output_dir:',output_dir)
            print('interval:',interval)
            print('Please check the above')
            exit()
        if visualize:
            print('info: option set for visualizing annotations')

        extractor_ = extractor(model_name=model_name,model_variant=model_variant,load=True)
        for input_video_file in input_video_files:
            name = '.'.join(input_video_file.split('/')[-1].split('.')[:-1])
            name = '-'+name if name else name
            zip_name_ = zip_name[:zip_name.rfind('.')]+ name + '.zip'
            print(input_video_file,class_labels_to_filter,interval,zip_name)
            output,total_duration = extractor_.process(input_video_file,class_labels_to_filter,interval,zip_name=zip_name_,visualize=visualize,dest_dir=output_dir)
            if output is None:
                print(input_video_file,' WAS NOT PROCESSED')
                return
            generate_report()
            ground_truth_ann_file = '.'.join(os.path.basename(input_video_file).split('.')[:-1])+'-every_'+str(interval)+'s.json'
            #ground_truth = get_ground_truth_ann(annotations_dir+ground_truth_ann_file)
            
            #compare(ground_truth,output)
            #compare(output,output) #just to test
    else:
        print("\nExample: python2 main.py -c class_labels.txt -t 10\n")
        print("\nExample: python2 main.py -i 'https://www.youtube.com/watch?v=7nnp55fO2dE' -c class_labels.txt -t 10 -m detectron\n")
        parser.print_help()

if __name__ == "__main__":
    main()
